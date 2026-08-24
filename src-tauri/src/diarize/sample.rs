//! A few seconds of one person's voice, so a name can be put to it.
//!
//! "Speaker 2" is a label nobody can check. The only way to know whether
//! Speaker 2 is Priya or Tom is to hear them, which is why this exists: pick the
//! best moment where one person is talking **on their own**, cut a short clip
//! out of the middle of it, and hand it over as a playable file.
//!
//! What makes a moment the best one:
//!
//! * **Theirs.** Segments attributed to that speaker, following merges — after
//!   "these two are the same person", both rows' segments are that person's.
//! * **Final.** Live partial text moves around as the pass revises it; the
//!   offline result is the one that owns who said what.
//! * **Alone.** A clip with two people talking over each other teaches nobody
//!   whose voice is whose. "Alone" is judged per channel, because the clip is cut
//!   from one channel only: someone on the far end talking at the same time as
//!   the person at the keyboard is on a different recording and cannot be heard
//!   in this one.
//! * **Long.** The longest such stretch is the one most likely to be a proper
//!   sentence rather than "mm-hm". Ties go to the more confident segment.
//!
//! Then the clip is cut from the *middle* of that stretch, skipping
//! [`LEAD_IN_MS`] at the start: the first fraction of a second of a turn is
//! where the click, the breath and the tail of whoever spoke before it live.
//! Length is capped at [`CLIP_MS`] — this is a sample to recognise a voice by,
//! not a way to re-listen to the meeting.
//!
//! Nothing is written to disk and nothing is cached: the clip is built in
//! memory, handed to the UI, and forgotten (mantra 1). The audio itself never
//! reaches the log — not the samples, not the words, not even how loud it was.

use std::collections::{BTreeSet, HashMap};
use std::io::Cursor;

use crate::audio::TARGET_SAMPLE_RATE;
use crate::db::{repo, Db};
use crate::types::{Channel, Segment, TranscriptQuery};

use super::pcm::ChunkPcm;
use super::DiarizeError;

/// Longest clip Echo will hand over. Enough to recognise a voice by; short
/// enough that it is obviously not a way to re-listen to the meeting.
pub const CLIP_MS: i64 = 6_000;

/// Skipped at the start of the chosen stretch — the click, the breath, and the
/// tail of whoever was talking a moment ago.
pub const LEAD_IN_MS: i64 = 300;

/// Shorter than this and there is no voice to recognise, only a syllable. Such a
/// segment is not a candidate at all.
pub const MIN_CLIP_MS: i64 = 700;

/// Below this peak the "audio" is silence, whatever the transcript remembers
/// happening there — a chunk that is no longer on disk reads back as a
/// perfectly quiet window of exactly the right length.
const SILENCE_FLOOR: f32 = 1e-4;

/// How many alias hops to follow before giving up, matching
/// `repo::resolve_speaker`.
const MAX_ALIAS_HOPS: usize = 8;

/// Where the clip is cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pick {
    /// Which recording it lives on — the clip is cut from this channel only.
    pub channel: Channel,
    pub from_ms: i64,
    pub to_ms: i64,
}

impl Pick {
    pub fn duration_ms(&self) -> i64 {
        (self.to_ms - self.from_ms).max(0)
    }
}

/// Least distance in time between two clips of the same voice before they stop
/// counting as separate moments.
///
/// Only used when several samples are being kept ([`pick_windows`]). Two clips
/// cut from ten seconds apart are the same sentence, the same sound of the room
/// and very nearly the same fingerprint; a profile made of those has one
/// observation in it however many rows it has.
pub const MIN_SPACING_MS: i64 = 10_000;

/// How short a clip may be and still be worth *playing back*.
///
/// Strictly a playback floor: [`MIN_CLIP_MS`] is what a clip must be before it
/// is worth teaching a profile from, and that bar does not move. But somebody
/// who said one short word-group under their name still deserves to hear which
/// voice the row belongs to, and four hundred milliseconds answers "is this
/// Priya or Tom" better than an error does. Never used for learning.
pub const PLAYABLE_CLIP_MS: i64 = 400;

/// How far the sliding clip window moves per step when placing a cut inside an
/// overlapped stretch. A quarter second is far finer than any real overlap
/// boundary and keeps the search trivially cheap.
const PLACEMENT_STEP_MS: i64 = 250;

/// Where everybody else is talking: `(channel, from_ms, to_ms)` for every final
/// attributed stretch that does not belong to `theirs`.
///
/// This is what "alone" is judged against, here and in [`ranked_candidates`] —
/// per channel, because someone on the far end talking while the person at the
/// keyboard talks is on a different recording and cannot be heard in this one.
fn busy_intervals(segments: &[Segment], theirs: &BTreeSet<String>) -> Vec<(Channel, i64, i64)> {
    segments
        .iter()
        .filter(|s| s.is_final && s.t_end_ms > s.t_start_ms)
        .filter(|s| {
            !s.speaker_id
                .as_deref()
                .is_some_and(|id| theirs.contains(id))
        })
        .map(|s| (s.channel, s.t_start_ms, s.t_end_ms))
        .collect()
}

/// The candidate segments of one voice at least `min_ms` long, best first.
///
/// "Best" is the ordering [`pick_window`] has always used: alone beats
/// overlapped, longer beats shorter, more confident beats less, and the earliest
/// of equals wins so the same meeting always answers the same way.
fn ranked_candidates_at_least<'a>(
    segments: &'a [Segment],
    theirs: &BTreeSet<String>,
    min_ms: i64,
) -> Vec<&'a Segment> {
    let final_with_speaker = |s: &&Segment| -> bool {
        s.is_final && s.speaker_id.is_some() && s.t_end_ms > s.t_start_ms
    };
    let is_theirs = |s: &Segment| -> bool {
        s.speaker_id
            .as_deref()
            .is_some_and(|id| theirs.contains(id))
    };

    let busy = busy_intervals(segments, theirs);

    // "Alone" is judged per channel: someone on the far end talking at the same
    // time as the person at the keyboard is on a different recording and cannot
    // be heard in this one.
    let alone = |s: &Segment| -> bool {
        !busy.iter().any(|(channel, start, end)| {
            *channel == s.channel && *start < s.t_end_ms && *end > s.t_start_ms
        })
    };

    let mut out: Vec<&Segment> = segments
        .iter()
        .filter(final_with_speaker)
        .filter(|s| is_theirs(s))
        .filter(|s| s.t_end_ms - s.t_start_ms >= min_ms)
        .collect();
    out.sort_by_key(|s| {
        std::cmp::Reverse((
            alone(s),
            s.t_end_ms - s.t_start_ms,
            (s.avg_confidence.unwrap_or(0.0) * 1_000.0) as i64,
            -s.t_start_ms,
        ))
    });
    out
}

/// Why there is no moment of this voice to hand over.
///
/// Two different answers, and they matter to the person asking. "No clear
/// moment" means Echo looked through this voice's lines and none of them is a
/// long enough stretch of them talking alone — a fact about how the meeting
/// went. [`DiarizeError::NoLines`] means there were no lines to look through:
/// the row exists but nothing in the transcript is on it, which on 2026-08-24
/// was every third and fourth speaker of a meeting forced to four people.
/// Saying "no clear moment" for that sends somebody looking for a recording
/// fault that is not there.
///
/// Called only once the answer is already known to be no.
pub fn why_no_moment(segments: &[Segment], theirs: &BTreeSet<String>) -> DiarizeError {
    let has_lines = segments.iter().any(|s| {
        s.is_final
            && s.speaker_id
                .as_deref()
                .is_some_and(|id| theirs.contains(id))
    });
    if has_lines {
        DiarizeError::NoVoiceSample
    } else {
        DiarizeError::NoLines
    }
}

/// The one moment to cut, or `None` when this person never talks for long
/// enough to be worth hearing.
///
/// `theirs` is every speaker id that resolves to this person, merges included.
/// `segments` is the whole meeting, not just theirs — the other people's
/// segments are what "alone" is judged against, and what a clip dodges inside
/// an overlapped stretch ([`window_within`]).
pub fn pick_window(segments: &[Segment], theirs: &BTreeSet<String>) -> Option<Pick> {
    pick_window_at_least(segments, theirs, MIN_CLIP_MS)
}

fn pick_window_at_least(
    segments: &[Segment],
    theirs: &BTreeSet<String>,
    min_ms: i64,
) -> Option<Pick> {
    let busy = busy_intervals(segments, theirs);
    ranked_candidates_at_least(segments, theirs, min_ms)
        .first()
        .map(|best| window_within(best, &busy))
}

/// Up to `max` moments of this voice, chosen to be *different* moments.
///
/// [`pick_window`] answers "let me hear this person"; this answers "keep enough
/// of this voice to recognise it again", which is a different question and wants
/// a different answer. What Echo remembers of a voice has to span the ways that
/// voice arrives, so:
///
/// * **Both conditions first.** The best candidate on each channel is taken
///   before anything else, because the same person in the room and the same
///   person down a call are two different sounds and a profile that only knows
///   one of them is half a profile (DESIGN §1: "keeping samples from different
///   conditions").
/// * **Then spread through the meeting.** The rest are filled in best-first,
///   skipping anything within [`MIN_SPACING_MS`] of a moment already taken.
/// * **Then, only if that left too few, close together.** A short meeting with
///   one long turn in it should still contribute something.
///
/// Deterministic: the same meeting always yields the same picks in the same
/// order, which is what makes the adaptation tests possible at all.
pub fn pick_windows(segments: &[Segment], theirs: &BTreeSet<String>, max: usize) -> Vec<Pick> {
    if max == 0 {
        return Vec::new();
    }
    /// Take this segment's window unless we are full, have it already, or it
    /// sits on top of a moment already taken.
    fn take<'a>(
        seg: &'a Segment,
        max: usize,
        spacing: i64,
        busy: &[(Channel, i64, i64)],
        picked: &mut Vec<Pick>,
        taken: &mut Vec<&'a str>,
    ) {
        if picked.len() >= max || taken.contains(&seg.id.as_str()) {
            return;
        }
        let window = window_within(seg, busy);
        let crowded = picked
            .iter()
            .any(|p| p.channel == window.channel && (p.from_ms - window.from_ms).abs() < spacing);
        if crowded {
            return;
        }
        picked.push(window);
        taken.push(seg.id.as_str());
    }

    let ranked = ranked_candidates_at_least(segments, theirs, MIN_CLIP_MS);
    let busy = busy_intervals(segments, theirs);
    let mut picked: Vec<Pick> = Vec::new();
    let mut taken: Vec<&str> = Vec::new();

    for channel in [Channel::Mic, Channel::System] {
        if let Some(best) = ranked.iter().find(|s| s.channel == channel) {
            take(best, max, MIN_SPACING_MS, &busy, &mut picked, &mut taken);
        }
    }
    for spacing in [MIN_SPACING_MS, 0] {
        for seg in &ranked {
            take(seg, max, spacing, &busy, &mut picked, &mut taken);
        }
    }
    picked
}

/// Where to cut inside one stretch.
///
/// A short stretch is taken whole (minus its opening); a long one is placed by
/// what the clip will contain. In a stretch nobody talks over, that is the
/// middle, where a sentence is at its most sentence-like. In an overlapped
/// stretch the window slides across and lands where the most of [`CLIP_MS`] is
/// this person alone — there is no point teaching or playing a profile clip
/// half full of somebody else when a cleaner quarter of the same turn exists.
fn window_within(segment: &Segment, busy: &[(Channel, i64, i64)]) -> Pick {
    let lead_in = if segment.t_end_ms - segment.t_start_ms > LEAD_IN_MS + MIN_CLIP_MS {
        LEAD_IN_MS
    } else {
        // Trimming this one would leave less voice than it is worth. Take it
        // whole rather than handing back a syllable.
        0
    };
    let from = segment.t_start_ms + lead_in;
    let usable = segment.t_end_ms - from;

    if usable <= CLIP_MS {
        return Pick {
            channel: segment.channel,
            from_ms: from,
            to_ms: segment.t_end_ms,
        };
    }

    // How much of [a, b) is free of everybody else on this channel.
    let solo_ms = |a: i64, b: i64| -> i64 {
        let covered = busy
            .iter()
            .filter(|(channel, s, e)| *channel == segment.channel && *s < b && *e > a)
            .map(|(_, s, e)| b.min(*e) - a.max(*s))
            .sum::<i64>();
        (CLIP_MS - covered).max(0)
    };

    // Slide across the stretch. The untouched-stretch answer — centred — is
    // where the search starts, and among equally clean windows the one nearest
    // it wins; the tail-aligned window is always considered too, because the
    // step grid can miss the one place the overlap ends exactly.
    let middle = (from + segment.t_end_ms) / 2;
    let last_start = segment.t_end_ms - CLIP_MS;
    let ideal_start = (middle - CLIP_MS / 2).clamp(from, last_start);
    let mut best_from = ideal_start;
    let mut best_solo = solo_ms(best_from, best_from + CLIP_MS);
    // Every step-grid window, plus the tail-aligned one the grid can miss.
    for candidate in (from..=last_start)
        .step_by(PLACEMENT_STEP_MS as usize)
        .chain(std::iter::once(last_start))
    {
        let solo = solo_ms(candidate, candidate + CLIP_MS);
        let nearer = (candidate - ideal_start).abs() < (best_from - ideal_start).abs();
        if solo > best_solo || (solo == best_solo && nearer) {
            best_solo = solo;
            best_from = candidate;
        }
    }
    Pick {
        channel: segment.channel,
        from_ms: best_from,
        to_ms: best_from + CLIP_MS,
    }
}

/// Every speaker id in this meeting that means the same person as `speaker_id`.
///
/// Merges are non-destructive — the merged row keeps its segments and points at
/// the survivor — so the person's voice is spread across every id in their alias
/// chain, and a clip should be allowed to come from any of them.
///
/// `None` when that id is not one of this meeting's speakers at all.
pub fn same_person(
    speakers: &[crate::types::Speaker],
    speaker_id: &str,
) -> Option<BTreeSet<String>> {
    if !speakers.iter().any(|s| s.id == speaker_id) {
        return None;
    }
    let alias_of: HashMap<&str, &str> = speakers
        .iter()
        .filter_map(|s| s.alias_of.as_deref().map(|to| (s.id.as_str(), to)))
        .collect();

    let resolve = |start: &str| -> String {
        let mut current = start;
        for _ in 0..MAX_ALIAS_HOPS {
            match alias_of.get(current) {
                Some(next) if *next != current => current = next,
                _ => break,
            }
        }
        current.to_string()
    };

    let root = resolve(speaker_id);
    Some(
        speakers
            .iter()
            .filter(|s| resolve(&s.id) == root)
            .map(|s| s.id.clone())
            .collect(),
    )
}

/// A 16 kHz mono WAV, in memory.
///
/// The same shape of file the rest of Echo writes, built here rather than on
/// disk because this one exists for as long as it takes to play it.
pub fn encode_wav_16k_mono(samples: &[f32]) -> Result<Vec<u8>, DiarizeError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| DiarizeError::Failed(format!("could not start a clip: {e}")))?;
        for sample in samples {
            writer
                .write_sample(to_i16(*sample))
                .map_err(|e| DiarizeError::Failed(format!("could not write a clip: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| DiarizeError::Failed(format!("could not finish a clip: {e}")))?;
    }
    Ok(cursor.into_inner())
}

/// Read a 16 kHz mono WAV back out of memory.
///
/// The inverse of [`encode_wav_16k_mono`], for the clips kept on a person's
/// samples: they went into the database as files and come back out as audio to
/// be fingerprinted again when the network changes. Written here rather than
/// through `audio::writer::read_wav_16k_mono` because there is no file — the
/// bytes never touch the disk on the way through.
pub fn decode_wav_16k_mono(bytes: &[u8]) -> Result<Vec<f32>, DiarizeError> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| DiarizeError::Failed(format!("could not read a stored clip: {e}")))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| DiarizeError::Failed(format!("could not read a stored clip: {e}")))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| DiarizeError::Failed(format!("could not read a stored clip: {e}")))?
                .into_iter()
                .map(|s| s as f32 * scale)
                .collect()
        }
    };
    let channels = usize::from(spec.channels.max(1));
    if channels == 1 {
        return Ok(raw);
    }
    Ok(raw
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect())
}

/// Cut the audio for several picks, opening each channel's recording once.
///
/// Picks whose audio is no longer on disk, or which read back as silence, are
/// dropped rather than returned as broken clips — the same judgement
/// [`speaker_sample`] makes, applied per pick, because one missing chunk in the
/// middle of a meeting should not cost the other three samples.
pub async fn clips_for(
    db: &Db,
    meeting_id: &str,
    picks: &[Pick],
) -> Result<Vec<(Pick, Vec<f32>)>, DiarizeError> {
    let mut out: Vec<(Pick, Vec<f32>)> = Vec::with_capacity(picks.len());
    for channel in [Channel::Mic, Channel::System] {
        if !picks.iter().any(|p| p.channel == channel) {
            continue;
        }
        let chunks = repo::list_chunks(db, meeting_id, Some(channel))
            .await
            .map_err(|e| DiarizeError::Failed(e.to_string()))?;
        let mut pcm = ChunkPcm::new(chunks);
        if pcm.is_empty() {
            continue;
        }
        for pick in picks.iter().filter(|p| p.channel == channel) {
            let Ok(samples) = pcm.window(pick.from_ms, pick.duration_ms()).await else {
                continue;
            };
            let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            if peak < SILENCE_FLOOR {
                continue;
            }
            out.push((*pick, samples));
        }
        pcm.release();
    }
    // Back into the order they were asked for, so "the best moment" stays first.
    out.sort_by_key(|(pick, _)| picks.iter().position(|p| p == pick).unwrap_or(usize::MAX));
    Ok(out)
}

/// Clamped, so a sum that went past full scale distorts rather than wrapping
/// around into a click.
fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Standard base64, because the clip travels to the UI as text inside a data
/// URL. Written out here rather than pulled in as a dependency: it is a table
/// and twenty lines, and this is the only thing in Echo that needs it.
pub fn to_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let b0 = group[0] as u32;
        let b1 = *group.get(1).unwrap_or(&0) as u32;
        let b2 = *group.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 0x3f] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 0x3f] as char);
        out.push(if group.len() > 1 {
            ALPHABET[(triple >> 6) as usize & 0x3f] as char
        } else {
            '='
        });
        out.push(if group.len() > 2 {
            ALPHABET[triple as usize & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

/// A short clip of one person talking alone, as base64 WAV.
///
/// Refuses — politely, and as a distinct error the UI can word properly —
/// when there is no such moment or when the audio is no longer on disk. Both
/// are ordinary states, not faults: a meeting whose recording was deleted still
/// has its transcript and its speakers.
pub async fn speaker_sample(
    db: &Db,
    meeting_id: &str,
    speaker_id: &str,
) -> Result<String, DiarizeError> {
    let speakers = repo::list_speakers(db, meeting_id)
        .await
        .map_err(|e| DiarizeError::Failed(e.to_string()))?;
    // A speaker that does not belong to this meeting and a speaker who never
    // talks alone come to the same thing for the person asking: there is no clip
    // to play, and nothing they could do differently.
    let Some(theirs) = same_person(&speakers, speaker_id) else {
        return Err(DiarizeError::NoVoiceSample);
    };

    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            limit: Some(50_000),
            ..Default::default()
        },
    )
    .await
    .map_err(|e| DiarizeError::Failed(e.to_string()))?;

    // Playback may go shorter than a profile clip: somebody who said one short
    // thing under their name still deserves to hear the voice. The learning
    // path keeps the strict floor; this is only ever a listen.
    let pick = match pick_window(&segments, &theirs) {
        Some(pick) => pick,
        None => match pick_window_at_least(&segments, &theirs, PLAYABLE_CLIP_MS) {
            Some(pick) => pick,
            None => return Err(why_no_moment(&segments, &theirs)),
        },
    };

    let chunks = repo::list_chunks(db, meeting_id, Some(pick.channel))
        .await
        .map_err(|e| DiarizeError::Failed(e.to_string()))?;
    let mut pcm = ChunkPcm::new(chunks);
    if pcm.is_empty() {
        return Err(DiarizeError::AudioForgotten);
    }

    let samples = pcm
        .window(pick.from_ms, pick.duration_ms())
        .await
        .map_err(|error| {
            // The words are still there, the audio is not. Nothing about the
            // clip itself goes to the log — only why there isn't one.
            tracing::debug!(%error, "could not read a voice sample off disk");
            DiarizeError::AudioForgotten
        })?;

    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak < SILENCE_FLOOR {
        // A window with nothing behind it comes back as silence of exactly the
        // right length, which would play as a broken clip rather than as a
        // missing one.
        return Err(DiarizeError::AudioForgotten);
    }

    tracing::debug!(
        channel = pick.channel.as_str(),
        ms = pick.duration_ms(),
        "handing over a voice sample"
    );
    Ok(to_base64(&encode_wav_16k_mono(&samples)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Speaker;

    fn segment(id: &str, speaker: &str, channel: Channel, from: i64, to: i64) -> Segment {
        Segment {
            id: id.into(),
            meeting_id: "m".into(),
            t_start_ms: from,
            t_end_ms: to,
            channel,
            speaker_id: Some(speaker.into()),
            text: String::new(),
            language: None,
            avg_confidence: Some(0.9),
            revision: 1,
            is_final: true,
            model_name: None,
            model_revision: None,
        }
    }

    fn theirs(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn speaker(id: &str, alias_of: Option<&str>) -> Speaker {
        Speaker {
            id: id.into(),
            meeting_id: "m".into(),
            cluster_key: id.into(),
            display_name: id.into(),
            alias_of: alias_of.map(|s| s.to_string()),
            is_self: false,
            speaking_ms: 0,
            person_id: None,
            suggested_person_id: None,
            suggestion_score: None,
        }
    }

    // -- picking the moment ------------------------------------------------

    #[test]
    fn the_longest_solo_stretch_wins() {
        let segments = vec![
            segment("a", "s1", Channel::Mic, 0, 4_000),
            segment("b", "s1", Channel::Mic, 10_000, 30_000),
            segment("c", "s1", Channel::Mic, 40_000, 45_000),
        ];
        let pick =
            pick_window(&segments, &theirs(&["s1"])).expect("there is plenty to choose from");
        assert!(
            pick.from_ms >= 10_000 && pick.to_ms <= 30_000,
            "the clip has to come out of the longest stretch, got {pick:?}"
        );
    }

    #[test]
    fn a_longer_stretch_with_someone_talking_over_it_loses_to_a_shorter_solo_one() {
        let segments = vec![
            // Twenty seconds, but the other person is talking across all of it.
            segment("a", "s1", Channel::Mic, 10_000, 30_000),
            segment("b", "s2", Channel::Mic, 12_000, 29_000),
            // Five clean seconds.
            segment("c", "s1", Channel::Mic, 60_000, 65_000),
        ];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert!(
            pick.from_ms >= 60_000,
            "a clip with two voices in it teaches nobody whose voice is whose, \
             got {pick:?}"
        );
    }

    #[test]
    fn someone_on_the_other_recording_is_not_talking_over_them() {
        // The far end and the person at the keyboard are on different channels,
        // so the clip cut from one cannot contain the other.
        let segments = vec![
            segment("a", "s1", Channel::Mic, 10_000, 30_000),
            segment("b", "s2", Channel::System, 10_000, 30_000),
        ];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert_eq!(pick.channel, Channel::Mic);
        assert!(pick.from_ms >= 10_000 && pick.to_ms <= 30_000);
    }

    #[test]
    fn the_clip_skips_the_start_and_sits_in_the_middle() {
        let segments = vec![segment("a", "s1", Channel::Mic, 10_000, 40_000)];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert_eq!(pick.duration_ms(), CLIP_MS, "capped, not the whole turn");
        assert!(
            pick.from_ms >= 10_000 + LEAD_IN_MS,
            "the opening of a turn is the throat-clearing part"
        );
        // Near the middle of the stretch, not hugging either end.
        let middle = (10_000 + 40_000) / 2;
        assert!((pick.from_ms + pick.to_ms) / 2 - middle < 500);
    }

    #[test]
    fn a_short_turn_is_taken_whole_rather_than_trimmed_to_nothing() {
        let segments = vec![segment("a", "s1", Channel::Mic, 1_000, 1_900)];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert_eq!(pick.from_ms, 1_000);
        assert_eq!(pick.to_ms, 1_900);
    }

    #[test]
    fn a_stretch_a_little_longer_than_the_lead_in_still_loses_its_opening() {
        let segments = vec![segment("a", "s1", Channel::Mic, 0, 3_000)];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert_eq!(pick.from_ms, LEAD_IN_MS);
        assert_eq!(pick.to_ms, 3_000);
    }

    /// A long stretch with somebody over part of it places the clip where the
    /// most of it is this person alone, rather than blindly in the middle.
    #[test]
    fn an_overlapped_stretch_places_the_clip_where_the_voice_is_alone() {
        let segments = vec![
            // Twenty-four seconds of s1; s2 talks over the middle ten.
            segment("a", "s1", Channel::Mic, 8_000, 32_000),
            segment("b", "s2", Channel::Mic, 14_000, 24_000),
        ];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        let overlaps_s2 = pick.to_ms > 14_000 && pick.from_ms < 24_000;
        assert!(
            !overlaps_s2,
            "eight clean seconds exist on the right; got {pick:?}"
        );
    }

    /// Playback may go shorter than a profile clip: one short line under the
    /// name is still something to listen to.
    #[test]
    fn playback_falls_back_to_a_shorter_moment_than_learning_takes() {
        let segments = vec![segment("a", "s1", Channel::Mic, 0, 500)];
        assert_eq!(pick_window(&segments, &theirs(&["s1"])), None);
        let pick =
            pick_window_at_least(&segments, &theirs(&["s1"]), PLAYABLE_CLIP_MS).expect("playable");
        assert_eq!(pick.from_ms, 0);
        assert_eq!(pick.to_ms, 500);
    }

    #[test]
    fn a_syllable_is_not_a_voice_sample() {
        let segments = vec![
            segment("a", "s1", Channel::Mic, 0, 200),
            segment("b", "s1", Channel::Mic, 1_000, 1_300),
        ];
        assert!(pick_window(&segments, &theirs(&["s1"])).is_none());
    }

    #[test]
    fn live_text_is_never_what_a_clip_is_cut_from() {
        let mut partial = segment("a", "s1", Channel::Mic, 0, 30_000);
        partial.is_final = false;
        let segments = vec![partial, segment("b", "s1", Channel::Mic, 60_000, 62_000)];
        let pick = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert!(pick.from_ms >= 60_000, "got {pick:?}");
    }

    #[test]
    fn a_speaker_with_nothing_attributed_to_them_has_no_clip() {
        let segments = vec![segment("a", "s2", Channel::Mic, 0, 30_000)];
        assert!(pick_window(&segments, &theirs(&["s1"])).is_none());
        assert!(pick_window(&[], &theirs(&["s1"])).is_none());
    }

    #[test]
    fn the_more_confident_of_two_equal_stretches_wins() {
        let mut quiet = segment("a", "s1", Channel::Mic, 0, 10_000);
        quiet.avg_confidence = Some(0.3);
        let mut clear = segment("b", "s1", Channel::Mic, 20_000, 30_000);
        clear.avg_confidence = Some(0.95);
        let pick = pick_window(&[quiet, clear], &theirs(&["s1"])).unwrap();
        assert!(pick.from_ms >= 20_000, "got {pick:?}");
    }

    #[test]
    fn the_same_meeting_always_gives_back_the_same_clip() {
        let segments = vec![
            segment("a", "s1", Channel::Mic, 0, 10_000),
            segment("b", "s1", Channel::Mic, 20_000, 30_000),
        ];
        let first = pick_window(&segments, &theirs(&["s1"])).unwrap();
        let again = pick_window(&segments, &theirs(&["s1"])).unwrap();
        assert_eq!(first, again);
        assert!(first.from_ms < 10_000, "ties go to the earlier stretch");
    }

    // -- merged speakers ---------------------------------------------------

    #[test]
    fn merged_speakers_are_one_persons_voice() {
        let speakers = vec![
            speaker("s1", None),
            speaker("s2", Some("s1")),
            speaker("s3", None),
        ];
        let both = same_person(&speakers, "s2").expect("s2 belongs to this meeting");
        assert_eq!(both, theirs(&["s1", "s2"]));
        // Asking about the survivor gives the same answer as asking about the
        // row that was merged into it.
        assert_eq!(same_person(&speakers, "s1").unwrap(), both);
        assert_eq!(same_person(&speakers, "s3").unwrap(), theirs(&["s3"]));
    }

    #[test]
    fn a_clip_can_come_from_either_half_of_a_merge() {
        let speakers = vec![speaker("s1", None), speaker("s2", Some("s1"))];
        let theirs = same_person(&speakers, "s1").unwrap();
        let segments = vec![
            segment("a", "s1", Channel::Mic, 0, 3_000),
            segment("b", "s2", Channel::Mic, 10_000, 30_000),
        ];
        let pick = pick_window(&segments, &theirs).unwrap();
        assert!(pick.from_ms >= 10_000, "got {pick:?}");
    }

    #[test]
    fn a_speaker_from_another_meeting_is_not_this_meetings_speaker() {
        let speakers = vec![speaker("s1", None)];
        assert!(same_person(&speakers, "somebody-else").is_none());
    }

    #[test]
    fn a_ring_of_aliases_cannot_spin_forever() {
        let speakers = vec![speaker("s1", Some("s2")), speaker("s2", Some("s1"))];
        let resolved = same_person(&speakers, "s1").expect("s1 is in this meeting");
        assert!(!resolved.is_empty());
    }

    // -- the file that comes out -------------------------------------------

    #[test]
    fn the_clip_is_a_wav_a_browser_will_play() {
        let samples = vec![0.25f32; TARGET_SAMPLE_RATE as usize]; // one second
        let wav = encode_wav_16k_mono(&samples).unwrap();

        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        let channels = u16::from_le_bytes([wav[22], wav[23]]);
        let rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        let bits = u16::from_le_bytes([wav[34], wav[35]]);
        assert_eq!(channels, 1);
        assert_eq!(rate, TARGET_SAMPLE_RATE);
        assert_eq!(bits, 16);

        // The header claims exactly as many bytes as the file carries.
        let riff_size = u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]);
        assert_eq!(riff_size as usize, wav.len() - 8);
        // One second of 16-bit mono at 16 kHz, plus a header of some size.
        assert!(wav.len() > samples.len() * 2);
    }

    #[test]
    fn an_empty_clip_is_still_a_valid_file() {
        let wav = encode_wav_16k_mono(&[]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert!(wav.len() >= 44);
    }

    #[test]
    fn loud_audio_is_clamped_rather_than_wrapped() {
        // A mixed-down window can sum past full scale; wrapping would turn that
        // into a bang.
        assert_eq!(to_i16(2.0), i16::MAX);
        assert_eq!(to_i16(-2.0), -i16::MAX);
        assert_eq!(to_i16(0.0), 0);
    }

    #[test]
    fn base64_matches_the_standard_alphabet_and_padding() {
        assert_eq!(to_base64(b""), "");
        assert_eq!(to_base64(b"f"), "Zg==");
        assert_eq!(to_base64(b"fo"), "Zm8=");
        assert_eq!(to_base64(b"foo"), "Zm9v");
        assert_eq!(to_base64(b"foob"), "Zm9vYg==");
        assert_eq!(to_base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(to_base64(b"foobar"), "Zm9vYmFy");
        // Bytes above 0x7f, which is most of a WAV.
        assert_eq!(to_base64(&[0xff, 0xfe, 0xfd]), "//79");
        assert_eq!(to_base64(&[0x00, 0x00, 0x00]), "AAAA");
    }

    #[test]
    fn a_clip_is_padded_to_a_whole_number_of_quads() {
        let wav = encode_wav_16k_mono(&[0.1, -0.1, 0.2]).unwrap();
        let encoded = to_base64(&wav);
        assert_eq!(encoded.len() % 4, 0);
        assert!(
            encoded.starts_with("UklGR"),
            "RIFF, in base64: {encoded:.8}"
        );
    }

    // -- end to end, against a real database and real files ----------------

    use crate::db::connect_in_memory;
    use crate::types::{MeetingStatus, SegmentDraft};

    /// A meeting with one long solo turn, and optionally the audio behind it.
    async fn meeting_with_one_voice(
        dir: &std::path::Path,
        with_audio: bool,
    ) -> (Db, String, String) {
        let db = connect_in_memory().await.unwrap();
        let meeting = repo::create_meeting(&db, "Weekly sync", dir.to_str().unwrap(), None)
            .await
            .unwrap();
        let speaker = repo::upsert_speaker(&db, &meeting.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();

        repo::insert_segments(
            &db,
            &[SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 10_000,
                t_end_ms: 40_000,
                channel: Channel::Mic,
                speaker_id: Some(speaker.id.clone()),
                text: "the part that never goes anywhere near the log".into(),
                revision: 1,
                is_final: true,
                ..Default::default()
            }],
        )
        .await
        .unwrap();

        if with_audio {
            // One chunk covering the whole turn, with something audible in it.
            let path = dir.join("mic-000000.wav");
            let tone: Vec<f32> = (0..(TARGET_SAMPLE_RATE as usize * 60))
                .map(|i| ((i as f32) * 0.05).sin() * 0.5)
                .collect();
            crate::audio::writer::write_wav_16k_mono(&path, &tone).unwrap();
            let id = repo::insert_chunk(
                &db,
                &meeting.id,
                Channel::Mic,
                0,
                path.to_str().unwrap(),
                0,
                60_000,
            )
            .await
            .unwrap();
            repo::commit_chunk(&db, &id, 60_000).await.unwrap();
        }

        repo::set_meeting_status(&db, &meeting.id, MeetingStatus::Complete)
            .await
            .unwrap();
        (db, meeting.id, speaker.id)
    }

    #[tokio::test]
    async fn a_voice_sample_comes_back_as_a_playable_clip() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, speaker_id) = meeting_with_one_voice(dir.path(), true).await;

        let encoded = speaker_sample(&db, &meeting_id, &speaker_id).await.unwrap();
        assert!(
            encoded.starts_with("UklGR"),
            "the UI puts this straight into a data URL, so it has to be a WAV"
        );
        // The cap, not the whole thirty-second turn.
        let bytes = encoded.len() / 4 * 3;
        let audio = (CLIP_MS as usize * TARGET_SAMPLE_RATE as usize / 1_000) * 2;
        assert!(
            bytes > audio && bytes < audio + 200,
            "{bytes} bytes is not {CLIP_MS}ms of 16-bit mono plus a header"
        );
    }

    #[tokio::test]
    async fn a_meeting_with_no_recording_left_says_so_instead_of_playing_silence() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, speaker_id) = meeting_with_one_voice(dir.path(), false).await;

        let err = speaker_sample(&db, &meeting_id, &speaker_id)
            .await
            .expect_err("there is no audio to cut a clip from");
        assert!(matches!(err, DiarizeError::AudioForgotten), "{err:?}");
    }

    #[tokio::test]
    async fn a_chunk_that_is_no_longer_on_disk_is_a_refusal_not_a_silent_clip() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, speaker_id) = meeting_with_one_voice(dir.path(), true).await;
        // The journal still remembers the chunk; the file is gone — someone
        // cleared the folder out, or a sync tool did.
        std::fs::remove_file(dir.path().join("mic-000000.wav")).unwrap();

        let err = speaker_sample(&db, &meeting_id, &speaker_id)
            .await
            .expect_err("a clip of silence is worse than an honest no");
        assert!(matches!(err, DiarizeError::AudioForgotten), "{err:?}");
    }

    /// A row with no line of transcript on it says *that*, not "no clear
    /// moment". The pass stopped making rows like this on 2026-08-24, but a
    /// meeting from before it, or a rename, can still leave one — and telling
    /// somebody there is no clear moment of a voice that never said anything
    /// sends them hunting for a recording fault that is not there.
    #[tokio::test]
    async fn a_speaker_nobody_has_ever_heard_says_it_has_no_lines() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, _) = meeting_with_one_voice(dir.path(), true).await;
        let silent = repo::upsert_speaker(&db, &meeting_id, "speaker-02", "Speaker 2", false)
            .await
            .unwrap();

        let err = speaker_sample(&db, &meeting_id, &silent.id)
            .await
            .expect_err("nothing is attributed to this speaker");
        assert!(matches!(err, DiarizeError::NoLines), "{err:?}");
    }

    /// The other half, and the reason the two are told apart: this voice does
    /// have lines, they are just all too short to recognise anybody *from*.
    /// Learning still refuses — a profile built on a syllable teaches nothing —
    /// but playback steps down to [`PLAYABLE_CLIP_MS`], because four hundred
    /// milliseconds of somebody's voice tells a person whose row this is.
    #[tokio::test]
    async fn a_speaker_who_only_ever_says_a_syllable_can_still_be_heard_but_not_learned() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, _) = meeting_with_one_voice(dir.path(), true).await;
        let brief = repo::upsert_speaker(&db, &meeting_id, "speaker-02", "Speaker 2", false)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[SegmentDraft {
                meeting_id: meeting_id.clone(),
                t_start_ms: 41_000,
                t_end_ms: 41_000 + MIN_CLIP_MS - 100,
                channel: Channel::Mic,
                speaker_id: Some(brief.id.clone()),
                text: "mm".into(),
                revision: 1,
                is_final: true,
                ..Default::default()
            }],
        )
        .await
        .unwrap();

        let theirs = theirs(&[&brief.id]);
        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting_id.clone(),
                limit: Some(50),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // Nothing at the learning floor...
        assert_eq!(pick_window(&segments, &theirs), None);
        // ...but playback hands over something listenable.
        let clip = speaker_sample(&db, &meeting_id, &brief.id)
            .await
            .expect("a short line is still worth hearing");
        assert!(clip.starts_with("UklGR"), "a wav, playable: {clip:?}");
    }

    // -- keeping several moments, for a profile ----------------------------

    #[test]
    fn a_confirmation_keeps_a_few_different_moments_and_no_more() {
        // Six clean turns spread through the meeting; four are kept, and no two
        // of them are the same moment.
        let segments: Vec<Segment> = (0..6)
            .map(|i| {
                segment(
                    &format!("s{i}"),
                    "s1",
                    Channel::Mic,
                    i * 60_000,
                    i * 60_000 + 20_000,
                )
            })
            .collect();
        let picks = pick_windows(&segments, &theirs(&["s1"]), 4);
        assert_eq!(picks.len(), 4);
        for (i, a) in picks.iter().enumerate() {
            for b in picks.iter().skip(i + 1) {
                assert!(
                    (a.from_ms - b.from_ms).abs() >= MIN_SPACING_MS,
                    "{a:?} and {b:?} are the same moment"
                );
            }
        }
        // And the best moment is still first, so "the freshest clip" means
        // something.
        assert_eq!(picks[0], pick_window(&segments, &theirs(&["s1"])).unwrap());
    }

    #[test]
    fn a_voice_heard_both_ways_contributes_both() {
        // The three longest turns are on the microphone and there is one shorter
        // one down the call. A profile that only knew the room would be half a
        // profile, so the call one is taken before the third microphone one.
        let mut segments: Vec<Segment> = (0..3)
            .map(|i| {
                segment(
                    &format!("mic{i}"),
                    "s1",
                    Channel::Mic,
                    i * 60_000,
                    i * 60_000 + 30_000,
                )
            })
            .collect();
        segments.push(segment("call", "s1", Channel::System, 200_000, 203_000));

        let picks = pick_windows(&segments, &theirs(&["s1"]), 2);
        assert_eq!(picks.len(), 2);
        assert!(
            picks.iter().any(|p| p.channel == Channel::System),
            "{picks:?}"
        );
        assert!(picks.iter().any(|p| p.channel == Channel::Mic), "{picks:?}");
    }

    #[test]
    fn one_long_turn_is_better_than_nothing() {
        let segments = vec![segment("a", "s1", Channel::Mic, 0, 30_000)];
        assert_eq!(pick_windows(&segments, &theirs(&["s1"]), 4).len(), 1);
    }

    #[test]
    fn moments_close_together_are_only_taken_when_there_is_nothing_else() {
        // Two turns three seconds apart. Spread out they are one moment; asked for
        // two, a short meeting still gets to contribute both.
        let segments = vec![
            segment("a", "s1", Channel::Mic, 0, 2_000),
            segment("b", "s1", Channel::Mic, 3_000, 5_000),
        ];
        assert_eq!(pick_windows(&segments, &theirs(&["s1"]), 2).len(), 2);
    }

    #[test]
    fn asking_for_no_moments_reads_no_audio() {
        let segments = vec![segment("a", "s1", Channel::Mic, 0, 30_000)];
        assert!(pick_windows(&segments, &theirs(&["s1"]), 0).is_empty());
        assert!(pick_windows(&[], &theirs(&["s1"]), 4).is_empty());
    }

    #[tokio::test]
    async fn the_clips_come_back_in_the_order_they_were_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, _) = meeting_with_one_voice(dir.path(), true).await;
        let picks = vec![
            Pick {
                channel: Channel::Mic,
                from_ms: 12_000,
                to_ms: 18_000,
            },
            Pick {
                channel: Channel::Mic,
                from_ms: 30_000,
                to_ms: 36_000,
            },
            // Past the end of the recording: dropped rather than returned as six
            // seconds of silence.
            Pick {
                channel: Channel::Mic,
                from_ms: 900_000,
                to_ms: 906_000,
            },
        ];
        let clips = clips_for(&db, &meeting_id, &picks).await.unwrap();
        assert_eq!(
            clips.len(),
            2,
            "{:?}",
            clips.iter().map(|c| c.0).collect::<Vec<_>>()
        );
        assert_eq!(clips[0].0, picks[0]);
        assert_eq!(clips[1].0, picks[1]);
        let expected = (6_000 * TARGET_SAMPLE_RATE as usize) / 1_000;
        assert_eq!(clips[0].1.len(), expected);
    }

    #[tokio::test]
    async fn a_meeting_whose_audio_is_gone_hands_back_nothing_rather_than_silence() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, _) = meeting_with_one_voice(dir.path(), false).await;
        let picks = vec![Pick {
            channel: Channel::Mic,
            from_ms: 12_000,
            to_ms: 18_000,
        }];
        assert!(clips_for(&db, &meeting_id, &picks)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_speaker_id_from_somewhere_else_is_turned_down_the_same_way() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, _) = meeting_with_one_voice(dir.path(), true).await;
        let err = speaker_sample(&db, &meeting_id, "not-a-speaker-in-this-meeting")
            .await
            .expect_err("that speaker is not in this meeting");
        assert!(matches!(err, DiarizeError::NoVoiceSample), "{err:?}");
    }
}
