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

/// The one moment to cut, or `None` when this person never talks alone for long
/// enough to be worth hearing.
///
/// `theirs` is every speaker id that resolves to this person, merges included.
/// `segments` is the whole meeting, not just theirs — the other people's
/// segments are what "alone" is judged against.
pub fn pick_window(segments: &[Segment], theirs: &BTreeSet<String>) -> Option<Pick> {
    let final_with_speaker = |s: &&Segment| -> bool {
        s.is_final && s.speaker_id.is_some() && s.t_end_ms > s.t_start_ms
    };
    let is_theirs = |s: &Segment| -> bool {
        s.speaker_id
            .as_deref()
            .is_some_and(|id| theirs.contains(id))
    };

    let others: Vec<&Segment> = segments
        .iter()
        .filter(final_with_speaker)
        .filter(|s| !is_theirs(s))
        .collect();

    let alone = |s: &Segment| -> bool {
        !others.iter().any(|other| {
            other.channel == s.channel
                && other.t_start_ms < s.t_end_ms
                && other.t_end_ms > s.t_start_ms
        })
    };

    // Longest, alone, confident — in that order of importance, and the earliest
    // of equals so the same meeting always gives back the same clip.
    let best = segments
        .iter()
        .filter(final_with_speaker)
        .filter(|s| is_theirs(s))
        .filter(|s| s.t_end_ms - s.t_start_ms >= MIN_CLIP_MS)
        .max_by_key(|s| {
            (
                alone(s),
                s.t_end_ms - s.t_start_ms,
                (s.avg_confidence.unwrap_or(0.0) * 1_000.0) as i64,
                -s.t_start_ms,
            )
        })?;

    Some(window_within(best))
}

/// The middle of a stretch, minus its opening, capped at [`CLIP_MS`].
fn window_within(segment: &Segment) -> Pick {
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

    // Long enough to choose within: sit the clip in the middle, where a
    // sentence is at its most sentence-like.
    let middle = (from + segment.t_end_ms) / 2;
    let from_ms = (middle - CLIP_MS / 2).max(from);
    Pick {
        channel: segment.channel,
        from_ms,
        to_ms: from_ms + CLIP_MS,
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

    let Some(pick) = pick_window(&segments, &theirs) else {
        return Err(DiarizeError::NoVoiceSample);
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

    #[tokio::test]
    async fn a_speaker_nobody_has_ever_heard_is_turned_down_politely() {
        let dir = tempfile::tempdir().unwrap();
        let (db, meeting_id, _) = meeting_with_one_voice(dir.path(), true).await;
        let silent = repo::upsert_speaker(&db, &meeting_id, "speaker-02", "Speaker 2", false)
            .await
            .unwrap();

        let err = speaker_sample(&db, &meeting_id, &silent.id)
            .await
            .expect_err("nothing is attributed to this speaker");
        assert!(matches!(err, DiarizeError::NoVoiceSample), "{err:?}");
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
