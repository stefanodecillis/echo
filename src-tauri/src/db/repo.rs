//! Every database operation Echo needs, one function per operation.
//!
//! Fully implemented (not a stub). Other modules must go through here instead
//! of writing SQL, so the schema stays owned in one place.
//!
//! Conventions:
//! * `Result<_, DbError>`; commands convert to `UiError` at the edge.
//! * Ids are generated here ([`new_id`]) unless the caller supplies one.
//! * Timestamps are written by [`now`] so every row uses the same format.
//! * Bulk inserts take a slice and use one transaction, never one statement
//!   per segment while a recording is live.

use std::collections::HashMap;

use sqlx::{QueryBuilder, Row, Sqlite};

use super::{Db, DbError};
use crate::types::{
    ActionItem, ActionItemPatch, AssetKind, AudioChunk, Channel, Correction, Id, Job, JobKind,
    JobPhase, JobQuery, JobStatus, Marker, MarkerKind, Meeting, MeetingDetail, MeetingQuery,
    MeetingStatus, MeetingSummary, ModelInfo, Provider, SearchHit, SearchQuery, Segment,
    SegmentDraft, Speaker, Summary, Template, TranscriptQuery,
};

/// Sentinels wrapped around FTS matches before HTML escaping, then swapped for
/// `<mark>`. Control characters never occur in transcript text.
const MARK_OPEN: &str = "\u{2}";
const MARK_CLOSE: &str = "\u{3}";

/// Fresh UUID v4, lowercase hyphenated.
pub fn new_id() -> Id {
    uuid::Uuid::new_v4().to_string()
}

/// Current time as an RFC3339 UTC string with millisecond precision.
pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// ===========================================================================
// meetings
// ===========================================================================

/// Insert the meeting row. Called **before** capture starts so a crash always
/// leaves a row to recover (DESIGN §3 Crash recovery).
pub async fn create_meeting(
    db: &Db,
    title: &str,
    audio_dir: &str,
    detected_app: Option<&str>,
) -> Result<Meeting, DbError> {
    let id = new_id();
    let started_at = now();
    sqlx::query(
        "INSERT INTO meetings (id, title, started_at, detected_app, status, audio_dir, duration_ms)
         VALUES (?1, ?2, ?3, ?4, 'created', ?5, 0)",
    )
    .bind(&id)
    .bind(title)
    .bind(&started_at)
    .bind(detected_app)
    .bind(audio_dir)
    .execute(db)
    .await?;

    get_meeting(db, &id)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("meeting {id}")))
}

pub async fn get_meeting(db: &Db, id: &str) -> Result<Option<Meeting>, DbError> {
    let row = sqlx::query(
        "SELECT id, title, started_at, ended_at, detected_app, language, status, audio_dir,
                mixed_path, duration_ms, deleted_at
         FROM meetings WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_meeting).transpose()
}

/// Home and History lists. Newest first.
pub async fn list_meetings(db: &Db, q: &MeetingQuery) -> Result<Vec<MeetingSummary>, DbError> {
    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT m.id, m.title, m.started_at, m.duration_ms, m.status, m.language,
                -- The person's own count of who was there wins over Echo's, so
                -- the list and the meeting itself never disagree about it.
                COALESCE(m.speaker_count_override,
                         (SELECT COUNT(*) FROM speakers s
                            WHERE s.meeting_id = m.id AND s.alias_of IS NULL)) AS speaker_count,
                (SELECT COUNT(*) FROM action_items a WHERE a.meeting_id = m.id) AS action_item_count,
                (SELECT COUNT(*) FROM summaries su WHERE su.meeting_id = m.id) AS summary_count,
                (SELECT COUNT(*) FROM audio_chunks c
                   WHERE c.meeting_id = m.id AND c.committed = 1) AS chunk_count,
                (SELECT g.text FROM segments g
                   WHERE g.meeting_id = m.id AND g.is_final = 1
                   ORDER BY g.t_start_ms LIMIT 1) AS first_text,
                (SELECT su2.content_md FROM summaries su2
                   WHERE su2.meeting_id = m.id
                   ORDER BY su2.created_at DESC LIMIT 1) AS recap_md
         FROM meetings m WHERE 1 = 1",
    );

    if !q.include_deleted.unwrap_or(false) {
        qb.push(" AND m.deleted_at IS NULL");
    }
    if let Some(status) = q.status {
        qb.push(" AND m.status = ")
            .push_bind(status.as_str().to_string());
    }
    if let Some(needle) = q.title_contains.as_ref().filter(|s| !s.trim().is_empty()) {
        qb.push(" AND m.title LIKE ")
            .push_bind(format!("%{}%", needle.trim().replace('%', "\\%")));
        qb.push(" ESCAPE '\\'");
    }
    qb.push(" ORDER BY m.started_at DESC LIMIT ")
        .push_bind(i64::from(q.limit.unwrap_or(50).min(500)))
        .push(" OFFSET ")
        .push_bind(i64::from(q.offset.unwrap_or(0)));

    let rows = qb.build().fetch_all(db).await?;
    rows.into_iter().map(row_to_meeting_summary).collect()
}

pub async fn count_meetings(db: &Db) -> Result<u32, DbError> {
    let (n,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM meetings WHERE deleted_at IS NULL")
        .fetch_one(db)
        .await?;
    Ok(n as u32)
}

pub async fn update_meeting_title(db: &Db, id: &str, title: &str) -> Result<(), DbError> {
    let res = sqlx::query("UPDATE meetings SET title = ?2 WHERE id = ?1")
        .bind(id)
        .bind(title)
        .execute(db)
        .await?;
    if res.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("meeting {id}")));
    }
    Ok(())
}

pub async fn set_meeting_status(db: &Db, id: &str, status: MeetingStatus) -> Result<(), DbError> {
    sqlx::query("UPDATE meetings SET status = ?2 WHERE id = ?1")
        .bind(id)
        .bind(status.as_str())
        .execute(db)
        .await?;
    Ok(())
}

/// Close out a meeting: end time, duration, dominant language, mixdown path.
pub async fn finish_meeting(
    db: &Db,
    id: &str,
    duration_ms: i64,
    language: Option<&str>,
    mixed_path: Option<&str>,
    status: MeetingStatus,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE meetings
         SET ended_at = ?2, duration_ms = ?3, language = COALESCE(?4, language),
             mixed_path = COALESCE(?5, mixed_path), status = ?6
         WHERE id = ?1",
    )
    .bind(id)
    .bind(now())
    .bind(duration_ms)
    .bind(language)
    .bind(mixed_path)
    .bind(status.as_str())
    .execute(db)
    .await?;
    Ok(())
}

/// Where this meeting's per-channel audio lives. `create_meeting` mints the id,
/// so the per-meeting directory can only be recorded once the row exists.
pub async fn set_meeting_audio_dir(db: &Db, id: &str, audio_dir: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE meetings SET audio_dir = ?2 WHERE id = ?1")
        .bind(id)
        .bind(audio_dir)
        .execute(db)
        .await?;
    Ok(())
}

/// Record the playback file the mixdown job produced, without touching
/// `ended_at` the way [`finish_meeting`] would.
pub async fn set_meeting_mixed_path(db: &Db, id: &str, mixed_path: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE meetings SET mixed_path = ?2 WHERE id = ?1")
        .bind(id)
        .bind(mixed_path)
        .execute(db)
        .await?;
    Ok(())
}

pub async fn set_meeting_language(db: &Db, id: &str, language: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE meetings SET language = ?2 WHERE id = ?1")
        .bind(id)
        .bind(language)
        .execute(db)
        .await?;
    Ok(())
}

/// Store the person's own answer to "how many people were in this meeting", or
/// clear it back to automatic with `None`.
///
/// The number is the total, counting whoever was at this computer — see
/// [`crate::diarize::remote_target`] for the arithmetic that turns it into a
/// number of remote voices, and the `0003` migration for why it is stored that
/// way round.
pub async fn set_speaker_count_override(
    db: &Db,
    id: &str,
    count: Option<u32>,
) -> Result<(), DbError> {
    let res = sqlx::query("UPDATE meetings SET speaker_count_override = ?2 WHERE id = ?1")
        .bind(id)
        .bind(count.map(i64::from))
        .execute(db)
        .await?;
    if res.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("meeting {id}")));
    }
    Ok(())
}

/// The person's correction, if they made one. `None` means automatic.
pub async fn speaker_count_override(db: &Db, id: &str) -> Result<Option<u32>, DbError> {
    let row: Option<(Option<i64>,)> =
        sqlx::query_as("SELECT speaker_count_override FROM meetings WHERE id = ?1")
            .bind(id)
            .fetch_optional(db)
            .await?;
    let Some((stored,)) = row else {
        return Err(DbError::NotFound(format!("meeting {id}")));
    };
    Ok(stored.map(|n| crate::diarize::clamp_people(n.clamp(0, i64::from(u32::MAX)) as u32)))
}

/// How many people Echo currently believes were in this meeting, and whether
/// that is the person's own answer.
///
/// Echo's own count is every speaker row that has not been merged into another
/// one: "You" plus each voice the pass separated out. An override replaces it
/// wholesale, because the control in the UI has to read back the number the
/// person typed into it.
pub async fn people_count(db: &Db, meeting_id: &str) -> Result<(u32, bool), DbError> {
    if let Some(count) = speaker_count_override(db, meeting_id).await? {
        return Ok((count, true));
    }
    Ok((count_people(db, meeting_id).await?, false))
}

/// Speaker rows for this meeting that are not merged into another one **and
/// have a line of the transcript on them**.
///
/// [`count_people`] is Echo's count of the people in the meeting, and "You" is
/// one of them whether or not the microphone caught a word — the person was
/// there. This answers the other question, the one the speaker pass reports and
/// the one "Echo can only hear N distinct voices" is about: how many voices are
/// audible in the recording. On a meeting somebody only listened to, those two
/// differ by exactly the silent "You" row, and counting it there would claim one
/// more voice than the recording holds.
///
/// A row somebody merged into another counts towards the row it was merged
/// into, so a person whose every line sits on the merged-away half is still one
/// voice. One hop, the same depth [`count_people`] reasons at.
pub async fn count_audible_people(db: &Db, meeting_id: &str) -> Result<u32, DbError> {
    let (n,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM speakers s
         WHERE s.meeting_id = ?1 AND s.alias_of IS NULL AND EXISTS (
             SELECT 1 FROM segments g
             JOIN speakers t ON t.id = g.speaker_id
             WHERE g.is_final = 1 AND (t.id = s.id OR t.alias_of = s.id)
         )",
    )
    .bind(meeting_id)
    .fetch_one(db)
    .await?;
    Ok(n as u32)
}

/// Speaker rows for this meeting that are not merged into another one.
pub async fn count_people(db: &Db, meeting_id: &str) -> Result<u32, DbError> {
    let (n,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM speakers WHERE meeting_id = ?1 AND alias_of IS NULL")
            .bind(meeting_id)
            .fetch_one(db)
            .await?;
    Ok(n as u32)
}

pub async fn set_meeting_duration(db: &Db, id: &str, duration_ms: i64) -> Result<(), DbError> {
    sqlx::query("UPDATE meetings SET duration_ms = ?2 WHERE id = ?1")
        .bind(id)
        .bind(duration_ms)
        .execute(db)
        .await?;
    Ok(())
}

/// "Delete audio, keep the text": drop the chunk journal and the mixdown
/// pointer, keep segments and recaps. Files are removed by the caller.
pub async fn forget_audio(db: &Db, meeting_id: &str) -> Result<Vec<String>, DbError> {
    let paths: Vec<(String,)> =
        sqlx::query_as("SELECT path FROM audio_chunks WHERE meeting_id = ?1")
            .bind(meeting_id)
            .fetch_all(db)
            .await?;
    let mut tx = db.begin().await?;
    sqlx::query("DELETE FROM audio_chunks WHERE meeting_id = ?1")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE meetings SET mixed_path = NULL WHERE id = ?1")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(paths.into_iter().map(|p| p.0).collect())
}

/// Ids of meetings whose capture is over (nothing recording or waiting on a
/// person), not deleted — the launch-time empty-husk sweep looks at these.
pub async fn finished_meeting_ids(db: &Db) -> Result<Vec<Id>, DbError> {
    let rows: Vec<(Id,)> = sqlx::query_as(
        "SELECT id FROM meetings
         WHERE deleted_at IS NULL AND status IN ('processing', 'complete', 'failed')",
    )
    .fetch_all(db)
    .await?;
    Ok(rows.into_iter().map(|r| r.0).collect())
}

/// Mark deleted without removing rows, so an undo is still possible.
pub async fn soft_delete_meeting(db: &Db, id: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE meetings SET deleted_at = ?2 WHERE id = ?1")
        .bind(id)
        .bind(now())
        .execute(db)
        .await?;
    Ok(())
}

/// Remove the meeting and everything derived from it. Returns every audio file
/// the caller must unlink: the per-channel chunks and the derived playback file.
///
/// The children are deleted explicitly, in one transaction, even though every
/// table cascades from `meetings`:
/// * the FTS indexes are kept in step by triggers on `segments`, and a trigger
///   firing for a row a foreign-key action removed is a SQLite build detail
///   ([`PRAGMA recursive_triggers`]) rather than something to bet a person's
///   search results on. Deleting segments here makes the trigger fire for
///   certain, so no deleted meeting can leave words behind in the index.
/// * `jobs` rows are removed with the meeting, so nothing is left queued
///   against something that no longer exists.
pub async fn delete_meeting(db: &Db, id: &str) -> Result<Vec<String>, DbError> {
    let mut paths: Vec<String> =
        sqlx::query_as("SELECT path FROM audio_chunks WHERE meeting_id = ?1")
            .bind(id)
            .fetch_all(db)
            .await?
            .into_iter()
            .map(|(p,): (String,)| p)
            .collect();
    let mixed: Option<(Option<String>,)> =
        sqlx::query_as("SELECT mixed_path FROM meetings WHERE id = ?1")
            .bind(id)
            .fetch_optional(db)
            .await?;
    if let Some(mixed) = mixed.and_then(|(p,)| p).filter(|p| !p.trim().is_empty()) {
        paths.push(mixed);
    }

    let mut tx = db.begin().await?;
    // Segments first: their delete trigger is what clears both FTS indexes.
    // Action items before summaries, speakers after segments, so no statement
    // depends on a cascade having run.
    for table in [
        "action_items",
        "summaries",
        "segments",
        "speakers",
        "markers",
        "audio_chunks",
        "jobs",
    ] {
        // `table` is one of the literals above, never user input.
        sqlx::query(&format!("DELETE FROM {table} WHERE meeting_id = ?1"))
            .bind(id)
            .execute(&mut *tx)
            .await?;
    }
    sqlx::query("DELETE FROM meetings WHERE id = ?1")
        .bind(id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;

    Ok(paths.into_iter().filter(|p| !p.is_empty()).collect())
}

/// Every meeting and the folder it was recorded into, including ones already
/// put out of the way. Used by delete-all so it can remove exactly the folders
/// Echo created and nothing else.
pub async fn all_meeting_audio_dirs(db: &Db) -> Result<Vec<(Id, String)>, DbError> {
    let rows: Vec<(String, Option<String>)> = sqlx::query_as("SELECT id, audio_dir FROM meetings")
        .fetch_all(db)
        .await?;
    Ok(rows
        .into_iter()
        .map(|(id, dir)| (id, dir.unwrap_or_default()))
        .collect())
}

/// Every audio file this database knows about: the per-channel chunks and the
/// derived playback files.
pub async fn all_audio_file_paths(db: &Db) -> Result<Vec<String>, DbError> {
    let chunks: Vec<(String,)> = sqlx::query_as("SELECT path FROM audio_chunks")
        .fetch_all(db)
        .await?;
    let mixed: Vec<(String,)> =
        sqlx::query_as("SELECT mixed_path FROM meetings WHERE mixed_path IS NOT NULL")
            .fetch_all(db)
            .await?;
    Ok(chunks
        .into_iter()
        .chain(mixed)
        .map(|(p,)| p)
        .filter(|p| !p.is_empty())
        .collect())
}

/// Delete-all, for Settings → Data.
///
/// Same reasoning as [`delete_meeting`]: the children go first and by name, so
/// the `segments` delete trigger clears the search indexes rather than leaving
/// that to whether this SQLite build fires triggers for cascaded rows.
pub async fn delete_all_meetings(db: &Db) -> Result<(), DbError> {
    let mut tx = db.begin().await?;
    for table in [
        "action_items",
        "summaries",
        "segments",
        "speakers",
        "markers",
        "audio_chunks",
        "jobs",
        "meetings",
    ] {
        sqlx::query(&format!("DELETE FROM {table}"))
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Meetings that were recording when the app went away.
pub async fn list_interrupted_meetings(db: &Db) -> Result<Vec<Meeting>, DbError> {
    let rows = sqlx::query(
        "SELECT id, title, started_at, ended_at, detected_app, language, status, audio_dir,
                mixed_path, duration_ms, deleted_at
         FROM meetings
         WHERE deleted_at IS NULL AND status IN ('recording', 'created', 'interrupted')
         ORDER BY started_at DESC",
    )
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_meeting).collect()
}

/// Everything the meeting-detail route needs in one round-trip.
pub async fn get_meeting_detail(db: &Db, id: &str) -> Result<MeetingDetail, DbError> {
    let meeting = get_meeting(db, id)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("meeting {id}")))?;

    let chunks = list_chunks(db, id, None).await?;
    let mut captured: Vec<Channel> = Vec::new();
    for c in &chunks {
        if c.channel != Channel::Mixed && !captured.contains(&c.channel) {
            captured.push(c.channel);
        }
    }

    let (people_count, people_count_is_override) = people_count(db, id).await?;

    Ok(MeetingDetail {
        speakers: list_speakers(db, id).await?,
        people_count,
        people_count_is_override,
        markers: list_markers(db, id).await?,
        summaries: list_summaries(db, id).await?,
        action_items: list_action_items(db, id).await?,
        jobs: list_jobs(
            db,
            &JobQuery {
                meeting_id: Some(id.to_string()),
                ..Default::default()
            },
        )
        .await?,
        audio_bytes: 0, // filled in by the command from the filesystem
        segment_count: count_segments(db, id).await?,
        captured_channels: captured,
        meeting,
    })
}

// ===========================================================================
// audio_chunks, crash-recovery journal
// ===========================================================================

/// Record a chunk. `committed` starts false; call [`commit_chunk`] once the
/// file is flushed and fsynced.
pub async fn insert_chunk(
    db: &Db,
    meeting_id: &str,
    channel: Channel,
    seq: i64,
    path: &str,
    t_start_ms: i64,
    t_end_ms: i64,
) -> Result<Id, DbError> {
    let id = new_id();
    sqlx::query(
        "INSERT INTO audio_chunks (id, meeting_id, channel, seq, path, t_start_ms, t_end_ms, committed)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0)",
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(channel.as_str())
    .bind(seq)
    .bind(path)
    .bind(t_start_ms)
    .bind(t_end_ms)
    .execute(db)
    .await?;
    Ok(id)
}

/// A chunk only counts as audio-on-disk once this returns.
pub async fn commit_chunk(db: &Db, id: &str, t_end_ms: i64) -> Result<(), DbError> {
    sqlx::query("UPDATE audio_chunks SET committed = 1, t_end_ms = ?2 WHERE id = ?1")
        .bind(id)
        .bind(t_end_ms)
        .execute(db)
        .await?;
    Ok(())
}

pub async fn list_chunks(
    db: &Db,
    meeting_id: &str,
    channel: Option<Channel>,
) -> Result<Vec<AudioChunk>, DbError> {
    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, meeting_id, channel, seq, path, t_start_ms, t_end_ms, committed
         FROM audio_chunks WHERE meeting_id = ",
    );
    qb.push_bind(meeting_id.to_string());
    if let Some(ch) = channel {
        qb.push(" AND channel = ")
            .push_bind(ch.as_str().to_string());
    }
    qb.push(" ORDER BY channel, seq");
    let rows = qb.build().fetch_all(db).await?;
    rows.into_iter().map(row_to_chunk).collect()
}

/// How far the committed audio reaches on this channel, where a catch-up pass
/// or a resume has to start from.
pub async fn last_committed_offset_ms(
    db: &Db,
    meeting_id: &str,
    channel: Channel,
) -> Result<i64, DbError> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT MAX(t_end_ms) FROM audio_chunks
         WHERE meeting_id = ?1 AND channel = ?2 AND committed = 1",
    )
    .bind(meeting_id)
    .bind(channel.as_str())
    .fetch_optional(db)
    .await?;
    Ok(row.map(|r| r.0).unwrap_or(0))
}

/// Chunks written but never committed, a crash left them half-done.
pub async fn list_uncommitted_chunks(
    db: &Db,
    meeting_id: &str,
) -> Result<Vec<AudioChunk>, DbError> {
    let rows = sqlx::query(
        "SELECT id, meeting_id, channel, seq, path, t_start_ms, t_end_ms, committed
         FROM audio_chunks WHERE meeting_id = ?1 AND committed = 0 ORDER BY channel, seq",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_chunk).collect()
}

pub async fn delete_chunk(db: &Db, id: &str) -> Result<(), DbError> {
    sqlx::query("DELETE FROM audio_chunks WHERE id = ?1")
        .bind(id)
        .execute(db)
        .await?;
    Ok(())
}

// ===========================================================================
// segments
// ===========================================================================

/// Insert one segment. Prefer [`insert_segments`] while recording.
pub async fn insert_segment(db: &Db, draft: &SegmentDraft) -> Result<Segment, DbError> {
    let ids = insert_segments(db, std::slice::from_ref(draft)).await?;
    let id = ids
        .into_iter()
        .next()
        .ok_or_else(|| DbError::Decode("segment insert returned no id".into()))?;
    get_segment(db, &id)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("segment {id}")))
}

/// Batched insert in a single transaction (DESIGN §3 "SQLite (batched)").
pub async fn insert_segments(db: &Db, drafts: &[SegmentDraft]) -> Result<Vec<Id>, DbError> {
    if drafts.is_empty() {
        return Ok(Vec::new());
    }
    let mut ids = Vec::with_capacity(drafts.len());
    let mut tx = db.begin().await?;
    for d in drafts {
        let id = new_id();
        sqlx::query(
            "INSERT INTO segments (id, meeting_id, t_start_ms, t_end_ms, channel, speaker_id, text,
                                   language, avg_confidence, revision, is_final, model_name,
                                   model_revision, corrections)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        )
        .bind(&id)
        .bind(&d.meeting_id)
        .bind(d.t_start_ms)
        .bind(d.t_end_ms)
        .bind(d.channel.as_str())
        .bind(d.speaker_id.as_deref())
        .bind(&d.text)
        .bind(d.language.as_deref())
        .bind(d.avg_confidence)
        .bind(d.revision.max(1))
        .bind(d.is_final)
        .bind(d.model_name.as_deref())
        .bind(d.model_revision.as_deref())
        .bind(corrections_json(&d.corrections))
        .execute(&mut *tx)
        .await?;
        ids.push(id);
    }
    tx.commit().await?;
    Ok(ids)
}

pub async fn get_segment(db: &Db, id: &str) -> Result<Option<Segment>, DbError> {
    let row = sqlx::query(
        "SELECT id, meeting_id, t_start_ms, t_end_ms, channel, speaker_id, text, language,
                avg_confidence, revision, is_final, model_name, model_revision, corrections
         FROM segments WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_segment).transpose()
}

pub async fn get_segments(db: &Db, q: &TranscriptQuery) -> Result<Vec<Segment>, DbError> {
    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, meeting_id, t_start_ms, t_end_ms, channel, speaker_id, text, language,
                avg_confidence, revision, is_final, model_name, model_revision, corrections
         FROM segments WHERE meeting_id = ",
    );
    qb.push_bind(q.meeting_id.clone());
    if !q.include_partial.unwrap_or(false) {
        qb.push(" AND is_final = 1");
    }
    if let Some(from) = q.from_ms {
        qb.push(" AND t_end_ms >= ").push_bind(from);
    }
    if let Some(to) = q.to_ms {
        qb.push(" AND t_start_ms <= ").push_bind(to);
    }
    qb.push(" ORDER BY t_start_ms, revision LIMIT ")
        .push_bind(i64::from(q.limit.unwrap_or(5_000).min(50_000)));
    let rows = qb.build().fetch_all(db).await?;
    rows.into_iter().map(row_to_segment).collect()
}

/// Replace the text/speaker of an existing segment with a better pass. The
/// revision must move forward, so an old pass can never overwrite a new one.
pub async fn revise_segment(
    db: &Db,
    id: &str,
    text: Option<&str>,
    speaker_id: Option<&str>,
    revision: i64,
    is_final: bool,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE segments
         SET text = COALESCE(?2, text),
             speaker_id = COALESCE(?3, speaker_id),
             revision = ?4,
             is_final = ?5
         WHERE id = ?1 AND revision <= ?4",
    )
    .bind(id)
    .bind(text)
    .bind(speaker_id)
    .bind(revision)
    .bind(is_final)
    .execute(db)
    .await?;
    Ok(())
}

/// Put one line back the way the engine wrote it, and forget the note about it.
///
/// The write half of "a repair can be put back"
/// (`migrations/0005_segment_corrections.sql`). Deliberately narrower than
/// [`revise_segment`], in three ways that are all the same way — this is not a
/// pass, it is a person saying *no, it heard that right*:
///
/// * **The revision does not move.** Nothing better has been read; the words are
///   going back to the ones the engine produced. Moving it would tell every
///   later pass that this line is newer than their work.
/// * **`was` is the whole guard.** The update only lands on a row whose text is
///   still character for character the text the caller reverted, so a
///   re-transcription, a split or a second click that got there first leaves the
///   row alone and this returns `false` instead of writing somebody else's words
///   over it.
/// * **`corrections IS NOT NULL`** makes a second call a no-op rather than a
///   rewrite: a line with nothing recorded against it has nothing to put back.
///
/// `false` means nothing was written and the caller should say so, not retry.
pub async fn undo_segment_corrections(
    db: &Db,
    id: &str,
    was: &str,
    original: &str,
) -> Result<bool, DbError> {
    let r = sqlx::query(
        "UPDATE segments SET text = ?3, corrections = NULL
         WHERE id = ?1 AND text = ?2 AND corrections IS NOT NULL",
    )
    .bind(id)
    .bind(was)
    .bind(original)
    .execute(db)
    .await?;
    Ok(r.rows_affected() > 0)
}

/// Point a batch of segments at a speaker, what the offline speaker pass does.
pub async fn assign_speaker(
    db: &Db,
    segment_ids: &[Id],
    speaker_id: &str,
    revision: i64,
) -> Result<u64, DbError> {
    if segment_ids.is_empty() {
        return Ok(0);
    }
    let mut affected = 0;
    let mut tx = db.begin().await?;
    for id in segment_ids {
        let r = sqlx::query(
            "UPDATE segments SET speaker_id = ?2, revision = ?3 WHERE id = ?1 AND revision <= ?3",
        )
        .bind(id)
        .bind(speaker_id)
        .bind(revision)
        .execute(&mut *tx)
        .await?;
        affected += r.rows_affected();
    }
    tx.commit().await?;
    Ok(affected)
}

/// Replace one transcript line with the pieces a speaker turn cut it into.
///
/// What the separation pass does to a line of fast dialogue that the speech
/// engine wrote as one row (see [`crate::diarize::split`]): the row goes and one
/// row per voice takes its place. Everything else about the line is carried over
/// unchanged — channel, language, the engine's own confidence, and the
/// provenance of the words — because the words were not re-transcribed. Only
/// where they were cut is Echo's own doing, and only the caller knows that.
///
/// Two details this depends on, the same two [`clear_transcript`] depends on:
///
/// * The old row is deleted **by name**, so the `segments_fts_ad` trigger fires
///   and both search indexes lose its words; the inserts then fire
///   `segments_fts_ai` and the indexes gain the pieces. A search for a sentence
///   that got cut in two still finds the half it is in.
/// * It is one transaction, so a crash can leave the line whole or leave it
///   split, and never leave the meeting with a hole where it was.
///
/// Refuses a "split" into fewer than two pieces — that is a rewrite, and
/// [`revise_segment`] is the function for those — and refuses one whose pieces
/// do not carry the revision forward, because a later pass must never be
/// overwritable by an earlier one.
pub async fn split_segment(
    db: &Db,
    id: &str,
    pieces: &[SegmentDraft],
    revision: i64,
) -> Result<Vec<Id>, DbError> {
    if pieces.len() < 2 {
        return Err(DbError::Invalid(
            "a split needs at least two pieces to put back".into(),
        ));
    }
    if pieces.iter().any(|p| p.revision < revision) {
        return Err(DbError::Invalid(
            "every piece of a split line has to carry the new revision".into(),
        ));
    }

    let mut tx = db.begin().await?;
    let gone = sqlx::query("DELETE FROM segments WHERE id = ?1 AND revision <= ?2")
        .bind(id)
        .bind(revision)
        .execute(&mut *tx)
        .await?
        .rows_affected();
    if gone == 0 {
        // Either it is already gone, or a later pass has moved past this one.
        // Both mean this split is not the current answer any more.
        tx.rollback().await?;
        return Ok(Vec::new());
    }

    let mut ids = Vec::with_capacity(pieces.len());
    for p in pieces {
        let piece_id = new_id();
        sqlx::query(
            "INSERT INTO segments (id, meeting_id, t_start_ms, t_end_ms, channel, speaker_id, text,
                                   language, avg_confidence, revision, is_final, model_name,
                                   model_revision, corrections)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        )
        .bind(&piece_id)
        .bind(&p.meeting_id)
        .bind(p.t_start_ms)
        .bind(p.t_end_ms)
        .bind(p.channel.as_str())
        .bind(p.speaker_id.as_deref())
        .bind(&p.text)
        .bind(p.language.as_deref())
        .bind(p.avg_confidence)
        .bind(p.revision.max(1))
        .bind(p.is_final)
        .bind(p.model_name.as_deref())
        .bind(p.model_revision.as_deref())
        .bind(corrections_json(&p.corrections))
        .execute(&mut *tx)
        .await?;
        ids.push(piece_id);
    }
    tx.commit().await?;
    Ok(ids)
}

/// Drop the live partials sitting over one stretch of one channel, because the
/// catch-up pass has just read that stretch back off the recording.
///
/// Scoped rather than wholesale for one reason: a pass can be stopped part-way
/// through (a recording starting takes the machine — see `asr::catchup`), and a
/// live guess is the only text its stretch has until some pass reads it. So the
/// guesses go where the real reading has actually happened and nowhere else. On
/// 2026-08-24 the whole meeting's guesses were dropped the moment the pass
/// began, and a pass that was interrupted three minutes in left twenty-seven
/// minutes of that meeting with no text of any kind.
///
/// Overlap, not containment: a guess that straddles the edge of the window is
/// about seconds that were just read, and leaving half-superseded text on the
/// transcript reads as a stutter.
pub async fn delete_partial_segments_in(
    db: &Db,
    meeting_id: &str,
    channel: Channel,
    from_ms: i64,
    to_ms: i64,
) -> Result<u64, DbError> {
    let r = sqlx::query(
        "DELETE FROM segments
         WHERE meeting_id = ?1 AND is_final = 0 AND channel = ?2
           AND t_start_ms < ?4 AND t_end_ms > ?3",
    )
    .bind(meeting_id)
    .bind(channel.as_str())
    .bind(from_ms)
    .bind(to_ms)
    .execute(db)
    .await?;
    Ok(r.rows_affected())
}

/// Drop live partials once the final pass replaced them.
pub async fn delete_partial_segments(db: &Db, meeting_id: &str) -> Result<u64, DbError> {
    let r = sqlx::query("DELETE FROM segments WHERE meeting_id = ?1 AND is_final = 0")
        .bind(meeting_id)
        .execute(db)
        .await?;
    Ok(r.rows_affected())
}

/// [`delete_partial_segments`], sparing the guesses that sit over stretches the
/// final pass could **not** read.
///
/// The sweep exists because a finished pass has replaced every guess with text
/// read off the recording. When a window would not decode, that is not true of
/// its seconds: nothing replaced them, and the live guess is the only text they
/// will ever have unless somebody reads the meeting again. Deleting it turns a
/// stretch a person watched appear on screen into nothing at all, which is
/// worse than the hole it was covering.
///
/// Overlap, not containment, for the same reason as
/// [`delete_partial_segments_in`]: a guess that straddles the edge of an unread
/// stretch is partly about seconds nothing replaced, and half a sentence is not
/// worth deleting the other half for.
pub async fn delete_partial_segments_except(
    db: &Db,
    meeting_id: &str,
    keep: &[(Channel, i64, i64)],
) -> Result<u64, DbError> {
    if keep.is_empty() {
        return delete_partial_segments(db, meeting_id).await;
    }
    let mut qb: QueryBuilder<Sqlite> =
        QueryBuilder::new("DELETE FROM segments WHERE is_final = 0 AND meeting_id = ");
    qb.push_bind(meeting_id);
    for (channel, from_ms, to_ms) in keep {
        qb.push(" AND NOT (channel = ");
        qb.push_bind(channel.as_str());
        qb.push(" AND t_start_ms < ");
        qb.push_bind(*to_ms);
        qb.push(" AND t_end_ms > ");
        qb.push_bind(*from_ms);
        qb.push(")");
    }
    let r = qb.build().execute(db).await?;
    Ok(r.rows_affected())
}

/// What [`clear_transcript`] took away.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClearedTranscript {
    pub segments_deleted: u64,
    pub speakers_deleted: u64,
    /// The meeting had a language written against it and no longer does, so the
    /// pass about to run will work it out again from the recording. False for a
    /// meeting that never settled on one.
    pub language_cleared: bool,
    /// Decisions not to transcribe ([`SuppressedSpan`]) that no longer stand,
    /// so the next pass judges those seconds again. The rows themselves are
    /// still there, holding what was measured about that audio — see
    /// [`measured_lag_ms`].
    pub spans_withdrawn: u64,
    /// One past the revision the transcript was on, so anything watching can
    /// tell that what it holds is stale.
    ///
    /// The stored revision lives on the segment rows ([`transcript_revision`] is
    /// `MAX(revision)`), so once they are gone there is nothing left to read: a
    /// wiped meeting is back at 0 until the pass that re-reads the audio writes
    /// its first row. This number is the one the *event* carries, and it only
    /// ever moves forward.
    pub revision: i64,
}

/// Throw away everything derived from this meeting's audio that a fresh pass
/// would write again: the transcript, the speakers (aliases included), and the
/// language those words were read in.
///
/// For "listen again" — the audio on disk is the truth (mantra 3), so a
/// transcript written by a broken pipeline is safe to delete and read back from
/// the recording. Deliberately **not** the recap or its task list: those are the
/// person's to keep or rewrite.
///
/// Three details this depends on, the first two the same as [`delete_meeting`]:
/// * segments are deleted by name so the `segments_fts_ad` trigger fires and
///   both search indexes lose the words. A cascade might not fire triggers, and
///   somebody's search results are not a thing to bet on a build detail.
/// * speakers go after segments, so nothing depends on `ON DELETE SET NULL`
///   having run first.
/// * `language` goes with them, in the same transaction. It is not something a
///   person typed: it is written from the words by the live pass
///   (`session::pipeline`) and recomputed from the finished segments by the
///   catch-up job (`session::jobs`), so it is derived from exactly the rows
///   being deleted here. Leaving it behind is what made a wrong language
///   permanent — catch-up reads the meeting's language as its prior
///   (`asr::catchup`), so "listen again" would pin the same wrong answer and
///   spend a whole pass reproducing the transcript it was asked to replace.
///   Clearing it in the same transaction means there is never a moment where
///   the words are gone and the language they were read in is still standing.
///
/// * the marks that say a stretch was heard and deliberately left without text
///   ([`SuppressedSpan`]) are **withdrawn**, in the same transaction. They are
///   a decision about the audio, and "listen again" is a person asking for the
///   audio to be decided again — the recording is still there to judge, and a
///   standing mark would mean one meeting's suppression outlived the transcript
///   it was part of while a button on screen promised a fresh reading. The cost
///   of being wrong this way round is a duplicated line, which is what the
///   button is for; the cost the other way round is a sentence that can never
///   come back.
///
///   Withdrawn, not deleted. The decision stops hiding those seconds from the
///   planner ([`suppressed_spans`] reads only what still stands), but what was
///   *measured* about them survives: a withdrawn row is still a true
///   measurement of this recording's audio, and [`measured_lag_ms`] hands the
///   delay it found to the pass about to read the meeting again. Deleting threw
///   that away and left the second reading — the one that is worse at this, see
///   `migrations/0008_withdrawn_suppressions.sql` — starting from nothing.
///
/// Chunks, markers, summaries and action items are left exactly where they are.
/// So is the person's own count of how many people were there: that is a fact
/// about the meeting, not something read out of the audio.
pub async fn clear_transcript(db: &Db, meeting_id: &str) -> Result<ClearedTranscript, DbError> {
    let mut tx = db.begin().await?;
    let previous: Option<(Option<i64>,)> =
        sqlx::query_as("SELECT MAX(revision) FROM segments WHERE meeting_id = ?1")
            .bind(meeting_id)
            .fetch_optional(&mut *tx)
            .await?;
    let previous = previous.and_then(|r| r.0).unwrap_or(0);

    let segments = sqlx::query("DELETE FROM segments WHERE meeting_id = ?1")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?
        .rows_affected();
    let speakers = sqlx::query("DELETE FROM speakers WHERE meeting_id = ?1")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?
        .rows_affected();
    // Only what still stands: a row an earlier "listen again" withdrew is
    // already withdrawn, and restamping it would say this clear did something
    // it did not.
    let spans_withdrawn = sqlx::query(
        "UPDATE suppressed_spans SET withdrawn_at = ?2
         WHERE meeting_id = ?1 AND withdrawn_at IS NULL",
    )
    .bind(meeting_id)
    .bind(now())
    .execute(&mut *tx)
    .await?
    .rows_affected();
    let language_cleared =
        sqlx::query("UPDATE meetings SET language = NULL WHERE id = ?1 AND language IS NOT NULL")
            .bind(meeting_id)
            .execute(&mut *tx)
            .await?
            .rows_affected()
            > 0;
    tx.commit().await?;

    Ok(ClearedTranscript {
        segments_deleted: segments,
        speakers_deleted: speakers,
        language_cleared,
        spans_withdrawn,
        revision: previous.saturating_add(1),
    })
}

/// Highest revision anywhere in this meeting, the "transcript revision" a
/// recap is pinned to.
pub async fn transcript_revision(db: &Db, meeting_id: &str) -> Result<i64, DbError> {
    let row: Option<(Option<i64>,)> =
        sqlx::query_as("SELECT MAX(revision) FROM segments WHERE meeting_id = ?1")
            .bind(meeting_id)
            .fetch_optional(db)
            .await?;
    Ok(row.and_then(|r| r.0).unwrap_or(0))
}

pub async fn count_segments(db: &Db, meeting_id: &str) -> Result<u32, DbError> {
    let (n,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM segments WHERE meeting_id = ?1")
        .bind(meeting_id)
        .fetch_one(db)
        .await?;
    Ok(n as u32)
}

/// Lines of transcript that are finished, i.e. not live partials waiting to be
/// replaced. This is "did this meeting produce any words", which is what decides
/// whether an ended meeting is worth keeping (see `session::recovery`'s
/// `meeting_has_content`, and [`crate::session::MIN_KEPT_MS`]).
pub async fn count_final_segments(db: &Db, meeting_id: &str) -> Result<u32, DbError> {
    let (n,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM segments
         WHERE meeting_id = ?1 AND is_final = 1 AND TRIM(text) <> ''",
    )
    .bind(meeting_id)
    .fetch_one(db)
    .await?;
    Ok(n as u32)
}

/// Languages seen in this meeting with how much time each covers, used to
/// pick the dominant meeting language.
/// Did this channel produce any words at all?
///
/// Not "was the channel recorded": the microphone is open for every meeting, so
/// the presence of mic audio says nothing about whether the person spoke. A
/// final segment with text in it does.
pub async fn has_channel_speech(
    db: &Db,
    meeting_id: &str,
    channel: Channel,
) -> Result<bool, DbError> {
    let (found,): (i64,) = sqlx::query_as(
        "SELECT EXISTS (
             SELECT 1 FROM segments
             WHERE meeting_id = ?1 AND channel = ?2 AND is_final = 1 AND TRIM(text) <> ''
         )",
    )
    .bind(meeting_id)
    .bind(channel.as_str())
    .fetch_one(db)
    .await?;
    Ok(found != 0)
}

pub async fn language_histogram(db: &Db, meeting_id: &str) -> Result<Vec<(String, i64)>, DbError> {
    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT language, SUM(t_end_ms - t_start_ms) AS ms
         FROM segments
         WHERE meeting_id = ?1 AND language IS NOT NULL AND is_final = 1
         GROUP BY language ORDER BY ms DESC",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    Ok(rows)
}

// ===========================================================================
// suppressed spans — seconds Echo heard and chose not to write down
// ===========================================================================

/// Why a stretch of a recording was deliberately left without text.
///
/// One variant today. It is an enum rather than a string because the column is
/// read back by machines as well as people, and "which mechanism decided this"
/// is not a thing to spell twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressionReason {
    /// The microphone's copy of what the computer played, already in the
    /// transcript from the computer's own side of the call
    /// ([`crate::audio::bleed`]).
    Bleed,
}

impl SuppressionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            SuppressionReason::Bleed => "bleed",
        }
    }
}

/// Which pass made the call.
///
/// Recorded because the two passes disagreeing is exactly what this table was
/// built out of (`migrations/0007_suppressed_spans.sql`): the live guard judges
/// against a ring aligned to the meeting clock and a delay this machine has
/// already measured, the offline one reads paged audio off disk and always
/// searches cold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecidedBy {
    /// The guard on the speech thread, while the meeting was happening.
    Live,
    /// The catch-up pass, reading the recording back afterwards.
    CatchUp,
}

impl DecidedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            DecidedBy::Live => "live",
            DecidedBy::CatchUp => "catchup",
        }
    }
}

/// One decision not to transcribe a stretch, with the measurement behind it.
///
/// The evidence fields are `Option` because a future reason may not have any;
/// bleed always does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuppressedSpan {
    pub channel: Channel,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub reason: SuppressionReason,
    pub decided_by: DecidedBy,
    /// How well the two loudness shapes agreed at the best delay, -1.0 to 1.0.
    pub correlation: Option<f32>,
    /// That delay: how much the mic copy was late relative to the system copy.
    pub lag_ms: Option<i64>,
    /// Milliseconds of the stretch where the far side was audibly playing.
    pub system_voice_ms: Option<i64>,
}

/// Write down that these seconds were heard and deliberately left alone.
///
/// Backwards or empty spans are refused rather than stored: a mark with no
/// width covers no audio and would only ever be a puzzle for whoever reads the
/// table next.
pub async fn record_suppressed_span(
    db: &Db,
    meeting_id: &str,
    span: &SuppressedSpan,
) -> Result<Id, DbError> {
    if span.t_end_ms <= span.t_start_ms {
        return Err(DbError::Invalid(format!(
            "suppressed span {}..{} has no width",
            span.t_start_ms, span.t_end_ms
        )));
    }
    let id = new_id();
    sqlx::query(
        "INSERT INTO suppressed_spans
             (id, meeting_id, channel, t_start_ms, t_end_ms, reason, decided_by,
              correlation, lag_ms, system_voice_ms, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(span.channel.as_str())
    .bind(span.t_start_ms)
    .bind(span.t_end_ms)
    .bind(span.reason.as_str())
    .bind(span.decided_by.as_str())
    .bind(span.correlation)
    .bind(span.lag_ms)
    .bind(span.system_voice_ms)
    .bind(now())
    .execute(db)
    .await?;
    Ok(id)
}

/// The stretches of this channel a pass already decided not to transcribe, as
/// merged spans on the meeting clock.
///
/// The catch-up planner subtracts these alongside the stretches that have text
/// against them ([`crate::asr::catchup`]), which is the whole point of the
/// table: a decision made once against better-aligned audio is not re-litigated
/// against worse.
///
/// **Only decisions that still stand.** One a "listen again" withdrew
/// ([`clear_transcript`]) hides nothing: those seconds are planned, read and
/// judged afresh, because the repair button has to be able to reach every
/// second of the recording — including a sentence Echo wrongly decided was an
/// echo of the far side. The row stays behind for what it measured
/// ([`measured_lag_ms`]), not for what it decided.
pub async fn suppressed_spans(
    db: &Db,
    meeting_id: &str,
    channel: Channel,
) -> Result<Vec<(i64, i64)>, DbError> {
    let rows: Vec<(i64, i64)> = sqlx::query_as(
        "SELECT t_start_ms, t_end_ms FROM suppressed_spans
         WHERE meeting_id = ?1 AND channel = ?2 AND t_end_ms > t_start_ms
           AND withdrawn_at IS NULL
         ORDER BY t_start_ms",
    )
    .bind(meeting_id)
    .bind(channel.as_str())
    .fetch_all(db)
    .await?;
    Ok(rows)
}

/// Every decision that still stands for this meeting, in clock order — the
/// diagnostic read, for a person asking why a stretch of their recording has no
/// words in it.
///
/// Withdrawn rows are left out because they are no longer an answer to that
/// question: the seconds they cover were judged again by a later pass, and
/// whatever that pass concluded is what the transcript now shows. What a
/// withdrawn row still holds is its measurement, and that is read through
/// [`measured_lag_ms`].
pub async fn list_suppressed_spans(
    db: &Db,
    meeting_id: &str,
) -> Result<Vec<SuppressedSpan>, DbError> {
    let rows = sqlx::query(
        "SELECT channel, t_start_ms, t_end_ms, reason, decided_by,
                correlation, lag_ms, system_voice_ms
         FROM suppressed_spans WHERE meeting_id = ?1 AND withdrawn_at IS NULL
         ORDER BY t_start_ms",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    rows.into_iter()
        .map(|row| {
            let channel: String = row.try_get("channel")?;
            let reason: String = row.try_get("reason")?;
            let decided_by: String = row.try_get("decided_by")?;
            Ok(SuppressedSpan {
                channel: Channel::parse(&channel)
                    .ok_or_else(|| DbError::Invalid(format!("channel {channel}")))?,
                t_start_ms: row.try_get("t_start_ms")?,
                t_end_ms: row.try_get("t_end_ms")?,
                reason: match reason.as_str() {
                    "bleed" => SuppressionReason::Bleed,
                    other => return Err(DbError::Invalid(format!("suppression reason {other}"))),
                },
                decided_by: match decided_by.as_str() {
                    "live" => DecidedBy::Live,
                    "catchup" => DecidedBy::CatchUp,
                    other => return Err(DbError::Invalid(format!("decided by {other}"))),
                },
                correlation: row.try_get("correlation")?,
                lag_ms: row.try_get("lag_ms")?,
                system_voice_ms: row.try_get("system_voice_ms")?,
            })
        })
        .collect()
}

/// The delay this meeting measured between what the speakers played and what
/// the microphone heard, in milliseconds — the median `lag_ms` of every
/// suppression written down for it, **withdrawn or not**.
///
/// A decision can be withdrawn; a measurement cannot. Every row here was a full
/// hit — the correlator only records one when the whole predicate agreed
/// ([`crate::audio::bleed::is_bleed`]) — so each is an independent statement
/// about this recording's audio, and pressing "listen again" did not change the
/// audio. That is why this reads through the withdrawal: what the next pass
/// inherits is the machine's delay, not the last pass's verdict.
///
/// **The median, not the mean**, for [`crate::audio::bleed::LagEstimate`]'s
/// reason: a drift correction moves the true delay in a single step and the odd
/// reading is nonsense, so the estimator has to follow a staircase and ignore an
/// outlier. With an even count this is the midpoint of the two middle
/// measurements, again matching `LagEstimate`.
///
/// One row is enough to answer. Live, [`crate::audio::bleed::LagEstimate`]
/// wants two before it will speak, because there the number *narrows the
/// search* and a single wrong reading would aim the search at the wrong delay.
/// Nothing narrows here — see [`crate::asr::catchup_bleed::OfflineBleed`], which
/// keeps searching cold whatever this returns — so the number is only ever
/// evidence that this recording has a real, measurable delay in it, and one full
/// hit is that.
///
/// `None` for a meeting nothing was ever measured on, which is every meeting
/// recorded before 2026-08-26 and every meeting where no copy was ever found.
pub async fn measured_lag_ms(db: &Db, meeting_id: &str) -> Result<Option<i64>, DbError> {
    let lags: Vec<(i64,)> = sqlx::query_as(
        "SELECT lag_ms FROM suppressed_spans
         WHERE meeting_id = ?1 AND lag_ms IS NOT NULL
         ORDER BY lag_ms",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    if lags.is_empty() {
        return Ok(None);
    }
    let mid = lags.len() / 2;
    Ok(Some(if lags.len() % 2 == 1 {
        lags[mid].0
    } else {
        (lags[mid - 1].0 + lags[mid].0) / 2
    }))
}

/// How many decisions about this meeting have been withdrawn — the count of
/// stretches a previous reading left without text and this one will judge
/// again.
///
/// Read by the catch-up pass so that it can say so in the log. A duplicated
/// line in a finished transcript is the thing this whole area gets diagnosed
/// for, and "which reading produced it" is the first question: a pass that
/// starts by naming how many decisions it is re-opening, and what delay it
/// inherited, answers it from the log alone.
pub async fn withdrawn_suppression_count(db: &Db, meeting_id: &str) -> Result<u64, DbError> {
    let (n,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM suppressed_spans
         WHERE meeting_id = ?1 AND withdrawn_at IS NOT NULL",
    )
    .bind(meeting_id)
    .fetch_one(db)
    .await?;
    Ok(n.max(0) as u64)
}

// ===========================================================================
// speakers
// ===========================================================================

/// Create the speaker for a cluster key, or return the existing one.
pub async fn upsert_speaker(
    db: &Db,
    meeting_id: &str,
    cluster_key: &str,
    display_name: &str,
    is_self: bool,
) -> Result<Speaker, DbError> {
    let id = new_id();
    sqlx::query(
        "INSERT INTO speakers (id, meeting_id, cluster_key, display_name, is_self, speaking_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, 0)
         ON CONFLICT (meeting_id, cluster_key) DO NOTHING",
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(cluster_key)
    .bind(display_name)
    .bind(is_self)
    .execute(db)
    .await?;

    let row = sqlx::query(
        "SELECT id, meeting_id, cluster_key, display_name, alias_of, is_self, speaking_ms,
                person_id, suggested_person_id, suggestion_score
         FROM speakers WHERE meeting_id = ?1 AND cluster_key = ?2",
    )
    .bind(meeting_id)
    .bind(cluster_key)
    .fetch_optional(db)
    .await?
    .ok_or_else(|| DbError::NotFound(format!("speaker {cluster_key}")))?;
    row_to_speaker(row)
}

pub async fn list_speakers(db: &Db, meeting_id: &str) -> Result<Vec<Speaker>, DbError> {
    let rows = sqlx::query(
        "SELECT id, meeting_id, cluster_key, display_name, alias_of, is_self, speaking_ms,
                person_id, suggested_person_id, suggestion_score
         FROM speakers WHERE meeting_id = ?1
         ORDER BY is_self DESC, speaking_ms DESC, cluster_key",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_speaker).collect()
}

pub async fn get_speaker(db: &Db, id: &str) -> Result<Option<Speaker>, DbError> {
    let row = sqlx::query(
        "SELECT id, meeting_id, cluster_key, display_name, alias_of, is_self, speaking_ms,
                person_id, suggested_person_id, suggestion_score
         FROM speakers WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_speaker).transpose()
}

pub async fn rename_speaker(db: &Db, id: &str, display_name: &str) -> Result<(), DbError> {
    let r = sqlx::query("UPDATE speakers SET display_name = ?2 WHERE id = ?1")
        .bind(id)
        .bind(display_name)
        .execute(db)
        .await?;
    if r.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("speaker {id}")));
    }
    Ok(())
}

/// Merge is an alias, never a delete: `from` keeps its row and points at
/// `into`, so the person can undo it. Existing aliases of `from` are
/// re-pointed so chains never form.
///
/// Two things are checked before anything is written, because the ids come from
/// the webview and a merge rewrites how a whole transcript reads:
///
/// * **Same meeting.** Aliasing a speaker from one meeting into another would
///   corrupt both transcripts' speaker chips and both speaking-time sums.
/// * **No cycles.** `into` is first resolved to the speaker it already points
///   at; if that root is `from`, the merge would make a ring that only the
///   hop limit in [`resolve_speaker`] survives, and both people would render as
///   merged away. Merging into where `from` already is, is a no-op.
pub async fn merge_speakers(db: &Db, from_id: &str, into_id: &str) -> Result<(), DbError> {
    if from_id == into_id {
        return Ok(());
    }
    let mut tx = db.begin().await?;

    let from: (String, Option<String>) =
        sqlx::query_as("SELECT meeting_id, alias_of FROM speakers WHERE id = ?1")
            .bind(from_id)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("speaker {from_id}")))?;
    let into: (String, Option<String>) =
        sqlx::query_as("SELECT meeting_id, alias_of FROM speakers WHERE id = ?1")
            .bind(into_id)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("speaker {into_id}")))?;
    if from.0 != into.0 {
        return Err(DbError::Invalid(
            "two speakers from different meetings cannot be the same person".into(),
        ));
    }

    // Follow `into`'s own aliases to the speaker that is actually displayed.
    let mut root = into_id.to_string();
    let mut next = into.1;
    let mut hops = 0;
    while let Some(candidate) = next {
        if candidate == root || hops >= 8 {
            break;
        }
        let row: Option<(Option<String>,)> =
            sqlx::query_as("SELECT alias_of FROM speakers WHERE id = ?1")
                .bind(&candidate)
                .fetch_optional(&mut *tx)
                .await?;
        let Some((alias_of,)) = row else { break };
        root = candidate;
        next = alias_of;
        hops += 1;
    }
    if root == from_id {
        // `into` already resolves to `from`: they are the same person already.
        return Ok(());
    }

    sqlx::query("UPDATE speakers SET alias_of = ?2 WHERE alias_of = ?1")
        .bind(from_id)
        .bind(&root)
        .execute(&mut *tx)
        .await?;
    let r = sqlx::query("UPDATE speakers SET alias_of = ?2 WHERE id = ?1")
        .bind(from_id)
        .bind(&root)
        .execute(&mut *tx)
        .await?;
    if r.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("speaker {from_id}")));
    }
    tx.commit().await?;
    Ok(())
}

pub async fn unmerge_speaker(db: &Db, id: &str) -> Result<(), DbError> {
    sqlx::query("UPDATE speakers SET alias_of = NULL WHERE id = ?1")
        .bind(id)
        .execute(db)
        .await?;
    Ok(())
}

/// Forget every speaker of this meeting.
///
/// The honest end of [`prune_speakers_except`], which refuses an empty `keep`
/// on purpose (see its docs) because that shape usually means a caller lost its
/// list. A caller that means it — the speaker pass on a meeting where no line of
/// transcript belongs to any voice it found — says so here instead, and the
/// lines those rows held go back to unattributed exactly as a partial prune
/// would leave them.
pub async fn forget_speakers(db: &Db, meeting_id: &str) -> Result<u64, DbError> {
    Ok(sqlx::query("DELETE FROM speakers WHERE meeting_id = ?1")
        .bind(meeting_id)
        .execute(db)
        .await?
        .rows_affected())
}

/// Forget the speakers of this meeting whose cluster keys are not in `keep`.
///
/// What the speaker pass calls after it has re-pointed the transcript: the keys
/// it produced this time, plus the microphone. Anything else is a voice a
/// previous run believed in and this one does not — a fourth person who turns
/// out to have been the second one twice, or the tail of a meeting the person has
/// since re-cut to fewer people.
///
/// Deliberately a delete rather than a soft flag, and safe because the schema
/// says what happens to everything that pointed at the row:
///
/// * `segments.speaker_id` is `ON DELETE SET NULL`, so a line that still carried
///   the old label goes back to unattributed. Honest: the voice it was given no
///   longer exists, and an unattributed line reads as one.
/// * `speakers.alias_of` is `ON DELETE SET NULL`, so a merge into a vanished
///   person is released rather than left dangling.
///
/// A name the person typed onto one of these rows goes with it. There is nothing
/// left for it to be the name of; the alternative — keeping empty rows so a name
/// has somewhere to live — is the ghost chip this exists to prevent.
///
/// Passing an empty `keep` would delete every speaker, so it does nothing
/// instead: that shape only ever arrives from a caller with a bug, and the honest
/// response to it is not to wipe a meeting's speakers.
pub async fn prune_speakers_except(
    db: &Db,
    meeting_id: &str,
    keep: &[String],
) -> Result<u64, DbError> {
    if keep.is_empty() {
        return Ok(0);
    }
    let mut qb: QueryBuilder<Sqlite> =
        QueryBuilder::new("DELETE FROM speakers WHERE meeting_id = ");
    qb.push_bind(meeting_id.to_string());
    qb.push(" AND cluster_key NOT IN (");
    let mut separated = qb.separated(", ");
    for key in keep {
        separated.push_bind(key.clone());
    }
    qb.push(")");
    Ok(qb.build().execute(db).await?.rows_affected())
}

/// Follow `alias_of` to the speaker that should actually be displayed.
pub async fn resolve_speaker(db: &Db, id: &str) -> Result<Option<Speaker>, DbError> {
    let mut current = get_speaker(db, id).await?;
    let mut hops = 0;
    while let Some(s) = current.clone() {
        match s.alias_of {
            Some(next) if hops < 8 => {
                hops += 1;
                current = get_speaker(db, &next).await?;
            }
            _ => return Ok(Some(s)),
        }
    }
    Ok(None)
}

/// Recompute speaking time per speaker from the final segments.
pub async fn recompute_speaking_time(db: &Db, meeting_id: &str) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE speakers SET speaking_ms = COALESCE((
             SELECT SUM(g.t_end_ms - g.t_start_ms) FROM segments g
             WHERE g.speaker_id = speakers.id AND g.is_final = 1
         ), 0)
         WHERE meeting_id = ?1",
    )
    .bind(meeting_id)
    .execute(db)
    .await?;
    Ok(())
}

/// Point this speaker at a known person, or release it.
///
/// Only the link. The display name is copied onto the row separately (through
/// [`rename_speaker`]) and is meeting-local from then on, which is why deleting a
/// person leaves the transcript reading the way the person last read it.
pub async fn set_speaker_person(
    db: &Db,
    speaker_id: &str,
    person_id: Option<&str>,
) -> Result<(), DbError> {
    // Linking settles the question, so any suggestion on the row goes with it:
    // "looks like Marco" next to "Marco" is noise.
    let r = sqlx::query(
        "UPDATE speakers
         SET person_id = ?2, suggested_person_id = NULL, suggestion_score = NULL
         WHERE id = ?1",
    )
    .bind(speaker_id)
    .bind(person_id)
    .execute(db)
    .await?;
    if r.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("speaker {speaker_id}")));
    }
    Ok(())
}

/// "Looks like Marco — confirm?", with the score that produced it. `None`
/// clears the suggestion.
pub async fn set_speaker_suggestion(
    db: &Db,
    speaker_id: &str,
    person_id: Option<&str>,
    score: Option<f32>,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE speakers SET suggested_person_id = ?2, suggestion_score = ?3 WHERE id = ?1",
    )
    .bind(speaker_id)
    .bind(person_id)
    .bind(person_id.and(score))
    .execute(db)
    .await?;
    Ok(())
}

/// Store this speaker's voice print for this meeting, as computed by the
/// offline pass. `None` forgets it.
///
/// This is what makes a recurring unnamed voice findable later without reading a
/// second of audio back, and what lets a person enrolled today be matched
/// against meetings that were separated before they existed.
pub async fn set_speaker_centroid(
    db: &Db,
    speaker_id: &str,
    centroid: Option<&[f32]>,
) -> Result<(), DbError> {
    sqlx::query("UPDATE speakers SET centroid = ?2 WHERE id = ?1")
        .bind(speaker_id)
        .bind(centroid.map(embedding_to_blob))
        .execute(db)
        .await?;
    Ok(())
}

// ===========================================================================
// known people (DESIGN §1 "Known people")
// ===========================================================================

/// A fingerprint as it is stored: f32 little-endian, no header.
///
/// Little-endian because every machine Echo ships on is, and a header would only
/// be a second place for the length to be wrong. A blob whose length is not a
/// multiple of four is a corrupt row, and [`blob_to_embedding`] returns what it
/// can rather than failing the whole query — a profile with a truncated vector
/// simply matches nothing.
pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(embedding.len() * 4);
    for x in embedding {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// The inverse of [`embedding_to_blob`].
pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    let (whole, _trailing) = blob.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

/// Row of `people`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PersonRow {
    pub id: Id,
    pub name: String,
    pub created_at: String,
    pub updated_at: String,
    /// Somebody merged this name into another one, and that other one is who
    /// they are now (`migrations/0006_person_merge.sql`). `None` for everybody
    /// Echo actually remembers, which is almost everybody.
    pub alias_of: Option<Id>,
}

/// Row of `person_samples`, without its audio.
///
/// The clip is the heavy half and almost nothing wants it, so it is fetched on
/// its own ([`person_sample_clips`], [`freshest_person_clip`]). Curation and
/// matching only ever look at the numbers.
#[derive(Debug, Clone, Default)]
pub struct PersonSampleRow {
    pub id: Id,
    pub person_id: Id,
    pub embedding: Vec<f32>,
    pub condition: Channel,
    pub source_meeting_id: Option<Id>,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub created_at: String,
}

/// A sample on its way in: both halves, because a sample without its audio
/// cannot survive a change of network and a sample without its numbers cannot
/// match anything today.
#[derive(Debug, Clone)]
pub struct NewPersonSample<'a> {
    pub person_id: &'a str,
    pub embedding: &'a [f32],
    /// 16 kHz mono WAV.
    pub clip: &'a [u8],
    pub condition: Channel,
    pub source_meeting_id: Option<&'a str>,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
}

/// Row of `person_profiles`, with the person's name for free.
#[derive(Debug, Clone, Default)]
pub struct PersonProfileRow {
    pub person_id: Id,
    pub name: String,
    pub centroid: Vec<f32>,
    pub sample_count: u32,
    /// The `models.id` of the network these numbers belong to.
    pub embedder_asset_id: String,
    pub updated_at: String,
}

/// One meeting's voice print for one speaker, for cross-meeting matching.
#[derive(Debug, Clone, Default)]
pub struct VoicePrint {
    pub speaker_id: Id,
    pub meeting_id: Id,
    pub meeting_title: String,
    pub started_at: String,
    pub display_name: String,
    pub speaking_ms: i64,
    pub centroid: Vec<f32>,
}

pub async fn create_person(db: &Db, name: &str) -> Result<PersonRow, DbError> {
    let row = PersonRow {
        id: new_id(),
        name: name.to_string(),
        created_at: now(),
        updated_at: now(),
        alias_of: None,
    };
    sqlx::query("INSERT INTO people (id, name, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&row.id)
        .bind(&row.name)
        .bind(&row.created_at)
        .bind(&row.updated_at)
        .execute(db)
        .await?;
    Ok(row)
}

pub async fn get_person(db: &Db, id: &str) -> Result<Option<PersonRow>, DbError> {
    let row =
        sqlx::query("SELECT id, name, created_at, updated_at, alias_of FROM people WHERE id = ?1")
            .bind(id)
            .fetch_optional(db)
            .await?;
    row.map(row_to_person).transpose()
}

/// Follow `alias_of` to the person somebody was merged into.
///
/// The twin of [`resolve_speaker`], hop-limited for the same reason: an id can
/// arrive from a window that has been open since before a merge, and the honest
/// answer to "who is this?" is the person they are now. A chain longer than the
/// limit — which [`merge_people`] cannot create — resolves to as far as it got
/// rather than looping.
pub async fn resolve_person(db: &Db, id: &str) -> Result<Option<PersonRow>, DbError> {
    let mut current = get_person(db, id).await?;
    let mut hops = 0;
    while let Some(person) = current.clone() {
        match person.alias_of {
            Some(next) if hops < 8 => {
                hops += 1;
                current = get_person(db, &next).await?;
            }
            _ => return Ok(Some(person)),
        }
    }
    Ok(None)
}

/// Everybody Echo remembers. Names that were merged into another one are not
/// people any more and do not appear.
pub async fn list_people(db: &Db) -> Result<Vec<PersonRow>, DbError> {
    let rows = sqlx::query(
        "SELECT id, name, created_at, updated_at, alias_of FROM people
         WHERE alias_of IS NULL ORDER BY name, created_at",
    )
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_person).collect()
}

pub async fn rename_person(db: &Db, id: &str, name: &str) -> Result<(), DbError> {
    let r = sqlx::query("UPDATE people SET name = ?2, updated_at = ?3 WHERE id = ?1")
        .bind(id)
        .bind(name)
        .bind(now())
        .execute(db)
        .await?;
    if r.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("person {id}")));
    }
    Ok(())
}

/// What one merge actually moved, so the caller can log it and say nothing
/// happened when nothing did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergedPeople {
    /// False when the two were already one person — a second click, or a merge
    /// somebody else's window already made. Everything else is zero then.
    pub merged: bool,
    pub samples_moved: u64,
    pub speakers_relinked: u64,
    pub suggestions_relinked: u64,
    /// Names that had already been merged into `merge_id` and now point at the
    /// survivor instead, so no chain is ever more than one hop long.
    pub aliases_flattened: u64,
}

/// Make one person out of two: `merge_id` becomes `keep_id`.
///
/// Every table that named the merged person is re-pointed and their row is left
/// behind pointing at the survivor (`migrations/0006_person_merge.sql` says why
/// the row stays). What moves:
///
/// * **`person_samples`** — the whole point. A duplicate deleted loses its
///   clips and its fingerprints; merged, both voices' samples end up in one set
///   for the caller to curate into one profile.
/// * **`speakers.person_id`** — every meeting that had been linked to the
///   merged name is now linked to the survivor. The *display names* on those
///   rows are not touched: they were copied onto the row when the link was made
///   and a meeting somebody has read does not rewrite itself (DESIGN §1, the
///   same rule [`rename_person`] and [`delete_person`] follow).
/// * **`speakers.suggested_person_id`** — a pending "looks like Marco?" about a
///   name that no longer exists would be unanswerable. A suggestion that lands
///   on the person the row is already linked to is dropped instead, because
///   "looks like Marco" beside "Marco" is noise ([`set_speaker_person`] takes
///   the same view).
/// * **`people.alias_of`** — anything already merged into `merge_id` is
///   re-pointed at the survivor, so chains stay one hop deep exactly as
///   [`merge_speakers`] keeps them.
///
/// The merged person's profile row goes: its samples are somebody else's now,
/// and a centroid with no samples under it is a number describing nothing.
/// Recomputing the **survivor's** profile is the caller's job — it needs to know
/// which network the numbers belong to, which this layer does not
/// (`diarize::people::merge`).
///
/// Refusals, both settled — the same request always gets the same answer:
///
/// * **A person cannot be merged into themselves.** Unlike [`merge_speakers`],
///   which shrugs at it, this is refused out loud: the only way to send the same
///   id twice is a UI that has lost track of which row is which, and silently
///   succeeding would tell it everything is fine.
/// * **No rings.** Both ids are first resolved through any merges already made
///   on them, so each one means "the person this name is now". If the two land
///   on the same person they already are one, and this is a no-op
///   (`merged: false`) — which is also what makes calling it twice safe, in
///   either order.
pub async fn merge_people(db: &Db, keep_id: &str, merge_id: &str) -> Result<MergedPeople, DbError> {
    if keep_id == merge_id {
        return Err(DbError::Invalid(
            "a person cannot be merged into themselves".into(),
        ));
    }
    let mut tx = db.begin().await?;

    // Both ids are read as "the person this name is now", so a merge asked for
    // from a window that has been open since an earlier one still means what it
    // says. Whichever way round, the work happens between the two survivors.
    let root = person_root(&mut tx, keep_id).await?;
    let doomed = person_root(&mut tx, merge_id).await?;
    if root == doomed {
        // Already one person — a second click, or a merge somebody else's window
        // already made. Merging now would be the ring the hop limit exists for.
        tx.rollback().await?;
        return Ok(MergedPeople::default());
    }
    let merge_id = doomed.as_str();

    let samples_moved =
        sqlx::query("UPDATE person_samples SET person_id = ?2 WHERE person_id = ?1")
            .bind(merge_id)
            .bind(&root)
            .execute(&mut *tx)
            .await?
            .rows_affected();
    let speakers_relinked = sqlx::query("UPDATE speakers SET person_id = ?2 WHERE person_id = ?1")
        .bind(merge_id)
        .bind(&root)
        .execute(&mut *tx)
        .await?
        .rows_affected();
    let suggestions_relinked =
        sqlx::query("UPDATE speakers SET suggested_person_id = ?2 WHERE suggested_person_id = ?1")
            .bind(merge_id)
            .bind(&root)
            .execute(&mut *tx)
            .await?
            .rows_affected();
    sqlx::query(
        "UPDATE speakers SET suggested_person_id = NULL, suggestion_score = NULL
         WHERE person_id = ?1 AND suggested_person_id = ?1",
    )
    .bind(&root)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM person_profiles WHERE person_id = ?1")
        .bind(merge_id)
        .execute(&mut *tx)
        .await?;
    let aliases_flattened = sqlx::query("UPDATE people SET alias_of = ?2 WHERE alias_of = ?1")
        .bind(merge_id)
        .bind(&root)
        .execute(&mut *tx)
        .await?
        .rows_affected();
    let r = sqlx::query("UPDATE people SET alias_of = ?2, updated_at = ?3 WHERE id = ?1")
        .bind(merge_id)
        .bind(&root)
        .bind(now())
        .execute(&mut *tx)
        .await?;
    if r.rows_affected() == 0 {
        tx.rollback().await?;
        return Err(DbError::NotFound(format!("person {merge_id}")));
    }
    tx.commit().await?;

    Ok(MergedPeople {
        merged: true,
        samples_moved,
        speakers_relinked,
        suggestions_relinked,
        aliases_flattened,
    })
}

/// Follow one person's merges to the name they are kept under now, inside a
/// transaction. Hop-limited so a hand-edited ring cannot spin here.
async fn person_root(tx: &mut sqlx::Transaction<'_, Sqlite>, id: &str) -> Result<String, DbError> {
    let start: (Option<String>,) = sqlx::query_as("SELECT alias_of FROM people WHERE id = ?1")
        .bind(id)
        .fetch_optional(&mut **tx)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("person {id}")))?;
    let mut root = id.to_string();
    let mut next = start.0;
    let mut hops = 0;
    while let Some(candidate) = next {
        if candidate == root || hops >= 8 {
            break;
        }
        let row: Option<(Option<String>,)> =
            sqlx::query_as("SELECT alias_of FROM people WHERE id = ?1")
                .bind(&candidate)
                .fetch_optional(&mut **tx)
                .await?;
        let Some((alias_of,)) = row else { break };
        root = candidate;
        next = alias_of;
        hops += 1;
    }
    Ok(root)
}

/// Forget a person: the profile, the samples and the clips.
///
/// Speaker links go to NULL by foreign key, and the names already copied onto
/// those speaker rows stay as plain text. That is the promise Settings makes —
/// "delete destroys profile, samples and clips" — and nothing more: a meeting
/// somebody has already read does not rewrite itself.
pub async fn delete_person(db: &Db, id: &str) -> Result<(), DbError> {
    let r = sqlx::query("DELETE FROM people WHERE id = ?1")
        .bind(id)
        .execute(db)
        .await?;
    if r.rows_affected() == 0 {
        return Err(DbError::NotFound(format!("person {id}")));
    }
    Ok(())
}

/// Settings → Data → delete everything, the people half.
///
/// Deliberately separate from [`delete_all_meetings`]: the two answer different
/// questions and `delete_all_data` asks both. Children first and by name, same
/// reasoning as everywhere else in this file.
pub async fn delete_all_people(db: &Db) -> Result<(), DbError> {
    let mut tx = db.begin().await?;
    for table in ["person_samples", "person_profiles", "people"] {
        sqlx::query(&format!("DELETE FROM {table}"))
            .execute(&mut *tx)
            .await?;
    }
    // The links live on speaker rows that survive delete-all only if their
    // meeting did; clearing them is belt and braces for a hand-edited database.
    sqlx::query(
        "UPDATE speakers SET person_id = NULL, suggested_person_id = NULL,
                suggestion_score = NULL",
    )
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(())
}

/// Everybody Echo remembers, as Settings → People shows them.
///
/// `current_embedder_asset_id` is the network in use right now; a profile tagged
/// with anything else is reported as needing a refresh, which is the honest
/// answer to "is this voice being matched at the moment" (DESIGN §1: "matching
/// silently skips when the tags disagree until a refresh runs"). A person with no
/// profile row at all — enrolled from a voice Echo could not fingerprint — needs
/// one too.
pub async fn list_person_infos(
    db: &Db,
    current_embedder_asset_id: &str,
) -> Result<Vec<crate::types::PersonInfo>, DbError> {
    let rows = sqlx::query(
        "SELECT p.id            AS id,
                p.name          AS name,
                (SELECT COUNT(*) FROM person_samples s WHERE s.person_id = p.id) AS sample_count,
                (SELECT MAX(m.started_at) FROM speakers k
                   JOIN meetings m ON m.id = k.meeting_id
                  WHERE k.person_id = p.id AND m.deleted_at IS NULL)             AS last_heard_at,
                f.embedder_asset_id AS embedder_asset_id
         FROM people p
         LEFT JOIN person_profiles f ON f.person_id = p.id
         WHERE p.alias_of IS NULL
         ORDER BY p.name, p.created_at",
    )
    .fetch_all(db)
    .await?;

    rows.into_iter()
        .map(|row| {
            let tag: Option<String> = row.try_get("embedder_asset_id").map_err(decode)?;
            let samples: i64 = row.try_get("sample_count").map_err(decode)?;
            Ok(crate::types::PersonInfo {
                id: row.try_get("id").map_err(decode)?,
                name: row.try_get("name").map_err(decode)?,
                sample_count: samples.max(0) as u32,
                last_heard_at: row.try_get("last_heard_at").map_err(decode)?,
                needs_refresh: tag.as_deref() != Some(current_embedder_asset_id),
            })
        })
        .collect()
}

pub async fn insert_person_sample(db: &Db, sample: &NewPersonSample<'_>) -> Result<Id, DbError> {
    let id = new_id();
    sqlx::query(
        "INSERT INTO person_samples
             (id, person_id, embedding, clip, condition, source_meeting_id,
              t_start_ms, t_end_ms, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
    )
    .bind(&id)
    .bind(sample.person_id)
    .bind(embedding_to_blob(sample.embedding))
    .bind(sample.clip)
    .bind(sample.condition.as_str())
    .bind(sample.source_meeting_id)
    .bind(sample.t_start_ms)
    .bind(sample.t_end_ms)
    .bind(now())
    .execute(db)
    .await?;
    Ok(id)
}

/// This person's samples, newest first, without their audio.
pub async fn list_person_samples(
    db: &Db,
    person_id: &str,
) -> Result<Vec<PersonSampleRow>, DbError> {
    let rows = sqlx::query(
        "SELECT id, person_id, embedding, condition, source_meeting_id,
                t_start_ms, t_end_ms, created_at
         FROM person_samples WHERE person_id = ?1
         ORDER BY created_at DESC, id",
    )
    .bind(person_id)
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_person_sample).collect()
}

/// Every sample's audio, for re-embedding when the network changes.
pub async fn person_sample_clips(db: &Db, person_id: &str) -> Result<Vec<(Id, Vec<u8>)>, DbError> {
    let rows: Vec<(String, Vec<u8>)> = sqlx::query_as(
        "SELECT id, clip FROM person_samples WHERE person_id = ?1 ORDER BY created_at, id",
    )
    .bind(person_id)
    .fetch_all(db)
    .await?;
    Ok(rows)
}

/// The most recent clip of this voice, for the Listen button.
pub async fn freshest_person_clip(db: &Db, person_id: &str) -> Result<Option<Vec<u8>>, DbError> {
    let row: Option<(Vec<u8>,)> = sqlx::query_as(
        "SELECT clip FROM person_samples WHERE person_id = ?1
         ORDER BY created_at DESC, id LIMIT 1",
    )
    .bind(person_id)
    .fetch_optional(db)
    .await?;
    Ok(row.map(|r| r.0))
}

pub async fn set_person_sample_embedding(
    db: &Db,
    sample_id: &str,
    embedding: &[f32],
) -> Result<(), DbError> {
    sqlx::query("UPDATE person_samples SET embedding = ?2 WHERE id = ?1")
        .bind(sample_id)
        .bind(embedding_to_blob(embedding))
        .execute(db)
        .await?;
    Ok(())
}

pub async fn delete_person_samples(db: &Db, ids: &[Id]) -> Result<u64, DbError> {
    if ids.is_empty() {
        return Ok(0);
    }
    let mut qb: QueryBuilder<Sqlite> =
        QueryBuilder::new("DELETE FROM person_samples WHERE id IN (");
    let mut sep = qb.separated(", ");
    for id in ids {
        sep.push_bind(id);
    }
    qb.push(")");
    Ok(qb.build().execute(db).await?.rows_affected())
}

/// Write the profile the samples add up to. One row per person; a second call
/// replaces it.
pub async fn upsert_person_profile(
    db: &Db,
    person_id: &str,
    centroid: &[f32],
    sample_count: u32,
    embedder_asset_id: &str,
) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO person_profiles
             (person_id, centroid, sample_count, embedder_asset_id, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT (person_id) DO UPDATE SET
             centroid = excluded.centroid,
             sample_count = excluded.sample_count,
             embedder_asset_id = excluded.embedder_asset_id,
             updated_at = excluded.updated_at",
    )
    .bind(person_id)
    .bind(embedding_to_blob(centroid))
    .bind(sample_count)
    .bind(embedder_asset_id)
    .bind(now())
    .execute(db)
    .await?;
    Ok(())
}

/// This person's profile row, or nothing when they have none — which is what an
/// enrolment Echo could not fingerprint looks like, and what a profile
/// [`crate::diarize::people::curate`] gave up on looks like.
pub async fn person_profile(db: &Db, person_id: &str) -> Result<Option<PersonProfileRow>, DbError> {
    let row = sqlx::query(
        "SELECT f.person_id, p.name, f.centroid, f.sample_count, f.embedder_asset_id,
                f.updated_at
         FROM person_profiles f JOIN people p ON p.id = f.person_id
         WHERE f.person_id = ?1",
    )
    .bind(person_id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_person_profile).transpose()
}

pub async fn delete_person_profile(db: &Db, person_id: &str) -> Result<(), DbError> {
    sqlx::query("DELETE FROM person_profiles WHERE person_id = ?1")
        .bind(person_id)
        .execute(db)
        .await?;
    Ok(())
}

/// Every profile there is, tags included. Callers decide which tags they can
/// compare against — [`crate::diarize::people`] does, and refuses the rest.
pub async fn list_person_profiles(db: &Db) -> Result<Vec<PersonProfileRow>, DbError> {
    let rows = sqlx::query(
        "SELECT f.person_id, p.name, f.centroid, f.sample_count, f.embedder_asset_id,
                f.updated_at
         FROM person_profiles f JOIN people p ON p.id = f.person_id
         WHERE p.alias_of IS NULL
         ORDER BY p.name, f.person_id",
    )
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_person_profile).collect()
}

/// Every stored voice print that still belongs to nobody.
///
/// The raw material for "this voice keeps turning up": one row per speaker the
/// offline pass fingerprinted, from live meetings only, skipping the ones already
/// linked to a person and the ones merged into another row (a merge means the
/// surviving row already speaks for this voice).
pub async fn list_unnamed_voice_prints(db: &Db) -> Result<Vec<VoicePrint>, DbError> {
    let rows = sqlx::query(
        "SELECT k.id AS speaker_id, k.meeting_id, k.display_name, k.speaking_ms, k.centroid,
                m.title AS meeting_title, m.started_at
         FROM speakers k JOIN meetings m ON m.id = k.meeting_id
         WHERE k.centroid IS NOT NULL
           AND k.person_id IS NULL
           AND k.alias_of IS NULL
           AND k.is_self = 0
           AND m.deleted_at IS NULL
         ORDER BY m.started_at, k.cluster_key",
    )
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_voice_print).collect()
}

// ===========================================================================
// markers
// ===========================================================================

pub async fn insert_marker(
    db: &Db,
    meeting_id: &str,
    t_ms: i64,
    kind: MarkerKind,
    note: Option<&str>,
) -> Result<Marker, DbError> {
    let id = new_id();
    sqlx::query(
        "INSERT INTO markers (id, meeting_id, t_ms, kind, note) VALUES (?1, ?2, ?3, ?4, ?5)",
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(t_ms)
    .bind(kind.as_str())
    .bind(note)
    .execute(db)
    .await?;
    Ok(Marker {
        id,
        meeting_id: meeting_id.to_string(),
        t_ms,
        kind,
        note: note.map(String::from),
    })
}

pub async fn list_markers(db: &Db, meeting_id: &str) -> Result<Vec<Marker>, DbError> {
    let rows = sqlx::query(
        "SELECT id, meeting_id, t_ms, kind, note FROM markers WHERE meeting_id = ?1 ORDER BY t_ms",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_marker).collect()
}

pub async fn delete_marker(db: &Db, id: &str) -> Result<(), DbError> {
    sqlx::query("DELETE FROM markers WHERE id = ?1")
        .bind(id)
        .execute(db)
        .await?;
    Ok(())
}

// ===========================================================================
// templates
// ===========================================================================

pub async fn list_templates(db: &Db) -> Result<Vec<Template>, DbError> {
    let rows = sqlx::query(
        "SELECT id, name, prompt_md, builtin FROM templates ORDER BY builtin DESC, name",
    )
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_template).collect()
}

pub async fn get_template(db: &Db, id: &str) -> Result<Option<Template>, DbError> {
    let row = sqlx::query("SELECT id, name, prompt_md, builtin FROM templates WHERE id = ?1")
        .bind(id)
        .fetch_optional(db)
        .await?;
    row.map(row_to_template).transpose()
}

/// The general recap, used when nothing else is selected.
pub async fn default_template(db: &Db) -> Result<Option<Template>, DbError> {
    let row = sqlx::query(
        "SELECT id, name, prompt_md, builtin FROM templates WHERE builtin = 1 ORDER BY id LIMIT 1",
    )
    .fetch_optional(db)
    .await?;
    row.map(row_to_template).transpose()
}

/// Create or update a custom template. Built-ins are read-only.
pub async fn upsert_template(
    db: &Db,
    id: Option<&str>,
    name: &str,
    prompt_md: &str,
) -> Result<Template, DbError> {
    match id {
        Some(existing) => {
            let current = get_template(db, existing)
                .await?
                .ok_or_else(|| DbError::NotFound(format!("template {existing}")))?;
            if current.builtin {
                return Err(DbError::Decode(
                    "built-in templates cannot be changed".into(),
                ));
            }
            sqlx::query("UPDATE templates SET name = ?2, prompt_md = ?3 WHERE id = ?1")
                .bind(existing)
                .bind(name)
                .bind(prompt_md)
                .execute(db)
                .await?;
            Ok(Template {
                id: existing.to_string(),
                name: name.to_string(),
                prompt_md: prompt_md.to_string(),
                builtin: false,
            })
        }
        None => {
            let id = new_id();
            sqlx::query(
                "INSERT INTO templates (id, name, prompt_md, builtin) VALUES (?1, ?2, ?3, 0)",
            )
            .bind(&id)
            .bind(name)
            .bind(prompt_md)
            .execute(db)
            .await?;
            Ok(Template {
                id,
                name: name.to_string(),
                prompt_md: prompt_md.to_string(),
                builtin: false,
            })
        }
    }
}

pub async fn delete_template(db: &Db, id: &str) -> Result<(), DbError> {
    let r = sqlx::query("DELETE FROM templates WHERE id = ?1 AND builtin = 0")
        .bind(id)
        .execute(db)
        .await?;
    if r.rows_affected() == 0 {
        return Err(DbError::Decode("that recap style can't be deleted".into()));
    }
    Ok(())
}

// ===========================================================================
// summaries
// ===========================================================================

#[allow(clippy::too_many_arguments)]
pub async fn insert_summary(
    db: &Db,
    meeting_id: &str,
    template_id: Option<&str>,
    template_snapshot: Option<&str>,
    provider: Provider,
    model: Option<&str>,
    language: Option<&str>,
    transcript_revision: i64,
    content_md: &str,
) -> Result<Summary, DbError> {
    let id = new_id();
    let created_at = now();
    sqlx::query(
        "INSERT INTO summaries (id, meeting_id, template_id, template_snapshot, provider, model,
                                language, transcript_revision, content_md, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(template_id)
    .bind(template_snapshot)
    .bind(provider.as_str())
    .bind(model)
    .bind(language)
    .bind(transcript_revision)
    .bind(content_md)
    .bind(&created_at)
    .execute(db)
    .await?;

    Ok(Summary {
        id,
        meeting_id: meeting_id.to_string(),
        template_id: template_id.map(String::from),
        template_snapshot: template_snapshot.map(String::from),
        provider,
        model: model.map(String::from),
        language: language.map(String::from),
        transcript_revision,
        content_md: content_md.to_string(),
        created_at,
    })
}

pub async fn list_summaries(db: &Db, meeting_id: &str) -> Result<Vec<Summary>, DbError> {
    let rows = sqlx::query(
        "SELECT id, meeting_id, template_id, template_snapshot, provider, model, language,
                transcript_revision, content_md, created_at
         FROM summaries WHERE meeting_id = ?1 ORDER BY created_at DESC",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_summary).collect()
}

pub async fn get_summary(db: &Db, id: &str) -> Result<Option<Summary>, DbError> {
    let row = sqlx::query(
        "SELECT id, meeting_id, template_id, template_snapshot, provider, model, language,
                transcript_revision, content_md, created_at
         FROM summaries WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_summary).transpose()
}

pub async fn latest_summary(db: &Db, meeting_id: &str) -> Result<Option<Summary>, DbError> {
    Ok(list_summaries(db, meeting_id).await?.into_iter().next())
}

pub async fn delete_summary(db: &Db, id: &str) -> Result<(), DbError> {
    sqlx::query("DELETE FROM summaries WHERE id = ?1")
        .bind(id)
        .execute(db)
        .await?;
    Ok(())
}

// ===========================================================================
// action_items
// ===========================================================================

/// Replace the action items belonging to one recap, keeping `done` flags for
/// items whose text did not change.
pub async fn replace_action_items(
    db: &Db,
    meeting_id: &str,
    summary_id: Option<&str>,
    items: &[ActionItem],
) -> Result<Vec<ActionItem>, DbError> {
    let existing = list_action_items(db, meeting_id).await?;
    let done_by_text: HashMap<String, bool> = existing
        .iter()
        .map(|i| (i.description.trim().to_lowercase(), i.done))
        .collect();

    let mut tx = db.begin().await?;
    match summary_id {
        Some(sid) => {
            sqlx::query("DELETE FROM action_items WHERE summary_id = ?1")
                .bind(sid)
                .execute(&mut *tx)
                .await?;
        }
        None => {
            sqlx::query("DELETE FROM action_items WHERE meeting_id = ?1 AND summary_id IS NULL")
                .bind(meeting_id)
                .execute(&mut *tx)
                .await?;
        }
    }

    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let id = new_id();
        let done = done_by_text
            .get(&item.description.trim().to_lowercase())
            .copied()
            .unwrap_or(item.done);
        sqlx::query(
            "INSERT INTO action_items (id, meeting_id, summary_id, description, owner, due_hint,
                                       done, external_url)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(&id)
        .bind(meeting_id)
        .bind(summary_id)
        .bind(&item.description)
        .bind(item.owner.as_deref())
        .bind(item.due_hint.as_deref())
        .bind(done)
        .bind(item.external_url.as_deref())
        .execute(&mut *tx)
        .await?;
        out.push(ActionItem {
            id,
            meeting_id: meeting_id.to_string(),
            summary_id: summary_id.map(String::from),
            done,
            ..item.clone()
        });
    }
    tx.commit().await?;
    Ok(out)
}

pub async fn list_action_items(db: &Db, meeting_id: &str) -> Result<Vec<ActionItem>, DbError> {
    let rows = sqlx::query(
        "SELECT id, meeting_id, summary_id, description, owner, due_hint, done, external_url
         FROM action_items WHERE meeting_id = ?1 ORDER BY done, rowid",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    rows.into_iter().map(row_to_action_item).collect()
}

pub async fn patch_action_item(db: &Db, patch: &ActionItemPatch) -> Result<ActionItem, DbError> {
    sqlx::query(
        "UPDATE action_items
         SET description = COALESCE(?2, description),
             owner = COALESCE(?3, owner),
             due_hint = COALESCE(?4, due_hint),
             done = COALESCE(?5, done),
             external_url = COALESCE(?6, external_url)
         WHERE id = ?1",
    )
    .bind(&patch.id)
    .bind(patch.description.as_deref())
    .bind(patch.owner.as_deref())
    .bind(patch.due_hint.as_deref())
    .bind(patch.done)
    .bind(patch.external_url.as_deref())
    .execute(db)
    .await?;

    let row = sqlx::query(
        "SELECT id, meeting_id, summary_id, description, owner, due_hint, done, external_url
         FROM action_items WHERE id = ?1",
    )
    .bind(&patch.id)
    .fetch_optional(db)
    .await?
    .ok_or_else(|| DbError::NotFound(format!("action item {}", patch.id)))?;
    row_to_action_item(row)
}

// ===========================================================================
// jobs
// ===========================================================================

pub async fn create_job(db: &Db, meeting_id: Option<&str>, kind: JobKind) -> Result<Job, DbError> {
    let id = new_id();
    let ts = now();
    sqlx::query(
        "INSERT INTO jobs (id, meeting_id, kind, status, progress, created_at, updated_at)
         VALUES (?1, ?2, ?3, 'queued', NULL, ?4, ?4)",
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(kind.as_str())
    .bind(&ts)
    .execute(db)
    .await?;
    Ok(Job {
        id,
        meeting_id: meeting_id.map(String::from),
        kind,
        status: JobStatus::Queued,
        progress: None,
        error: None,
        created_at: ts.clone(),
        updated_at: ts,
        phase: None,
    })
}

/// One job of a kind per meeting: reuse an unfinished one instead of piling up
/// duplicates when the person double-clicks (IPC must be idempotent, §3).
pub async fn ensure_job(db: &Db, meeting_id: Option<&str>, kind: JobKind) -> Result<Job, DbError> {
    ensure_job_with_payload(db, meeting_id, kind, None).await
}

/// [`ensure_job`], remembering what was asked for.
///
/// `payload` is opaque JSON the handler for this kind understands. When an
/// unfinished job of the same kind already exists its payload is *replaced*, so
/// asking again with a different recap style uses the newer request rather than
/// quietly running the older one.
pub async fn ensure_job_with_payload(
    db: &Db,
    meeting_id: Option<&str>,
    kind: JobKind,
    payload: Option<&str>,
) -> Result<Job, DbError> {
    let existing = list_jobs(
        db,
        &JobQuery {
            meeting_id: meeting_id.map(String::from),
            kind: Some(kind),
            active_only: Some(true),
            ..Default::default()
        },
    )
    .await?;
    if let Some(job) = existing.into_iter().next() {
        if payload.is_some() {
            set_job_payload(db, &job.id, payload).await?;
        }
        return Ok(job);
    }
    let job = create_job(db, meeting_id, kind).await?;
    if payload.is_some() {
        set_job_payload(db, &job.id, payload).await?;
    }
    Ok(job)
}

/// Store what a job was asked for. Never reaches the UI: [`Job`] deliberately
/// has no payload field, because none of it is anything a person reads.
pub async fn set_job_payload(db: &Db, id: &str, payload: Option<&str>) -> Result<(), DbError> {
    sqlx::query("UPDATE jobs SET payload = ?2 WHERE id = ?1")
        .bind(id)
        .bind(payload)
        .execute(db)
        .await?;
    Ok(())
}

/// Read back what a job was asked for.
pub async fn get_job_payload(db: &Db, id: &str) -> Result<Option<String>, DbError> {
    let row: Option<(Option<String>,)> = sqlx::query_as("SELECT payload FROM jobs WHERE id = ?1")
        .bind(id)
        .fetch_optional(db)
        .await?;
    Ok(row.and_then(|(p,)| p))
}

pub async fn get_job(db: &Db, id: &str) -> Result<Option<Job>, DbError> {
    let row = sqlx::query(
        "SELECT id, meeting_id, kind, status, progress, error, created_at, updated_at, phase
         FROM jobs WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_job).transpose()
}

pub async fn list_jobs(db: &Db, q: &JobQuery) -> Result<Vec<Job>, DbError> {
    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, meeting_id, kind, status, progress, error, created_at, updated_at, phase
         FROM jobs WHERE 1 = 1",
    );
    if let Some(m) = &q.meeting_id {
        qb.push(" AND meeting_id = ").push_bind(m.clone());
    }
    if let Some(k) = q.kind {
        qb.push(" AND kind = ").push_bind(k.as_str().to_string());
    }
    if let Some(s) = q.status {
        qb.push(" AND status = ").push_bind(s.as_str().to_string());
    }
    if q.active_only.unwrap_or(false) {
        qb.push(" AND status IN ('queued', 'running', 'paused')");
    }
    qb.push(" ORDER BY created_at DESC LIMIT ")
        .push_bind(i64::from(q.limit.unwrap_or(200).min(1_000)));
    let rows = qb.build().fetch_all(db).await?;
    rows.into_iter().map(row_to_job).collect()
}

/// Write down where a job stands, and drop whatever stage it was in.
///
/// The stage is only ever true of a job that is running this second, so every
/// change of status ends it: starting, finishing, failing, being cancelled,
/// being parked for a recording, being put back in the queue. Clearing it in
/// the same statement is what makes it safe for a screen to read the stage off
/// the row without asking how old it is.
pub async fn set_job_status(
    db: &Db,
    id: &str,
    status: JobStatus,
    error: Option<&str>,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE jobs SET status = ?2, error = ?3, phase = NULL, updated_at = ?4 WHERE id = ?1",
    )
    .bind(id)
    .bind(status.as_str())
    .bind(error)
    .bind(now())
    .execute(db)
    .await?;
    Ok(())
}

pub async fn set_job_progress(db: &Db, id: &str, progress: f32) -> Result<(), DbError> {
    sqlx::query("UPDATE jobs SET progress = ?2, updated_at = ?3 WHERE id = ?1")
        .bind(id)
        .bind(progress.clamp(0.0, 1.0))
        .bind(now())
        .execute(db)
        .await?;
    Ok(())
}

/// Put the job into a stage, or take it out of one.
///
/// `None` is not a tidy-up nobody needs: a job that has finished waiting for
/// the engine and started reporting fractions again is no longer in that stage,
/// and a sentence about a one-time setup left standing over a moving bar is a
/// lie the same size as the missing one this column was added to fix.
///
/// `updated_at` is deliberately not touched. The stage is about what is
/// happening, and it is announced in the same breath it is written; moving the
/// timestamp would make a fifteen-minute silence look like fifteen minutes of
/// activity to anything reading the table for signs of life.
pub async fn set_job_phase(db: &Db, id: &str, phase: Option<JobPhase>) -> Result<(), DbError> {
    sqlx::query("UPDATE jobs SET phase = ?2 WHERE id = ?1")
        .bind(id)
        .bind(phase.map(|p| p.as_str()))
        .execute(db)
        .await?;
    Ok(())
}

/// Recording has absolute priority: park everything that is running.
pub async fn pause_active_jobs(db: &Db) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE jobs SET status = 'paused', phase = NULL, updated_at = ?1
         WHERE status IN ('running', 'queued')",
    )
    .bind(now())
    .execute(db)
    .await?;
    Ok(r.rows_affected())
}

/// Recording stopped; let background work continue.
pub async fn resume_paused_jobs(db: &Db) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE jobs SET status = 'queued', phase = NULL, updated_at = ?1
             WHERE status = 'paused'",
    )
    .bind(now())
    .execute(db)
    .await?;
    Ok(r.rows_affected())
}

/// A crash leaves jobs marked running, and a crash *during a recording* leaves
/// them marked paused. Neither can be true at launch: nothing is running, and
/// "a recording owns the machine" is an in-memory flag that starts out clear.
/// Both go back in the queue, keeping their progress, or the work a recording
/// parked would sit there for good and its meeting would never leave
/// "Processing".
///
/// One kind of work is settled rather than requeued, in the same breath so that
/// no caller can get the order wrong: a `prepare_engine` row that was still
/// marked running was, in all likelihood, in the middle of the one-time compile
/// that took a quarter of an hour on 2026-08-24 — and whatever took the process
/// down in the middle of it (an out-of-memory, a driver fault, somebody
/// force-quitting an app that looked frozen) would do exactly the same thing to
/// the next attempt, at this launch and every launch after it. So it is left
/// failed, wearing `setup_interrupted` as its reason. Nothing is lost: the
/// meeting that needs the engine still loads it, and the settings marker written
/// before the load is what keeps anything from queueing a fresh row on its own.
///
/// Returns how many rows were touched either way.
pub async fn requeue_orphaned_jobs(db: &Db, setup_interrupted: &str) -> Result<u64, DbError> {
    let settled = sqlx::query(
        "UPDATE jobs SET status = 'failed', error = ?2, phase = NULL, updated_at = ?1
         WHERE status = 'running' AND kind = ?3",
    )
    .bind(now())
    .bind(setup_interrupted)
    .bind(JobKind::PrepareEngine.as_str())
    .execute(db)
    .await?
    .rows_affected();
    let r = sqlx::query(
        "UPDATE jobs SET status = 'queued', phase = NULL, updated_at = ?1
         WHERE status IN ('running', 'paused')",
    )
    .bind(now())
    .execute(db)
    .await?;
    Ok(settled + r.rows_affected())
}

/// Give the one-time speech setup another go, at launch and only at launch.
///
/// Scoped to the weights named by `payload`, so a row that failed for a model
/// somebody has since moved on from stays where it is. The caller decides
/// whether a retry is allowed at all — the marker in settings says whether the
/// last attempt ever reached the compile — and calls this at most once per
/// launch. A launch apart is the right spacing for a retry of something that
/// failed for a reason nobody can see from here: it is long enough to be a
/// different day, a different disk and a different amount of free memory, and it
/// cannot become the tight loop that queueing on every readiness poll would.
pub async fn requeue_failed_setup_jobs(db: &Db, payload: &str) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE jobs SET status = 'queued', error = NULL, phase = NULL, updated_at = ?1
         WHERE status = 'failed' AND kind = ?2 AND payload = ?3",
    )
    .bind(now())
    .bind(JobKind::PrepareEngine.as_str())
    .bind(payload)
    .execute(db)
    .await?;
    Ok(r.rows_affected())
}

/// Whether a job of this kind has already failed carrying exactly this payload.
///
/// The record that this precise piece of work has been tried and did not come
/// off — which is what stops the caller asking for it again a few milliseconds
/// later, and again, for as long as the app is open.
pub async fn has_failed_job_with_payload(
    db: &Db,
    kind: JobKind,
    payload: &str,
) -> Result<bool, DbError> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT 1 FROM jobs WHERE kind = ?1 AND status = 'failed' AND payload = ?2 LIMIT 1",
    )
    .bind(kind.as_str())
    .bind(payload)
    .fetch_optional(db)
    .await?;
    Ok(row.is_some())
}

/// The queued job to run next, honouring the priority order in DESIGN §3.
///
/// Kind first — catch-up before speakers before the mixdown before the recap —
/// and then **the newest meeting first**.
///
/// The tie-break used to be the oldest, which sounds fair and is not what a
/// person wants: finishing a meeting costs about a sixth of its length, so on a
/// day of back-to-back meetings the queue always held work from more than one.
/// Oldest-first meant the meeting somebody had just walked out of — the one they
/// are about to open, the one they want the recap of — waited behind a meeting
/// from two hours ago they have already read. Newest-first serves the meeting
/// they are looking at, and the older one loses nothing by waiting: it keeps its
/// place, its progress and its parked scan.
///
/// This is only safe to say because an interrupted catch-up now parks rather
/// than dropping what it had; before that, changing this order changed which
/// meeting lost transcript.
pub async fn next_queued_job(db: &Db) -> Result<Option<Job>, DbError> {
    let row = sqlx::query(
        "SELECT id, meeting_id, kind, status, progress, error, created_at, updated_at, phase
         FROM jobs WHERE status = 'queued'
         ORDER BY CASE kind
                    WHEN 'download' THEN 0
                    WHEN 'transcribe_catchup' THEN 1
                    WHEN 'diarize' THEN 2
                    WHEN 'mixdown' THEN 3
                    WHEN 'summarize' THEN 4
                    ELSE 5
                  END,
                  created_at DESC
         LIMIT 1",
    )
    .fetch_optional(db)
    .await?;
    row.map(row_to_job).transpose()
}

pub async fn delete_finished_jobs(db: &Db) -> Result<u64, DbError> {
    let r = sqlx::query("DELETE FROM jobs WHERE status IN ('done', 'cancelled')")
        .execute(db)
        .await?;
    Ok(r.rows_affected())
}

// ===========================================================================
// models (the download catalog)
// ===========================================================================

pub async fn upsert_model(db: &Db, m: &ModelInfo) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO models (id, kind, name, url, sha256, bytes, license, revision, installed, path)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT (id) DO UPDATE SET
             kind = excluded.kind,
             name = excluded.name,
             url = excluded.url,
             sha256 = excluded.sha256,
             bytes = excluded.bytes,
             license = excluded.license,
             revision = excluded.revision",
    )
    .bind(&m.id)
    .bind(m.kind.as_str())
    .bind(&m.name)
    .bind(&m.url)
    .bind(&m.sha256)
    .bind(m.bytes)
    .bind(m.license.as_deref())
    .bind(m.revision.as_deref())
    .bind(m.installed)
    .bind(m.path.as_deref())
    .execute(db)
    .await?;
    Ok(())
}

pub async fn list_models(db: &Db, kind: Option<AssetKind>) -> Result<Vec<ModelInfo>, DbError> {
    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, kind, name, url, sha256, bytes, license, revision, installed, path
         FROM models WHERE 1 = 1",
    );
    if let Some(k) = kind {
        qb.push(" AND kind = ").push_bind(k.as_str().to_string());
    }
    qb.push(" ORDER BY kind, name");
    let rows = qb.build().fetch_all(db).await?;
    rows.into_iter().map(row_to_model).collect()
}

pub async fn get_model(db: &Db, id: &str) -> Result<Option<ModelInfo>, DbError> {
    let row = sqlx::query(
        "SELECT id, kind, name, url, sha256, bytes, license, revision, installed, path
         FROM models WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_model).transpose()
}

pub async fn set_model_installed(
    db: &Db,
    id: &str,
    installed: bool,
    path: Option<&str>,
) -> Result<(), DbError> {
    sqlx::query("UPDATE models SET installed = ?2, path = ?3 WHERE id = ?1")
        .bind(id)
        .bind(installed)
        .bind(path)
        .execute(db)
        .await?;
    Ok(())
}

pub async fn installed_model_bytes(db: &Db) -> Result<u64, DbError> {
    let row: Option<(Option<i64>,)> =
        sqlx::query_as("SELECT SUM(bytes) FROM models WHERE installed = 1")
            .fetch_optional(db)
            .await?;
    Ok(row.and_then(|r| r.0).unwrap_or(0).max(0) as u64)
}

// ===========================================================================
// settings (raw key/value; see settings.rs for the typed view)
// ===========================================================================

pub async fn get_setting(db: &Db, key: &str) -> Result<Option<String>, DbError> {
    let row: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key = ?1")
        .bind(key)
        .fetch_optional(db)
        .await?;
    Ok(row.map(|r| r.0))
}

pub async fn set_setting(db: &Db, key: &str, value: &str) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO settings (key, value) VALUES (?1, ?2)
         ON CONFLICT (key) DO UPDATE SET value = excluded.value",
    )
    .bind(key)
    .bind(value)
    .execute(db)
    .await?;
    Ok(())
}

pub async fn set_settings(db: &Db, pairs: &[(String, String)]) -> Result<(), DbError> {
    if pairs.is_empty() {
        return Ok(());
    }
    let mut tx = db.begin().await?;
    for (k, v) in pairs {
        sqlx::query(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        )
        .bind(k)
        .bind(v)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

pub async fn all_settings(db: &Db) -> Result<HashMap<String, String>, DbError> {
    let rows: Vec<(String, String)> = sqlx::query_as("SELECT key, value FROM settings")
        .fetch_all(db)
        .await?;
    Ok(rows.into_iter().collect())
}

pub async fn delete_setting(db: &Db, key: &str) -> Result<(), DbError> {
    sqlx::query("DELETE FROM settings WHERE key = ?1")
        .bind(key)
        .execute(db)
        .await?;
    Ok(())
}

// ===========================================================================
// search
// ===========================================================================

/// Which index to use. The word index handles most languages; the trigram
/// index is the only thing that can find substrings in scripts without spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtsIndex {
    Words,
    Trigram,
}

/// Turn raw user input into a safe FTS5 MATCH expression.
///
/// Every token becomes a quoted phrase, so `*`, `-`, `NEAR`, `AND`, `:` and
/// unbalanced quotes are literal text rather than syntax. Returns an empty
/// string when there is nothing searchable, callers must then return no
/// results instead of running a MATCH.
pub fn escape_fts_query(input: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for token in input.split_whitespace() {
        let cleaned: String = token.chars().filter(|c| !c.is_control()).collect();
        if cleaned.is_empty() {
            continue;
        }
        let mut quoted = String::with_capacity(cleaned.len() + 2);
        quoted.push('"');
        for ch in cleaned.chars() {
            if ch == '"' {
                quoted.push('"'); // FTS5 escapes a quote by doubling it
            }
            quoted.push(ch);
        }
        quoted.push('"');
        parts.push(quoted);
    }
    parts.join(" AND ")
}

/// Trigram is needed when the text has no word boundaries a tokenizer can see.
pub fn pick_index(input: &str) -> FtsIndex {
    let scriptless = input.chars().any(|c| {
        let u = c as u32;
        (0x3040..=0x30FF).contains(&u)      // Hiragana, Katakana
            || (0x3400..=0x4DBF).contains(&u) // CJK ext A
            || (0x4E00..=0x9FFF).contains(&u) // CJK
            || (0xF900..=0xFAFF).contains(&u) // CJK compat
            || (0x0E00..=0x0E7F).contains(&u) // Thai
    });
    if scriptless {
        FtsIndex::Trigram
    } else {
        FtsIndex::Words
    }
}

/// Full-text search across meetings (or inside one).
pub async fn search_segments(db: &Db, q: &SearchQuery) -> Result<Vec<SearchHit>, DbError> {
    let expr = escape_fts_query(&q.text);
    if expr.is_empty() {
        return Ok(Vec::new());
    }
    let table = match pick_index(&q.text) {
        FtsIndex::Words => "segments_fts",
        FtsIndex::Trigram => "segments_fts_trigram",
    };

    // `table` is one of two literals chosen above, never user input.
    let sql = format!(
        "SELECT g.id AS segment_id, g.meeting_id, g.t_start_ms,
                m.title, m.started_at,
                sp.display_name AS speaker_name,
                snippet({table}, 0, '{MARK_OPEN}', '{MARK_CLOSE}', '…', 14) AS snip
         FROM {table} f
         JOIN segments g ON g.rowid = f.rowid
         JOIN meetings m ON m.id = g.meeting_id
         LEFT JOIN speakers sp ON sp.id = g.speaker_id
         WHERE {table} MATCH ?1
           AND m.deleted_at IS NULL
           AND (?2 IS NULL OR g.meeting_id = ?2)
         ORDER BY bm25({table}), g.t_start_ms
         LIMIT ?3 OFFSET ?4"
    );

    let rows = sqlx::query(&sql)
        .bind(&expr)
        .bind(q.meeting_id.as_deref())
        .bind(i64::from(q.limit.unwrap_or(50).min(500)))
        .bind(i64::from(q.offset.unwrap_or(0)))
        .fetch_all(db)
        .await?;

    rows.into_iter()
        .map(|row| {
            let snip: String = row.try_get("snip").map_err(decode)?;
            Ok(SearchHit {
                meeting_id: row.try_get("meeting_id").map_err(decode)?,
                meeting_title: row.try_get("title").map_err(decode)?,
                started_at: row.try_get("started_at").map_err(decode)?,
                segment_id: row.try_get("segment_id").map_err(decode)?,
                t_start_ms: row.try_get("t_start_ms").map_err(decode)?,
                snippet_html: marked_snippet_to_html(&snip),
                speaker_name: row.try_get("speaker_name").map_err(decode)?,
            })
        })
        .collect()
}

/// HTML-escape the snippet first, then turn the control-character sentinels
/// into `<mark>`, so transcript text can never inject markup.
fn marked_snippet_to_html(snippet: &str) -> String {
    let escaped = snippet
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;");
    escaped
        .replace(MARK_OPEN, "<mark>")
        .replace(MARK_CLOSE, "</mark>")
}

// ===========================================================================
// row mappers
// ===========================================================================

fn decode(e: sqlx::Error) -> DbError {
    DbError::Decode(e.to_string())
}

fn bad_enum(column: &str, value: &str) -> DbError {
    DbError::Decode(format!("unknown {column} value {value:?}"))
}

fn row_to_meeting(row: sqlx::sqlite::SqliteRow) -> Result<Meeting, DbError> {
    let status: String = row.try_get("status").map_err(decode)?;
    Ok(Meeting {
        id: row.try_get("id").map_err(decode)?,
        title: row.try_get("title").map_err(decode)?,
        started_at: row.try_get("started_at").map_err(decode)?,
        ended_at: row.try_get("ended_at").map_err(decode)?,
        detected_app: row.try_get("detected_app").map_err(decode)?,
        language: row.try_get("language").map_err(decode)?,
        status: MeetingStatus::parse(&status).ok_or_else(|| bad_enum("status", &status))?,
        audio_dir: row.try_get("audio_dir").map_err(decode)?,
        mixed_path: row.try_get("mixed_path").map_err(decode)?,
        duration_ms: row.try_get("duration_ms").map_err(decode)?,
        deleted_at: row.try_get("deleted_at").map_err(decode)?,
    })
}

fn row_to_meeting_summary(row: sqlx::sqlite::SqliteRow) -> Result<MeetingSummary, DbError> {
    let status: String = row.try_get("status").map_err(decode)?;
    let recap_md: Option<String> = row.try_get("recap_md").map_err(decode)?;
    let first_text: Option<String> = row.try_get("first_text").map_err(decode)?;
    let summary_count: i64 = row.try_get("summary_count").map_err(decode)?;
    let chunk_count: i64 = row.try_get("chunk_count").map_err(decode)?;
    Ok(MeetingSummary {
        id: row.try_get("id").map_err(decode)?,
        title: row.try_get("title").map_err(decode)?,
        started_at: row.try_get("started_at").map_err(decode)?,
        duration_ms: row.try_get("duration_ms").map_err(decode)?,
        status: MeetingStatus::parse(&status).ok_or_else(|| bad_enum("status", &status))?,
        language: row.try_get("language").map_err(decode)?,
        snippet: recap_md
            .as_deref()
            .and_then(first_prose_line)
            .or_else(|| first_text.as_deref().map(shorten)),
        has_recap: summary_count > 0,
        has_audio: chunk_count > 0,
        speaker_count: row.try_get::<i64, _>("speaker_count").map_err(decode)? as u32,
        action_item_count: row.try_get::<i64, _>("action_item_count").map_err(decode)? as u32,
    })
}

/// First line of a recap that is actual prose (skip headings, bullets, blanks).
fn first_prose_line(md: &str) -> Option<String> {
    md.lines()
        .map(str::trim)
        .find(|l| {
            !l.is_empty() && !l.starts_with('#') && !l.starts_with('-') && !l.starts_with('*')
        })
        .map(shorten)
}

fn shorten(s: &str) -> String {
    const MAX: usize = 180;
    let s = s.trim();
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let mut out: String = s.chars().take(MAX).collect();
    out.push('…');
    out
}

fn row_to_chunk(row: sqlx::sqlite::SqliteRow) -> Result<AudioChunk, DbError> {
    let channel: String = row.try_get("channel").map_err(decode)?;
    Ok(AudioChunk {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        channel: Channel::parse(&channel).ok_or_else(|| bad_enum("channel", &channel))?,
        seq: row.try_get("seq").map_err(decode)?,
        path: row.try_get("path").map_err(decode)?,
        t_start_ms: row.try_get("t_start_ms").map_err(decode)?,
        t_end_ms: row.try_get("t_end_ms").map_err(decode)?,
        committed: row.try_get("committed").map_err(decode)?,
    })
}

fn row_to_segment(row: sqlx::sqlite::SqliteRow) -> Result<Segment, DbError> {
    let channel: String = row.try_get("channel").map_err(decode)?;
    Ok(Segment {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        t_start_ms: row.try_get("t_start_ms").map_err(decode)?,
        t_end_ms: row.try_get("t_end_ms").map_err(decode)?,
        channel: Channel::parse(&channel).ok_or_else(|| bad_enum("channel", &channel))?,
        speaker_id: row.try_get("speaker_id").map_err(decode)?,
        text: row.try_get("text").map_err(decode)?,
        language: row.try_get("language").map_err(decode)?,
        avg_confidence: row.try_get("avg_confidence").map_err(decode)?,
        revision: row.try_get("revision").map_err(decode)?,
        is_final: row.try_get("is_final").map_err(decode)?,
        model_name: row.try_get("model_name").map_err(decode)?,
        model_revision: row.try_get("model_revision").map_err(decode)?,
        corrections: corrections_of(row.try_get("corrections").map_err(decode)?),
    })
}

/// The corrections column, as JSON — or nothing at all, which is what the
/// overwhelming majority of lines have and is stored as NULL rather than as an
/// empty array.
fn corrections_json(corrections: &[Correction]) -> Option<String> {
    if corrections.is_empty() {
        return None;
    }
    serde_json::to_string(corrections).ok()
}

/// A line whose corrections column is NULL, or holds something unreadable, is a
/// line nothing was changed in as far as anyone reading it is concerned. The
/// note about a repair is worth nothing next to the words themselves, so a
/// corrupt one is dropped rather than allowed to fail the query carrying them.
fn corrections_of(stored: Option<String>) -> Vec<Correction> {
    stored
        .as_deref()
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_default()
}

fn row_to_speaker(row: sqlx::sqlite::SqliteRow) -> Result<Speaker, DbError> {
    Ok(Speaker {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        cluster_key: row.try_get("cluster_key").map_err(decode)?,
        display_name: row.try_get("display_name").map_err(decode)?,
        alias_of: row.try_get("alias_of").map_err(decode)?,
        is_self: row.try_get("is_self").map_err(decode)?,
        speaking_ms: row.try_get("speaking_ms").map_err(decode)?,
        person_id: row.try_get("person_id").map_err(decode)?,
        suggested_person_id: row.try_get("suggested_person_id").map_err(decode)?,
        suggestion_score: row.try_get("suggestion_score").map_err(decode)?,
    })
}

fn row_to_person(row: sqlx::sqlite::SqliteRow) -> Result<PersonRow, DbError> {
    Ok(PersonRow {
        id: row.try_get("id").map_err(decode)?,
        name: row.try_get("name").map_err(decode)?,
        created_at: row.try_get("created_at").map_err(decode)?,
        updated_at: row.try_get("updated_at").map_err(decode)?,
        alias_of: row.try_get("alias_of").map_err(decode)?,
    })
}

fn row_to_person_sample(row: sqlx::sqlite::SqliteRow) -> Result<PersonSampleRow, DbError> {
    let condition: String = row.try_get("condition").map_err(decode)?;
    let embedding: Vec<u8> = row.try_get("embedding").map_err(decode)?;
    Ok(PersonSampleRow {
        id: row.try_get("id").map_err(decode)?,
        person_id: row.try_get("person_id").map_err(decode)?,
        embedding: blob_to_embedding(&embedding),
        condition: Channel::parse(&condition).ok_or_else(|| bad_enum("condition", &condition))?,
        source_meeting_id: row.try_get("source_meeting_id").map_err(decode)?,
        t_start_ms: row.try_get("t_start_ms").map_err(decode)?,
        t_end_ms: row.try_get("t_end_ms").map_err(decode)?,
        created_at: row.try_get("created_at").map_err(decode)?,
    })
}

fn row_to_person_profile(row: sqlx::sqlite::SqliteRow) -> Result<PersonProfileRow, DbError> {
    let centroid: Vec<u8> = row.try_get("centroid").map_err(decode)?;
    let count: i64 = row.try_get("sample_count").map_err(decode)?;
    Ok(PersonProfileRow {
        person_id: row.try_get("person_id").map_err(decode)?,
        name: row.try_get("name").map_err(decode)?,
        centroid: blob_to_embedding(&centroid),
        sample_count: count.max(0) as u32,
        embedder_asset_id: row.try_get("embedder_asset_id").map_err(decode)?,
        updated_at: row.try_get("updated_at").map_err(decode)?,
    })
}

fn row_to_voice_print(row: sqlx::sqlite::SqliteRow) -> Result<VoicePrint, DbError> {
    let centroid: Vec<u8> = row.try_get("centroid").map_err(decode)?;
    Ok(VoicePrint {
        speaker_id: row.try_get("speaker_id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        meeting_title: row.try_get("meeting_title").map_err(decode)?,
        started_at: row.try_get("started_at").map_err(decode)?,
        display_name: row.try_get("display_name").map_err(decode)?,
        speaking_ms: row.try_get("speaking_ms").map_err(decode)?,
        centroid: blob_to_embedding(&centroid),
    })
}

fn row_to_marker(row: sqlx::sqlite::SqliteRow) -> Result<Marker, DbError> {
    let kind: String = row.try_get("kind").map_err(decode)?;
    Ok(Marker {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        t_ms: row.try_get("t_ms").map_err(decode)?,
        kind: MarkerKind::parse(&kind).ok_or_else(|| bad_enum("kind", &kind))?,
        note: row.try_get("note").map_err(decode)?,
    })
}

fn row_to_template(row: sqlx::sqlite::SqliteRow) -> Result<Template, DbError> {
    Ok(Template {
        id: row.try_get("id").map_err(decode)?,
        name: row.try_get("name").map_err(decode)?,
        prompt_md: row.try_get("prompt_md").map_err(decode)?,
        builtin: row.try_get("builtin").map_err(decode)?,
    })
}

fn row_to_summary(row: sqlx::sqlite::SqliteRow) -> Result<Summary, DbError> {
    let provider: String = row.try_get("provider").map_err(decode)?;
    Ok(Summary {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        template_id: row.try_get("template_id").map_err(decode)?,
        template_snapshot: row.try_get("template_snapshot").map_err(decode)?,
        provider: Provider::parse(&provider).ok_or_else(|| bad_enum("provider", &provider))?,
        model: row.try_get("model").map_err(decode)?,
        language: row.try_get("language").map_err(decode)?,
        transcript_revision: row.try_get("transcript_revision").map_err(decode)?,
        content_md: row.try_get("content_md").map_err(decode)?,
        created_at: row.try_get("created_at").map_err(decode)?,
    })
}

fn row_to_action_item(row: sqlx::sqlite::SqliteRow) -> Result<ActionItem, DbError> {
    Ok(ActionItem {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        summary_id: row.try_get("summary_id").map_err(decode)?,
        description: row.try_get("description").map_err(decode)?,
        owner: row.try_get("owner").map_err(decode)?,
        due_hint: row.try_get("due_hint").map_err(decode)?,
        done: row.try_get("done").map_err(decode)?,
        external_url: row.try_get("external_url").map_err(decode)?,
    })
}

fn row_to_job(row: sqlx::sqlite::SqliteRow) -> Result<Job, DbError> {
    let kind: String = row.try_get("kind").map_err(decode)?;
    let status: String = row.try_get("status").map_err(decode)?;
    let phase: Option<String> = row.try_get("phase").map_err(decode)?;
    let phase = phase
        .map(|p| JobPhase::parse(&p).ok_or_else(|| bad_enum("phase", &p)))
        .transpose()?;
    Ok(Job {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        kind: JobKind::parse(&kind).ok_or_else(|| bad_enum("kind", &kind))?,
        status: JobStatus::parse(&status).ok_or_else(|| bad_enum("status", &status))?,
        progress: row.try_get("progress").map_err(decode)?,
        error: row.try_get("error").map_err(decode)?,
        created_at: row.try_get("created_at").map_err(decode)?,
        updated_at: row.try_get("updated_at").map_err(decode)?,
        phase,
    })
}

fn row_to_model(row: sqlx::sqlite::SqliteRow) -> Result<ModelInfo, DbError> {
    let kind: String = row.try_get("kind").map_err(decode)?;
    Ok(ModelInfo {
        id: row.try_get("id").map_err(decode)?,
        kind: AssetKind::parse(&kind).ok_or_else(|| bad_enum("kind", &kind))?,
        name: row.try_get("name").map_err(decode)?,
        url: row.try_get("url").map_err(decode)?,
        sha256: row.try_get("sha256").map_err(decode)?,
        bytes: row.try_get("bytes").map_err(decode)?,
        license: row.try_get("license").map_err(decode)?,
        revision: row.try_get("revision").map_err(decode)?,
        installed: row.try_get("installed").map_err(decode)?,
        path: row.try_get("path").map_err(decode)?,
    })
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::connect_in_memory;

    async fn seeded() -> (Db, Meeting) {
        let db = connect_in_memory().await.unwrap();
        let m = create_meeting(&db, "Weekly sync", "/tmp/audio/m1", Some("zoom.us"))
            .await
            .unwrap();
        (db, m)
    }

    fn draft(meeting_id: &str, start: i64, text: &str) -> SegmentDraft {
        SegmentDraft {
            meeting_id: meeting_id.to_string(),
            t_start_ms: start,
            t_end_ms: start + 1_000,
            channel: Channel::Mic,
            text: text.to_string(),
            revision: 1,
            is_final: true,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn meeting_round_trip() {
        let (db, m) = seeded().await;
        assert_eq!(m.status, MeetingStatus::Created);
        assert_eq!(m.detected_app.as_deref(), Some("zoom.us"));

        update_meeting_title(&db, &m.id, "Weekly sync (renamed)")
            .await
            .unwrap();
        finish_meeting(
            &db,
            &m.id,
            61_000,
            Some("en"),
            Some("/tmp/mixed.flac"),
            MeetingStatus::Complete,
        )
        .await
        .unwrap();

        let after = get_meeting(&db, &m.id).await.unwrap().unwrap();
        assert_eq!(after.title, "Weekly sync (renamed)");
        assert_eq!(after.duration_ms, 61_000);
        assert_eq!(after.language.as_deref(), Some("en"));
        assert_eq!(after.status, MeetingStatus::Complete);
        assert!(after.ended_at.is_some());
    }

    #[tokio::test]
    async fn list_meetings_hides_deleted_and_carries_counts() {
        let (db, m) = seeded().await;
        insert_segments(&db, &[draft(&m.id, 0, "hello there")])
            .await
            .unwrap();
        upsert_speaker(&db, &m.id, "mic", "You", true)
            .await
            .unwrap();

        let list = list_meetings(&db, &MeetingQuery::default()).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].speaker_count, 1);
        assert_eq!(list[0].snippet.as_deref(), Some("hello there"));
        assert!(!list[0].has_recap);

        soft_delete_meeting(&db, &m.id).await.unwrap();
        assert!(list_meetings(&db, &MeetingQuery::default())
            .await
            .unwrap()
            .is_empty());
        let with_deleted = MeetingQuery {
            include_deleted: Some(true),
            ..Default::default()
        };
        assert_eq!(list_meetings(&db, &with_deleted).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn chunks_journal_tracks_committed_offset() {
        let (db, m) = seeded().await;
        let c1 = insert_chunk(&db, &m.id, Channel::Mic, 0, "/a/0.flac", 0, 10_000)
            .await
            .unwrap();
        let _c2 = insert_chunk(&db, &m.id, Channel::Mic, 1, "/a/1.flac", 10_000, 20_000)
            .await
            .unwrap();

        // Nothing counts until it is committed.
        assert_eq!(
            last_committed_offset_ms(&db, &m.id, Channel::Mic)
                .await
                .unwrap(),
            0
        );
        commit_chunk(&db, &c1, 10_000).await.unwrap();
        assert_eq!(
            last_committed_offset_ms(&db, &m.id, Channel::Mic)
                .await
                .unwrap(),
            10_000
        );

        assert_eq!(list_uncommitted_chunks(&db, &m.id).await.unwrap().len(), 1);
        assert_eq!(
            list_chunks(&db, &m.id, Some(Channel::Mic))
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            list_chunks(&db, &m.id, Some(Channel::System))
                .await
                .unwrap()
                .len(),
            0
        );
    }

    /// The sweep after a finished catch-up pass takes the guesses the pass
    /// replaced, and leaves the ones it did not.
    ///
    /// A window that would not read leaves its seconds with no text of their
    /// own. The guess over them is the only thing a person watched appear there,
    /// so deleting it turns a rough sentence into nothing at all.
    #[tokio::test]
    async fn the_guesses_over_a_stretch_nothing_read_survive_the_sweep() {
        let (db, m) = seeded().await;
        let guess = |start: i64, channel: Channel| SegmentDraft {
            channel,
            is_final: false,
            ..draft(&m.id, start, "half heard gue")
        };
        // Two on the microphone: one over the stretch nothing read, one clear of
        // it. One on the computer's side at the same time as the unread stretch,
        // which is a different channel and so a different set of seconds.
        insert_segment(&db, &guess(30_000, Channel::Mic))
            .await
            .unwrap();
        insert_segment(&db, &guess(90_000, Channel::Mic))
            .await
            .unwrap();
        insert_segment(&db, &guess(30_000, Channel::System))
            .await
            .unwrap();
        // And a real line, which is never a guess and never swept.
        insert_segment(&db, &draft(&m.id, 90_000, "read off the recording"))
            .await
            .unwrap();

        let taken = delete_partial_segments_except(&db, &m.id, &[(Channel::Mic, 25_000, 45_000)])
            .await
            .unwrap();
        assert_eq!(taken, 2);

        let left = get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: m.id.clone(),
                include_partial: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let guesses: Vec<(Channel, i64)> = left
            .iter()
            .filter(|s| !s.is_final)
            .map(|s| (s.channel, s.t_start_ms))
            .collect();
        assert_eq!(
            guesses,
            vec![(Channel::Mic, 30_000)],
            "only the guess over seconds nothing replaced stays"
        );
        assert!(
            left.iter().any(|s| s.is_final),
            "the sweep is about guesses and touches nothing else"
        );
    }

    /// Nothing was left unread, so nothing is spared: a pass that read the whole
    /// meeting back clears every guess, exactly as it always did.
    #[tokio::test]
    async fn with_nothing_left_unread_every_guess_goes() {
        let (db, m) = seeded().await;
        for start in [0, 30_000, 60_000] {
            insert_segment(
                &db,
                &SegmentDraft {
                    is_final: false,
                    ..draft(&m.id, start, "half heard gue")
                },
            )
            .await
            .unwrap();
        }

        assert_eq!(
            delete_partial_segments_except(&db, &m.id, &[])
                .await
                .unwrap(),
            3
        );
        assert_eq!(count_segments(&db, &m.id).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn revising_a_segment_never_goes_backwards() {
        let (db, m) = seeded().await;
        let seg = insert_segment(&db, &draft(&m.id, 0, "first guess"))
            .await
            .unwrap();

        revise_segment(&db, &seg.id, Some("better text"), None, 3, true)
            .await
            .unwrap();
        assert_eq!(
            get_segment(&db, &seg.id).await.unwrap().unwrap().text,
            "better text"
        );

        // A stale pass arriving late must not win.
        revise_segment(&db, &seg.id, Some("stale text"), None, 2, true)
            .await
            .unwrap();
        assert_eq!(
            get_segment(&db, &seg.id).await.unwrap().unwrap().text,
            "better text"
        );
        assert_eq!(transcript_revision(&db, &m.id).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn merge_is_non_destructive_and_chains_are_flattened() {
        let (db, m) = seeded().await;
        let a = upsert_speaker(&db, &m.id, "spk1", "Speaker 1", false)
            .await
            .unwrap();
        let b = upsert_speaker(&db, &m.id, "spk2", "Speaker 2", false)
            .await
            .unwrap();
        let c = upsert_speaker(&db, &m.id, "spk3", "Speaker 3", false)
            .await
            .unwrap();

        merge_speakers(&db, &a.id, &b.id).await.unwrap();
        merge_speakers(&db, &b.id, &c.id).await.unwrap();

        // a was re-pointed at c rather than left chained through b.
        let a_after = get_speaker(&db, &a.id).await.unwrap().unwrap();
        assert_eq!(a_after.alias_of.as_deref(), Some(c.id.as_str()));
        assert_eq!(resolve_speaker(&db, &a.id).await.unwrap().unwrap().id, c.id);

        // Rows survive, so undo is possible.
        assert_eq!(list_speakers(&db, &m.id).await.unwrap().len(), 3);
        unmerge_speaker(&db, &a.id).await.unwrap();
        assert!(get_speaker(&db, &a.id)
            .await
            .unwrap()
            .unwrap()
            .alias_of
            .is_none());
    }

    #[tokio::test]
    async fn merging_back_the_other_way_does_not_make_a_ring() {
        let (db, m) = seeded().await;
        let a = upsert_speaker(&db, &m.id, "spk1", "Speaker 1", false)
            .await
            .unwrap();
        let b = upsert_speaker(&db, &m.id, "spk2", "Speaker 2", false)
            .await
            .unwrap();

        merge_speakers(&db, &a.id, &b.id).await.unwrap();
        // Somebody clicks the other way round. That is already true, so it must
        // change nothing rather than point a at itself.
        merge_speakers(&db, &b.id, &a.id).await.unwrap();

        let a_after = get_speaker(&db, &a.id).await.unwrap().unwrap();
        let b_after = get_speaker(&db, &b.id).await.unwrap().unwrap();
        assert_eq!(a_after.alias_of.as_deref(), Some(b.id.as_str()));
        assert!(b_after.alias_of.is_none(), "b must stay a real speaker");
        assert_eq!(resolve_speaker(&db, &a.id).await.unwrap().unwrap().id, b.id);
        assert_eq!(resolve_speaker(&db, &b.id).await.unwrap().unwrap().id, b.id);
    }

    #[tokio::test]
    async fn speakers_from_two_meetings_are_never_the_same_person() {
        let (db, m) = seeded().await;
        let other = create_meeting(&db, "Client call", "/tmp/audio/m2", None)
            .await
            .unwrap();
        let mine = upsert_speaker(&db, &m.id, "spk1", "Speaker 1", false)
            .await
            .unwrap();
        let theirs = upsert_speaker(&db, &other.id, "spk1", "Speaker 1", false)
            .await
            .unwrap();

        let err = merge_speakers(&db, &mine.id, &theirs.id).await.unwrap_err();
        assert!(matches!(err, DbError::Invalid(_)), "{err:?}");
        assert!(get_speaker(&db, &mine.id)
            .await
            .unwrap()
            .unwrap()
            .alias_of
            .is_none());
    }

    #[tokio::test]
    async fn merging_a_speaker_that_does_not_exist_is_an_error() {
        let (db, m) = seeded().await;
        let a = upsert_speaker(&db, &m.id, "spk1", "Speaker 1", false)
            .await
            .unwrap();
        let err = merge_speakers(&db, &a.id, &new_id()).await.unwrap_err();
        assert!(matches!(err, DbError::NotFound(_)), "{err:?}");
    }

    #[tokio::test]
    async fn upsert_speaker_is_idempotent_per_cluster_key() {
        let (db, m) = seeded().await;
        let first = upsert_speaker(&db, &m.id, "mic", "You", true)
            .await
            .unwrap();
        let again = upsert_speaker(&db, &m.id, "mic", "Ignored", true)
            .await
            .unwrap();
        assert_eq!(first.id, again.id);
        assert_eq!(again.display_name, "You");
    }

    // -----------------------------------------------------------------------
    // How many people were in the meeting
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn a_fresh_meeting_has_no_correction_and_so_counts_its_speakers() {
        let (db, m) = seeded().await;
        assert_eq!(speaker_count_override(&db, &m.id).await.unwrap(), None);
        assert_eq!(people_count(&db, &m.id).await.unwrap(), (0, false));

        upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();
        upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        assert_eq!(people_count(&db, &m.id).await.unwrap(), (2, false));
    }

    /// A meeting that existed before the column did. `ALTER TABLE ADD COLUMN`
    /// with no default leaves NULL on every row it finds, and NULL is exactly the
    /// behaviour those meetings already had: automatic.
    #[tokio::test]
    async fn a_meeting_written_without_the_column_reads_back_as_automatic() {
        let (db, _) = seeded().await;
        let id = new_id();
        sqlx::query(
            "INSERT INTO meetings (id, title, started_at, status, audio_dir, duration_ms)
             VALUES (?1, 'Older meeting', '2026-01-01T00:00:00Z', 'complete', '/tmp/old', 60000)",
        )
        .bind(&id)
        .execute(&db)
        .await
        .unwrap();
        assert_eq!(speaker_count_override(&db, &id).await.unwrap(), None);
        assert_eq!(people_count(&db, &id).await.unwrap(), (0, false));
    }

    #[tokio::test]
    async fn a_correction_is_stored_read_back_and_cleared() {
        let (db, m) = seeded().await;
        set_speaker_count_override(&db, &m.id, Some(5))
            .await
            .unwrap();
        assert_eq!(speaker_count_override(&db, &m.id).await.unwrap(), Some(5));
        assert_eq!(
            people_count(&db, &m.id).await.unwrap(),
            (5, true),
            "the person's answer is the one the UI shows"
        );

        // Null is automatic, and it really does go back to counting.
        set_speaker_count_override(&db, &m.id, None).await.unwrap();
        assert_eq!(speaker_count_override(&db, &m.id).await.unwrap(), None);
        assert_eq!(people_count(&db, &m.id).await.unwrap(), (0, false));
    }

    /// Merging two chips is the person saying "those two were one person", so it
    /// has to move Echo's count as well as the transcript.
    #[tokio::test]
    async fn merging_two_speakers_takes_one_off_echos_count() {
        let (db, m) = seeded().await;
        let a = upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let b = upsert_speaker(&db, &m.id, "speaker-02", "Speaker 2", false)
            .await
            .unwrap();
        assert_eq!(count_people(&db, &m.id).await.unwrap(), 2);
        merge_speakers(&db, &b.id, &a.id).await.unwrap();
        assert_eq!(count_people(&db, &m.id).await.unwrap(), 1);
    }

    /// A number nobody could act on cannot come back out of the database, however
    /// it got in.
    #[tokio::test]
    async fn a_stored_number_out_of_range_is_read_back_inside_it() {
        let (db, m) = seeded().await;
        sqlx::query("UPDATE meetings SET speaker_count_override = 900 WHERE id = ?1")
            .bind(&m.id)
            .execute(&db)
            .await
            .unwrap();
        assert_eq!(
            speaker_count_override(&db, &m.id).await.unwrap(),
            Some(crate::diarize::MAX_PEOPLE)
        );
        sqlx::query("UPDATE meetings SET speaker_count_override = 0 WHERE id = ?1")
            .bind(&m.id)
            .execute(&db)
            .await
            .unwrap();
        assert_eq!(
            speaker_count_override(&db, &m.id).await.unwrap(),
            Some(crate::diarize::MIN_PEOPLE)
        );
    }

    #[tokio::test]
    async fn a_correction_for_a_meeting_that_does_not_exist_is_not_found() {
        let (db, _) = seeded().await;
        let err = set_speaker_count_override(&db, &new_id(), Some(2))
            .await
            .unwrap_err();
        assert!(matches!(err, DbError::NotFound(_)), "{err:?}");
        let err = speaker_count_override(&db, &new_id()).await.unwrap_err();
        assert!(matches!(err, DbError::NotFound(_)), "{err:?}");
    }

    #[tokio::test]
    async fn the_meeting_detail_carries_the_count_and_where_it_came_from() {
        let (db, m) = seeded().await;
        upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();
        let detail = get_meeting_detail(&db, &m.id).await.unwrap();
        assert_eq!(detail.people_count, 1);
        assert!(!detail.people_count_is_override);

        set_speaker_count_override(&db, &m.id, Some(3))
            .await
            .unwrap();
        let detail = get_meeting_detail(&db, &m.id).await.unwrap();
        assert_eq!(detail.people_count, 3);
        assert!(detail.people_count_is_override);
    }

    /// The history list and the meeting itself must never disagree about how many
    /// people were there.
    #[tokio::test]
    async fn the_history_list_shows_the_persons_count_when_there_is_one() {
        let (db, m) = seeded().await;
        upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();

        let listed = list_meetings(&db, &MeetingQuery::default()).await.unwrap();
        assert_eq!(listed[0].speaker_count, 1);

        set_speaker_count_override(&db, &m.id, Some(4))
            .await
            .unwrap();
        let listed = list_meetings(&db, &MeetingQuery::default()).await.unwrap();
        let detail = get_meeting_detail(&db, &m.id).await.unwrap();
        assert_eq!(listed[0].speaker_count, 4);
        assert_eq!(listed[0].speaker_count, detail.people_count);
    }

    /// The microphone is always recorded, so only words tell you whether the
    /// person actually spoke.
    #[tokio::test]
    async fn a_channel_has_speech_when_it_has_words_not_when_it_has_audio() {
        let (db, m) = seeded().await;
        let chunk = insert_chunk(&db, &m.id, Channel::Mic, 0, "/tmp/mic-0.flac", 0, 30_000)
            .await
            .unwrap();
        commit_chunk(&db, &chunk, 30_000).await.unwrap();
        assert!(
            !has_channel_speech(&db, &m.id, Channel::Mic).await.unwrap(),
            "an open microphone is not somebody speaking"
        );

        insert_segments(&db, &[draft(&m.id, 0, "   ")])
            .await
            .unwrap();
        assert!(
            !has_channel_speech(&db, &m.id, Channel::Mic).await.unwrap(),
            "an empty line is not somebody speaking either"
        );

        insert_segments(&db, &[draft(&m.id, 2_000, "morning")])
            .await
            .unwrap();
        assert!(has_channel_speech(&db, &m.id, Channel::Mic).await.unwrap());
        assert!(!has_channel_speech(&db, &m.id, Channel::System)
            .await
            .unwrap());
    }

    /// Pruning is how a re-run stops leaving ghosts behind. Everything that
    /// pointed at the row it removes has to end up somewhere honest.
    #[tokio::test]
    async fn pruning_forgets_the_voices_a_new_pass_did_not_produce() {
        let (db, m) = seeded().await;
        let me = upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();
        let first = upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let second = upsert_speaker(&db, &m.id, "speaker-02", "Dana", false)
            .await
            .unwrap();
        let third = upsert_speaker(&db, &m.id, "speaker-03", "Speaker 3", false)
            .await
            .unwrap();
        // Somebody merged the third voice into the second.
        merge_speakers(&db, &third.id, &second.id).await.unwrap();
        // And a line is still attributed to the second one.
        let ids = insert_segments(&db, &[draft(&m.id, 0, "we ship on Friday")])
            .await
            .unwrap();
        assign_speaker(&db, &ids, &second.id, 2).await.unwrap();

        let removed =
            prune_speakers_except(&db, &m.id, &["you".to_string(), "speaker-01".to_string()])
                .await
                .unwrap();
        assert_eq!(removed, 2);

        let left = list_speakers(&db, &m.id).await.unwrap();
        let ids_left: Vec<&str> = left.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids_left.len(), 2);
        assert!(ids_left.contains(&me.id.as_str()));
        assert!(ids_left.contains(&first.id.as_str()));

        // The line that pointed at a voice which no longer exists reads as
        // unattributed rather than as somebody who was never there.
        let segment = get_segment(&db, &ids[0]).await.unwrap().unwrap();
        assert_eq!(segment.speaker_id, None);
        assert_eq!(segment.text, "we ship on Friday", "the words are untouched");
    }

    #[tokio::test]
    async fn pruning_with_nothing_to_keep_refuses_to_wipe_the_meeting() {
        let (db, m) = seeded().await;
        upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();
        assert_eq!(prune_speakers_except(&db, &m.id, &[]).await.unwrap(), 0);
        assert_eq!(count_people(&db, &m.id).await.unwrap(), 1);
    }

    /// The deliberate end of the same road, and the one destructive query with
    /// no `keep` list to hold it back: a pass that found nobody at all says so
    /// here. Every row of *this* meeting goes, the lines they held read as
    /// unattributed, and no other meeting is touched.
    #[tokio::test]
    async fn forgetting_takes_this_meeting_s_voices_and_only_this_meeting_s() {
        let (db, m) = seeded().await;
        let other = create_meeting(&db, "Retro", "/tmp/audio/m2", None)
            .await
            .unwrap();
        upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();
        let voice = upsert_speaker(&db, &m.id, "speaker-01", "Dana", false)
            .await
            .unwrap();
        upsert_speaker(&db, &other.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let ids = insert_segments(&db, &[draft(&m.id, 0, "we ship on Friday")])
            .await
            .unwrap();
        assign_speaker(&db, &ids, &voice.id, 2).await.unwrap();

        assert_eq!(forget_speakers(&db, &m.id).await.unwrap(), 2);
        assert_eq!(count_people(&db, &m.id).await.unwrap(), 0);
        assert_eq!(count_people(&db, &other.id).await.unwrap(), 1);
        let segment = get_segment(&db, &ids[0]).await.unwrap().unwrap();
        assert_eq!(segment.speaker_id, None);
        assert_eq!(segment.text, "we ship on Friday", "the words are untouched");
        // Saying it twice is not an error, it is just nothing left to forget.
        assert_eq!(forget_speakers(&db, &m.id).await.unwrap(), 0);
    }

    /// Who Echo can actually hear, which is not the same question as who was in
    /// the meeting: a row nobody's words landed on — "You" on a meeting the
    /// person only listened to — is not a voice in the recording.
    #[tokio::test]
    async fn the_audible_count_leaves_out_rows_with_no_lines_on_them() {
        let (db, m) = seeded().await;
        upsert_speaker(&db, &m.id, "you", "You", true)
            .await
            .unwrap();
        let voice = upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let merged = upsert_speaker(&db, &m.id, "speaker-02", "Speaker 2", false)
            .await
            .unwrap();
        let ids = insert_segments(&db, &[draft(&m.id, 0, "morning")])
            .await
            .unwrap();
        assign_speaker(&db, &ids, &voice.id, 2).await.unwrap();

        assert_eq!(count_people(&db, &m.id).await.unwrap(), 3);
        assert_eq!(count_audible_people(&db, &m.id).await.unwrap(), 1);

        // A merged row's lines count towards the row it was merged into, so a
        // person is not made inaudible by somebody tidying the list.
        let more = insert_segments(&db, &[draft(&m.id, 5_000, "and then we shipped")])
            .await
            .unwrap();
        assign_speaker(&db, &more, &merged.id, 3).await.unwrap();
        merge_speakers(&db, &merged.id, &voice.id).await.unwrap();
        assert_eq!(count_people(&db, &m.id).await.unwrap(), 2);
        assert_eq!(count_audible_people(&db, &m.id).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn pruning_never_reaches_into_another_meeting() {
        let (db, m) = seeded().await;
        let other = create_meeting(&db, "Retro", "/tmp/audio/m2", None)
            .await
            .unwrap();
        upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        upsert_speaker(&db, &other.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();

        prune_speakers_except(&db, &m.id, &["you".to_string()])
            .await
            .unwrap();
        assert_eq!(count_people(&db, &m.id).await.unwrap(), 0);
        assert_eq!(count_people(&db, &other.id).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn jobs_pause_for_recording_and_resume_after() {
        let (db, m) = seeded().await;
        let j = create_job(&db, Some(&m.id), JobKind::Summarize)
            .await
            .unwrap();
        set_job_status(&db, &j.id, JobStatus::Running, None)
            .await
            .unwrap();

        assert_eq!(pause_active_jobs(&db).await.unwrap(), 1);
        assert_eq!(
            get_job(&db, &j.id).await.unwrap().unwrap().status,
            JobStatus::Paused
        );
        assert_eq!(resume_paused_jobs(&db).await.unwrap(), 1);
        assert_eq!(
            get_job(&db, &j.id).await.unwrap().unwrap().status,
            JobStatus::Queued
        );
    }

    #[tokio::test]
    async fn a_crash_during_a_recording_does_not_strand_parked_work() {
        let (db, m) = seeded().await;
        let running = create_job(&db, Some(&m.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        let parked = create_job(&db, Some(&m.id), JobKind::Summarize)
            .await
            .unwrap();
        set_job_status(&db, &running.id, JobStatus::Running, None)
            .await
            .unwrap();
        set_job_status(&db, &parked.id, JobStatus::Paused, None)
            .await
            .unwrap();
        let done = create_job(&db, Some(&m.id), JobKind::Mixdown)
            .await
            .unwrap();
        set_job_status(&db, &done.id, JobStatus::Done, None)
            .await
            .unwrap();

        // Nothing can legitimately be running or parked at launch: the flag that
        // parks work lives in memory only.
        assert_eq!(requeue_orphaned_jobs(&db, "interrupted").await.unwrap(), 2);
        for id in [&running.id, &parked.id] {
            assert_eq!(
                get_job(&db, id).await.unwrap().unwrap().status,
                JobStatus::Queued
            );
        }
        assert_eq!(
            get_job(&db, &done.id).await.unwrap().unwrap().status,
            JobStatus::Done,
            "finished work is left alone"
        );
    }

    /// The one exception, and the reason it exists: a setup row that was running
    /// when the process went down is not started again by the launch that
    /// follows. Whatever ended the app was most likely the compile itself, and
    /// putting the row back in the queue is how one bad compile becomes a
    /// quarter of an hour of it at every launch, forever.
    #[tokio::test]
    async fn an_interrupted_speech_setup_is_settled_rather_than_started_again() {
        let (db, m) = seeded().await;
        let setup = ensure_job_with_payload(&db, None, JobKind::PrepareEngine, Some("\"w.bin\""))
            .await
            .unwrap();
        set_job_status(&db, &setup.id, JobStatus::Running, None)
            .await
            .unwrap();
        // A setup row a *recording* parked is an ordinary park, and goes back in
        // the queue like anything else: nothing crashed, so nothing is feared.
        let parked = create_job(&db, None, JobKind::PrepareEngine).await.unwrap();
        set_job_status(&db, &parked.id, JobStatus::Paused, None)
            .await
            .unwrap();
        let other = create_job(&db, Some(&m.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        set_job_status(&db, &other.id, JobStatus::Running, None)
            .await
            .unwrap();

        assert_eq!(requeue_orphaned_jobs(&db, "interrupted").await.unwrap(), 3);
        let settled = get_job(&db, &setup.id).await.unwrap().unwrap();
        assert_eq!(settled.status, JobStatus::Failed);
        assert_eq!(settled.error.as_deref(), Some("interrupted"));
        assert_eq!(
            get_job(&db, &parked.id).await.unwrap().unwrap().status,
            JobStatus::Queued,
            "a park is not a crash"
        );
        assert_eq!(
            get_job(&db, &other.id).await.unwrap().unwrap().status,
            JobStatus::Queued,
            "every other kind of work is picked back up as it always was"
        );

        // And the launch after that can hand it back, once, when the caller says
        // the weights are still owed a setup — those weights, by name: a row
        // that failed for a model somebody has moved on from stays where it is.
        assert_eq!(
            requeue_failed_setup_jobs(&db, "\"other.bin\"")
                .await
                .unwrap(),
            0,
            "another model's failed row is not this model's business"
        );
        assert_eq!(
            requeue_failed_setup_jobs(&db, "\"w.bin\"").await.unwrap(),
            1
        );
        let revived = get_job(&db, &setup.id).await.unwrap().unwrap();
        assert_eq!(revived.status, JobStatus::Queued);
        assert_eq!(revived.error, None, "an old reason is not still on it");
    }

    /// The queue itself is the memory of what has been tried: a failed row for
    /// exactly these weights is what keeps the same job from being asked for
    /// several times a second for the rest of the launch.
    #[tokio::test]
    async fn a_failed_row_is_found_by_the_payload_it_carried() {
        let (db, _m) = seeded().await;
        let job = ensure_job_with_payload(
            &db,
            None,
            JobKind::PrepareEngine,
            Some("\"whisper-large.bin\""),
        )
        .await
        .unwrap();

        assert!(
            !has_failed_job_with_payload(&db, JobKind::PrepareEngine, "\"whisper-large.bin\"")
                .await
                .unwrap(),
            "a job that is still queued has not failed at anything yet"
        );

        set_job_status(&db, &job.id, JobStatus::Failed, Some("no"))
            .await
            .unwrap();
        assert!(
            has_failed_job_with_payload(&db, JobKind::PrepareEngine, "\"whisper-large.bin\"")
                .await
                .unwrap()
        );
        assert!(
            !has_failed_job_with_payload(&db, JobKind::PrepareEngine, "\"other-weights.bin\"")
                .await
                .unwrap(),
            "another model's bad day says nothing about this one"
        );
        assert!(
            !has_failed_job_with_payload(&db, JobKind::Download, "\"whisper-large.bin\"")
                .await
                .unwrap(),
            "and neither does another kind of work"
        );
    }

    #[tokio::test]
    async fn ensure_job_does_not_duplicate_on_double_click() {
        let (db, m) = seeded().await;
        let a = ensure_job(&db, Some(&m.id), JobKind::Diarize)
            .await
            .unwrap();
        let b = ensure_job(&db, Some(&m.id), JobKind::Diarize)
            .await
            .unwrap();
        assert_eq!(a.id, b.id);
        set_job_status(&db, &a.id, JobStatus::Done, None)
            .await
            .unwrap();
        let c = ensure_job(&db, Some(&m.id), JobKind::Diarize)
            .await
            .unwrap();
        assert_ne!(a.id, c.id);
    }

    /// The contract of the queue, both halves of it: **kind first, then the
    /// newest meeting**.
    ///
    /// Kind is the important half and has not changed — a transcript before
    /// speakers before a recap, because each is worth less without the one
    /// before it. The tie-break within a kind is the half that did: the newest
    /// work goes first, so the meeting somebody has just walked out of is not
    /// stuck behind one they finished two hours ago.
    #[tokio::test]
    async fn next_queued_job_prefers_catchup_over_summarize() {
        let (db, m) = seeded().await;
        create_job(&db, Some(&m.id), JobKind::Summarize)
            .await
            .unwrap();
        let catchup = create_job(&db, Some(&m.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        assert_eq!(
            next_queued_job(&db).await.unwrap().unwrap().id,
            catchup.id,
            "kind outranks age: a recap never goes before a transcript"
        );
    }

    #[tokio::test]
    async fn within_one_kind_the_meeting_you_just_left_goes_first() {
        let (db, earlier) = seeded().await;
        let later = create_meeting(&db, "The one that just ended", "/tmp/audio/m2", None)
            .await
            .unwrap();

        let old_work = create_job(&db, Some(&earlier.id), JobKind::Diarize)
            .await
            .unwrap();
        // `created_at` is stamped to the millisecond, and these two rows would
        // otherwise be indistinguishable in age. Real meetings are minutes
        // apart.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let new_work = create_job(&db, Some(&later.id), JobKind::Diarize)
            .await
            .unwrap();

        assert_eq!(next_queued_job(&db).await.unwrap().unwrap().id, new_work.id);

        // And the older meeting is next, not forgotten: it keeps its place, its
        // progress and whatever its parked pass already worked out.
        set_job_status(&db, &new_work.id, JobStatus::Done, None)
            .await
            .unwrap();
        assert_eq!(next_queued_job(&db).await.unwrap().unwrap().id, old_work.id);
    }

    /// Newest-first inside a kind must not reorder the kinds themselves: a
    /// recap of the meeting that just ended still waits for the transcript of
    /// the one before it, because a recap is written from a transcript.
    #[tokio::test]
    async fn the_newest_meeting_still_does_not_jump_ahead_of_an_earlier_kind() {
        let (db, earlier) = seeded().await;
        let later = create_meeting(&db, "The one that just ended", "/tmp/audio/m2", None)
            .await
            .unwrap();
        let catchup = create_job(&db, Some(&earlier.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        create_job(&db, Some(&later.id), JobKind::Summarize)
            .await
            .unwrap();
        assert_eq!(next_queued_job(&db).await.unwrap().unwrap().id, catchup.id);
    }

    #[tokio::test]
    async fn action_items_keep_done_flags_across_regeneration() {
        let (db, m) = seeded().await;
        let s = insert_summary(
            &db,
            &m.id,
            None,
            None,
            Provider::OnThisComputer,
            None,
            Some("en"),
            1,
            "# Recap\n\nWe shipped it.",
        )
        .await
        .unwrap();

        let items = vec![ActionItem {
            description: "Send the notes".into(),
            ..Default::default()
        }];
        let stored = replace_action_items(&db, &m.id, Some(&s.id), &items)
            .await
            .unwrap();
        patch_action_item(
            &db,
            &ActionItemPatch {
                id: stored[0].id.clone(),
                done: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let regenerated = replace_action_items(&db, &m.id, Some(&s.id), &items)
            .await
            .unwrap();
        assert!(
            regenerated[0].done,
            "a re-run must not un-tick what the person ticked"
        );
        assert_eq!(list_action_items(&db, &m.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn summary_snippet_prefers_recap_prose() {
        let (db, m) = seeded().await;
        insert_segments(&db, &[draft(&m.id, 0, "transcript line")])
            .await
            .unwrap();
        insert_summary(
            &db,
            &m.id,
            None,
            None,
            Provider::Gemini,
            Some("g"),
            Some("en"),
            1,
            "# Recap\n\n- bullet\n\nWe agreed to ship on Friday.",
        )
        .await
        .unwrap();
        let list = list_meetings(&db, &MeetingQuery::default()).await.unwrap();
        assert_eq!(
            list[0].snippet.as_deref(),
            Some("We agreed to ship on Friday.")
        );
        assert!(list[0].has_recap);
    }

    #[tokio::test]
    async fn templates_builtins_are_protected() {
        let db = connect_in_memory().await.unwrap();
        let all = list_templates(&db).await.unwrap();
        assert_eq!(all.len(), 6);
        let builtin = all.iter().find(|t| t.builtin).unwrap();
        assert!(delete_template(&db, &builtin.id).await.is_err());
        assert!(upsert_template(&db, Some(&builtin.id), "x", "y")
            .await
            .is_err());

        let custom = upsert_template(&db, None, "My style", "Do it my way")
            .await
            .unwrap();
        assert!(!custom.builtin);
        upsert_template(&db, Some(&custom.id), "My style 2", "Better")
            .await
            .unwrap();
        delete_template(&db, &custom.id).await.unwrap();
        assert_eq!(list_templates(&db).await.unwrap().len(), 6);
    }

    #[tokio::test]
    async fn settings_are_key_value_and_upsert() {
        let db = connect_in_memory().await.unwrap();
        assert!(get_setting(&db, "nope").await.unwrap().is_none());
        set_setting(&db, "detection_enabled", "true").await.unwrap();
        set_setting(&db, "detection_enabled", "false")
            .await
            .unwrap();
        assert_eq!(
            get_setting(&db, "detection_enabled")
                .await
                .unwrap()
                .unwrap(),
            "false"
        );
        set_settings(&db, &[("a".into(), "1".into()), ("b".into(), "2".into())])
            .await
            .unwrap();
        assert_eq!(all_settings(&db).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn models_upsert_preserves_installed_state() {
        let db = connect_in_memory().await.unwrap();
        let m = ModelInfo {
            id: "speech-everyday".into(),
            kind: AssetKind::Speech,
            name: "everyday".into(),
            url: "https://example.invalid/a.bin".into(),
            sha256: "abc".into(),
            bytes: 1_600_000_000,
            ..Default::default()
        };
        upsert_model(&db, &m).await.unwrap();
        set_model_installed(&db, &m.id, true, Some("/tmp/a.bin"))
            .await
            .unwrap();
        // A catalog refresh must not un-install what is on disk.
        upsert_model(
            &db,
            &ModelInfo {
                bytes: 1_600_000_001,
                ..m.clone()
            },
        )
        .await
        .unwrap();
        let after = get_model(&db, &m.id).await.unwrap().unwrap();
        assert!(after.installed);
        assert_eq!(after.bytes, 1_600_000_001);
        assert_eq!(installed_model_bytes(&db).await.unwrap(), 1_600_000_001);
    }

    #[test]
    fn fts_escaping_neutralises_operators() {
        assert_eq!(escape_fts_query("hello"), "\"hello\"");
        assert_eq!(
            escape_fts_query("  hello   world "),
            "\"hello\" AND \"world\""
        );
        assert_eq!(
            escape_fts_query("NEAR AND *"),
            "\"NEAR\" AND \"AND\" AND \"*\""
        );
        assert_eq!(escape_fts_query("say \"hi\""), "\"say\" AND \"\"\"hi\"\"\"");
        assert_eq!(escape_fts_query("   "), "");
        assert_eq!(escape_fts_query(""), "");
    }

    #[test]
    fn trigram_is_picked_for_scripts_without_spaces() {
        assert_eq!(pick_index("hello world"), FtsIndex::Words);
        assert_eq!(pick_index("réunion"), FtsIndex::Words);
        assert_eq!(pick_index("会議の記録"), FtsIndex::Trigram);
        assert_eq!(pick_index("ประชุม"), FtsIndex::Trigram);
    }

    #[tokio::test]
    async fn search_finds_words_and_escapes_html() {
        let (db, m) = seeded().await;
        insert_segments(
            &db,
            &[
                draft(
                    &m.id,
                    0,
                    "We should ship the <script>alert(1)</script> feature",
                ),
                draft(&m.id, 2_000, "Totally unrelated chatter"),
            ],
        )
        .await
        .unwrap();

        let hits = search_segments(
            &db,
            &SearchQuery {
                text: "ship".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].snippet_html.contains("<mark>ship</mark>"),
            "{}",
            hits[0].snippet_html
        );
        assert!(
            hits[0].snippet_html.contains("&lt;script&gt;"),
            "{}",
            hits[0].snippet_html
        );
        assert!(!hits[0].snippet_html.contains("<script>"));
        assert_eq!(hits[0].meeting_title, "Weekly sync");
    }

    #[tokio::test]
    async fn search_survives_operator_soup_and_empty_input() {
        let (db, m) = seeded().await;
        insert_segments(&db, &[draft(&m.id, 0, "budget review")])
            .await
            .unwrap();

        for nasty in ["\"", "*", "NEAR(", "a OR", "-- drop", "budget\"*"] {
            let hits = search_segments(
                &db,
                &SearchQuery {
                    text: nasty.to_string(),
                    ..Default::default()
                },
            )
            .await;
            assert!(hits.is_ok(), "query {nasty:?} must not error: {hits:?}");
        }
        assert!(search_segments(&db, &SearchQuery::default())
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn search_index_follows_updates_and_deletes() {
        let (db, m) = seeded().await;
        let ids = insert_segments(&db, &[draft(&m.id, 0, "pineapple")])
            .await
            .unwrap();

        let found = |db: Db, term: &'static str| async move {
            search_segments(
                &db,
                &SearchQuery {
                    text: term.into(),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .len()
        };
        assert_eq!(found(db.clone(), "pineapple").await, 1);

        revise_segment(&db, &ids[0], Some("watermelon"), None, 2, true)
            .await
            .unwrap();
        assert_eq!(found(db.clone(), "pineapple").await, 0);
        assert_eq!(found(db.clone(), "watermelon").await, 1);

        sqlx::query("DELETE FROM segments WHERE id = ?1")
            .bind(&ids[0])
            .execute(&db)
            .await
            .unwrap();
        assert_eq!(found(db.clone(), "watermelon").await, 0);
    }

    /// What "listen again" leans on: the transcript, the speakers and the
    /// language all go, the search index goes with them (via the delete
    /// trigger, not a cascade), the revision it reports moves forward — and the
    /// recording, the markers, the recap and the person's own people count are
    /// all still there afterwards.
    #[tokio::test]
    async fn clearing_a_transcript_leaves_the_recording_and_the_recap_alone() {
        let (db, m) = seeded().await;
        // A meeting that settled on the wrong language: the 75 minutes of
        // Italian recorded as Danish on 2026-08-24. Reading it again has to be
        // able to reach a different answer.
        set_meeting_language(&db, &m.id, "da").await.unwrap();
        set_speaker_count_override(&db, &m.id, Some(4))
            .await
            .unwrap();
        let chunk = insert_chunk(&db, &m.id, Channel::Mic, 0, "/a/0.flac", 0, 30_000)
            .await
            .unwrap();
        commit_chunk(&db, &chunk, 30_000).await.unwrap();
        insert_marker(&db, &m.id, 1_000, MarkerKind::ActionItem, None)
            .await
            .unwrap();
        let speaker = upsert_speaker(&db, &m.id, "mic", "You", true)
            .await
            .unwrap();
        let alias = upsert_speaker(&db, &m.id, "c2", "Speaker 2", false)
            .await
            .unwrap();
        merge_speakers(&db, &alias.id, &speaker.id).await.unwrap();
        let ids = insert_segments(&db, &[draft(&m.id, 0, "kumquat")])
            .await
            .unwrap();
        revise_segment(&db, &ids[0], Some("kumquat again"), None, 4, true)
            .await
            .unwrap();
        let summary = insert_summary(
            &db,
            &m.id,
            None,
            None,
            Provider::OnThisComputer,
            Some("test-model"),
            Some("en"),
            4,
            "## Recap",
        )
        .await
        .unwrap();

        // A stretch a pass heard and deliberately left without text.
        record_suppressed_span(
            &db,
            &m.id,
            &SuppressedSpan {
                channel: Channel::Mic,
                t_start_ms: 4_000,
                t_end_ms: 12_000,
                reason: SuppressionReason::Bleed,
                decided_by: DecidedBy::Live,
                correlation: Some(0.91),
                lag_ms: Some(210),
                system_voice_ms: Some(7_400),
            },
        )
        .await
        .unwrap();

        let cleared = clear_transcript(&db, &m.id).await.unwrap();
        assert_eq!(cleared.segments_deleted, 1);
        assert_eq!(cleared.speakers_deleted, 2, "the alias row goes too");
        assert!(cleared.language_cleared);
        assert_eq!(cleared.revision, 5, "one past where the transcript was");
        assert_eq!(
            get_meeting(&db, &m.id).await.unwrap().unwrap().language,
            None,
            "the wrong language must not be the prior the next pass reads"
        );

        assert!(get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: m.id.clone(),
                ..Default::default()
            }
        )
        .await
        .unwrap()
        .is_empty());
        assert!(list_speakers(&db, &m.id).await.unwrap().is_empty());
        assert!(search_segments(
            &db,
            &SearchQuery {
                text: "kumquat".into(),
                ..Default::default()
            }
        )
        .await
        .unwrap()
        .is_empty());

        // Everything that is not derived from a transcription pass survives.
        assert_eq!(
            last_committed_offset_ms(&db, &m.id, Channel::Mic)
                .await
                .unwrap(),
            30_000
        );
        assert_eq!(list_markers(&db, &m.id).await.unwrap().len(), 1);
        assert_eq!(
            latest_summary(&db, &m.id).await.unwrap().map(|s| s.id),
            Some(summary.id)
        );
        assert_eq!(
            people_count(&db, &m.id).await.unwrap(),
            (4, true),
            "how many people were here is the person's answer, not the pass's"
        );
    }

    /// The record that says "Echo heard this and chose not to write it down
    /// twice", which is a different claim from "Echo never heard it".
    ///
    /// It has to survive being read back — the catch-up planner subtracts these
    /// spans from its work — it has to be per channel, and it has to stop
    /// standing when the transcript goes, because "listen again" is a person
    /// asking for the recording to be judged afresh.
    #[tokio::test]
    async fn a_decision_not_to_transcribe_is_kept_with_its_evidence_and_withdrawn_with_the_words() {
        let db = connect_in_memory().await.unwrap();
        let m = create_meeting(&db, "Bleed", "/audio", None).await.unwrap();

        for (channel, from_ms, to_ms) in [
            (Channel::Mic, 4_000, 12_000),
            (Channel::Mic, 20_000, 26_000),
            (Channel::System, 40_000, 44_000),
        ] {
            record_suppressed_span(
                &db,
                &m.id,
                &SuppressedSpan {
                    channel,
                    t_start_ms: from_ms,
                    t_end_ms: to_ms,
                    reason: SuppressionReason::Bleed,
                    decided_by: DecidedBy::CatchUp,
                    correlation: Some(0.83),
                    lag_ms: Some(210),
                    system_voice_ms: Some(to_ms - from_ms),
                },
            )
            .await
            .unwrap();
        }

        assert_eq!(
            suppressed_spans(&db, &m.id, Channel::Mic).await.unwrap(),
            vec![(4_000, 12_000), (20_000, 26_000)],
            "in clock order, and only this channel's"
        );
        assert_eq!(
            suppressed_spans(&db, &m.id, Channel::System).await.unwrap(),
            vec![(40_000, 44_000)]
        );
        let all = list_suppressed_spans(&db, &m.id).await.unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].decided_by, DecidedBy::CatchUp);
        assert_eq!(all[0].reason, SuppressionReason::Bleed);
        assert_eq!(all[0].correlation, Some(0.83));
        assert_eq!(all[0].lag_ms, Some(210));

        // A mark with no width covers no audio and would only ever be a puzzle.
        assert!(matches!(
            record_suppressed_span(
                &db,
                &m.id,
                &SuppressedSpan {
                    channel: Channel::Mic,
                    t_start_ms: 5_000,
                    t_end_ms: 5_000,
                    reason: SuppressionReason::Bleed,
                    decided_by: DecidedBy::Live,
                    correlation: None,
                    lag_ms: None,
                    system_voice_ms: None,
                },
            )
            .await,
            Err(DbError::Invalid(_))
        ));

        let cleared = clear_transcript(&db, &m.id).await.unwrap();
        assert_eq!(cleared.spans_withdrawn, 3);
        assert!(
            list_suppressed_spans(&db, &m.id).await.unwrap().is_empty(),
            "a withdrawn decision is no longer why a stretch has no words"
        );
        assert!(
            suppressed_spans(&db, &m.id, Channel::Mic)
                .await
                .unwrap()
                .is_empty(),
            "and it must not hide those seconds from the planner: the sentence \
             a person pressed the button for might be one of them"
        );

        // …but the measurement is still there, because the audio did not change
        // when somebody pressed a button.
        assert_eq!(withdrawn_suppression_count(&db, &m.id).await.unwrap(), 3);
        assert_eq!(measured_lag_ms(&db, &m.id).await.unwrap(), Some(210));

        // A second clear withdraws nothing: they are already withdrawn, and
        // saying otherwise would put a number in the log for work not done.
        let again = clear_transcript(&db, &m.id).await.unwrap();
        assert_eq!(again.spans_withdrawn, 0);
        assert_eq!(withdrawn_suppression_count(&db, &m.id).await.unwrap(), 3);
    }

    /// A withdrawn decision still measures the audio it was made about.
    ///
    /// The delay between what the speakers played and what the microphone heard
    /// is a property of the machine that recorded the meeting, and every row
    /// holding one was a full hit. So the median is taken over the meeting's
    /// rows whether they still stand or not — that is the whole reason "listen
    /// again" withdraws instead of deleting.
    #[tokio::test]
    async fn the_delay_a_meeting_measured_is_read_through_its_withdrawals() {
        let db = connect_in_memory().await.unwrap();
        let m = create_meeting(&db, "Delay", "/audio", None).await.unwrap();
        assert_eq!(
            measured_lag_ms(&db, &m.id).await.unwrap(),
            None,
            "a meeting nothing was measured on inherits nothing"
        );

        let mark = |from_ms: i64, lag_ms: Option<i64>| SuppressedSpan {
            channel: Channel::Mic,
            t_start_ms: from_ms,
            t_end_ms: from_ms + 4_000,
            reason: SuppressionReason::Bleed,
            decided_by: DecidedBy::Live,
            correlation: Some(0.9),
            lag_ms,
            system_voice_ms: Some(3_800),
        };

        // Two measurements: the midpoint of the middle pair, as `LagEstimate`
        // takes it.
        record_suppressed_span(&db, &m.id, &mark(0, Some(180)))
            .await
            .unwrap();
        record_suppressed_span(&db, &m.id, &mark(10_000, Some(220)))
            .await
            .unwrap();
        assert_eq!(measured_lag_ms(&db, &m.id).await.unwrap(), Some(200));

        // A third, wildly out: the median ignores it, which is why it is a
        // median.
        record_suppressed_span(&db, &m.id, &mark(20_000, Some(9_999)))
            .await
            .unwrap();
        assert_eq!(measured_lag_ms(&db, &m.id).await.unwrap(), Some(220));

        // A reason with no measurement behind it says nothing about the delay.
        record_suppressed_span(&db, &m.id, &mark(30_000, None))
            .await
            .unwrap();
        assert_eq!(measured_lag_ms(&db, &m.id).await.unwrap(), Some(220));

        clear_transcript(&db, &m.id).await.unwrap();
        assert_eq!(
            measured_lag_ms(&db, &m.id).await.unwrap(),
            Some(220),
            "the delay survives the clear that withdrew every decision"
        );
        assert!(suppressed_spans(&db, &m.id, Channel::Mic)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn search_can_be_scoped_to_one_meeting() {
        let db = connect_in_memory().await.unwrap();
        let a = create_meeting(&db, "A", "/a", None).await.unwrap();
        let b = create_meeting(&db, "B", "/b", None).await.unwrap();
        insert_segments(&db, &[draft(&a.id, 0, "shared keyword")])
            .await
            .unwrap();
        insert_segments(&db, &[draft(&b.id, 0, "shared keyword")])
            .await
            .unwrap();

        let all = search_segments(
            &db,
            &SearchQuery {
                text: "keyword".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(all.len(), 2);
        let scoped = search_segments(
            &db,
            &SearchQuery {
                text: "keyword".into(),
                meeting_id: Some(a.id.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].meeting_id, a.id);
    }

    #[tokio::test]
    async fn deleting_a_meeting_returns_its_audio_paths() {
        let (db, m) = seeded().await;
        insert_chunk(&db, &m.id, Channel::Mic, 0, "/a/0.flac", 0, 1_000)
            .await
            .unwrap();
        insert_chunk(&db, &m.id, Channel::System, 0, "/a/s0.flac", 0, 1_000)
            .await
            .unwrap();
        set_meeting_mixed_path(&db, &m.id, "/a/mixed.flac")
            .await
            .unwrap();

        let paths = delete_meeting(&db, &m.id).await.unwrap();
        assert!(get_meeting(&db, &m.id).await.unwrap().is_none());
        assert_eq!(
            paths.len(),
            3,
            "both channels and the playback file: {paths:?}"
        );
        assert!(paths.contains(&"/a/mixed.flac".to_string()));
    }

    /// One confirmation, and then nothing is left: no rows anywhere, and no
    /// words left in the search index either.
    #[tokio::test]
    async fn deleting_a_meeting_leaves_nothing_behind() {
        let (db, m) = seeded().await;
        let other = create_meeting(&db, "Keep me", "/tmp/audio/m2", None)
            .await
            .unwrap();

        insert_chunk(&db, &m.id, Channel::Mic, 0, "/a/0.flac", 0, 1_000)
            .await
            .unwrap();
        insert_segments(&db, &[draft(&m.id, 0, "pomegranate season")])
            .await
            .unwrap();
        insert_segments(&db, &[draft(&other.id, 0, "quince season")])
            .await
            .unwrap();
        let speaker = upsert_speaker(&db, &m.id, "mic", "You", true)
            .await
            .unwrap();
        let mine = get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: m.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assign_speaker(&db, &[mine[0].id.clone()], &speaker.id, 2)
            .await
            .unwrap();
        insert_marker(&db, &m.id, 500, MarkerKind::ActionItem, Some("chase this"))
            .await
            .unwrap();
        let summary = insert_summary(
            &db,
            &m.id,
            None,
            None,
            Provider::OnThisComputer,
            None,
            Some("en"),
            1,
            "# Recap",
        )
        .await
        .unwrap();
        replace_action_items(
            &db,
            &m.id,
            Some(&summary.id),
            &[ActionItem {
                id: String::new(),
                meeting_id: m.id.clone(),
                summary_id: Some(summary.id.clone()),
                description: "send the notes".into(),
                owner: None,
                due_hint: None,
                done: false,
                external_url: None,
            }],
        )
        .await
        .unwrap();
        create_job(&db, Some(&m.id), JobKind::Summarize)
            .await
            .unwrap();

        delete_meeting(&db, &m.id).await.unwrap();

        for (table, column) in [
            ("segments", "meeting_id"),
            ("speakers", "meeting_id"),
            ("markers", "meeting_id"),
            ("summaries", "meeting_id"),
            ("action_items", "meeting_id"),
            ("audio_chunks", "meeting_id"),
            ("jobs", "meeting_id"),
        ] {
            let (left,): (i64,) =
                sqlx::query_as(&format!("SELECT COUNT(*) FROM {table} WHERE {column} = ?1"))
                    .bind(&m.id)
                    .fetch_one(&db)
                    .await
                    .unwrap();
            assert_eq!(left, 0, "{table} still has rows for a deleted meeting");
        }

        // The search index is external content, so a stale entry does not show
        // up as a row anywhere — read the index itself.
        for index in ["segments_fts", "segments_fts_trigram"] {
            let (terms, still_there) = index_terms(&db, index, "pomegranate").await;
            assert_eq!(
                still_there, 0,
                "{index} still knows a deleted meeting's words"
            );
            assert!(
                terms > 0,
                "{index} lost the meeting that was not deleted ({terms} terms)"
            );
        }

        // And the meeting nobody deleted is untouched.
        assert_eq!(count_segments(&db, &other.id).await.unwrap(), 1);
        assert_eq!(
            search_segments(
                &db,
                &SearchQuery {
                    text: "quince".into(),
                    ..Default::default()
                }
            )
            .await
            .unwrap()
            .len(),
            1
        );
    }

    /// A repaired line has to come back saying it was repaired — and a line
    /// nobody touched has to come back saying nothing, which is the state every
    /// row written before this column existed is in.
    #[tokio::test]
    async fn what_echo_put_right_travels_with_the_line_and_survives_a_bad_row() {
        let (db, m) = seeded().await;
        let draft = |text: &str, corrections: Vec<Correction>| SegmentDraft {
            meeting_id: m.id.clone(),
            t_start_ms: 0,
            t_end_ms: 2_000,
            channel: Channel::Mic,
            text: text.into(),
            revision: 1,
            is_final: true,
            corrections,
            ..Default::default()
        };
        let repaired = insert_segment(
            &db,
            &draft(
                "Allora Langola è quello che usiamo.",
                vec![Correction {
                    from: "Nongula".into(),
                    to: "Langola".into(),
                }],
            ),
        )
        .await
        .unwrap();
        assert_eq!(repaired.corrections.len(), 1);
        assert_eq!(repaired.corrections[0].from, "Nongula");
        // Round trip through a fresh read, not just the insert's own answer.
        let read_back = get_segment(&db, &repaired.id).await.unwrap().unwrap();
        assert_eq!(read_back.corrections, repaired.corrections);

        let untouched = insert_segment(&db, &draft("niente da segnalare", Vec::new()))
            .await
            .unwrap();
        assert!(untouched.corrections.is_empty());
        let (stored,): (Option<String>,) =
            sqlx::query_as("SELECT corrections FROM segments WHERE id = ?1")
                .bind(&untouched.id)
                .fetch_one(&db)
                .await
                .unwrap();
        assert_eq!(stored, None, "nothing changed is NULL, not an empty list");

        // And a row somebody's disk or an older build left unreadable is a line
        // with no note on it, never a query that fails and takes the words with
        // it.
        sqlx::query("UPDATE segments SET corrections = ?2 WHERE id = ?1")
            .bind(&repaired.id)
            .bind("{not json")
            .execute(&db)
            .await
            .unwrap();
        let salvaged = get_segment(&db, &repaired.id).await.unwrap().unwrap();
        assert_eq!(salvaged.text, "Allora Langola è quello che usiamo.");
        assert!(salvaged.corrections.is_empty());
    }

    /// The other half of the promise: a repaired line can be put back, exactly
    /// once, and only while it is still the line that was repaired.
    #[tokio::test]
    async fn a_repaired_line_can_be_put_back_once_and_only_while_it_is_that_line() {
        let (db, m) = seeded().await;
        let repaired = insert_segment(
            &db,
            &SegmentDraft {
                meeting_id: m.id.clone(),
                t_start_ms: 0,
                t_end_ms: 2_000,
                channel: Channel::Mic,
                text: "Allora Langola è quello che usiamo.".into(),
                revision: 3,
                is_final: true,
                corrections: vec![Correction {
                    from: "Nongula".into(),
                    to: "Langola".into(),
                }],
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(undo_segment_corrections(
            &db,
            &repaired.id,
            "Allora Langola è quello che usiamo.",
            "Allora Nongula è quello che usiamo.",
        )
        .await
        .unwrap());

        let after = get_segment(&db, &repaired.id).await.unwrap().unwrap();
        assert_eq!(after.text, "Allora Nongula è quello che usiamo.");
        assert!(after.corrections.is_empty(), "the note goes with the words");
        assert_eq!(
            after.revision, 3,
            "putting a line back is not a later, better pass"
        );

        // Twice is a no-op, not a second rewrite: there is nothing left to undo.
        assert!(!undo_segment_corrections(
            &db,
            &repaired.id,
            "Allora Nongula è quello che usiamo.",
            "something else entirely",
        )
        .await
        .unwrap());
        assert_eq!(
            get_segment(&db, &repaired.id).await.unwrap().unwrap().text,
            "Allora Nongula è quello che usiamo."
        );

        // And the words that are there now are the ones search finds.
        let hits = search_segments(
            &db,
            &SearchQuery {
                text: "Nongula".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), 1, "the index followed the line back");
    }

    /// A pass got to the row between the read and the write. The row is theirs;
    /// putting a misheard word over it would be the one thing an undo must never
    /// do.
    #[tokio::test]
    async fn a_line_that_moved_under_the_undo_keeps_the_words_it_has_now() {
        let (db, m) = seeded().await;
        let repaired = insert_segment(
            &db,
            &SegmentDraft {
                meeting_id: m.id.clone(),
                t_start_ms: 0,
                t_end_ms: 2_000,
                channel: Channel::Mic,
                text: "Allora Langola è quello che usiamo.".into(),
                revision: 1,
                is_final: true,
                corrections: vec![Correction {
                    from: "Nongula".into(),
                    to: "Langola".into(),
                }],
                ..Default::default()
            },
        )
        .await
        .unwrap();
        revise_segment(&db, &repaired.id, Some("un'altra frase"), None, 2, true)
            .await
            .unwrap();

        assert!(!undo_segment_corrections(
            &db,
            &repaired.id,
            "Allora Langola è quello che usiamo.",
            "Allora Nongula è quello che usiamo.",
        )
        .await
        .unwrap());
        assert_eq!(
            get_segment(&db, &repaired.id).await.unwrap().unwrap().text,
            "un'altra frase"
        );
    }

    // -----------------------------------------------------------------------
    // Two names, one voice
    // -----------------------------------------------------------------------

    /// Everything that named the merged person moves, and the meetings that
    /// have already been read are left reading exactly as they read.
    #[tokio::test]
    async fn merging_two_people_moves_the_voice_and_leaves_the_meetings_alone() {
        let (db, m) = seeded().await;
        let keep = create_person(&db, "Marco").await.unwrap();
        let dup = create_person(&db, "Marco (call)").await.unwrap();
        insert_person_sample(
            &db,
            &NewPersonSample {
                person_id: &dup.id,
                embedding: &[0.5f32; 4],
                clip: b"wav",
                condition: Channel::System,
                source_meeting_id: None,
                t_start_ms: 0,
                t_end_ms: 6_000,
            },
        )
        .await
        .unwrap();
        let speaker = upsert_speaker(&db, &m.id, "c1", "Speaker 1", false)
            .await
            .unwrap();
        set_speaker_person(&db, &speaker.id, Some(&dup.id))
            .await
            .unwrap();
        rename_speaker(&db, &speaker.id, "Marco").await.unwrap();
        let other = upsert_speaker(&db, &m.id, "c2", "Speaker 2", false)
            .await
            .unwrap();
        set_speaker_suggestion(&db, &other.id, Some(&dup.id), Some(0.6))
            .await
            .unwrap();

        let done = merge_people(&db, &keep.id, &dup.id).await.unwrap();
        assert!(done.merged);
        assert_eq!(done.samples_moved, 1);
        assert_eq!(done.speakers_relinked, 1);
        assert_eq!(done.suggestions_relinked, 1);

        assert_eq!(list_person_samples(&db, &keep.id).await.unwrap().len(), 1);
        assert!(list_person_samples(&db, &dup.id).await.unwrap().is_empty());
        let linked = get_speaker(&db, &speaker.id).await.unwrap().unwrap();
        assert_eq!(linked.person_id.as_deref(), Some(keep.id.as_str()));
        assert_eq!(
            linked.display_name, "Marco",
            "a meeting somebody has read does not rewrite itself"
        );
        assert_eq!(
            get_speaker(&db, &other.id)
                .await
                .unwrap()
                .unwrap()
                .suggested_person_id
                .as_deref(),
            Some(keep.id.as_str())
        );

        // The merged name stops being somebody, and still resolves to whoever
        // it is now.
        let people = list_people(&db).await.unwrap();
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].id, keep.id);
        assert_eq!(
            resolve_person(&db, &dup.id).await.unwrap().map(|p| p.id),
            Some(keep.id.clone())
        );

        // Twice, and the other way round, both change nothing.
        assert!(!merge_people(&db, &keep.id, &dup.id).await.unwrap().merged);
        assert!(!merge_people(&db, &dup.id, &keep.id).await.unwrap().merged);
        assert_eq!(list_person_samples(&db, &keep.id).await.unwrap().len(), 1);
        assert_eq!(list_people(&db).await.unwrap().len(), 1);
    }

    /// A person is never merged into themselves, and a chain never gets longer
    /// than one hop — the same two rules `merge_speakers` keeps.
    #[tokio::test]
    async fn a_person_is_never_merged_into_themselves_and_chains_stay_flat() {
        let db = connect_in_memory().await.unwrap();
        let a = create_person(&db, "A").await.unwrap();
        let b = create_person(&db, "B").await.unwrap();
        let c = create_person(&db, "C").await.unwrap();

        let err = merge_people(&db, &a.id, &a.id).await.unwrap_err();
        assert!(matches!(err, DbError::Invalid(_)), "{err:?}");
        assert!(get_person(&db, &a.id)
            .await
            .unwrap()
            .unwrap()
            .alias_of
            .is_none());

        merge_people(&db, &b.id, &c.id).await.unwrap();
        merge_people(&db, &a.id, &b.id).await.unwrap();
        for merged in [&b, &c] {
            assert_eq!(
                get_person(&db, &merged.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .alias_of
                    .as_deref(),
                Some(a.id.as_str()),
                "every merged name points straight at the survivor"
            );
        }
        assert_eq!(list_people(&db).await.unwrap().len(), 1);

        // A name that has already been merged means the person it is now, so
        // merging it again lands on the survivor rather than making a ring.
        let d = create_person(&db, "D").await.unwrap();
        merge_people(&db, &d.id, &c.id).await.unwrap();
        assert_eq!(
            resolve_person(&db, &c.id).await.unwrap().map(|p| p.id),
            Some(d.id.clone())
        );
        assert!(!merge_people(&db, &c.id, &d.id).await.unwrap().merged);
    }

    /// A suggestion that would land on the person the row is already linked to
    /// is noise, so it goes — the same thing `set_speaker_person` does.
    #[tokio::test]
    async fn a_suggestion_that_the_merge_makes_redundant_is_dropped() {
        let (db, m) = seeded().await;
        let keep = create_person(&db, "Marco").await.unwrap();
        let dup = create_person(&db, "Marco again").await.unwrap();
        let speaker = upsert_speaker(&db, &m.id, "c1", "Speaker 1", false)
            .await
            .unwrap();
        set_speaker_person(&db, &speaker.id, Some(&keep.id))
            .await
            .unwrap();
        set_speaker_suggestion(&db, &speaker.id, Some(&dup.id), Some(0.55))
            .await
            .unwrap();

        merge_people(&db, &keep.id, &dup.id).await.unwrap();

        let after = get_speaker(&db, &speaker.id).await.unwrap().unwrap();
        assert_eq!(after.person_id.as_deref(), Some(keep.id.as_str()));
        assert_eq!(after.suggested_person_id, None);
        assert_eq!(after.suggestion_score, None);
    }

    /// Forgetting the survivor forgets the name that was folded into them.
    /// Unlike a merged speaker, a merged person owns nothing of their own, so
    /// releasing the row would put an empty person back in the list.
    #[tokio::test]
    async fn forgetting_the_survivor_forgets_the_name_folded_into_them() {
        let db = connect_in_memory().await.unwrap();
        let keep = create_person(&db, "Marco").await.unwrap();
        let dup = create_person(&db, "Marco (call)").await.unwrap();
        merge_people(&db, &keep.id, &dup.id).await.unwrap();

        delete_person(&db, &keep.id).await.unwrap();

        assert!(get_person(&db, &dup.id).await.unwrap().is_none());
        assert!(list_people(&db).await.unwrap().is_empty());
    }

    /// Cutting a line in two has to leave search knowing about both halves and
    /// nothing about the line they came from. The index is external content, so a
    /// stale entry is invisible as a row — it has to be read out of the index.
    #[tokio::test]
    async fn splitting_a_line_moves_its_words_into_the_search_index_in_halves() {
        let (db, m) = seeded().await;
        let whole = insert_segment(
            &db,
            &SegmentDraft {
                meeting_id: m.id.clone(),
                t_start_ms: 0,
                t_end_ms: 8_000,
                channel: Channel::Mic,
                speaker_id: None,
                text: "pomegranate then quince".into(),
                language: Some("it".into()),
                avg_confidence: Some(0.8),
                revision: 3,
                is_final: true,
                model_name: Some("whisper large-v3 (ggml)".into()),
                model_revision: Some("abc123".into()),
                corrections: Vec::new(),
            },
        )
        .await
        .unwrap();

        let piece = |from: i64, to: i64, text: &str| SegmentDraft {
            meeting_id: m.id.clone(),
            t_start_ms: from,
            t_end_ms: to,
            channel: Channel::Mic,
            speaker_id: None,
            text: text.into(),
            language: Some("it".into()),
            avg_confidence: Some(0.8),
            revision: 4,
            is_final: true,
            model_name: Some("whisper large-v3 (ggml)".into()),
            model_revision: Some("abc123".into()),
            corrections: Vec::new(),
        };
        let ids = split_segment(
            &db,
            &whole.id,
            &[
                piece(0, 4_000, "pomegranate then"),
                piece(4_000, 8_000, "quince"),
            ],
            4,
        )
        .await
        .unwrap();
        assert_eq!(ids.len(), 2);
        assert!(get_segment(&db, &whole.id).await.unwrap().is_none());

        // Both halves are findable, and the words are each in exactly one of
        // them.
        for word in ["pomegranate", "quince"] {
            let hits = search_segments(
                &db,
                &SearchQuery {
                    text: word.into(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_eq!(hits.len(), 1, "search lost {word}");
        }
        // The word index knows whole words; the trigram index knows threes.
        for (index, term) in [
            ("segments_fts", "pomegranate"),
            ("segments_fts_trigram", "pom"),
        ] {
            let (_, hits) = index_terms(&db, index, term).await;
            assert_eq!(hits, 1, "{index} has the wrong idea about the split");
        }

        // Provenance and the revision came across; the pieces tile the line.
        let after = get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: m.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].t_start_ms, 0);
        assert_eq!(after[1].t_end_ms, 8_000);
        assert!(after
            .iter()
            .all(|s| s.revision == 4 && s.model_revision.as_deref() == Some("abc123")));
    }

    #[tokio::test]
    async fn a_split_is_refused_rather_than_half_done() {
        let (db, m) = seeded().await;
        let whole = insert_segment(&db, &draft(&m.id, 0, "kumquat"))
            .await
            .unwrap();
        let piece = |text: &str, revision: i64| SegmentDraft {
            meeting_id: m.id.clone(),
            t_start_ms: 0,
            t_end_ms: 500,
            channel: Channel::Mic,
            text: text.into(),
            revision,
            is_final: true,
            ..Default::default()
        };

        // One piece is not a split.
        assert!(split_segment(&db, &whole.id, &[piece("kumquat", 2)], 2)
            .await
            .is_err());
        // Nor is a piece that would let an older pass overwrite a newer one.
        assert!(
            split_segment(&db, &whole.id, &[piece("kum", 1), piece("quat", 2)], 2)
                .await
                .is_err()
        );
        // Both refusals left the line exactly where it was.
        assert_eq!(
            get_segment(&db, &whole.id).await.unwrap().unwrap().text,
            "kumquat"
        );

        // A line a later pass has already moved past is not split at all, and
        // that is not an error — it is no longer the current answer.
        let newer = insert_segment(
            &db,
            &SegmentDraft {
                revision: 5,
                ..draft(&m.id, 1_000, "loquat")
            },
        )
        .await
        .unwrap();
        assert!(
            split_segment(&db, &newer.id, &[piece("lo", 4), piece("quat", 4)], 4)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            get_segment(&db, &newer.id).await.unwrap().unwrap().text,
            "loquat"
        );

        // And a line that is not there any more is a no-op, not a panic.
        assert!(
            split_segment(&db, "no-such-line", &[piece("a", 9), piece("b", 9)], 9)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn delete_all_clears_the_search_index_too() {
        let (db, m) = seeded().await;
        insert_segments(&db, &[draft(&m.id, 0, "tangerine")])
            .await
            .unwrap();
        delete_all_meetings(&db).await.unwrap();

        let (terms, _) = index_terms(&db, "segments_fts", "tangerine").await;
        assert_eq!(terms, 0, "delete-all left words in the search index");
    }

    /// "Delete everything" has to include the remembered voices, and they are the
    /// one thing that would otherwise survive it: a person's samples hold their
    /// own copy of the audio, so deleting every meeting leaves them untouched
    /// (DESIGN §1: "`delete_all_data` includes people").
    #[tokio::test]
    async fn delete_all_takes_the_remembered_voices_and_their_audio_too() {
        let (db, m) = seeded().await;
        let speaker = upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let person = create_person(&db, "Marco").await.unwrap();
        insert_person_sample(
            &db,
            &NewPersonSample {
                person_id: &person.id,
                embedding: &[0.5, 0.5, 0.5, 0.5],
                clip: b"RIFF....WAVE",
                condition: Channel::Mic,
                source_meeting_id: Some(&m.id),
                t_start_ms: 0,
                t_end_ms: 6_000,
            },
        )
        .await
        .unwrap();
        upsert_person_profile(&db, &person.id, &[0.5, 0.5, 0.5, 0.5], 1, "network")
            .await
            .unwrap();
        set_speaker_person(&db, &speaker.id, Some(&person.id))
            .await
            .unwrap();

        // Meetings alone are not enough — the samples keep their own audio.
        delete_all_meetings(&db).await.unwrap();
        assert_eq!(list_person_samples(&db, &person.id).await.unwrap().len(), 1);

        delete_all_people(&db).await.unwrap();
        assert!(list_people(&db).await.unwrap().is_empty());
        assert!(list_person_samples(&db, &person.id)
            .await
            .unwrap()
            .is_empty());
        assert!(list_person_profiles(&db).await.unwrap().is_empty());
        for table in ["people", "person_samples", "person_profiles"] {
            let (rows,): (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM {table}"))
                .fetch_one(&db)
                .await
                .unwrap();
            assert_eq!(rows, 0, "{table} still holds something");
        }
    }

    #[tokio::test]
    async fn a_voice_print_survives_the_trip_through_a_blob() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32) / 512.0 - 0.25).collect();
        let blob = embedding_to_blob(&original);
        assert_eq!(blob.len(), original.len() * 4);
        assert_eq!(blob_to_embedding(&blob), original);
        // A truncated row gives back what it can rather than failing a query: a
        // profile with a short vector simply matches nothing.
        assert_eq!(blob_to_embedding(&blob[..9]).len(), 2);
        assert!(blob_to_embedding(&[]).is_empty());
    }

    #[tokio::test]
    async fn a_person_reports_when_echo_last_heard_them_and_whether_it_can_match_them() {
        let (db, m) = seeded().await;
        let person = create_person(&db, "Marco").await.unwrap();
        insert_person_sample(
            &db,
            &NewPersonSample {
                person_id: &person.id,
                embedding: &[1.0, 0.0],
                clip: b"clip",
                condition: Channel::System,
                source_meeting_id: None,
                t_start_ms: 0,
                t_end_ms: 6_000,
            },
        )
        .await
        .unwrap();
        upsert_person_profile(&db, &person.id, &[1.0, 0.0], 1, "network-a")
            .await
            .unwrap();

        // Never heard in a meeting yet.
        let info = &list_person_infos(&db, "network-a").await.unwrap()[0];
        assert_eq!(info.sample_count, 1);
        assert_eq!(info.last_heard_at, None);
        assert!(!info.needs_refresh);

        let speaker = upsert_speaker(&db, &m.id, "speaker-01", "Marco", false)
            .await
            .unwrap();
        set_speaker_person(&db, &speaker.id, Some(&person.id))
            .await
            .unwrap();
        let info = &list_person_infos(&db, "network-a").await.unwrap()[0];
        assert_eq!(info.last_heard_at.as_deref(), Some(m.started_at.as_str()));

        // A different network: still remembered, not currently matched.
        let info = &list_person_infos(&db, "network-b").await.unwrap()[0];
        assert!(info.needs_refresh);
    }

    #[tokio::test]
    async fn linking_a_speaker_takes_any_suggestion_off_the_row() {
        let (db, m) = seeded().await;
        let speaker = upsert_speaker(&db, &m.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let person = create_person(&db, "Marco").await.unwrap();
        set_speaker_suggestion(&db, &speaker.id, Some(&person.id), Some(0.55))
            .await
            .unwrap();
        assert!(get_speaker(&db, &speaker.id)
            .await
            .unwrap()
            .unwrap()
            .suggested_person_id
            .is_some());

        set_speaker_person(&db, &speaker.id, Some(&person.id))
            .await
            .unwrap();
        let row = get_speaker(&db, &speaker.id).await.unwrap().unwrap();
        assert_eq!(row.person_id.as_deref(), Some(person.id.as_str()));
        assert!(
            row.suggested_person_id.is_none(),
            "answered, so stop asking"
        );
        assert!(row.suggestion_score.is_none());

        // A score without a person is not a suggestion, and must not be stored as
        // one.
        set_speaker_suggestion(&db, &speaker.id, None, Some(0.9))
            .await
            .unwrap();
        assert!(get_speaker(&db, &speaker.id)
            .await
            .unwrap()
            .unwrap()
            .suggestion_score
            .is_none());
    }

    #[tokio::test]
    async fn renaming_or_deleting_a_person_that_is_not_there_says_so() {
        let db = connect_in_memory().await.unwrap();
        assert!(matches!(
            rename_person(&db, &new_id(), "Marco").await,
            Err(DbError::NotFound(_))
        ));
        assert!(matches!(
            delete_person(&db, &new_id()).await,
            Err(DbError::NotFound(_))
        ));
        assert!(matches!(
            set_speaker_person(&db, &new_id(), None).await,
            Err(DbError::NotFound(_))
        ));
    }

    /// How many terms one search index holds, and how many of them are `needle`.
    /// The indexes are external-content FTS5 tables, so a stale entry is
    /// invisible from the `segments` side — `fts5vocab` reads the index itself.
    async fn index_terms(db: &Db, index: &str, needle: &str) -> (i64, i64) {
        let view = format!("v_{index}");
        sqlx::query(&format!("DROP TABLE IF EXISTS {view}"))
            .execute(db)
            .await
            .unwrap();
        sqlx::query(&format!(
            "CREATE VIRTUAL TABLE {view} USING fts5vocab({index}, 'row')"
        ))
        .execute(db)
        .await
        .unwrap();
        let (terms,): (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM {view}"))
            .fetch_one(db)
            .await
            .unwrap();
        let (matching,): (i64,) =
            sqlx::query_as(&format!("SELECT COUNT(*) FROM {view} WHERE term = ?1"))
                .bind(needle)
                .fetch_one(db)
                .await
                .unwrap();
        sqlx::query(&format!("DROP TABLE {view}"))
            .execute(db)
            .await
            .unwrap();
        (terms, matching)
    }

    /// Browse-all needs every meeting the person has, including ones that have
    /// not been given a recap (or even a transcript) yet.
    #[tokio::test]
    async fn listing_includes_meetings_with_no_recap_yet() {
        let db = connect_in_memory().await.unwrap();
        let fresh = create_meeting(&db, "Still processing", "/tmp/a", None)
            .await
            .unwrap();
        set_meeting_status(&db, &fresh.id, MeetingStatus::Processing)
            .await
            .unwrap();
        let gone = create_meeting(&db, "Thrown away", "/tmp/b", None)
            .await
            .unwrap();
        soft_delete_meeting(&db, &gone.id).await.unwrap();

        let listed = list_meetings(&db, &MeetingQuery::default()).await.unwrap();
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].id, fresh.id);
        assert!(!listed[0].has_recap);
        assert_eq!(count_meetings(&db).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn forget_audio_keeps_the_text() {
        let (db, m) = seeded().await;
        insert_chunk(&db, &m.id, Channel::Mic, 0, "/a/0.flac", 0, 1_000)
            .await
            .unwrap();
        insert_segments(&db, &[draft(&m.id, 0, "keep me")])
            .await
            .unwrap();

        let paths = forget_audio(&db, &m.id).await.unwrap();
        assert_eq!(paths, vec!["/a/0.flac".to_string()]);
        assert_eq!(count_segments(&db, &m.id).await.unwrap(), 1);
        assert!(list_chunks(&db, &m.id, None).await.unwrap().is_empty());
        assert!(get_meeting(&db, &m.id)
            .await
            .unwrap()
            .unwrap()
            .mixed_path
            .is_none());
    }

    #[tokio::test]
    async fn language_histogram_orders_by_time() {
        let (db, m) = seeded().await;
        let mut it = draft(&m.id, 0, "ciao");
        it.language = Some("it".into());
        it.t_end_ms = 5_000;
        let mut en = draft(&m.id, 5_000, "hello");
        en.language = Some("en".into());
        en.t_end_ms = 7_000;
        insert_segments(&db, &[it, en]).await.unwrap();
        let hist = language_histogram(&db, &m.id).await.unwrap();
        assert_eq!(hist[0].0, "it");
        assert_eq!(hist[0].1, 5_000);
    }

    #[tokio::test]
    async fn markers_and_details_come_back_together() {
        let (db, m) = seeded().await;
        insert_marker(
            &db,
            &m.id,
            12_000,
            MarkerKind::ActionItem,
            Some("follow up"),
        )
        .await
        .unwrap();
        insert_chunk(&db, &m.id, Channel::Mic, 0, "/a/0.flac", 0, 1_000)
            .await
            .unwrap();
        insert_segments(&db, &[draft(&m.id, 0, "hi")])
            .await
            .unwrap();

        let detail = get_meeting_detail(&db, &m.id).await.unwrap();
        assert_eq!(detail.markers.len(), 1);
        assert_eq!(detail.segment_count, 1);
        assert_eq!(detail.captured_channels, vec![Channel::Mic]);
    }

    #[tokio::test]
    async fn speaking_time_is_recomputed_from_segments() {
        let (db, m) = seeded().await;
        let sp = upsert_speaker(&db, &m.id, "mic", "You", true)
            .await
            .unwrap();
        let mut d = draft(&m.id, 0, "hello");
        d.speaker_id = Some(sp.id.clone());
        d.t_end_ms = 4_000;
        insert_segments(&db, &[d]).await.unwrap();
        recompute_speaking_time(&db, &m.id).await.unwrap();
        assert_eq!(
            get_speaker(&db, &sp.id).await.unwrap().unwrap().speaking_ms,
            4_000
        );
    }

    #[tokio::test]
    async fn assign_speaker_respects_revision_ordering() {
        let (db, m) = seeded().await;
        let sp = upsert_speaker(&db, &m.id, "spk1", "Speaker 1", false)
            .await
            .unwrap();
        let ids = insert_segments(&db, &[draft(&m.id, 0, "a"), draft(&m.id, 1_000, "b")])
            .await
            .unwrap();
        assert_eq!(assign_speaker(&db, &ids, &sp.id, 2).await.unwrap(), 2);
        // Same ids, older revision: refused.
        assert_eq!(assign_speaker(&db, &ids, &sp.id, 1).await.unwrap(), 0);
        assert_eq!(assign_speaker(&db, &[], &sp.id, 5).await.unwrap(), 0);
    }

    /// A "write this recap again, but differently" click has to survive the trip
    /// through the table, because the handler runs later and settings may say
    /// something else by then.
    #[tokio::test]
    async fn a_job_remembers_what_it_was_asked_for() {
        let (db, m) = seeded().await;

        let job = ensure_job_with_payload(&db, Some(&m.id), JobKind::Summarize, Some("{\"a\":1}"))
            .await
            .unwrap();
        assert_eq!(
            get_job_payload(&db, &job.id).await.unwrap().as_deref(),
            Some("{\"a\":1}")
        );

        // Asking again reuses the row but takes the newer request: the person
        // changed their mind before the first one ran.
        let again =
            ensure_job_with_payload(&db, Some(&m.id), JobKind::Summarize, Some("{\"a\":2}"))
                .await
                .unwrap();
        assert_eq!(again.id, job.id, "one job of a kind per meeting");
        assert_eq!(
            get_job_payload(&db, &job.id).await.unwrap().as_deref(),
            Some("{\"a\":2}")
        );

        // Queueing with nothing to remember must not wipe what is already there.
        ensure_job(&db, Some(&m.id), JobKind::Summarize)
            .await
            .unwrap();
        assert_eq!(
            get_job_payload(&db, &job.id).await.unwrap().as_deref(),
            Some("{\"a\":2}")
        );

        // Jobs nobody passed a payload for have none, not an empty string.
        let plain = ensure_job(&db, Some(&m.id), JobKind::Diarize)
            .await
            .unwrap();
        assert!(get_job_payload(&db, &plain.id).await.unwrap().is_none());
    }

    /// The stage a job is in has to survive being announced, or the screen that
    /// mounts a minute later has nothing to read.
    ///
    /// This is the launch case in full: the one-time setup is queued before the
    /// window has finished loading, so the announcement of its stage reaches
    /// nobody, and for the next quarter of an hour the row is the only thing
    /// that knows what the machine is busy with.
    #[tokio::test]
    async fn the_stage_a_job_is_in_is_on_the_row_and_comes_off_it() {
        let (db, m) = seeded().await;
        let job = ensure_job(&db, Some(&m.id), JobKind::PrepareEngine)
            .await
            .unwrap();
        assert!(job.phase.is_none(), "a job starts in no stage at all");

        set_job_phase(&db, &job.id, Some(JobPhase::PreparingEngine))
            .await
            .unwrap();
        assert_eq!(
            get_job(&db, &job.id).await.unwrap().unwrap().phase,
            Some(JobPhase::PreparingEngine),
            "read back by id"
        );
        let listed = list_jobs(
            &db,
            &JobQuery {
                meeting_id: Some(m.id.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            listed.first().and_then(|j| j.phase),
            Some(JobPhase::PreparingEngine),
            "and by the listing every screen loads on mount"
        );

        // Put back. Nothing is in a stage it has left.
        set_job_phase(&db, &job.id, None).await.unwrap();
        assert!(get_job(&db, &job.id)
            .await
            .unwrap()
            .unwrap()
            .phase
            .is_none());
    }

    /// A stage that outlived the moment it described would be a new lie, not a
    /// fix — "Finishing one-time setup…" over a job that has finished, failed,
    /// or is parked behind a recording. Every write that moves a job's status
    /// ends its stage, including the bulk ones nothing calls one row at a time.
    #[tokio::test]
    async fn no_stage_survives_the_job_moving_on() {
        let (db, m) = seeded().await;
        let in_a_stage = |db: Db, id: String| async move {
            set_job_phase(&db, &id, Some(JobPhase::PreparingEngine))
                .await
                .unwrap();
        };
        let stage_of =
            |db: Db, id: String| async move { get_job(&db, &id).await.unwrap().unwrap().phase };

        let job = ensure_job(&db, Some(&m.id), JobKind::PrepareEngine)
            .await
            .unwrap();

        // Finished, however it finished.
        in_a_stage(db.clone(), job.id.clone()).await;
        set_job_status(&db, &job.id, JobStatus::Done, None)
            .await
            .unwrap();
        assert!(stage_of(db.clone(), job.id.clone()).await.is_none());

        // Parked for a recording, and let go again afterwards.
        set_job_status(&db, &job.id, JobStatus::Running, None)
            .await
            .unwrap();
        in_a_stage(db.clone(), job.id.clone()).await;
        pause_active_jobs(&db).await.unwrap();
        assert!(stage_of(db.clone(), job.id.clone()).await.is_none());
        in_a_stage(db.clone(), job.id.clone()).await;
        resume_paused_jobs(&db).await.unwrap();
        assert!(stage_of(db.clone(), job.id.clone()).await.is_none());

        // And the launch sweep, which is what catches a stage left by a crash:
        // the process went down mid-compile, so nothing was ever able to clear
        // it from inside the job.
        set_job_status(&db, &job.id, JobStatus::Running, None)
            .await
            .unwrap();
        in_a_stage(db.clone(), job.id.clone()).await;
        requeue_orphaned_jobs(&db, "interrupted").await.unwrap();
        assert!(stage_of(db.clone(), job.id.clone()).await.is_none());

        // The one-a-launch retry of the setup, same rule.
        set_job_payload(&db, &job.id, Some("\"w.bin\""))
            .await
            .unwrap();
        set_job_status(&db, &job.id, JobStatus::Failed, Some("no"))
            .await
            .unwrap();
        in_a_stage(db.clone(), job.id.clone()).await;
        requeue_failed_setup_jobs(&db, "\"w.bin\"").await.unwrap();
        assert!(stage_of(db.clone(), job.id.clone()).await.is_none());
    }

    /// The row has to be able to name its own directory and its playback file
    /// without `finish_meeting` rewriting when the meeting ended.
    #[tokio::test]
    async fn a_meeting_records_its_folder_and_playback_file_independently() {
        let (db, m) = seeded().await;

        set_meeting_audio_dir(&db, &m.id, "/tmp/audio/m1/abc")
            .await
            .unwrap();
        set_meeting_mixed_path(&db, &m.id, "/tmp/audio/m1/abc/mixed.wav")
            .await
            .unwrap();

        let back = get_meeting(&db, &m.id).await.unwrap().unwrap();
        assert_eq!(back.audio_dir, "/tmp/audio/m1/abc");
        assert_eq!(
            back.mixed_path.as_deref(),
            Some("/tmp/audio/m1/abc/mixed.wav")
        );
        assert!(
            back.ended_at.is_none(),
            "recording the playback file must not look like the meeting ended"
        );
    }
}
