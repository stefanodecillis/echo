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
    ActionItem, ActionItemPatch, AssetKind, AudioChunk, Channel, Id, Job, JobKind, JobQuery,
    JobStatus, Marker, MarkerKind, Meeting, MeetingDetail, MeetingQuery, MeetingStatus,
    MeetingSummary, ModelInfo, Provider, SearchHit, SearchQuery, Segment, SegmentDraft, Speaker,
    Summary, Template, TranscriptQuery,
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
                (SELECT COUNT(*) FROM speakers s
                   WHERE s.meeting_id = m.id AND s.alias_of IS NULL) AS speaker_count,
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
    let mut paths: Vec<String> = sqlx::query_as("SELECT path FROM audio_chunks WHERE meeting_id = ?1")
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
    let rows: Vec<(String, Option<String>)> =
        sqlx::query_as("SELECT id, audio_dir FROM meetings").fetch_all(db).await?;
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

    Ok(MeetingDetail {
        speakers: list_speakers(db, id).await?,
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
                                   model_revision)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
                avg_confidence, revision, is_final, model_name, model_revision
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
                avg_confidence, revision, is_final, model_name, model_revision
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

/// Drop live partials once the final pass replaced them.
pub async fn delete_partial_segments(db: &Db, meeting_id: &str) -> Result<u64, DbError> {
    let r = sqlx::query("DELETE FROM segments WHERE meeting_id = ?1 AND is_final = 0")
        .bind(meeting_id)
        .execute(db)
        .await?;
    Ok(r.rows_affected())
}

/// What [`clear_transcript`] took away.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClearedTranscript {
    pub segments_deleted: u64,
    pub speakers_deleted: u64,
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
/// would write again: the transcript and the speakers, aliases included.
///
/// For "listen again" — the audio on disk is the truth (mantra 3), so a
/// transcript written by a broken pipeline is safe to delete and read back from
/// the recording. Deliberately **not** the recap or its task list: those are the
/// person's to keep or rewrite.
///
/// Two details this depends on, both the same as [`delete_meeting`]:
/// * segments are deleted by name so the `segments_fts_ad` trigger fires and
///   both search indexes lose the words. A cascade might not fire triggers, and
///   somebody's search results are not a thing to bet on a build detail.
/// * speakers go after segments, so nothing depends on `ON DELETE SET NULL`
///   having run first.
///
/// Chunks, markers, summaries and action items are left exactly where they are.
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
    tx.commit().await?;

    Ok(ClearedTranscript {
        segments_deleted: segments,
        speakers_deleted: speakers,
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
        "SELECT id, meeting_id, cluster_key, display_name, alias_of, is_self, speaking_ms
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
        "SELECT id, meeting_id, cluster_key, display_name, alias_of, is_self, speaking_ms
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
        "SELECT id, meeting_id, cluster_key, display_name, alias_of, is_self, speaking_ms
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
        "SELECT id, meeting_id, kind, status, progress, error, created_at, updated_at
         FROM jobs WHERE id = ?1",
    )
    .bind(id)
    .fetch_optional(db)
    .await?;
    row.map(row_to_job).transpose()
}

pub async fn list_jobs(db: &Db, q: &JobQuery) -> Result<Vec<Job>, DbError> {
    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
        "SELECT id, meeting_id, kind, status, progress, error, created_at, updated_at
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

pub async fn set_job_status(
    db: &Db,
    id: &str,
    status: JobStatus,
    error: Option<&str>,
) -> Result<(), DbError> {
    sqlx::query("UPDATE jobs SET status = ?2, error = ?3, updated_at = ?4 WHERE id = ?1")
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

/// Recording has absolute priority: park everything that is running.
pub async fn pause_active_jobs(db: &Db) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE jobs SET status = 'paused', updated_at = ?1 WHERE status IN ('running', 'queued')",
    )
    .bind(now())
    .execute(db)
    .await?;
    Ok(r.rows_affected())
}

/// Recording stopped; let background work continue.
pub async fn resume_paused_jobs(db: &Db) -> Result<u64, DbError> {
    let r =
        sqlx::query("UPDATE jobs SET status = 'queued', updated_at = ?1 WHERE status = 'paused'")
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
pub async fn requeue_orphaned_jobs(db: &Db) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE jobs SET status = 'queued', updated_at = ?1
         WHERE status IN ('running', 'paused')",
    )
    .bind(now())
    .execute(db)
    .await?;
    Ok(r.rows_affected())
}

/// Oldest queued job, honouring the priority order in DESIGN §3.
pub async fn next_queued_job(db: &Db) -> Result<Option<Job>, DbError> {
    let row = sqlx::query(
        "SELECT id, meeting_id, kind, status, progress, error, created_at, updated_at
         FROM jobs WHERE status = 'queued'
         ORDER BY CASE kind
                    WHEN 'download' THEN 0
                    WHEN 'transcribe_catchup' THEN 1
                    WHEN 'diarize' THEN 2
                    WHEN 'mixdown' THEN 3
                    WHEN 'summarize' THEN 4
                    ELSE 5
                  END,
                  created_at
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
    })
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
    Ok(Job {
        id: row.try_get("id").map_err(decode)?,
        meeting_id: row.try_get("meeting_id").map_err(decode)?,
        kind: JobKind::parse(&kind).ok_or_else(|| bad_enum("kind", &kind))?,
        status: JobStatus::parse(&status).ok_or_else(|| bad_enum("status", &status))?,
        progress: row.try_get("progress").map_err(decode)?,
        error: row.try_get("error").map_err(decode)?,
        created_at: row.try_get("created_at").map_err(decode)?,
        updated_at: row.try_get("updated_at").map_err(decode)?,
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

        let err = merge_speakers(&db, &mine.id, &theirs.id)
            .await
            .unwrap_err();
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
        let done = create_job(&db, Some(&m.id), JobKind::Mixdown).await.unwrap();
        set_job_status(&db, &done.id, JobStatus::Done, None)
            .await
            .unwrap();

        // Nothing can legitimately be running or parked at launch: the flag that
        // parks work lives in memory only.
        assert_eq!(requeue_orphaned_jobs(&db).await.unwrap(), 2);
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

    #[tokio::test]
    async fn next_queued_job_prefers_catchup_over_summarize() {
        let (db, m) = seeded().await;
        create_job(&db, Some(&m.id), JobKind::Summarize)
            .await
            .unwrap();
        let catchup = create_job(&db, Some(&m.id), JobKind::TranscribeCatchup)
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

    /// What "listen again" leans on: the transcript and the speakers go, the
    /// search index goes with them (via the delete trigger, not a cascade), the
    /// revision it reports moves forward — and the recording, the markers and
    /// the recap are all still there afterwards.
    #[tokio::test]
    async fn clearing_a_transcript_leaves_the_recording_and_the_recap_alone() {
        let (db, m) = seeded().await;
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

        let cleared = clear_transcript(&db, &m.id).await.unwrap();
        assert_eq!(cleared.segments_deleted, 1);
        assert_eq!(cleared.speakers_deleted, 2, "the alias row goes too");
        assert_eq!(cleared.revision, 5, "one past where the transcript was");

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
        set_meeting_mixed_path(&db, &m.id, "/a/mixed.flac").await.unwrap();

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
        let speaker = upsert_speaker(&db, &m.id, "mic", "You", true).await.unwrap();
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
        create_job(&db, Some(&m.id), JobKind::Summarize).await.unwrap();

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
