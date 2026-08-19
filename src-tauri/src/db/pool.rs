//! SQLite pool creation and migrations.
//!
//! WAL + `synchronous = NORMAL` so a recording never stalls behind a database
//! write (mantra 3). Foreign keys are on, so deleting a meeting cascades to
//! everything derived from it.

use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;

use super::DbError;

/// The one pool shared by the whole app.
pub type Db = SqlitePool;

/// Embedded migrations from `src-tauri/migrations`. Applied transactionally.
pub static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Open (creating if needed) the database at `path` and apply migrations.
pub async fn connect(path: &Path) -> Result<Db, DbError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DbError::Open(e.to_string()))?;
    }

    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .foreign_keys(true)
        .busy_timeout(Duration::from_secs(10))
        // ~16 MB page cache; keeps transcript scrolling off the disk.
        .pragma("cache_size", "-16000")
        .pragma("temp_store", "MEMORY");

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .min_connections(1)
        .acquire_timeout(Duration::from_secs(15))
        .connect_with(options)
        .await?;

    migrate(&pool).await?;
    Ok(pool)
}

/// In-memory database with migrations applied. Tests only, a single
/// connection keeps the database alive for the pool's lifetime.
pub async fn connect_in_memory() -> Result<Db, DbError> {
    let options = SqliteConnectOptions::from_str("sqlite::memory:")?
        .foreign_keys(true)
        .busy_timeout(Duration::from_secs(5));

    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect_with(options)
        .await?;

    migrate(&pool).await?;
    Ok(pool)
}

/// Apply any migration the database has not seen yet.
pub async fn migrate(db: &Db) -> Result<(), DbError> {
    MIGRATOR.run(db).await?;
    Ok(())
}

/// Ask SQLite to fold the WAL back into the main file. Cheap enough to call on
/// quit and after a bulk delete.
pub async fn checkpoint(db: &Db) -> Result<(), DbError> {
    sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(db)
        .await?;
    Ok(())
}

/// Reclaim space after the person deletes meetings. Blocking-ish, so only run
/// it when nothing is recording.
pub async fn vacuum(db: &Db) -> Result<(), DbError> {
    sqlx::query("VACUUM").execute(db).await?;
    Ok(())
}

/// Size of the database file plus its WAL, for the storage report.
pub fn database_bytes(path: &Path) -> u64 {
    let mut total = crate::paths::file_size_bytes(path);
    for suffix in ["-wal", "-shm"] {
        let mut p = path.as_os_str().to_os_string();
        p.push(suffix);
        total += crate::paths::file_size_bytes(Path::new(&p));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn migrations_apply_to_a_fresh_in_memory_database() {
        let db = connect_in_memory().await.unwrap();
        let tables: Vec<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name",
        )
        .fetch_all(&db)
        .await
        .unwrap();
        let names: Vec<String> = tables.into_iter().map(|t| t.0).collect();
        for expected in [
            "action_items",
            "audio_chunks",
            "jobs",
            "markers",
            "meetings",
            "models",
            "segments",
            "segments_fts",
            "segments_fts_trigram",
            "settings",
            "speakers",
            "summaries",
            "templates",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "missing {expected} in {names:?}"
            );
        }
    }

    #[tokio::test]
    async fn migrations_are_idempotent() {
        let db = connect_in_memory().await.unwrap();
        migrate(&db).await.unwrap();
        migrate(&db).await.unwrap();
    }

    #[tokio::test]
    async fn six_builtin_templates_are_seeded() {
        let db = connect_in_memory().await.unwrap();
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM templates WHERE builtin = 1")
            .fetch_one(&db)
            .await
            .unwrap();
        assert_eq!(count, 6);
    }

    #[tokio::test]
    async fn foreign_keys_cascade_from_meetings() {
        let db = connect_in_memory().await.unwrap();
        sqlx::query("INSERT INTO meetings (id, started_at, audio_dir) VALUES ('m', '2026-01-01T00:00:00Z', '/tmp/m')")
            .execute(&db)
            .await
            .unwrap();
        sqlx::query("INSERT INTO segments (id, meeting_id, t_start_ms, t_end_ms, channel, text) VALUES ('s', 'm', 0, 10, 'mic', 'hello')")
            .execute(&db)
            .await
            .unwrap();
        sqlx::query("DELETE FROM meetings WHERE id = 'm'")
            .execute(&db)
            .await
            .unwrap();
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM segments")
            .fetch_one(&db)
            .await
            .unwrap();
        assert_eq!(count, 0, "segments should cascade with their meeting");
    }
}
