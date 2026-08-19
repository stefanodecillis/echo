//! Storage. Owned and fully implemented by the scaffold. Other modules call
//! [`repo`] functions rather than writing SQL of their own.
//!
//! * [`pool`]: connection pool, PRAGMAs, migrations.
//! * [`repo`]: one function per operation, over every table in DESIGN §3.
//!
//! Writes are batched by callers (`repo::insert_segments`) and must never sit
//! on the audio path (mantra 3).

pub mod pool;
pub mod repo;

pub use pool::{connect, connect_in_memory, migrate, Db};

use crate::types::{UiError, UiErrorKind};

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("database error: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("could not apply a database update: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("could not open the database: {0}")]
    Open(String),
    #[error("stored value could not be read back: {0}")]
    Decode(String),
    #[error("{0} does not exist")]
    NotFound(String),
    /// The request itself does not make sense — joining two people who were in
    /// different meetings, for instance. Not a fault, and not something a retry
    /// would fix.
    #[error("{0}")]
    Invalid(String),
}

impl From<DbError> for UiError {
    fn from(err: DbError) -> Self {
        match err {
            DbError::NotFound(what) => UiError::new(
                UiErrorKind::NotFound,
                "Echo couldn't find that. It may have been deleted.",
            )
            .with_detail(what),
            DbError::Invalid(what) => {
                UiError::invalid("Echo can't do that with these two.").with_detail(what)
            }
            other => {
                let detail = other.to_string();
                // Disk-full surfaces as a plain SQLite error; tell the person
                // something they can act on.
                if detail.contains("disk is full") || detail.contains("database or disk is full") {
                    UiError::new(
                        UiErrorKind::Storage,
                        "This computer is out of space. Free some up and Echo will pick up where it left off.",
                    )
                    .with_detail(detail)
                } else {
                    UiError::unexpected(detail)
                }
            }
        }
    }
}
