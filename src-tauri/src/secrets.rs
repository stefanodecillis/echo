//! Keychain access (macOS Keychain / Linux Secret Service).
//!
//! Fully implemented (not a stub). Non-negotiables from DESIGN §2:
//!
//! * **Never a plaintext fallback.** If the keychain cannot store a secret, the
//!   secret is not stored, Echo asks again next time.
//! * **Off the UI thread.** Secret Service can block for seconds while a dialog
//!   is up, so every call goes through `spawn_blocking`.
//! * **Three failure modes are distinct**, because the person has to do
//!   something different in each: [`SecretError::Absent`] (nothing saved yet),
//!   [`SecretError::Locked`] (keychain locked, unlock it),
//!   [`SecretError::Cancelled`] (the prompt was dismissed, try again).
//! * Recording never depends on any of this. A missing key only stops recaps
//!   from a remote service.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::types::{UiError, UiErrorAction, UiErrorKind};

/// Keychain service name. Same string on both platforms.
pub const SERVICE: &str = "app.echo.desktop";

/// Account names. One per secret; add here rather than inline.
pub mod accounts {
    /// Google AI Studio key for Gemini recaps.
    pub const GEMINI_API_KEY: &str = "gemini-api-key";
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SecretError {
    /// No secret has been saved under this account.
    #[error("nothing is saved for {0}")]
    Absent(String),
    /// The keychain exists but is locked.
    #[error("the keychain is locked")]
    Locked,
    /// The person dismissed the system prompt.
    #[error("the keychain prompt was dismissed")]
    Cancelled,
    /// No keychain on this system at all (headless Linux, no Secret Service).
    #[error("this computer has no place to keep secrets safely: {0}")]
    Unavailable(String),
    /// Anything else the backend reported.
    #[error("the keychain refused: {0}")]
    Backend(String),
}

impl From<SecretError> for UiError {
    fn from(err: SecretError) -> Self {
        match err {
            SecretError::Absent(_) => UiError::new(
                UiErrorKind::Credential,
                "Echo doesn't have that key saved yet. Add it in Settings and it will remember.",
            )
            .with_action(UiErrorAction::OpenSummarySettings),
            SecretError::Locked => UiError::new(
                UiErrorKind::Credential,
                "Your keychain is locked. Unlock it and Echo will try again.",
            )
            .with_action(UiErrorAction::Retry),
            SecretError::Cancelled => UiError::new(
                UiErrorKind::Credential,
                "Echo needs your permission to read the saved key.",
            )
            .with_action(UiErrorAction::Retry),
            SecretError::Unavailable(detail) => UiError::new(
                UiErrorKind::Credential,
                "This computer has no secure place to keep the key, so Echo won't store it. Local recaps still work.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::OpenSummarySettings),
            SecretError::Backend(detail) => UiError::new(
                UiErrorKind::Credential,
                "Echo couldn't reach your keychain. Try again in a moment.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::Retry),
        }
    }
}

// ---------------------------------------------------------------------------
// Public async API, always off the calling thread.
// ---------------------------------------------------------------------------

/// Read a secret. `Err(Absent)` when nothing is saved.
pub async fn get(account: &str) -> Result<String, SecretError> {
    let account = account.to_string();
    run(move || get_blocking(&account)).await
}

/// Store (or replace) a secret.
pub async fn set(account: &str, secret: &str) -> Result<(), SecretError> {
    if secret.is_empty() {
        return Err(SecretError::Backend(
            "refusing to store an empty secret".into(),
        ));
    }
    let account = account.to_string();
    let secret = secret.to_string();
    run(move || set_blocking(&account, &secret)).await
}

/// Remove a secret. Already-absent is success.
pub async fn delete(account: &str) -> Result<(), SecretError> {
    let account = account.to_string();
    run(move || match delete_blocking(&account) {
        Err(SecretError::Absent(_)) => Ok(()),
        other => other,
    })
    .await
}

/// Is a secret saved? Distinguishes "no" from "couldn't tell", so the UI can
/// show a masked value without ever reading the secret itself.
pub async fn has(account: &str) -> Result<bool, SecretError> {
    match get(account).await {
        Ok(_) => Ok(true),
        Err(SecretError::Absent(_)) => Ok(false),
        Err(other) => Err(other),
    }
}

async fn run<T, F>(f: F) -> Result<T, SecretError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, SecretError> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(join) => Err(SecretError::Backend(format!(
            "keychain task failed: {join}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

static USE_MEMORY: AtomicBool = AtomicBool::new(false);

fn memory_store() -> &'static Mutex<HashMap<String, String>> {
    static STORE: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Swap in an in-process store. Tests and CI only, there is no keychain on a
/// headless runner and we must never fall back to disk.
pub fn use_in_memory_store() {
    USE_MEMORY.store(true, Ordering::SeqCst);
    if let Ok(mut m) = memory_store().lock() {
        m.clear();
    }
}

fn get_blocking(account: &str) -> Result<String, SecretError> {
    if USE_MEMORY.load(Ordering::SeqCst) {
        let store = memory_store()
            .lock()
            .map_err(|_| SecretError::Backend("poisoned".into()))?;
        return store
            .get(account)
            .cloned()
            .ok_or_else(|| SecretError::Absent(account.to_string()));
    }
    let entry = keyring::Entry::new(SERVICE, account).map_err(|e| classify(e, account))?;
    entry.get_password().map_err(|e| classify(e, account))
}

fn set_blocking(account: &str, secret: &str) -> Result<(), SecretError> {
    if USE_MEMORY.load(Ordering::SeqCst) {
        let mut store = memory_store()
            .lock()
            .map_err(|_| SecretError::Backend("poisoned".into()))?;
        store.insert(account.to_string(), secret.to_string());
        return Ok(());
    }
    let entry = keyring::Entry::new(SERVICE, account).map_err(|e| classify(e, account))?;
    entry.set_password(secret).map_err(|e| classify(e, account))
}

fn delete_blocking(account: &str) -> Result<(), SecretError> {
    if USE_MEMORY.load(Ordering::SeqCst) {
        let mut store = memory_store()
            .lock()
            .map_err(|_| SecretError::Backend("poisoned".into()))?;
        return match store.remove(account) {
            Some(_) => Ok(()),
            None => Err(SecretError::Absent(account.to_string())),
        };
    }
    let entry = keyring::Entry::new(SERVICE, account).map_err(|e| classify(e, account))?;
    entry.delete_credential().map_err(|e| classify(e, account))
}

/// Map a backend error onto the taxonomy the UI reacts to.
///
/// The platforms do not agree on how they report a locked keychain or a
/// dismissed prompt, so we look at the message as well as the variant.
fn classify(err: keyring::Error, account: &str) -> SecretError {
    use keyring::Error as K;
    match err {
        K::NoEntry => SecretError::Absent(account.to_string()),
        K::NoStorageAccess(inner) => {
            let text = inner.to_string();
            classify_text(&text).unwrap_or(SecretError::Locked)
        }
        K::PlatformFailure(inner) => {
            let text = inner.to_string();
            classify_text(&text).unwrap_or(SecretError::Backend(text))
        }
        K::Ambiguous(_) => SecretError::Backend(
            "more than one saved key matched; remove the duplicates in your keychain".into(),
        ),
        other => SecretError::Backend(other.to_string()),
    }
}

fn classify_text(text: &str) -> Option<SecretError> {
    let lower = text.to_ascii_lowercase();
    // macOS: errSecUserCanceled (-128). Linux: "prompt dismissed"/"cancelled".
    if lower.contains("cancel") || lower.contains("dismiss") || lower.contains("-128") {
        return Some(SecretError::Cancelled);
    }
    if lower.contains("locked") {
        return Some(SecretError::Locked);
    }
    // No Secret Service on the bus at all.
    if lower.contains("was not provided by any .service")
        || lower.contains("servicename")
        || lower.contains("no such interface")
        || lower.contains("connection refused")
        || lower.contains("dbus")
    {
        return Some(SecretError::Unavailable(text.to_string()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() {
        use_in_memory_store();
    }

    #[tokio::test]
    async fn absent_is_distinct_from_a_failure() {
        fresh();
        let err = get("no-such-account").await.unwrap_err();
        assert!(matches!(err, SecretError::Absent(_)));
        assert!(!has("no-such-account").await.unwrap());
    }

    #[tokio::test]
    async fn set_get_delete_round_trip() {
        fresh();
        set(accounts::GEMINI_API_KEY, "sk-test-123").await.unwrap();
        assert!(has(accounts::GEMINI_API_KEY).await.unwrap());
        assert_eq!(get(accounts::GEMINI_API_KEY).await.unwrap(), "sk-test-123");

        set(accounts::GEMINI_API_KEY, "sk-test-456").await.unwrap();
        assert_eq!(get(accounts::GEMINI_API_KEY).await.unwrap(), "sk-test-456");

        delete(accounts::GEMINI_API_KEY).await.unwrap();
        assert!(!has(accounts::GEMINI_API_KEY).await.unwrap());
    }

    #[tokio::test]
    async fn deleting_something_absent_is_success() {
        fresh();
        delete("never-existed").await.unwrap();
    }

    #[tokio::test]
    async fn empty_secrets_are_refused() {
        fresh();
        assert!(set(accounts::GEMINI_API_KEY, "").await.is_err());
    }

    #[test]
    fn error_taxonomy_maps_to_distinct_user_messages() {
        let absent: UiError = SecretError::Absent("x".into()).into();
        let locked: UiError = SecretError::Locked.into();
        let cancelled: UiError = SecretError::Cancelled.into();
        for e in [&absent, &locked, &cancelled] {
            assert_eq!(e.kind, UiErrorKind::Credential);
        }
        assert_ne!(absent.message, locked.message);
        assert_ne!(locked.message, cancelled.message);
        // Zero jargon in what the person reads.
        for e in [&absent, &locked, &cancelled] {
            let lower = e.message.to_lowercase();
            for banned in ["keyring", "secret service", "dbus", "token", "api"] {
                assert!(
                    !lower.contains(banned),
                    "{:?} leaks jargon: {banned}",
                    e.message
                );
            }
        }
    }

    #[test]
    fn platform_messages_are_classified() {
        assert_eq!(
            classify_text("User canceled the operation (-128)"),
            Some(SecretError::Cancelled)
        );
        assert_eq!(
            classify_text("Prompt was dismissed"),
            Some(SecretError::Cancelled)
        );
        assert_eq!(
            classify_text("The collection is locked"),
            Some(SecretError::Locked)
        );
        assert!(matches!(
            classify_text("org.freedesktop.secrets was not provided by any .service files"),
            Some(SecretError::Unavailable(_))
        ));
        assert_eq!(classify_text("something else entirely"), None);
    }
}
