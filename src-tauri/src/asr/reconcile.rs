//! Making the weights on disk agree with the weights Echo wants.
//!
//! IMPLEMENTED-BY: asr agent, product decision of 2026-08-20.
//!
//! # Why this exists
//!
//! Echo used to ship a model and hope. When the model changed, a person who
//! already had the old one had two files, no way to know which was being used,
//! and no way to get rid of the one that was not. Worse, the naive fix — delete
//! the old, download the new — turns "Echo got better at understanding speech"
//! into "Echo cannot record for the next forty minutes", which is a promise
//! Echo does not get to break (mantra 3: nothing downstream may block capture).
//!
//! So the app reconciles instead. It looks at what it wants, looks at what is
//! there, and works out the one next thing to do. It runs at launch and every
//! time anything asks whether Echo can understand speech, and it is safe to run
//! at any moment, twice, or halfway.
//!
//! # The rule that matters
//!
//! **Nothing is ever deleted before its replacement is verified installed.**
//!
//! That single rule is what makes an upgrade in flight harmless. While large-v3
//! is downloading, turbo is still on disk and still serving: a meeting started
//! in the middle of the upgrade records, captions and transcribes on turbo, from
//! second zero, exactly as it did yesterday. Only once every wanted file is
//! present and checked does the engine's config move to the new weights and the
//! old files go. There is no window in which Echo has no model.
//!
//! The same rule covers the two speaker files, which changed for the first time
//! on 2026-08-20 when the voice-print network moved from CAM++ to WeSpeaker
//! ResNet34-LM. Nothing about that is a special case: yesterday's voice prints
//! stay on disk and stay usable until today's are verified, and then they go
//! ([`is_supersedable`]). The one asset with no replacement candidate in the
//! catalog — the speech detector — is excluded, because for it "supersede" could
//! only ever mean "delete the last copy".
//!
//! # The matrix
//!
//! Two questions, and the answer is the cell they meet in.
//!
//! ```text
//!                     │ no old weights        │ old weights on disk
//! ────────────────────┼───────────────────────┼──────────────────────────────
//!  every wanted file  │ Steady                │ Switch:
//!  is installed       │ nothing to do         │ point the engine at the new
//!                     │                       │ model, then delete the old
//! ────────────────────┼───────────────────────┼──────────────────────────────
//!  something wanted   │ FirstRun:             │ Upgrading:
//!  is missing         │ download; recording   │ download; keep recording on
//!                     │ waits (nothing to     │ the old model; delete nothing
//!                     │ record with)          │
//! ```
//!
//! [`Plan`] is that table, computed by [`plan`] from two lists of ids and
//! nothing else — no database, no disk, no clock. That is deliberate: the
//! interesting behaviour is the decision, so the decision is a pure function
//! with a test per cell. [`crate::asr::models::plan_reconcile`] gathers the two
//! lists, and [`crate::asr::models::reconcile`] carries the plan out.
//!
//! # Why the deleting is not optional
//!
//! Leaving the old weights alone forever would be tempting — they cost only disk
//! space, and disk space is cheap. It is not only disk space. On Apple silicon
//! the encoder companion is compiled for the machine on first load and cached by
//! the OS, and that cache is not big enough for two of these at once: measured
//! on 2026-08-20, loading the old model's encoder evicted the new one's, and the
//! next load spent **eighteen minutes** recompiling it. Two models left side by
//! side on one disk do not sit quietly; they take turns throwing each other out
//! of the cache. Removing the one nothing uses is what keeps the one everything
//! uses fast.
//!
//! # Crash safety
//!
//! Every state is re-derivable from the disk, so there is no progress to lose
//! and nothing to roll back:
//!
//! * **Half-downloaded upgrade.** The partial file and its sidecar are still
//!   there, the wanted asset still reads as missing, and the plan is `Upgrading`
//!   again — the download resumes from where it stopped
//!   ([`crate::asr::models::fetch_verified`] does the resuming).
//! * **Crash mid-cleanup.** Deletion is one obsolete asset at a time, and each
//!   one is only deleted while a wanted asset of the same kind is installed
//!   ([`Plan::deletable`]). So an interrupted cleanup leaves *redundant* files —
//!   never zero. The next reconcile finishes the job.
//! * **A file deleted behind Echo's back.** The installed set is read from the
//!   disk, not from the table, so the plan simply says to fetch it again.

use crate::types::AssetKind;

use super::catalog;

/// Which cell of the matrix we are in. Names what is happening to the person,
/// because that is what decides whether they are told anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// The wanted model is installed and nothing else is. The common case.
    Steady,
    /// Nothing usable is installed yet: a fresh install, or someone cleared the
    /// folder. Recording has to wait for the download — there is nothing to
    /// record *with*.
    FirstRun,
    /// The wanted model is still arriving, and an older one is carrying the
    /// weight until it does. Recording works, on the old model.
    Upgrading,
    /// Every wanted file is here and an older model still is too. This is the
    /// one transition a person is told about.
    Switch,
}

impl State {
    /// Is Echo mid-change, rather than settled? Used only for logging.
    pub fn is_settled(self) -> bool {
        matches!(self, State::Steady)
    }
}

/// What to do about it. Derived entirely from the two id lists handed to
/// [`plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub state: State,
    /// The level this was computed for.
    pub level_id: &'static str,
    /// Wanted assets that are not installed, in the order the catalog wants them
    /// fetched. Empty means the level is complete.
    pub missing: Vec<&'static str>,
    /// Installed assets the level does not want.
    ///
    /// Non-empty does **not** mean "delete these": see [`Plan::deletable`]. In
    /// `Upgrading` these are the files that are keeping Echo working.
    pub obsolete: Vec<&'static str>,
    /// The speech file the engine should load right now — the level's own, or
    /// the best installed older one while an upgrade is in flight.
    ///
    /// `None` means there is nothing to transcribe with at all.
    pub serving: Option<&'static str>,
}

impl Plan {
    /// The obsolete assets it is safe to delete *now*.
    ///
    /// Empty in every state but `Switch`. In `Switch`, [`obsolete`] has already
    /// been narrowed by [`plan`] twice: to kinds a model can supersede at all
    /// ([`is_supersedable`]), and to assets with a wanted, installed replacement
    /// **of their own kind** — a speech file for a speech file, a companion for
    /// a companion. This is what survived both.
    ///
    /// That narrowing is the invariant behind the crash-safety claim: whatever
    /// order these are deleted in, and wherever the deleting stops, a complete
    /// set of what Echo wants is already on the disk. The worst an interruption
    /// can do is leave a file that takes up room.
    ///
    /// [`obsolete`]: Plan::obsolete
    pub fn deletable(&self) -> Vec<&'static str> {
        if self.state != State::Switch {
            return Vec::new();
        }
        self.obsolete.clone()
    }

    /// Can a recording start? True whenever *something* can transcribe.
    ///
    /// Note what this does not consult: whether the download has finished. An
    /// upgrade in flight never blocks recording (mantra 3).
    pub fn can_serve(&self) -> bool {
        self.serving.is_some()
    }

    /// Is the engine about to be pointed at different weights? The one moment
    /// worth a word to the person.
    pub fn switching(&self) -> bool {
        self.state == State::Switch
    }

    /// Bytes still to fetch, for the progress the person watches.
    pub fn remaining_bytes(&self) -> i64 {
        self.missing
            .iter()
            .filter_map(|id| catalog::entry(id))
            .map(|e| e.bytes)
            .sum()
    }
}

/// Work out what to do, from what is wanted and what is there.
///
/// `installed` is every asset id verified present on disk — including ones the
/// level does not want, which is the whole point. Anything not catalogued is
/// ignored: Echo does not delete files it cannot account for.
pub fn plan(level_id: &str, installed: &[&str]) -> Plan {
    let level = catalog::preset_or_default(level_id);
    let is_installed = |id: &str| installed.contains(&id);

    let wanted = catalog::preset_asset_ids(level.id);
    let missing: Vec<&'static str> = wanted
        .iter()
        .copied()
        .filter(|id| !is_installed(id))
        .collect();
    let obsolete: Vec<&'static str> = catalog::obsolete_asset_ids(level.id)
        .into_iter()
        .filter(|id| is_installed(id))
        .collect();

    // What can transcribe right now. The wanted model if it is here, otherwise
    // the best installed one — catalog order is best-first, so "first installed"
    // is "best installed".
    let serving = catalog::speech_ids_best_first()
        .into_iter()
        .find(|id| is_installed(id));

    let state = match (missing.is_empty(), obsolete.is_empty()) {
        (true, true) => State::Steady,
        (true, false) => State::Switch,
        (false, _) if serving.is_some() => State::Upgrading,
        (false, _) => State::FirstRun,
    };

    // The invariant, asserted rather than assumed. `Switch` means every wanted
    // asset is installed, so every obsolete asset has a same-kind replacement
    // on disk by construction — but "by construction" is how a later edit
    // introduces a dead end, so the code checks instead of trusting.
    let obsolete = if state == State::Switch {
        obsolete
            .into_iter()
            .filter(|id| catalog::entry(id).is_some_and(|e| is_supersedable(e.kind)))
            .filter(|id| replacement_installed(id, &wanted, &is_installed))
            .collect()
    } else {
        obsolete
    };
    // Filtering may have emptied it, which would make this Steady after all.
    let state = if state == State::Switch && obsolete.is_empty() {
        State::Steady
    } else {
        state
    };

    Plan {
        state,
        level_id: level.id,
        missing,
        obsolete,
        serving,
    }
}

/// Is there a wanted, installed asset of the same kind as `victim`?
fn replacement_installed(
    victim: &str,
    wanted: &[&'static str],
    is_installed: &dyn Fn(&str) -> bool,
) -> bool {
    let Some(kind) = catalog::entry(victim).map(|e| e.kind) else {
        return false;
    };
    wanted
        .iter()
        .filter_map(|id| catalog::entry(id))
        .any(|e| e.kind == kind && is_installed(e.id))
}

/// The only kinds a newer model can supersede.
///
/// A second guard on the one destructive path in this module, and deliberately
/// redundant: the obsolete set is derived from the level, and the level wants one
/// asset of each of these kinds, so nothing the level still needs can land in it
/// anyway. But "can never" is a claim about code somebody will edit later, and
/// the cost of being wrong is deleting the file that lets Echo hear anybody. So
/// the deletion also checks, from the other direction, that what it is about to
/// remove is a kind of thing a newer model replaces.
///
/// The two speaker files are on the list as of 2026-08-20, when the voice-print
/// network changed for the first time. A change of voice-print or segmentation
/// network is the same event as a change of speech model — new bytes wanted, old
/// bytes on thousands of disks with nothing able to account for them — so it gets
/// the same treatment and the same guarantee: nothing is deleted until a wanted
/// asset **of its own kind** is verified installed ([`replacement_installed`]).
///
/// The speech detector is deliberately *not* here. It is the one file with no
/// replacement candidate in the catalog at all, so listing it could only ever
/// authorise deleting the last copy of it.
pub fn is_supersedable(kind: AssetKind) -> bool {
    matches!(
        kind,
        AssetKind::Speech
            | AssetKind::SpeechAccelerator
            | AssetKind::SpeakerSegmenter
            | AssetKind::SpeakerEmbedder
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::{ids, CatalogEntry};

    /// Every asset the level wants on this platform.
    fn all_wanted() -> Vec<&'static str> {
        catalog::preset_asset_ids(catalog::DEFAULT_PRESET_ID)
    }

    /// The shared files, which are installed in nearly every real situation.
    fn shared() -> Vec<&'static str> {
        catalog::SHARED_ASSET_IDS.to_vec()
    }

    fn plan_for(installed: &[&str]) -> Plan {
        plan(catalog::DEFAULT_PRESET_ID, installed)
    }

    // -----------------------------------------------------------------
    // The four cells
    // -----------------------------------------------------------------

    #[test]
    fn nothing_installed_is_a_first_run_that_downloads_everything() {
        let p = plan_for(&[]);
        assert_eq!(p.state, State::FirstRun);
        assert_eq!(p.missing, all_wanted());
        assert!(p.obsolete.is_empty());
        assert_eq!(p.serving, None);
        assert!(!p.can_serve(), "there is nothing to record with");
        assert!(p.deletable().is_empty());
        assert!(!p.switching());
        assert_eq!(p.remaining_bytes(), catalog::preset_total_bytes(p.level_id));
    }

    #[test]
    fn everything_wanted_and_nothing_else_is_steady_and_silent() {
        let installed = all_wanted();
        let p = plan_for(&installed);
        assert_eq!(p.state, State::Steady);
        assert!(p.missing.is_empty());
        assert!(p.obsolete.is_empty());
        assert_eq!(p.serving, Some(ids::SPEECH));
        assert!(p.can_serve());
        assert!(!p.switching(), "a settled install says nothing to anybody");
        assert!(p.deletable().is_empty());
        assert_eq!(p.remaining_bytes(), 0);
        assert!(p.state.is_settled());
    }

    /// Yesterday's install, first launch of the new build: the exact situation
    /// this whole module was written for.
    #[test]
    fn yesterdays_model_alone_keeps_serving_while_the_new_one_downloads() {
        let mut installed = shared();
        installed.push(ids::SPEECH_TURBO);
        if cfg!(target_os = "macos") {
            installed.push(ids::ACCEL_TURBO);
        }
        let p = plan_for(&installed);

        assert_eq!(p.state, State::Upgrading);
        assert!(
            p.missing.contains(&ids::SPEECH),
            "the new weights are what is missing"
        );
        assert!(p.obsolete.contains(&ids::SPEECH_TURBO));
        assert_eq!(
            p.serving,
            Some(ids::SPEECH_TURBO),
            "a meeting started now records on the model that is actually here"
        );
        assert!(p.can_serve(), "an upgrade in flight never blocks recording");
        assert!(
            p.deletable().is_empty(),
            "deleting turbo now would leave the person with nothing"
        );
        assert!(!p.switching(), "nothing has switched yet, so nothing is said");
    }

    #[test]
    fn once_the_new_model_is_verified_the_old_one_goes_and_the_person_is_told() {
        let mut installed = all_wanted();
        installed.push(ids::SPEECH_TURBO);
        if cfg!(target_os = "macos") {
            installed.push(ids::ACCEL_TURBO);
        }
        let p = plan_for(&installed);

        assert_eq!(p.state, State::Switch);
        assert!(p.missing.is_empty());
        assert_eq!(
            p.serving,
            Some(ids::SPEECH),
            "the engine moves to the new weights in the same step"
        );
        assert!(p.switching(), "this is the one moment worth a notice");

        let doomed = p.deletable();
        assert!(doomed.contains(&ids::SPEECH_TURBO));
        if cfg!(target_os = "macos") {
            assert!(
                doomed.contains(&ids::ACCEL_TURBO),
                "the old companion goes with the old weights"
            );
        }
        for id in &doomed {
            assert!(
                !all_wanted().contains(id),
                "{id} is wanted and must not be deleted"
            );
        }
    }

    // -----------------------------------------------------------------
    // The no-dead-ends invariant
    // -----------------------------------------------------------------

    /// The heart of it: for every asset the plan would delete, a wanted
    /// replacement of the same kind is already on disk. Stopping the deletions
    /// at any point therefore leaves a complete set plus leftovers.
    #[test]
    fn a_deletion_always_has_its_replacement_already_in_place() {
        let mut installed = all_wanted();
        installed.extend(catalog::OBSOLETE_ASSET_IDS.iter().copied());
        installed.retain(|id| catalog::entry(id).is_some_and(CatalogEntry::applies_here));
        let p = plan_for(&installed);
        assert_eq!(p.state, State::Switch);

        for victim in p.deletable() {
            let kind = catalog::entry(victim).unwrap().kind;
            assert!(
                is_supersedable(kind),
                "{victim} is not the kind of thing a new model replaces"
            );
            let replacement = all_wanted()
                .into_iter()
                .filter_map(catalog::entry)
                .find(|e| e.kind == kind);
            let replacement = replacement.unwrap_or_else(|| {
                panic!("{victim} would be deleted with nothing of its kind wanted")
            });
            assert!(
                installed.contains(&replacement.id),
                "{victim} would be deleted before {} arrived",
                replacement.id
            );
        }
    }

    /// A cleanup that stopped halfway is not a special case: run the plan again
    /// and it picks up the rest.
    #[test]
    fn a_cleanup_interrupted_halfway_finishes_on_the_next_run() {
        let mut installed = all_wanted();
        installed.push(ids::SPEECH_TURBO);
        installed.push(ids::SPEECH_TINY);

        let first = plan_for(&installed);
        assert_eq!(first.state, State::Switch);
        let doomed = first.deletable();
        assert!(doomed.len() >= 2);

        // Pretend we deleted exactly one of them and then the power went out.
        installed.retain(|id| *id != doomed[0]);
        let second = plan_for(&installed);
        assert_eq!(second.state, State::Switch, "still work to do");
        assert!(
            second.deletable().contains(&doomed[1]),
            "the one we did not get to is still queued for deletion"
        );
        assert!(
            second.can_serve(),
            "and there was never a moment with nothing to transcribe with"
        );

        // Finish the job.
        installed.retain(|id| !doomed.contains(id));
        let third = plan_for(&installed);
        assert_eq!(third.state, State::Steady);
        assert!(third.deletable().is_empty());
    }

    #[test]
    fn the_plan_is_the_same_however_many_times_it_is_asked() {
        for installed in [
            vec![],
            all_wanted(),
            {
                let mut v = shared();
                v.push(ids::SPEECH_TURBO);
                v
            },
            {
                let mut v = all_wanted();
                v.push(ids::SPEECH_TINY);
                v
            },
        ] {
            let once = plan_for(&installed);
            let twice = plan_for(&installed);
            assert_eq!(once, twice, "reconciling is a question, not a change");
        }
    }

    // -----------------------------------------------------------------
    // Awkward corners
    // -----------------------------------------------------------------

    /// Three generations of weights on one disk. The best available serves, and
    /// everything older than what is wanted is swept up together.
    #[test]
    fn several_old_models_are_all_recognised_and_the_best_one_serves() {
        let mut installed = shared();
        installed.push(ids::SPEECH_TINY);
        installed.push(ids::SPEECH_SMALL);
        let p = plan_for(&installed);
        assert_eq!(p.state, State::Upgrading);
        assert_eq!(
            p.serving,
            Some(ids::SPEECH_SMALL),
            "small beats tiny, so small is what a meeting gets"
        );
        assert_eq!(p.obsolete.len(), 2);
        assert!(p.deletable().is_empty(), "nothing wanted has arrived yet");
    }

    /// The speech file arrived; the companion has not. The engine can and should
    /// switch — a missing companion costs speed, never correctness — but until
    /// every wanted file is here we do not start deleting, because the old
    /// companion is the only accelerated thing on the disk.
    #[test]
    fn the_new_weights_without_their_companion_do_not_trigger_a_cleanup_yet() {
        if !cfg!(target_os = "macos") {
            return;
        }
        let mut installed = shared();
        installed.push(ids::SPEECH);
        installed.push(ids::SPEECH_TURBO);
        installed.push(ids::ACCEL_TURBO);
        let p = plan_for(&installed);

        assert_eq!(p.state, State::Upgrading);
        assert_eq!(p.missing, vec![ids::ACCEL]);
        assert_eq!(
            p.serving,
            Some(ids::SPEECH),
            "the best installed weights serve, companion or not"
        );
        assert!(
            p.deletable().is_empty(),
            "the switch waits for the whole set"
        );
    }

    /// Only the shared files are here. There is nothing to transcribe with, but
    /// there is also nothing old to keep — a fresh install that got partway.
    #[test]
    fn a_download_that_stopped_after_the_small_files_is_still_a_first_run() {
        let p = plan_for(&shared());
        assert_eq!(p.state, State::FirstRun);
        assert!(!p.can_serve());
        assert!(p.missing.contains(&ids::SPEECH));
        assert!(p.obsolete.is_empty());
        assert!(p.remaining_bytes() > 3_000_000_000);
    }

    /// The shared files are never obsolete, whatever else is going on. Sweeping
    /// up the speech detector would stop Echo hearing anybody.
    #[test]
    fn the_shared_files_are_never_swept_up() {
        let mut installed = all_wanted();
        installed.extend(catalog::OBSOLETE_ASSET_IDS.iter().copied());
        let p = plan_for(&installed);
        for id in catalog::SHARED_ASSET_IDS {
            assert!(!p.obsolete.contains(id), "{id} must never be obsolete");
            assert!(!p.deletable().contains(id));
        }
    }

    /// Files Echo did not put there are not Echo's to remove. Nothing here
    /// names them, which is the point — an id the catalog does not know cannot
    /// reach `deletable`.
    #[test]
    fn an_unrecognised_file_is_left_completely_alone() {
        let mut installed = all_wanted();
        installed.push("someones-own-experiment");
        let p = plan_for(&installed);
        assert_eq!(p.state, State::Steady);
        assert!(p.obsolete.is_empty());
        assert!(p.deletable().is_empty());
    }

    /// Somebody deleted the old weights but not the old companion — a gigabyte
    /// of encoder with nothing to encode for. There is nothing to serve, so this
    /// is a first run; the orphan is recognised but not touched until the new
    /// set is complete, because until then there is no rule that says it is
    /// safe to remove.
    #[test]
    fn an_old_companion_left_without_its_weights_waits_its_turn() {
        if !cfg!(target_os = "macos") {
            return;
        }
        let mut installed = shared();
        installed.push(ids::ACCEL_TURBO);
        let p = plan_for(&installed);

        assert_eq!(p.state, State::FirstRun);
        assert_eq!(p.serving, None, "an encoder alone cannot transcribe");
        assert!(p.obsolete.contains(&ids::ACCEL_TURBO), "but it is recognised");
        assert!(
            p.deletable().is_empty(),
            "and left alone until the new set is complete"
        );

        // Once everything wanted is here, it goes.
        let mut installed = all_wanted();
        installed.push(ids::ACCEL_TURBO);
        let after = plan_for(&installed);
        assert_eq!(after.state, State::Switch);
        assert_eq!(after.deletable(), vec![ids::ACCEL_TURBO]);
    }

    /// A settings row naming a level that no longer exists (a downgrade, or a
    /// hand-edited database) must not make the plan give up.
    #[test]
    fn an_unknown_level_falls_back_to_the_one_that_exists() {
        let p = plan("a-level-from-the-future", &all_wanted());
        assert_eq!(p.level_id, catalog::DEFAULT_PRESET_ID);
        assert_eq!(p.state, State::Steady);
    }

    #[test]
    fn a_companion_follows_whatever_is_serving_not_whatever_is_wanted() {
        if !cfg!(target_os = "macos") {
            return;
        }
        let mut installed = shared();
        installed.push(ids::SPEECH_TURBO);
        installed.push(ids::ACCEL_TURBO);
        let p = plan_for(&installed);
        let serving = p.serving.unwrap();
        assert_eq!(
            catalog::accelerator_for(serving).map(|e| e.id),
            Some(ids::ACCEL_TURBO),
            "the old weights load the old companion, not the new one's"
        );
    }

    /// The speech detector is the one file with no replacement candidate in the
    /// catalog, so nothing may ever authorise deleting it.
    #[test]
    fn everything_a_newer_model_replaces_can_be_superseded_except_the_detector() {
        assert!(is_supersedable(AssetKind::Speech));
        assert!(is_supersedable(AssetKind::SpeechAccelerator));
        assert!(is_supersedable(AssetKind::SpeakerSegmenter));
        assert!(is_supersedable(AssetKind::SpeakerEmbedder));
        assert!(!is_supersedable(AssetKind::SpeechDetector));
    }

    // -----------------------------------------------------------------
    // A change of voice-print network
    // -----------------------------------------------------------------

    /// The 2026-08-20 swap, walked through cell by cell. This is the same
    /// machinery the speech model uses, and the point of the test is that it is
    /// the same: no new path, no special case, and above all no window in which
    /// Echo has no way to tell voices apart.
    #[test]
    fn yesterdays_voice_prints_keep_working_until_todays_are_verified() {
        // Yesterday's install: everything current except the voice prints, which
        // are still CAM++.
        let mut installed = all_wanted();
        installed.retain(|id| *id != ids::EMBEDDER);
        installed.push(ids::EMBEDDER_CAMPLUS);

        let mid = plan_for(&installed);
        assert_eq!(mid.state, State::Upgrading);
        assert_eq!(mid.missing, vec![ids::EMBEDDER]);
        assert!(mid.obsolete.contains(&ids::EMBEDDER_CAMPLUS));
        assert!(
            mid.deletable().is_empty(),
            "deleting CAM++ now would leave the pass with no way to tell voices apart"
        );
        assert!(
            mid.can_serve(),
            "and the speech model is untouched, so meetings still transcribe"
        );

        // The 26 MB arrives.
        installed.push(ids::EMBEDDER);
        let after = plan_for(&installed);
        assert_eq!(after.state, State::Switch);
        assert_eq!(after.deletable(), vec![ids::EMBEDDER_CAMPLUS]);

        // And once it is gone there is nothing left to do.
        installed.retain(|id| *id != ids::EMBEDDER_CAMPLUS);
        assert_eq!(plan_for(&installed).state, State::Steady);
    }

    /// Both models changing at once — a person who skipped a release. Neither
    /// cleanup waits on the other's kind, and neither happens early.
    #[test]
    fn a_speech_change_and_a_voice_print_change_do_not_block_each_other() {
        let mut installed = shared();
        installed.retain(|id| *id != ids::EMBEDDER);
        installed.push(ids::EMBEDDER_CAMPLUS);
        installed.push(ids::SPEECH_TURBO);
        if cfg!(target_os = "macos") {
            installed.push(ids::ACCEL_TURBO);
        }

        let p = plan_for(&installed);
        assert_eq!(p.state, State::Upgrading);
        assert_eq!(p.serving, Some(ids::SPEECH_TURBO));
        assert!(p.deletable().is_empty());

        // Everything wanted arrives; both generations of leftovers go together.
        let mut installed = all_wanted();
        installed.push(ids::EMBEDDER_CAMPLUS);
        installed.push(ids::SPEECH_TURBO);
        let doomed = plan_for(&installed).deletable();
        assert!(doomed.contains(&ids::EMBEDDER_CAMPLUS));
        assert!(doomed.contains(&ids::SPEECH_TURBO));
        for id in &doomed {
            assert!(!all_wanted().contains(id), "{id} is wanted");
        }
    }
}
