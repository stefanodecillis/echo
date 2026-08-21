-- Known people: the voices the person chose to have Echo remember.
--
-- DESIGN §1 "Known people (voice enrollment)". A person is a name, a centroid
-- and a capped, condition-diverse set of samples — and the samples keep their
-- audio as well as their fingerprint, which is the whole reason this is four
-- tables rather than one column.
--
-- ## Why the clips are stored
--
-- A fingerprint only means something inside the network that produced it. The
-- day Echo ships a better voice-print network, every centroid computed with the
-- old one describes a geometry the new one has never been in — the exact failure
-- `diarize::cluster::DISTANCE_THRESHOLD` documents at length. A profile that
-- kept only its numbers would have to be thrown away and re-enrolled by hand.
--
-- So each sample keeps its ~6 s of audio. `person_profiles.embedder_asset_id`
-- records which network the numbers belong to; matching silently skips a profile
-- whose tag disagrees with the network in use, and a refresh re-embeds the clips
-- and puts the profile back in service. Nobody has to be asked to say their name
-- into a microphone twice.
--
-- ## What is deliberately *not* here
--
-- No "confidence", no "is_enrolled" flag on speakers, no history of matches. A
-- link is a link; a suggestion is a suggestion with the score that produced it,
-- so the row can be re-decided when thresholds move. Enrollment is per-person
-- opt-in and there is nothing to record for the people who were never enrolled.

-- ---------------------------------------------------------------------------
-- people — the name, and nothing else
--
-- No uniqueness on the name. Two colleagues really are both called Marco, and a
-- constraint here would turn "remember this voice" into an error message about a
-- name the person cannot see the other of. The voice is the identity; the name is
-- a label on it.
-- ---------------------------------------------------------------------------
CREATE TABLE people (
    id         TEXT PRIMARY KEY NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_people_name ON people (name);

-- ---------------------------------------------------------------------------
-- person_samples — a few seconds of this voice, kept twice
--
-- `embedding` is f32 little-endian, exactly as `diarize::embedding` produced it
-- (unit length). `clip` is a 16 kHz mono WAV of about six seconds — the same
-- file `diarize::sample` hands the UI to play. `condition` is which recording it
-- came off ('mic' or 'system'), because a voice arriving down a call sounds
-- different from the same voice in the room and a profile wants both.
--
-- `source_meeting_id` is provenance only, and it is ON DELETE SET NULL on
-- purpose: deleting a meeting must never quietly degrade a profile. The audio
-- Echo needs is the copy in this row.
-- ---------------------------------------------------------------------------
CREATE TABLE person_samples (
    id                TEXT    PRIMARY KEY NOT NULL,
    person_id         TEXT    NOT NULL REFERENCES people (id) ON DELETE CASCADE,
    -- f32 little-endian, unit length
    embedding         BLOB    NOT NULL,
    -- 16 kHz mono WAV, ~6 s
    clip              BLOB    NOT NULL,
    -- mic | system
    condition         TEXT    NOT NULL,
    source_meeting_id TEXT    REFERENCES meetings (id) ON DELETE SET NULL,
    t_start_ms        INTEGER NOT NULL,
    t_end_ms          INTEGER NOT NULL,
    created_at        TEXT    NOT NULL
);

CREATE INDEX idx_person_samples_person ON person_samples (person_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- person_profiles — the one vector matching actually compares against
--
-- Derived entirely from `person_samples` (mean of the embeddings, back on the
-- unit sphere), so it can always be rebuilt and never has to be migrated. One
-- row per person, created the moment the first sample lands.
--
-- `embedder_asset_id` is the `models.id` of the network the centroid was
-- computed with. It is the only thing standing between a profile and being
-- compared in a space it was never measured in.
-- ---------------------------------------------------------------------------
CREATE TABLE person_profiles (
    person_id         TEXT    PRIMARY KEY NOT NULL REFERENCES people (id) ON DELETE CASCADE,
    centroid          BLOB    NOT NULL,
    sample_count      INTEGER NOT NULL DEFAULT 0,
    embedder_asset_id TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- speakers gains four columns
--
-- `person_id`     — this voice is that known person. ON DELETE SET NULL:
--                   deleting a person releases the link and leaves the speaker
--                   row alone. The display name was *copied* onto the row when
--                   the link was made, so the transcript keeps saying "Marco"
--                   as plain text; deleting the profile deletes the voice data,
--                   not the meeting's words. That asymmetry is deliberate — a
--                   meeting that has already been read should not silently
--                   rewrite itself.
-- `suggested_person_id`, `suggestion_score`
--                 — "Looks like Marco — confirm?". Never an assignment: the
--                   score is kept so the row can be re-decided if the
--                   thresholds in `diarize::people` ever move.
-- `centroid`      — this speaker's own voice print for this meeting, f32
--                   little-endian, written by the offline pass. It is what makes
--                   "this unnamed voice has now turned up in four meetings"
--                   answerable without re-reading a single second of audio, and
--                   what lets a later enrollment be matched against meetings
--                   that were separated before the person existed.
--
-- Plain ADD COLUMNs with no defaults: every existing speaker row gets NULL,
-- which is exactly "nobody knows who this is" — the behaviour those rows
-- already had.
-- ---------------------------------------------------------------------------
ALTER TABLE speakers ADD COLUMN person_id TEXT REFERENCES people (id) ON DELETE SET NULL;
ALTER TABLE speakers ADD COLUMN suggested_person_id TEXT REFERENCES people (id) ON DELETE SET NULL;
ALTER TABLE speakers ADD COLUMN suggestion_score REAL;
ALTER TABLE speakers ADD COLUMN centroid BLOB;

CREATE INDEX idx_speakers_person ON speakers (person_id);
CREATE INDEX idx_speakers_suggested ON speakers (suggested_person_id);
