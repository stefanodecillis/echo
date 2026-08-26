-- Seconds Echo heard and deliberately did not write down.
--
-- This table exists because of the live verification of 2026-08-26. The live
-- bleed guard suppressed six stretches of the microphone that were the far side
-- coming back through the speakers. Suppression wrote nothing to the database,
-- on the reasoning that the catch-up pass would re-derive the same verdict from
-- the same audio. It re-derived five of the six. The sixth — "One risk is the
-- vendor contract, it has not been signed yet." — went onto the microphone
-- channel as the one duplicate that survived into the finished transcript.
--
-- The disagreement is structural, not a threshold: the live pass judges against
-- a ring aligned to the meeting clock and a delay this machine has already
-- measured, the offline pass reads paged audio off disk and always searches
-- cold (`src/asr/catchup_bleed.rs`). Two judgements of the same seconds, and
-- both had to agree.
--
-- So the decision is written down instead. The catch-up pass plans its work as
-- "audio on disk minus what has text against it"; it now subtracts these spans
-- as well, and never asks a second time about seconds already decided. That
-- also stops it re-reading them, which is the engine time live suppression was
-- always supposed to save and did not.
--
-- ## What a row claims
--
-- "Echo heard this and chose not to write it down twice." That is a different
-- claim from "Echo never heard it", and the difference is why the evidence is
-- on the row: `correlation` is how well the microphone's loudness shape matched
-- the computer's own at the best delay, `lag_ms` is that delay, and
-- `system_voice_ms` is how much of the stretch the far side was audibly
-- playing. Somebody reading this database a year from now can see why those
-- seconds have no words, and can disagree with the measurement rather than
-- guess at it. The audio itself is untouched on disk and can be judged again.
--
-- `reason` says which mechanism decided ('bleed' is the only one today).
-- `decided_by` says which pass decided ('live' or 'catchup'), because the two
-- disagreeing is precisely the thing this table was built out of and a support
-- question a year from now will start by asking which one wrote the row.
--
-- ## What clears it
--
-- "Listen again" (`db::repo::clear_transcript`). A person asking for a fresh
-- reading of the recording gets one: the marks go with the transcript, in the
-- same transaction, and the next pass judges the audio from scratch. Anything
-- else would make one meeting's suppression permanent while offering a button
-- that says otherwise.
--
-- Deleting the meeting takes them by cascade, like everything else derived from
-- the audio.
CREATE TABLE suppressed_spans (
    id              TEXT    PRIMARY KEY NOT NULL,
    meeting_id      TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    -- mic | system
    channel         TEXT    NOT NULL,
    t_start_ms      INTEGER NOT NULL,
    t_end_ms        INTEGER NOT NULL,
    -- bleed
    reason          TEXT    NOT NULL,
    -- live | catchup
    decided_by      TEXT    NOT NULL,
    correlation     REAL,
    lag_ms          INTEGER,
    system_voice_ms INTEGER,
    created_at      TEXT    NOT NULL
);

-- The one query that runs against this: the catch-up planner asking for one
-- meeting's marks on one channel, in clock order.
CREATE INDEX idx_suppressed_spans_meeting ON suppressed_spans (meeting_id, channel, t_start_ms);
