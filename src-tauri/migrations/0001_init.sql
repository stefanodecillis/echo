-- Echo initial schema. Mirrors docs/DESIGN.md §3 "Data model" exactly.
--
-- Conventions:
--   * ids are UUID v4 text
--   * wall-clock times are RFC3339 UTC text
--   * in-meeting offsets are integer milliseconds on the monotonic meeting clock
--   * enums are stored as snake_case text (see src/types.rs `as_str`/`parse`)
--
-- Raw per-channel audio on disk is the source of truth (mantra 3); every table
-- here is derivable from it, so ON DELETE CASCADE is safe throughout.

-- ---------------------------------------------------------------------------
-- meetings
-- ---------------------------------------------------------------------------
CREATE TABLE meetings (
    id           TEXT    PRIMARY KEY NOT NULL,
    title        TEXT    NOT NULL DEFAULT '',
    started_at   TEXT    NOT NULL,
    ended_at     TEXT,
    detected_app TEXT,
    language     TEXT,
    -- created | recording | processing | complete | interrupted | failed
    status       TEXT    NOT NULL DEFAULT 'created',
    audio_dir    TEXT    NOT NULL,
    mixed_path   TEXT,
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    deleted_at   TEXT
);

CREATE INDEX idx_meetings_started_at ON meetings (started_at DESC);
CREATE INDEX idx_meetings_status ON meetings (status);
CREATE INDEX idx_meetings_live ON meetings (deleted_at, started_at DESC);

-- ---------------------------------------------------------------------------
-- audio_chunks — the crash-recovery journal. A chunk counts only once
-- committed = 1, i.e. flushed and fsynced.
-- ---------------------------------------------------------------------------
CREATE TABLE audio_chunks (
    id         TEXT    PRIMARY KEY NOT NULL,
    meeting_id TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    -- mic | system | mixed
    channel    TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    path       TEXT    NOT NULL,
    t_start_ms INTEGER NOT NULL,
    t_end_ms   INTEGER NOT NULL,
    committed  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (meeting_id, channel, seq)
);

CREATE INDEX idx_audio_chunks_meeting ON audio_chunks (meeting_id, channel, seq);
CREATE INDEX idx_audio_chunks_committed ON audio_chunks (meeting_id, committed, t_end_ms);

-- ---------------------------------------------------------------------------
-- speakers — merge is an alias, never a delete (non-destructive).
-- ---------------------------------------------------------------------------
CREATE TABLE speakers (
    id           TEXT    PRIMARY KEY NOT NULL,
    meeting_id   TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    cluster_key  TEXT    NOT NULL,
    display_name TEXT    NOT NULL,
    alias_of     TEXT    REFERENCES speakers (id) ON DELETE SET NULL,
    is_self      INTEGER NOT NULL DEFAULT 0,
    speaking_ms  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (meeting_id, cluster_key)
);

CREATE INDEX idx_speakers_meeting ON speakers (meeting_id);
CREATE INDEX idx_speakers_alias ON speakers (alias_of);

-- ---------------------------------------------------------------------------
-- segments — revision grows as better passes land
-- (live partial → final → diarization-refined).
-- ---------------------------------------------------------------------------
CREATE TABLE segments (
    id             TEXT    PRIMARY KEY NOT NULL,
    meeting_id     TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    t_start_ms     INTEGER NOT NULL,
    t_end_ms       INTEGER NOT NULL,
    channel        TEXT    NOT NULL,
    speaker_id     TEXT    REFERENCES speakers (id) ON DELETE SET NULL,
    text           TEXT    NOT NULL,
    language       TEXT,
    avg_confidence REAL,
    revision       INTEGER NOT NULL DEFAULT 1,
    is_final       INTEGER NOT NULL DEFAULT 0,
    -- provenance: which engine + revision produced this text
    model_name     TEXT,
    model_revision TEXT
);

CREATE INDEX idx_segments_meeting_time ON segments (meeting_id, t_start_ms);
CREATE INDEX idx_segments_speaker ON segments (speaker_id);
CREATE INDEX idx_segments_final ON segments (meeting_id, is_final, t_start_ms);

-- ---------------------------------------------------------------------------
-- markers — live "flag action item"
-- ---------------------------------------------------------------------------
CREATE TABLE markers (
    id         TEXT    PRIMARY KEY NOT NULL,
    meeting_id TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    t_ms       INTEGER NOT NULL,
    -- action_item | highlight | system
    kind       TEXT    NOT NULL DEFAULT 'action_item',
    note       TEXT
);

CREATE INDEX idx_markers_meeting ON markers (meeting_id, t_ms);

-- ---------------------------------------------------------------------------
-- templates — 6 built-ins + custom
-- ---------------------------------------------------------------------------
CREATE TABLE templates (
    id        TEXT    PRIMARY KEY NOT NULL,
    name      TEXT    NOT NULL,
    prompt_md TEXT    NOT NULL,
    builtin   INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX idx_templates_name ON templates (name);

-- ---------------------------------------------------------------------------
-- summaries — template_snapshot + provider/model/revision keep an old recap
-- explainable after templates or engines change.
-- ---------------------------------------------------------------------------
CREATE TABLE summaries (
    id                  TEXT    PRIMARY KEY NOT NULL,
    meeting_id          TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    template_id         TEXT    REFERENCES templates (id) ON DELETE SET NULL,
    template_snapshot   TEXT,
    -- onThisComputer | gemini
    provider            TEXT    NOT NULL,
    model               TEXT,
    language            TEXT,
    transcript_revision INTEGER NOT NULL DEFAULT 1,
    content_md          TEXT    NOT NULL,
    created_at          TEXT    NOT NULL
);

CREATE INDEX idx_summaries_meeting ON summaries (meeting_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- action_items
-- ---------------------------------------------------------------------------
CREATE TABLE action_items (
    id           TEXT    PRIMARY KEY NOT NULL,
    meeting_id   TEXT    NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    summary_id   TEXT    REFERENCES summaries (id) ON DELETE CASCADE,
    description  TEXT    NOT NULL,
    owner        TEXT,
    due_hint     TEXT,
    done         INTEGER NOT NULL DEFAULT 0,
    external_url TEXT
);

CREATE INDEX idx_action_items_meeting ON action_items (meeting_id, done);
CREATE INDEX idx_action_items_summary ON action_items (summary_id);

-- ---------------------------------------------------------------------------
-- jobs — persisted so background work survives a restart. Recording preempts
-- everything, so a running job may go back to 'paused'.
-- ---------------------------------------------------------------------------
CREATE TABLE jobs (
    id         TEXT    PRIMARY KEY NOT NULL,
    meeting_id TEXT    REFERENCES meetings (id) ON DELETE CASCADE,
    -- transcribe_catchup | diarize | summarize | export | download | mixdown
    kind       TEXT    NOT NULL,
    -- queued | running | paused | done | failed | cancelled
    status     TEXT    NOT NULL DEFAULT 'queued',
    progress   REAL,
    error      TEXT,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE INDEX idx_jobs_status ON jobs (status, created_at);
CREATE INDEX idx_jobs_meeting ON jobs (meeting_id, kind);

-- ---------------------------------------------------------------------------
-- models — the signed catalog of everything Echo downloads, including the
-- Apple encoder companions. url/sha256/license/revision give provenance.
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    id        TEXT    PRIMARY KEY NOT NULL,
    -- speech | speech_accelerator | speech_detector
    --       | speaker_segmenter | speaker_embedder
    kind      TEXT    NOT NULL,
    name      TEXT    NOT NULL,
    url       TEXT    NOT NULL,
    sha256    TEXT    NOT NULL,
    bytes     INTEGER NOT NULL DEFAULT 0,
    license   TEXT,
    revision  TEXT,
    installed INTEGER NOT NULL DEFAULT 0,
    path      TEXT
);

CREATE INDEX idx_models_kind ON models (kind, installed);

-- ---------------------------------------------------------------------------
-- settings — non-secret key/value only. Secrets live in the OS keychain.
-- ---------------------------------------------------------------------------
CREATE TABLE settings (
    key   TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Full-text search over segments.
--
-- Two external-content indexes over the same rows (DESIGN §2 Storage):
--   segments_fts          unicode61, diacritic-insensitive — word search that
--                         works across most languages
--   segments_fts_trigram  trigram — substring/CJK search where word breaks do
--                         not exist
-- MATCH input is escaped in Rust (db::repo::escape_fts_query); never
-- interpolate raw user text into a MATCH expression.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE segments_fts USING fts5 (
    text,
    content = 'segments',
    content_rowid = 'rowid',
    tokenize = "unicode61 remove_diacritics 2"
);

CREATE VIRTUAL TABLE segments_fts_trigram USING fts5 (
    text,
    content = 'segments',
    content_rowid = 'rowid',
    tokenize = "trigram"
);

CREATE TRIGGER segments_fts_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts (rowid, text) VALUES (new.rowid, new.text);
    INSERT INTO segments_fts_trigram (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER segments_fts_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts (segments_fts, rowid, text)
        VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_fts_trigram (segments_fts_trigram, rowid, text)
        VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER segments_fts_au AFTER UPDATE OF text ON segments BEGIN
    INSERT INTO segments_fts (segments_fts, rowid, text)
        VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_fts (rowid, text) VALUES (new.rowid, new.text);
    INSERT INTO segments_fts_trigram (segments_fts_trigram, rowid, text)
        VALUES ('delete', old.rowid, old.text);
    INSERT INTO segments_fts_trigram (rowid, text) VALUES (new.rowid, new.text);
END;

-- ---------------------------------------------------------------------------
-- Built-in summary templates (DESIGN §3: general recap, standup, client call,
-- retro, 1:1, interview). Prompts are plain instructions; the person never
-- has to read them.
-- ---------------------------------------------------------------------------
INSERT INTO templates (id, name, prompt_md, builtin) VALUES
('00000000-0000-4000-8000-000000000001', 'General recap',
'Write a short recap of this meeting.

- Start with two or three sentences on what the meeting was about.
- Then list the decisions that were made.
- Then list the open questions.
- Finally list every task someone agreed to do, with who owns it and any timing that was mentioned.

Only use what was actually said. If something is unclear, say so instead of guessing.', 1),

('00000000-0000-4000-8000-000000000002', 'Daily standup',
'Summarise this standup.

For each person who spoke, list: what they finished, what they are working on next, and anything blocking them.
Then list the tasks that came out of the meeting with their owners.

Keep it tight. Only use what was actually said.', 1),

('00000000-0000-4000-8000-000000000003', 'Client call',
'Summarise this client call.

- What the client asked for, in their words where possible.
- What was promised to them, and by when.
- Concerns or objections they raised.
- Anything about budget, scope or timing.
- Every follow-up task, with its owner.

Only use what was actually said. Do not soften or invent commitments.', 1),

('00000000-0000-4000-8000-000000000004', 'Retrospective',
'Summarise this retrospective.

- What went well.
- What did not go well.
- Ideas people suggested for improving.
- The experiments or changes the group agreed to try, with owners.

Group similar points together. Only use what was actually said.', 1),

('00000000-0000-4000-8000-000000000005', 'One-on-one',
'Summarise this one-on-one.

- The main topics discussed.
- Feedback given in either direction.
- Anything about growth, goals or career.
- Agreements and follow-ups, with owners and any timing mentioned.

Be discreet and factual. Only use what was actually said.', 1),

('00000000-0000-4000-8000-000000000006', 'Interview',
'Summarise this interview.

- The role and the candidate''s background as described.
- Their answers to the main questions, grouped by topic.
- Strengths and concerns that came up, quoting briefly where useful.
- Questions the candidate asked.
- Agreed next steps.

Stay factual and avoid judgements that were not voiced. Only use what was actually said.', 1);
