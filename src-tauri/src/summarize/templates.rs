//! Recap styles: the six built-ins plus whatever the person writes.
//!
//! IMPLEMENTED-BY: summarize agent (M5).
//!
//! The built-in rows are seeded by `migrations/0001_init.sql`, so they exist
//! before any code runs — that seed is what the settings screen lists and
//! what `TemplateDraft` name-collision checks compare against. But the
//! *instructions* a builtin actually renders come from the markdown files in
//! `resources/templates/`, embedded at compile time with `include_str!`. That
//! way the wording that ships in the binary can never drift from what a
//! migration happened to insert years ago; the db row is really just a
//! stable id + display name for a compiled-in prompt.
//!
//! Built-ins are read-only. `db::repo::upsert_template` and `delete_template`
//! already refuse to touch them.
//!
//! A recap stores `template_snapshot`, the prompt exactly as used, so editing a
//! style later does not rewrite the history of what produced an old recap.

use crate::db::{repo, Db, DbError};
use crate::summarize::SummarizeError;
use crate::types::{Speaker, SummaryLanguage, Template, TemplateDraft};

/// Ids of the seeded built-ins, in the order the settings screen lists them.
pub const BUILTIN_IDS: [&str; 6] = [
    "00000000-0000-4000-8000-000000000001", // General recap
    "00000000-0000-4000-8000-000000000002", // Daily standup
    "00000000-0000-4000-8000-000000000003", // Client call
    "00000000-0000-4000-8000-000000000004", // Retrospective
    "00000000-0000-4000-8000-000000000005", // One-on-one
    "00000000-0000-4000-8000-000000000006", // Interview
];

/// Names of the built-ins, same order as [`BUILTIN_IDS`]. A custom style may
/// not take one of these (case-insensitively), so the settings list never has
/// two rows that read the same.
pub const BUILTIN_NAMES: [&str; 6] = [
    "General recap",
    "Daily standup",
    "Client call",
    "Retrospective",
    "One-on-one",
    "Interview",
];

/// The instructions for each builtin, embedded at compile time so they can
/// never go missing or drift from what shipped (DESIGN §3: "6 built-ins ...
/// embedded via include_str!").
const GENERAL_RECAP_MD: &str = include_str!("../../resources/templates/general-recap.md");
const STANDUP_MD: &str = include_str!("../../resources/templates/standup.md");
const CLIENT_CALL_MD: &str = include_str!("../../resources/templates/client-call.md");
const RETRO_MD: &str = include_str!("../../resources/templates/retro.md");
const ONE_ON_ONE_MD: &str = include_str!("../../resources/templates/one-on-one.md");
const INTERVIEW_MD: &str = include_str!("../../resources/templates/interview.md");

/// Longest custom prompt we accept, so one template cannot eat the whole
/// context window.
pub const MAX_PROMPT_CHARS: usize = 4_000;

/// The compiled-in instructions for a builtin id, or `None` for a custom
/// template (whose instructions live only in the database).
fn builtin_prompt(id: &str) -> Option<&'static str> {
    match id {
        "00000000-0000-4000-8000-000000000001" => Some(GENERAL_RECAP_MD),
        "00000000-0000-4000-8000-000000000002" => Some(STANDUP_MD),
        "00000000-0000-4000-8000-000000000003" => Some(CLIENT_CALL_MD),
        "00000000-0000-4000-8000-000000000004" => Some(RETRO_MD),
        "00000000-0000-4000-8000-000000000005" => Some(ONE_ON_ONE_MD),
        "00000000-0000-4000-8000-000000000006" => Some(INTERVIEW_MD),
        _ => None,
    }
}

/// The instructions to actually use: the compiled-in copy for a builtin, the
/// database row for a custom one. Public so a recap's `template_snapshot` can
/// record exactly what was rendered, not just what the database happens to
/// hold for that id.
pub fn prompt_for(template: &Template) -> String {
    builtin_prompt(&template.id)
        .map(str::to_string)
        .unwrap_or_else(|| template.prompt_md.clone())
}

/// Everything needed to turn a template into a prompt.
#[derive(Debug, Clone, Default)]
pub struct RenderContext {
    pub meeting_title: String,
    pub started_at: String,
    pub duration_ms: i64,
    /// Detected language of the meeting, for the "same as the meeting" setting.
    pub meeting_language: Option<String>,
    /// A second language a real part of the meeting was held in, when there was
    /// one. A meeting that ran a third in English is not the same job as one
    /// that ran entirely in Italian, and the instruction should say so rather
    /// than leave the model to average the two.
    pub also_spoken: Option<String>,
    /// What language to write the recap in.
    pub output_language: SummaryLanguage,
    pub speakers: Vec<Speaker>,
    /// The transcript piece this prompt covers.
    pub transcript_chunk: String,
    /// Which piece this is, for map-reduce prompts.
    pub chunk_index: u32,
    pub chunk_count: u32,
}

/// The languages Echo can name outright, by the code speech detection reports.
///
/// Whisper reports a language as a two-letter code, and a code is what gets
/// stored on a segment and on a recap. An instruction that says "write this in
/// (it)" is asking a model to guess; "write this in Italian" is not. This is the
/// list of codes worth spelling out — the common ones plus the ones Whisper
/// detects most often. Anything not here falls back to the code itself, which is
/// still better than nothing and never lies about what was asked for.
const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("bg", "Bulgarian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("ms", "Malay"),
    ("nb", "Norwegian"),
    ("nl", "Dutch"),
    ("nn", "Norwegian"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sr", "Serbian"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
];

/// The few region-tagged codes where the region is the point: a recap asked for
/// in `pt-BR` should not come back in European Portuguese.
const REGIONAL_LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("pt-br", "Brazilian Portuguese"),
    ("pt-pt", "European Portuguese"),
    ("zh-tw", "Traditional Chinese"),
    ("zh-hant", "Traditional Chinese"),
    ("zh-hk", "Traditional Chinese"),
    ("zh-cn", "Simplified Chinese"),
    ("zh-hans", "Simplified Chinese"),
];

/// The name of a language, given whatever Echo has: a code (`it`), a
/// region-tagged code (`pt-BR`), or a name somebody typed themselves
/// ("Italian", "Bavarian German").
///
/// A value that is already a name comes back untouched — the point is only to
/// stop a bare code reaching a prompt as if it were a word.
pub fn language_name(language: &str) -> String {
    let trimmed = language.trim();
    let lower = trimmed.to_ascii_lowercase().replace('_', "-");

    if let Some((_, name)) = REGIONAL_LANGUAGE_NAMES.iter().find(|(c, _)| *c == lower) {
        return (*name).to_string();
    }
    // A region we have no special name for ("de-AT") still names its language.
    let base = lower.split('-').next().unwrap_or(&lower);
    if let Some((_, name)) = LANGUAGE_NAMES.iter().find(|(c, _)| *c == base) {
        return (*name).to_string();
    }
    trimmed.to_string()
}

/// Plain-English instruction for what language to write in. No jargon here
/// either — this text can end up quoted in a diagnostics export.
fn language_directive(ctx: &RenderContext) -> String {
    language_directive_for(ctx, "the recap")
}

/// The same instruction about `what` to write — a recap, or a task list.
fn language_directive_for(ctx: &RenderContext, what: &str) -> String {
    match &ctx.output_language {
        SummaryLanguage::SameAsMeeting => match &ctx.meeting_language {
            Some(lang) => {
                let mut directive = format!(
                    "Write {what} in the same language the meeting was held in, {}.",
                    language_name(lang)
                );
                if let Some(also) = ctx.also_spoken.as_deref().filter(|a| a != &lang.as_str()) {
                    directive.push_str(&format!(
                        " Parts of it were held in {}: write {what} in {} anyway, and keep names \
                         and quoted words as they were said.",
                        language_name(also),
                        language_name(lang)
                    ));
                }
                directive
            }
            None => format!("Write {what} in the same language the meeting was held in."),
        },
        SummaryLanguage::English => format!("Write {what} in English."),
        SummaryLanguage::Fixed(lang) => {
            format!("Write {what} in {}.", language_name(lang))
        }
    }
}

fn meeting_header(ctx: &RenderContext) -> String {
    let mut header = format!(
        "Meeting: {}\nStarted: {}\n",
        if ctx.meeting_title.trim().is_empty() {
            "Untitled meeting"
        } else {
            ctx.meeting_title.trim()
        },
        ctx.started_at
    );
    if !ctx.speakers.is_empty() {
        let names: Vec<&str> = ctx
            .speakers
            .iter()
            .map(|s| s.display_name.as_str())
            .collect();
        header.push_str(&format!("Speakers: {}\n", names.join(", ")));
    }
    header
}

/// Fill a template into a prompt for one transcript chunk.
///
/// When there is more than one chunk this asks for working notes rather than
/// a finished recap — [`render_reduce`] folds the notes together afterwards.
pub fn render(template: &Template, ctx: &RenderContext) -> String {
    let mut out = String::new();
    out.push_str(&prompt_for(template));
    out.push_str("\n\n");
    out.push_str(&language_directive(ctx));
    out.push_str("\n\n");
    out.push_str(&meeting_header(ctx));

    if ctx.chunk_count > 1 {
        out.push_str(&format!(
            "\nThis is part {} of {} of the meeting's transcript. Write working notes for just \
             this part — the same kind of information the instructions above ask for (decisions, \
             open questions, tasks with owners) — as plain prose or a list. Do not write a \
             finished recap yet; a later pass folds every part together.\n\n",
            ctx.chunk_index + 1,
            ctx.chunk_count
        ));
    }

    out.push_str("Transcript:\n---\n");
    out.push_str(&ctx.transcript_chunk);
    out.push_str("\n---\n");
    out
}

/// The reduce step: fold per-chunk notes into one recap.
pub fn render_reduce(
    template: &Template,
    chunk_summaries: &[String],
    ctx: &RenderContext,
) -> String {
    let mut out = String::new();
    out.push_str(&prompt_for(template));
    out.push_str("\n\n");
    out.push_str(&language_directive(ctx));
    out.push_str("\n\n");
    out.push_str(&meeting_header(ctx));
    out.push_str(&format!(
        "\nBelow are working notes written from {} separate parts of this meeting's transcript, \
         in order. Combine them into one recap that follows the instructions above. Merge \
         duplicate points, keep the chronological sense of decisions, and never mention that the \
         transcript was split into parts.\n\n",
        chunk_summaries.len()
    ));
    for (i, notes) in chunk_summaries.iter().enumerate() {
        out.push_str(&format!("Notes from part {}:\n{}\n\n", i + 1, notes));
    }
    out
}

/// The strict-JSON prompt for action items, with the schema inline.
pub fn render_action_items(recap_md: &str, ctx: &RenderContext) -> String {
    let schema_text = serde_json::to_string_pretty(&action_item_schema()).unwrap_or_default();
    format!(
        "Read this meeting recap and pull out every task someone agreed to do.\n\n\
         Reply with ONLY JSON matching this schema, and nothing else — no markdown code fences, \
         no commentary before or after it:\n{schema}\n\n\
         {due_hint_language}\n\
         If there are no tasks, reply with {{\"items\": []}}.\n\n\
         Recap:\n---\n{recap}\n---\n",
        schema = schema_text,
        due_hint_language = language_directive_for(ctx, "every task"),
        recap = recap_md,
    )
}

/// JSON schema action items are validated against locally, whether or not the
/// backend supports a JSON mode.
pub fn action_item_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "required": ["items"],
        "additionalProperties": false,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["description"],
                    "additionalProperties": false,
                    "properties": {
                        "description": { "type": "string", "minLength": 1 },
                        "owner": { "type": ["string", "null"] },
                        "dueHint": { "type": ["string", "null"] }
                    }
                }
            }
        }
    })
}

/// Check a custom style before it is saved. Rejects an empty name, a name that
/// collides with a built-in, and an over-long prompt.
pub fn validate_draft(draft: &TemplateDraft) -> Result<(), SummarizeError> {
    let name = draft.name.trim();
    if name.is_empty() {
        return Err(SummarizeError::Failed("a recap style needs a name".into()));
    }
    if name.chars().count() > 80 {
        return Err(SummarizeError::Failed("that name is too long".into()));
    }
    let prompt = draft.prompt_md.trim();
    if prompt.is_empty() {
        return Err(SummarizeError::Failed(
            "a recap style needs instructions".into(),
        ));
    }
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(SummarizeError::Failed(
            "those instructions are too long".into(),
        ));
    }
    // Only a brand-new draft can collide; editing an existing custom template
    // by its own id is fine even if the name happens to match one it already had.
    if draft.id.is_none()
        && BUILTIN_NAMES
            .iter()
            .any(|builtin| builtin.eq_ignore_ascii_case(name))
    {
        return Err(SummarizeError::Failed(
            "that name is already used by a built-in style".into(),
        ));
    }
    Ok(())
}

fn db_err(e: DbError) -> SummarizeError {
    SummarizeError::Failed(e.to_string())
}

/// The style to use for a meeting: the request's, then the setting, then the
/// general recap.
pub async fn resolve_template(
    db: &Db,
    requested: Option<&str>,
) -> Result<Template, SummarizeError> {
    if let Some(id) = requested {
        if let Some(t) = repo::get_template(db, id).await.map_err(db_err)? {
            return Ok(t);
        }
        // An id that no longer exists falls through to the safe default
        // rather than failing the whole recap (mantra: defaults are safe).
    }

    if let Ok(settings) = crate::settings::load(db).await {
        if let Some(id) = &settings.summary_template_id {
            if let Some(t) = repo::get_template(db, id).await.map_err(db_err)? {
                return Ok(t);
            }
        }
    }

    repo::default_template(db)
        .await
        .map_err(db_err)?
        .ok_or_else(|| SummarizeError::Failed("no recap style is available".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_six_builtin_ids_and_they_are_unique() {
        let unique: std::collections::HashSet<_> = BUILTIN_IDS.iter().collect();
        assert_eq!(unique.len(), 6);
    }

    #[test]
    fn every_builtin_id_has_compiled_in_instructions() {
        for id in BUILTIN_IDS {
            let prompt = builtin_prompt(id).expect("builtin id should have compiled-in text");
            assert!(!prompt.trim().is_empty());
        }
        assert!(builtin_prompt("not-a-real-id").is_none());
    }

    #[test]
    fn the_action_item_schema_requires_a_description() {
        let schema = action_item_schema();
        let item = &schema["properties"]["items"]["items"];
        assert_eq!(item["required"][0], "description");
        assert_eq!(item["additionalProperties"], false);
    }

    fn general_recap() -> Template {
        Template {
            id: BUILTIN_IDS[0].to_string(),
            name: "General recap".to_string(),
            prompt_md: GENERAL_RECAP_MD.to_string(),
            builtin: true,
        }
    }

    #[test]
    fn render_uses_the_compiled_in_prompt_for_a_builtin() {
        let ctx = RenderContext {
            meeting_title: "Weekly sync".into(),
            transcript_chunk: "You: hello\nSpeaker 1: hi".into(),
            chunk_count: 1,
            ..Default::default()
        };
        let out = render(&general_recap(), &ctx);
        assert!(out.contains("Write a short recap of this meeting."));
        assert!(out.contains("Weekly sync"));
        assert!(out.contains("You: hello"));
        assert!(!out.contains("part 1 of"));
    }

    #[test]
    fn render_asks_for_working_notes_when_chunked() {
        let ctx = RenderContext {
            chunk_index: 1,
            chunk_count: 3,
            transcript_chunk: "...".into(),
            ..Default::default()
        };
        let out = render(&general_recap(), &ctx);
        assert!(out.contains("part 2 of 3"));
        assert!(out.to_lowercase().contains("working notes"));
    }

    #[test]
    fn render_reduce_lists_every_part_in_order() {
        let ctx = RenderContext::default();
        let out = render_reduce(
            &general_recap(),
            &["first notes".into(), "second notes".into()],
            &ctx,
        );
        assert!(out.contains("first notes"));
        assert!(out.contains("second notes"));
        assert!(out.find("first notes").unwrap() < out.find("second notes").unwrap());
    }

    #[test]
    fn language_directive_names_the_fixed_language() {
        let ctx = RenderContext {
            output_language: SummaryLanguage::Fixed("Italian".into()),
            ..Default::default()
        };
        assert_eq!(language_directive(&ctx), "Write the recap in Italian.");
    }

    // -- language names, not language codes --------------------------------
    //
    // Speech detection reports "it"; a prompt that says "write this in (it)"
    // is asking the model to guess what that means.

    #[test]
    fn a_language_code_is_spelled_out_as_its_name() {
        for (code, name) in [
            ("it", "Italian"),
            ("IT", "Italian"),
            (" it ", "Italian"),
            ("en", "English"),
            ("de", "German"),
            ("pt", "Portuguese"),
            ("zh", "Chinese"),
            ("nb", "Norwegian"),
        ] {
            assert_eq!(language_name(code), name, "{code}");
        }
    }

    #[test]
    fn a_region_is_kept_when_it_changes_the_answer_and_dropped_when_it_does_not() {
        assert_eq!(language_name("pt-BR"), "Brazilian Portuguese");
        assert_eq!(language_name("pt_BR"), "Brazilian Portuguese");
        assert_eq!(language_name("zh-TW"), "Traditional Chinese");
        assert_eq!(language_name("zh-Hans"), "Simplified Chinese");
        // No special name for Austrian German; it is still German.
        assert_eq!(language_name("de-AT"), "German");
    }

    #[test]
    fn something_that_is_already_a_language_name_is_left_alone() {
        assert_eq!(language_name("Italian"), "Italian");
        assert_eq!(language_name("Swiss German"), "Swiss German");
        // An unknown code is no worse off than before: it goes through as-is
        // rather than being turned into a guess.
        assert_eq!(language_name("xx"), "xx");
        assert_eq!(language_name(""), "");
    }

    #[test]
    fn a_recap_prompt_asks_for_a_named_language_never_a_code() {
        let ctx = RenderContext {
            output_language: SummaryLanguage::Fixed("it".into()),
            ..Default::default()
        };
        assert_eq!(language_directive(&ctx), "Write the recap in Italian.");
        assert!(!render(&general_recap(), &ctx).contains("(it)"));

        let same = RenderContext {
            output_language: SummaryLanguage::SameAsMeeting,
            meeting_language: Some("it".into()),
            ..Default::default()
        };
        assert_eq!(
            language_directive(&same),
            "Write the recap in the same language the meeting was held in, Italian."
        );

        let unknown = RenderContext {
            output_language: SummaryLanguage::SameAsMeeting,
            ..Default::default()
        };
        assert_eq!(
            language_directive(&unknown),
            "Write the recap in the same language the meeting was held in."
        );
    }

    /// A meeting a real part of which was held in another language: the prompt
    /// says so, and still asks for one recap in one language. Silently averaging
    /// the two is how a bilingual meeting gets written up in neither.
    #[test]
    fn a_meeting_held_in_two_languages_says_so_in_the_prompt() {
        let ctx = RenderContext {
            output_language: SummaryLanguage::SameAsMeeting,
            meeting_language: Some("it".into()),
            also_spoken: Some("en".into()),
            ..Default::default()
        };
        assert_eq!(
            language_directive(&ctx),
            "Write the recap in the same language the meeting was held in, Italian. Parts of it \
             were held in English: write the recap in Italian anyway, and keep names and quoted \
             words as they were said."
        );
        // No codes anywhere in the prompt a model actually reads.
        let prompt = render(&general_recap(), &ctx);
        assert!(!prompt.contains("(en)"), "{prompt}");
    }

    #[test]
    fn the_task_list_prompt_asks_for_the_same_language_by_name() {
        let ctx = RenderContext {
            output_language: SummaryLanguage::Fixed("it".into()),
            ..Default::default()
        };
        let prompt = render_action_items("## Decisioni\n\n- Spedire il venerdì.", &ctx);
        assert!(prompt.contains("Write every task in Italian."), "{prompt}");
        assert!(!prompt.contains("(it)"), "{prompt}");
    }

    #[test]
    fn validate_draft_rejects_empty_and_builtin_names() {
        assert!(validate_draft(&TemplateDraft {
            id: None,
            name: "  ".into(),
            prompt_md: "do something".into(),
        })
        .is_err());
        assert!(validate_draft(&TemplateDraft {
            id: None,
            name: "general recap".into(), // case-insensitive collision
            prompt_md: "do something".into(),
        })
        .is_err());
        assert!(validate_draft(&TemplateDraft {
            id: Some(BUILTIN_IDS[0].into()),
            name: "General recap".into(), // editing itself is fine
            prompt_md: "do something".into(),
        })
        .is_ok());
        assert!(validate_draft(&TemplateDraft {
            id: None,
            name: "My style".into(),
            prompt_md: "x".repeat(MAX_PROMPT_CHARS + 1),
        })
        .is_err());
        assert!(validate_draft(&TemplateDraft {
            id: None,
            name: "My style".into(),
            prompt_md: "do something".into(),
        })
        .is_ok());
    }

    #[tokio::test]
    async fn resolve_template_falls_back_to_the_general_recap() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let t = resolve_template(&db, None).await.unwrap();
        assert_eq!(t.id, BUILTIN_IDS[0]);
    }

    #[tokio::test]
    async fn resolve_template_prefers_an_explicit_request() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let t = resolve_template(&db, Some(BUILTIN_IDS[2])).await.unwrap();
        assert_eq!(t.id, BUILTIN_IDS[2]);
    }

    #[tokio::test]
    async fn resolve_template_ignores_a_stale_requested_id() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let t = resolve_template(&db, Some("00000000-0000-4000-8000-00000000dead"))
            .await
            .unwrap();
        assert_eq!(t.id, BUILTIN_IDS[0]);
    }
}
