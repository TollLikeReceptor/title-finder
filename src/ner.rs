use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use orp::params::RuntimeParameters;
use uuid::Uuid;

pub use crate::models::{Attribution, ExtractedTitle};
use crate::models::{GlinerRequest, GlinerSpan, SearchHit};

pub const MIN_VERIFICATION_SCORE: f64 = 0.70;

pub const TITLE_LABEL: &str = "job title";
pub const PERSON_LABEL: &str = "person";

/// GLiNER's minimum span probability (gline-rs's default).
pub const THRESHOLD: f32 = 0.5;

#[derive(Debug)]
pub struct ExtractError(pub String);

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GLiNER extraction failed: {}", self.0)
    }
}

impl std::error::Error for ExtractError {}

/// What GLiNER found in one extraction.
#[derive(Debug, Clone)]
pub struct ExtractionOutput {
    /// Every span found, titles and people, with title attributions.
    pub spans: Vec<GlinerSpan>,
    /// The searched person's most confident title, if any.
    pub chosen: Option<ExtractedTitle>,
}

/// One GLiNER run: what it was given and what came back. The request is
/// returned even on failure so it can always be saved.
#[derive(Debug)]
pub struct ExtractionAttempt {
    pub request: GlinerRequest,
    pub duration_ms: u64,
    pub result: Result<ExtractionOutput, ExtractError>,
}

/// Finds the searched person's job title in search hits. Behind a trait so
/// tests can substitute a stub instead of loading a 1.8 GB model.
#[async_trait]
pub trait TitleExtractor: Send + Sync {
    async fn title_for(&self, hits: &[SearchHit], name: &str) -> ExtractionAttempt;
}

/// Local GLiNER inference through ONNX Runtime.
pub struct GlinerExtractor {
    model: Arc<GLiNER<SpanMode>>,
    /// Model directory name, recorded with each extraction.
    model_name: String,
}

impl GlinerExtractor {
    /// Loads `tokenizer.json` and `onnx/model.onnx` from `dir`, the layout
    /// `scripts/download-gliner.sh` produces. Blocking; call at startup.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, ExtractError> {
        let dir = dir.as_ref();
        let tokenizer = dir.join("tokenizer.json");
        let onnx = dir.join("onnx").join("model.onnx");

        for file in [&tokenizer, &onnx] {
            if !file.is_file() {
                return Err(ExtractError(format!(
                    "{} not found — run ./scripts/download-gliner.sh",
                    file.display()
                )));
            }
        }

        let model = GLiNER::<SpanMode>::new(
            Parameters::default().with_threshold(THRESHOLD),
            RuntimeParameters::default(),
            path_str(&tokenizer)?,
            path_str(&onnx)?,
        )
        .map_err(|error| ExtractError(error.to_string()))?;

        let model_name = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string());

        Ok(Self {
            model: Arc::new(model),
            model_name,
        })
    }
}

fn path_str(path: &Path) -> Result<&str, ExtractError> {
    path.to_str()
        .ok_or_else(|| ExtractError(format!("path is not UTF-8: {}", path.display())))
}

/// One text per hit, in hit order (answer box first). Kept per-hit rather than
/// joined so each stays under GLiNER's 512-token limit and so a title is only
/// ever attributed to a person in the same hit. The link is left out: it never
/// contains a title.
pub fn hits_to_sequences(hits: &[SearchHit]) -> Vec<String> {
    hits.iter()
        .map(|hit| {
            [hit.title.as_str(), hit.snippet.as_str()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty())
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Title,
    Person,
}

/// A span GLiNER found, reduced to what attribution needs.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Which hit it came from.
    pub sequence: usize,
    /// Byte offsets within that hit's text.
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub kind: SpanKind,
    pub probability: f32,
}

/// Honorifics ignored when comparing a person span with the searched name.
const HONORIFICS: &[&str] = &["mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame"];

/// Lowercased name tokens, punctuation and honorifics stripped.
fn name_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| character.is_whitespace() || character == ',')
        .map(|token| {
            token
                .trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty() && !HONORIFICS.contains(&token.as_str()))
        .collect()
}

/// Whether a person span refers to the searched person: every word of the span
/// is part of the searched name, so "Nadella", "Satya Nadella" and
/// "Mr. Nadella" all match "Satya Nadella", but "Steve Ballmer" does not.
pub fn refers_to(person: &str, name: &str) -> bool {
    let wanted = name_tokens(name);
    let found = name_tokens(person);
    !found.is_empty() && found.iter().all(|token| wanted.contains(token))
}

/// Byte ranges where `needle` appears in `haystack` as whole words, ignoring
/// ASCII case. ASCII lowercasing keeps byte offsets aligned with the original.
fn word_occurrences(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    let haystack_lower = haystack.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    let mut found = Vec::new();

    if needle_lower.is_empty() {
        return found;
    }

    let mut offset = 0;
    while let Some(at) = haystack_lower[offset..].find(&needle_lower) {
        let start = offset + at;
        let end = start + needle_lower.len();
        let before = haystack[..start].chars().next_back();
        let after = haystack[end..].chars().next();

        if !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric) {
            found.push((start, end));
        }
        offset = end;
    }

    found
}

/// Places the searched person is mentioned in a hit: GLiNER's matching person
/// spans, plus literal occurrences of the full name and surname in case GLiNER
/// missed one. First names alone aren't searched for literally — too common.
fn searched_mentions(text: &str, people: &[&Candidate], name: &str) -> Vec<(usize, usize)> {
    let mut mentions: Vec<(usize, usize)> = people
        .iter()
        .filter(|person| refers_to(&person.text, name))
        .map(|person| (person.start, person.end))
        .collect();

    mentions.extend(word_occurrences(text, name.trim()));

    let tokens: Vec<&str> = name.split_whitespace().collect();
    if tokens.len() > 1
        && let Some(surname) = tokens.last()
        && surname.len() >= 3
    {
        mentions.extend(word_occurrences(text, surname));
    }

    mentions
}

/// Gap in bytes between two ranges; 0 when they touch or overlap.
fn gap(a: (usize, usize), b: (usize, usize)) -> usize {
    b.0.saturating_sub(a.1).max(a.0.saturating_sub(b.1))
}

fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Attributes a title to the nearest person mentioned in the same hit. Ties go
/// to the searched person.
fn attribute(
    title: (usize, usize),
    searched: &[(usize, usize)],
    others: &[(usize, usize)],
) -> Attribution {
    let nearest = |ranges: &[(usize, usize)]| ranges.iter().map(|range| gap(title, *range)).min();

    match (nearest(searched), nearest(others)) {
        (None, None) => Attribution::Nobody,
        (Some(_), None) => Attribution::SearchedPerson,
        (None, Some(_)) => Attribution::SomeoneElse,
        (Some(to_searched), Some(to_other)) if to_searched <= to_other => {
            Attribution::SearchedPerson
        }
        _ => Attribution::SomeoneElse,
    }
}

/// Attributes every title span to a person; `None` for person spans. The
/// result lines up index-for-index with `candidates`.
pub fn attribute_all(
    candidates: &[Candidate],
    sequences: &[String],
    name: &str,
) -> Vec<Option<Attribution>> {
    let mut attributions = vec![None; candidates.len()];

    for (sequence, text) in sequences.iter().enumerate() {
        let people: Vec<&Candidate> = candidates
            .iter()
            .filter(|candidate| {
                candidate.sequence == sequence && candidate.kind == SpanKind::Person
            })
            .collect();

        let searched = searched_mentions(text, &people, name);
        let others: Vec<(usize, usize)> = people
            .iter()
            .map(|person| (person.start, person.end))
            .filter(|range| !searched.iter().any(|mention| overlaps(*range, *mention)))
            .collect();

        for (index, title) in candidates.iter().enumerate() {
            if title.sequence == sequence && title.kind == SpanKind::Title {
                attributions[index] = Some(attribute((title.start, title.end), &searched, &others));
            }
        }
    }

    attributions
}

/// The most confident title belonging to the searched person.
///
/// Titles nearest the searched person win. If none are, titles from hits that
/// mention nobody are used as a fallback. Titles nearer someone else are never
/// used. Within a tier the highest GLiNER probability wins; ties go to the
/// earlier hit, then the earlier position.
pub fn pick_for_person(
    candidates: &[Candidate],
    sequences: &[String],
    name: &str,
) -> Option<ExtractedTitle> {
    pick_attributed(candidates, &attribute_all(candidates, sequences, name))
}

fn pick_attributed(
    candidates: &[Candidate],
    attributions: &[Option<Attribution>],
) -> Option<ExtractedTitle> {
    let mut best: [Option<&Candidate>; 2] = [None, None];

    // Candidates arrive in hit order, then position, so "strictly greater"
    // below leaves ties with the earlier one.
    for (title, attribution) in candidates.iter().zip(attributions) {
        if title.text.trim().is_empty() {
            continue;
        }

        let tier = match attribution {
            Some(Attribution::SearchedPerson) => 0,
            Some(Attribution::Nobody) => 1,
            Some(Attribution::SomeoneElse) | None => continue,
        };

        if best[tier].is_none_or(|current| title.probability > current.probability) {
            best[tier] = Some(title);
        }
    }

    best.into_iter()
        .flatten()
        .next()
        .map(|title| ExtractedTitle {
            text: title.text.trim().to_string(),
            probability: title.probability,
        })
}

#[async_trait]
impl TitleExtractor for GlinerExtractor {
    async fn title_for(&self, hits: &[SearchHit], name: &str) -> ExtractionAttempt {
        let request = GlinerRequest {
            id: Uuid::new_v4(),
            model: self.model_name.clone(),
            labels: vec![TITLE_LABEL.to_string(), PERSON_LABEL.to_string()],
            threshold: THRESHOLD,
            searched_name: name.to_string(),
            inputs: hits_to_sequences(hits),
            started_at: Utc::now(),
        };

        let started = Instant::now();
        let result = self.run(&request).await;

        ExtractionAttempt {
            request,
            duration_ms: started.elapsed().as_millis() as u64,
            result,
        }
    }
}

impl GlinerExtractor {
    async fn run(&self, request: &GlinerRequest) -> Result<ExtractionOutput, ExtractError> {
        if request.inputs.is_empty() {
            return Ok(ExtractionOutput {
                spans: Vec::new(),
                chosen: None,
            });
        }

        let model = Arc::clone(&self.model);
        let inputs = request.inputs.clone();
        let name = request.searched_name.clone();

        // Inference is CPU-bound and synchronous; keep it off the async workers.
        tokio::task::spawn_blocking(move || {
            let texts: Vec<&str> = inputs.iter().map(String::as_str).collect();
            let input = TextInput::from_str(&texts, &[TITLE_LABEL, PERSON_LABEL])
                .map_err(|error| ExtractError(error.to_string()))?;
            let output = model
                .inference(input)
                .map_err(|error| ExtractError(error.to_string()))?;

            let candidates: Vec<Candidate> = output
                .spans
                .into_iter()
                .flatten()
                .filter_map(|span| {
                    let kind = match span.class() {
                        TITLE_LABEL => SpanKind::Title,
                        PERSON_LABEL => SpanKind::Person,
                        _ => return None,
                    };
                    let (start, end) = span.offsets();

                    Some(Candidate {
                        sequence: span.sequence(),
                        start,
                        end,
                        text: span.text().to_string(),
                        kind,
                        probability: span.probability(),
                    })
                })
                .collect();

            let attributions = attribute_all(&candidates, &inputs, &name);
            let chosen = pick_attributed(&candidates, &attributions);

            let spans = candidates
                .into_iter()
                .zip(attributions)
                .map(|(candidate, attribution)| GlinerSpan {
                    input: candidate.sequence,
                    start: candidate.start,
                    end: candidate.end,
                    text: candidate.text,
                    label: match candidate.kind {
                        SpanKind::Title => TITLE_LABEL,
                        SpanKind::Person => PERSON_LABEL,
                    }
                    .to_string(),
                    probability: candidate.probability,
                    attribution,
                })
                .collect();

            Ok(ExtractionOutput { spans, chosen })
        })
        .await
        .map_err(|error| ExtractError(format!("inference task panicked: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds candidates from `(kind, text, start, end, probability)` in one hit.
    /// The fixtures below are real gliner_large-v2.1 outputs.
    fn spans(sequence: usize, rows: &[(SpanKind, &str, usize, usize, f32)]) -> Vec<Candidate> {
        rows.iter()
            .map(|(kind, text, start, end, probability)| Candidate {
                sequence,
                start: *start,
                end: *end,
                text: text.to_string(),
                kind: *kind,
                probability: *probability,
            })
            .collect()
    }

    use SpanKind::{Person, Title};

    fn pick(
        text: &str,
        rows: &[(SpanKind, &str, usize, usize, f32)],
        name: &str,
    ) -> Option<String> {
        pick_for_person(&spans(0, rows), &[text.to_string()], name).map(|title| title.text)
    }

    #[test]
    fn takes_the_most_confident_of_the_persons_titles() {
        let text = "Satya Nadella - Microsoft\nSatya Nadella is the Chairman and Chief Executive Officer of Microsoft and joined the company in 1992.";
        let rows = [
            (Person, "Satya Nadella", 0, 13, 1.00),
            (Person, "Satya Nadella", 26, 39, 1.00),
            (Title, "Chairman", 47, 55, 0.72),
            (Title, "Chief Executive Officer", 60, 83, 0.91),
        ];

        assert_eq!(
            pick(text, &rows, "Satya Nadella").as_deref(),
            Some("Chief Executive Officer")
        );
    }

    #[test]
    fn skips_titles_nearer_someone_else() {
        let text = "Former CFO Mark Chen left in 2021; Lisa Park now leads finance as Chief Financial Officer.";
        let rows = [
            (Title, "Former CFO", 0, 10, 0.60),
            (Person, "Mark Chen", 11, 20, 0.95),
            (Person, "Lisa Park", 35, 44, 0.99),
            (Title, "Chief Financial Officer", 66, 89, 0.94),
        ];

        assert_eq!(
            pick(text, &rows, "Lisa Park").as_deref(),
            Some("Chief Financial Officer")
        );
        assert_eq!(
            pick(text, &rows, "Mark Chen").as_deref(),
            Some("Former CFO")
        );
    }

    #[test]
    fn separates_titles_listed_side_by_side() {
        let text = "Amy Hood, CFO, and Satya Nadella, CEO, presented the results.";
        let rows = [
            (Person, "Amy Hood", 0, 8, 0.99),
            (Title, "CFO", 10, 13, 0.98),
            (Person, "Satya Nadella", 19, 32, 0.99),
            (Title, "CEO", 34, 37, 0.98),
        ];

        assert_eq!(pick(text, &rows, "Satya Nadella").as_deref(), Some("CEO"));
        assert_eq!(pick(text, &rows, "Amy Hood").as_deref(), Some("CFO"));
    }

    #[test]
    fn never_takes_another_persons_title() {
        let text = "Jane reports to John Smith, the VP of Sales at Acme.";
        let rows = [
            (Person, "Jane", 0, 4, 0.97),
            (Person, "John Smith", 16, 26, 0.97),
            (Title, "VP of Sales", 32, 43, 0.97),
        ];

        assert_eq!(pick(text, &rows, "Jane Doe"), None);
        assert_eq!(
            pick(text, &rows, "John Smith").as_deref(),
            Some("VP of Sales")
        );
    }

    #[test]
    fn known_limitation_nearest_person_can_be_wrong() {
        // "succeeded Steve Ballmer as CEO": CEO is Nadella's, but Ballmer is
        // nearer. Pinned so a smarter rule shows up as a deliberate change.
        let text =
            "Satya Nadella, who succeeded Steve Ballmer as CEO in 2014, has reshaped Microsoft.";
        let rows = [
            (Person, "Satya Nadella", 0, 13, 0.99),
            (Person, "Steve Ballmer", 29, 42, 0.97),
            (Title, "CEO", 46, 49, 0.97),
        ];

        assert_eq!(pick(text, &rows, "Satya Nadella"), None);
    }

    #[test]
    fn a_title_with_nobody_mentioned_is_only_a_fallback() {
        let answer_box = "Chief Executive Officer".to_string();
        let profile = "Satya Nadella - Executive Vice President".to_string();

        let mut candidates = spans(0, &[(Title, "Chief Executive Officer", 0, 23, 0.95)]);
        candidates.extend(spans(
            1,
            &[
                (Person, "Satya Nadella", 0, 13, 0.99),
                (Title, "Executive Vice President", 16, 40, 0.60),
            ],
        ));

        // The attributed title wins even though the unattributed one scores higher.
        let picked = pick_for_person(&candidates, &[answer_box.clone(), profile], "Satya Nadella");
        assert_eq!(
            picked.map(|title| title.text).as_deref(),
            Some("Executive Vice President")
        );

        // With nothing attributed, the unattributed title is used.
        let alone = pick_for_person(&candidates[..1], &[answer_box], "Satya Nadella");
        assert_eq!(
            alone.map(|title| title.text).as_deref(),
            Some("Chief Executive Officer")
        );
    }

    #[test]
    fn finds_the_person_by_literal_name_when_gliner_misses_them() {
        // No person span at all, but the surname appears in the text.
        let text = "Nadella: Chief Executive Officer since 2014";
        let rows = [(Title, "Chief Executive Officer", 9, 32, 0.9)];

        assert_eq!(
            pick(text, &rows, "Satya Nadella").as_deref(),
            Some("Chief Executive Officer")
        );
    }

    #[test]
    fn name_matching() {
        assert!(refers_to("Satya Nadella", "Satya Nadella"));
        assert!(refers_to("Nadella", "Satya Nadella"));
        assert!(refers_to("Mr. Nadella", "Satya Nadella"));
        assert!(refers_to("satya nadella", "Satya Nadella"));
        assert!(!refers_to("Steve Ballmer", "Satya Nadella"));
        assert!(!refers_to("Satya Nadella Jr", "Satya Nadella"));
        assert!(!refers_to("Mr.", "Satya Nadella"));
    }

    #[test]
    fn word_occurrences_respect_boundaries() {
        assert_eq!(
            word_occurrences("Park Lane and Lisa Park", "park"),
            vec![(0, 4), (19, 23)]
        );
        assert!(word_occurrences("Parkinson", "park").is_empty());
    }

    #[test]
    fn sequences_follow_hit_order_and_skip_links() {
        let hits = vec![
            SearchHit {
                title: "Jane Doe - Acme".to_string(),
                link: "https://example.com".to_string(),
                snippet: "Jane is a Staff Engineer".to_string(),
            },
            SearchHit {
                title: String::new(),
                link: "https://example.com/empty".to_string(),
                snippet: String::new(),
            },
            SearchHit {
                title: String::new(),
                link: String::new(),
                snippet: "Second".to_string(),
            },
        ];

        assert_eq!(
            hits_to_sequences(&hits),
            vec!["Jane Doe - Acme\nJane is a Staff Engineer", "Second"]
        );
    }
}
