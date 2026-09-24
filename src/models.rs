use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
pub struct TitleRecord {
    pub id: Uuid,
    pub name: String,
    pub company: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub link: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerpRequest {
    pub id: Uuid,
    pub endpoint: String,
    pub engine: String,
    pub query: String,
    pub num: u32,
    pub sent_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RequestOutcome {
    Succeeded { hit_count: usize },
    Failed { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerpRequestLog {
    pub request: SerpRequest,
    pub http_status: Option<u16>,
    pub outcome: RequestOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jev: Option<JevRequestLog>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gliner: Option<GlinerRequestLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTitle {
    pub text: String,
    pub probability: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attribution {
    SearchedPerson,
    Nobody,
    SomeoneElse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlinerSpan {
    pub input: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub label: String,
    pub probability: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<Attribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlinerRequest {
    pub id: Uuid,
    pub model: String,
    pub labels: Vec<String>,
    pub threshold: f32,
    pub searched_name: String,
    pub inputs: Vec<String>,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GlinerOutcome {
    Succeeded {
        spans: Vec<GlinerSpan>,
        chosen: Option<ExtractedTitle>,
    },
    Failed {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlinerRequestLog {
    pub serp_request_id: Uuid,
    pub request: GlinerRequest,
    pub duration_ms: u64,
    pub outcome: GlinerOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevRequest {
    pub id: Uuid,
    pub endpoint: String,
    pub model: String,
    pub question_key: String,
    pub instructions: String,
    pub state: String,
    pub sent_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum JevOutcome {
    Succeeded {
        score: f64,
        answers_question: bool,
        model: String,
    },
    Failed {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevRequestLog {
    pub serp_request_id: Uuid,
    pub request: JevRequest,
    pub http_status: Option<u16>,
    pub outcome: JevOutcome,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Verification {
    Checked {
        score: f64,
        answers_question: bool,
        model: String,
        checked_at: DateTime<Utc>,
    },
    Failed {
        message: String,
    },
    Skipped {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TitleSource {
    Directory,
    Gliner,
    Unverified,
    NotFound,
    ExtractionFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Directory,
    SerpApi,
    SerpApiCached,
}

#[derive(Debug, Clone, Serialize)]
pub struct TitleLookup {
    pub name: String,
    pub company: String,
    pub title: Option<String>,
    pub title_source: TitleSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_confidence: Option<f32>,
    pub source: Source,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<SearchHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification: Option<Verification>,
    pub retrieved_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct TitleQuery {
    pub name: String,
    pub company: String,
    #[serde(default)]
    pub refresh: bool,
}

#[derive(Debug, Deserialize)]
pub struct RequestListQuery {
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct NewTitle {
    pub name: String,
    pub company: String,
    pub title: String,
}

#[derive(Debug, Serialize)]
pub struct Health {
    pub status: &'static str,
    pub version: &'static str,
}
