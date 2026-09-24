use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;

use uuid::Uuid;

use crate::models::{JevRequest, SearchHit, Verification};

/// Scores at or above this count as "the search answered the question".
pub const ANSWER_THRESHOLD: f64 = 0.5;

/// Key for our one question in the Jev request; the response is keyed the same way.
const QUESTION_KEY: &str = "answers_job_title";

/// The question Jev is asked about the search text.
pub fn build_instructions(name: &str, company: &str) -> String {
    format!("Does this text answer the question What is the job title of {name} at {company}?")
}

/// Flattens search hits into the plain text Jev reads as `state`.
pub fn hits_to_text(hits: &[SearchHit]) -> String {
    hits.iter()
        .map(|hit| {
            [hit.title.as_str(), hit.link.as_str(), hit.snippet.as_str()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|block| !block.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[derive(Debug)]
pub enum VerifyError {
    /// The request never completed (DNS, TLS, timeout, connection refused).
    Transport(String),
    /// Jev answered with an error (bad key, bad request, ...).
    Upstream { status: u16, message: String },
    /// Jev answered with a body we could not understand.
    Decode(String),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::Transport(message) => write!(f, "Jev request failed: {message}"),
            VerifyError::Upstream { status, message } => {
                write!(f, "Jev returned {status}: {message}")
            }
            VerifyError::Decode(message) => write!(f, "could not parse Jev response: {message}"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// One call to Jev: what was sent and what came back. The request is returned
/// even on failure so it can always be saved.
#[derive(Debug)]
pub struct JevAttempt {
    pub request: JevRequest,
    /// HTTP status, when a response arrived at all.
    pub http_status: Option<u16>,
    pub result: Result<Verification, VerifyError>,
}

/// Asks whether search text actually answers the job-title question. Behind a
/// trait so tests can substitute a stub instead of calling Jev.
#[async_trait]
pub trait AnswerCheck: Send + Sync {
    async fn check(&self, text: &str, name: &str, company: &str) -> JevAttempt;
}

/// Jev (typesafe.ai `systemone`) client.
pub struct JevClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl JevClient {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.typesafe.ai";
    const MODEL: &'static str = "jev-latest";

    pub fn new(api_key: String) -> Result<Self, reqwest::Error> {
        Self::with_base_url(api_key, Self::DEFAULT_BASE_URL.to_string())
    }

    /// `base_url` is injectable so it can be pointed at a local stub server.
    pub fn with_base_url(api_key: String, base_url: String) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent(concat!("title-finder/", env!("CARGO_PKG_VERSION")))
            .build()?;

        Ok(Self {
            http,
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

#[async_trait]
impl AnswerCheck for JevClient {
    async fn check(&self, text: &str, name: &str, company: &str) -> JevAttempt {
        let request = JevRequest {
            id: Uuid::new_v4(),
            endpoint: format!("{}/v1/systemone", self.base_url),
            model: Self::MODEL.to_string(),
            question_key: QUESTION_KEY.to_string(),
            instructions: build_instructions(name, company),
            state: text.to_string(),
            sent_at: Utc::now(),
        };

        let (http_status, result) = self.send(&request).await;

        JevAttempt {
            request,
            http_status,
            result,
        }
    }
}

impl JevClient {
    async fn send(&self, request: &JevRequest) -> (Option<u16>, Result<Verification, VerifyError>) {
        let body = json!({
            "state": request.state,
            "model": request.model,
            "questions": {
                request.question_key.as_str(): {
                    "type": "noul",
                    "instructions": request.instructions,
                }
            }
        });

        let response = match self
            .http
            .post(&request.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => return (None, Err(VerifyError::Transport(error.to_string()))),
        };

        let status = response.status().as_u16();
        let result = match response.text().await {
            Ok(text) => parse_response(status, &text),
            Err(error) => Err(VerifyError::Transport(error.to_string())),
        };

        (Some(status), result)
    }
}

#[derive(Debug, Deserialize)]
struct JevResponse {
    model: String,
    answers: std::collections::HashMap<String, JevAnswer>,
}

#[derive(Debug, Deserialize)]
struct JevAnswer {
    noul: f64,
}

/// Error body, e.g. `{"detail":{"error_type":"authentication_error","message":"..."}}`.
#[derive(Debug, Deserialize)]
struct JevErrorResponse {
    detail: JevErrorDetail,
}

#[derive(Debug, Deserialize)]
struct JevErrorDetail {
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

/// Turns a Jev HTTP response into a verification. Separate from the client so
/// it can be tested against recorded payloads.
pub fn parse_response(status: u16, body: &str) -> Result<Verification, VerifyError> {
    if !(200..300).contains(&status) {
        let message = serde_json::from_str::<JevErrorResponse>(body)
            .ok()
            .map(
                |error| match (error.detail.error_type, error.detail.message) {
                    (Some(kind), Some(message)) => format!("{kind}: {message}"),
                    (None, Some(message)) => message,
                    (Some(kind), None) => kind,
                    (None, None) => "unknown error".to_string(),
                },
            )
            .unwrap_or_else(|| body.chars().take(200).collect());

        return Err(VerifyError::Upstream { status, message });
    }

    let payload: JevResponse =
        serde_json::from_str(body).map_err(|error| VerifyError::Decode(error.to_string()))?;

    let score = payload
        .answers
        .get(QUESTION_KEY)
        .map(|answer| answer.noul)
        .ok_or_else(|| VerifyError::Decode(format!("response has no `{QUESTION_KEY}` answer")))?;

    if !(0.0..=1.0).contains(&score) {
        return Err(VerifyError::Decode(format!(
            "score {score} is outside 0..=1"
        )));
    }

    Ok(Verification::Checked {
        score,
        answers_question: score >= ANSWER_THRESHOLD,
        model: payload.model,
        checked_at: Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Payloads recorded from the live API.
    const ANSWERED: &str = r#"{"model":"jev-1.13.0","answers":{"answers_job_title":{"type":"noul","noul":0.98}},"usage":{"input_tokens":324,"output_tokens":22}}"#;
    const NOT_ANSWERED: &str = r#"{"model":"jev-1.13.0","answers":{"answers_job_title":{"type":"noul","noul":0.01}},"usage":{"input_tokens":319,"output_tokens":22}}"#;
    const BAD_KEY: &str = r#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}"#;

    #[test]
    fn high_score_answers_the_question() {
        match parse_response(200, ANSWERED).expect("parses") {
            Verification::Checked {
                score,
                answers_question,
                model,
                ..
            } => {
                assert_eq!(score, 0.98);
                assert!(answers_question);
                assert_eq!(model, "jev-1.13.0");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn low_score_does_not_answer_the_question() {
        assert!(matches!(
            parse_response(200, NOT_ANSWERED).expect("parses"),
            Verification::Checked {
                answers_question: false,
                ..
            }
        ));
    }

    #[test]
    fn auth_error_carries_the_message() {
        let error = parse_response(401, BAD_KEY).expect_err("should fail");
        let text = error.to_string();
        assert!(text.contains("401"), "{text}");
        assert!(text.contains("authentication_error"), "{text}");
    }

    #[test]
    fn missing_answer_key_is_a_decode_error() {
        let body = r#"{"model":"jev-1.13.0","answers":{}}"#;
        assert!(matches!(
            parse_response(200, body),
            Err(VerifyError::Decode(_))
        ));
    }

    #[test]
    fn flattens_hits_into_text() {
        let hits = vec![
            SearchHit {
                title: "Jane Doe - Acme".to_string(),
                link: "https://example.com".to_string(),
                snippet: "Jane is a Staff Engineer".to_string(),
            },
            SearchHit {
                title: String::new(),
                link: String::new(),
                snippet: "Second".to_string(),
            },
        ];

        assert_eq!(
            hits_to_text(&hits),
            "Jane Doe - Acme\nhttps://example.com\nJane is a Staff Engineer\n\nSecond"
        );
    }
}
