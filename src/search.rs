use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use uuid::Uuid;

use crate::models::{SearchHit, SerpRequest};

/// The question we ask the search engine.
pub fn build_query(name: &str, company: &str) -> String {
    format!("What is the job title for {name} at {company}?")
}

#[derive(Debug)]
pub enum SearchError {
    /// The request never completed (DNS, TLS, timeout, connection refused).
    Transport(String),
    /// SerpAPI answered, but with an error (bad key, quota exhausted, ...).
    Upstream { status: u16, message: String },
    /// SerpAPI answered with a body we could not understand.
    Decode(String),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::Transport(message) => write!(f, "search request failed: {message}"),
            SearchError::Upstream { status, message } => {
                write!(f, "search provider returned {status}: {message}")
            }
            SearchError::Decode(message) => write!(f, "could not parse search response: {message}"),
        }
    }
}

impl std::error::Error for SearchError {}

/// One call to the search provider: what was sent and what came back. The
/// request is returned even on failure so it can always be saved.
#[derive(Debug)]
pub struct SearchAttempt {
    pub request: SerpRequest,
    /// HTTP status, when a response arrived at all.
    pub http_status: Option<u16>,
    pub result: Result<Vec<SearchHit>, SearchError>,
}

/// Abstracts the search provider so handlers don't depend on SerpAPI directly
/// and tests can substitute a stub instead of making network calls.
#[async_trait]
pub trait TitleSearch: Send + Sync {
    async fn search(&self, query: &str) -> SearchAttempt;
}

/// SerpAPI (<https://serpapi.com>) Google search client.
pub struct SerpApiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl SerpApiClient {
    pub const DEFAULT_BASE_URL: &'static str = "https://serpapi.com";
    const ENGINE: &'static str = "google";
    const NUM_RESULTS: u32 = 10;

    pub fn new(api_key: String) -> Result<Self, reqwest::Error> {
        Self::with_base_url(api_key, Self::DEFAULT_BASE_URL.to_string())
    }

    /// `base_url` is injectable so it can be pointed at a local stub server.
    pub fn with_base_url(api_key: String, base_url: String) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent(concat!("title-finder/", env!("CARGO_PKG_VERSION")))
            .build()?;

        Ok(Self {
            http,
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

/// The slice of the SerpAPI response we care about. Unknown fields are ignored,
/// so extra keys in their payload won't break us.
#[derive(Debug, Deserialize)]
struct SerpApiResponse {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    answer_box: Option<AnswerBox>,
    #[serde(default)]
    organic_results: Vec<OrganicResult>,
}

#[derive(Debug, Deserialize)]
struct AnswerBox {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    link: Option<String>,
    /// SerpAPI uses `answer`, `snippet` or `result` depending on the box type.
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    snippet: Option<String>,
    #[serde(default)]
    result: Option<String>,
}

impl AnswerBox {
    fn text(&self) -> Option<&str> {
        self.answer
            .as_deref()
            .or(self.snippet.as_deref())
            .or(self.result.as_deref())
    }
}

#[derive(Debug, Deserialize)]
struct OrganicResult {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    link: Option<String>,
    #[serde(default)]
    snippet: Option<String>,
}

#[async_trait]
impl TitleSearch for SerpApiClient {
    async fn search(&self, query: &str) -> SearchAttempt {
        let request = SerpRequest {
            id: Uuid::new_v4(),
            endpoint: format!("{}/search", self.base_url),
            engine: Self::ENGINE.to_string(),
            query: query.to_string(),
            num: Self::NUM_RESULTS,
            sent_at: Utc::now(),
        };

        let (http_status, result) = self.send(&request).await;

        SearchAttempt {
            request,
            http_status,
            result,
        }
    }
}

impl SerpApiClient {
    async fn send(
        &self,
        request: &SerpRequest,
    ) -> (Option<u16>, Result<Vec<SearchHit>, SearchError>) {
        let response = match self
            .http
            .get(&request.endpoint)
            .query(&[
                ("engine", request.engine.as_str()),
                ("q", request.query.as_str()),
                ("num", &request.num.to_string()),
                ("api_key", self.api_key.as_str()),
            ])
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => return (None, Err(SearchError::Transport(error.to_string()))),
        };

        let status = response.status();
        (Some(status.as_u16()), Self::parse(response).await)
    }

    async fn parse(response: reqwest::Response) -> Result<Vec<SearchHit>, SearchError> {
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| SearchError::Transport(error.to_string()))?;

        let payload: SerpApiResponse = serde_json::from_str(&body).map_err(|error| {
            if status.is_success() {
                SearchError::Decode(error.to_string())
            } else {
                // A non-JSON error page; surface the status rather than the parse failure.
                SearchError::Upstream {
                    status: status.as_u16(),
                    message: body.chars().take(200).collect(),
                }
            }
        })?;

        // SerpAPI reports quota and key problems in an `error` field, sometimes
        // alongside a 200.
        if let Some(message) = payload.error {
            return Err(SearchError::Upstream {
                status: status.as_u16(),
                message,
            });
        }

        if !status.is_success() {
            return Err(SearchError::Upstream {
                status: status.as_u16(),
                message: "search provider rejected the request".to_string(),
            });
        }

        let mut hits = Vec::with_capacity(payload.organic_results.len() + 1);

        // The answer box, when present, is the strongest signal — keep it first.
        if let Some(answer_box) = payload.answer_box
            && let Some(text) = answer_box.text()
        {
            hits.push(SearchHit {
                title: answer_box.title.clone().unwrap_or_default(),
                link: answer_box.link.clone().unwrap_or_default(),
                snippet: text.to_string(),
            });
        }

        hits.extend(payload.organic_results.into_iter().map(|result| SearchHit {
            title: result.title.unwrap_or_default(),
            link: result.link.unwrap_or_default(),
            snippet: result.snippet.unwrap_or_default(),
        }));

        Ok(hits)
    }
}
