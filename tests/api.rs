use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::Utc;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use title_finder::models::{Attribution, GlinerOutcome, GlinerRequest, GlinerSpan};
use title_finder::models::{
    JevOutcome, JevRequest, RequestOutcome, SearchHit, SerpRequest, Verification,
};
use title_finder::ner::{
    ExtractError, ExtractedTitle, ExtractionAttempt, ExtractionOutput, TitleExtractor,
};
use title_finder::search::{SearchAttempt, SearchError, TitleSearch};
use title_finder::store::Store;
use title_finder::verify::{AnswerCheck, JevAttempt, VerifyError};
use title_finder::{AppState, app};
use tower::ServiceExt;
use uuid::Uuid;

/// A stand-in for SerpAPI: returns canned hits and counts how often it is hit,
/// so tests never touch the network.
struct StubSearch {
    hits: Vec<SearchHit>,
    error: Option<&'static str>,
    calls: AtomicUsize,
}

impl StubSearch {
    fn returning(snippet: &str) -> Self {
        Self {
            hits: vec![SearchHit {
                title: "Acme Corp Leadership".to_string(),
                link: "https://example.com/team".to_string(),
                snippet: snippet.to_string(),
            }],
            error: None,
            calls: AtomicUsize::new(0),
        }
    }

    fn empty() -> Self {
        Self {
            hits: Vec::new(),
            error: None,
            calls: AtomicUsize::new(0),
        }
    }

    fn failing(message: &'static str) -> Self {
        Self {
            hits: Vec::new(),
            error: Some(message),
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TitleSearch for StubSearch {
    async fn search(&self, query: &str) -> SearchAttempt {
        self.calls.fetch_add(1, Ordering::SeqCst);

        // Every lookup must ask the question the spec calls for.
        assert!(
            query.starts_with("What is the job title for ") && query.ends_with('?'),
            "unexpected query sent to search provider: {query}"
        );

        let request = SerpRequest {
            id: Uuid::new_v4(),
            endpoint: "https://serpapi.com/search".to_string(),
            engine: "google".to_string(),
            query: query.to_string(),
            num: 10,
            sent_at: Utc::now(),
        };

        match self.error {
            Some(message) => SearchAttempt {
                request,
                http_status: Some(401),
                result: Err(SearchError::Upstream {
                    status: 401,
                    message: message.to_string(),
                }),
            },
            None => SearchAttempt {
                request,
                http_status: Some(200),
                result: Ok(self.hits.clone()),
            },
        }
    }
}

/// A stand-in for Jev: returns a fixed score (or error) and records what it
/// was asked, so tests never touch the network.
struct StubVerifier {
    score: Option<f64>,
    calls: AtomicUsize,
    last_text: std::sync::Mutex<Option<String>>,
}

impl StubVerifier {
    fn scoring(score: f64) -> Self {
        Self {
            score: Some(score),
            calls: AtomicUsize::new(0),
            last_text: std::sync::Mutex::new(None),
        }
    }

    fn failing() -> Self {
        Self {
            score: None,
            calls: AtomicUsize::new(0),
            last_text: std::sync::Mutex::new(None),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl AnswerCheck for StubVerifier {
    async fn check(&self, text: &str, name: &str, company: &str) -> JevAttempt {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_text.lock().unwrap() = Some(text.to_string());

        let request = JevRequest {
            id: Uuid::new_v4(),
            endpoint: "https://api.typesafe.ai/v1/systemone".to_string(),
            model: "jev-latest".to_string(),
            question_key: "answers_job_title".to_string(),
            instructions: title_finder::verify::build_instructions(name, company),
            state: text.to_string(),
            sent_at: Utc::now(),
        };

        match self.score {
            Some(score) => JevAttempt {
                request,
                http_status: Some(200),
                result: Ok(Verification::Checked {
                    score,
                    answers_question: score >= title_finder::verify::ANSWER_THRESHOLD,
                    model: "jev-test".to_string(),
                    checked_at: Utc::now(),
                }),
            },
            None => JevAttempt {
                request,
                http_status: Some(401),
                result: Err(VerifyError::Upstream {
                    status: 401,
                    message: "authentication_error: bad key".to_string(),
                }),
            },
        }
    }
}

/// A stand-in for GLiNER: returns a fixed title (or nothing, or an error) and
/// records whose title it was asked for.
struct StubExtractor {
    title: Option<&'static str>,
    fail: bool,
    calls: AtomicUsize,
    last_name: std::sync::Mutex<Option<String>>,
}

impl StubExtractor {
    fn finding(title: &'static str) -> Self {
        Self {
            title: Some(title),
            fail: false,
            calls: AtomicUsize::new(0),
            last_name: std::sync::Mutex::new(None),
        }
    }

    fn finding_nothing() -> Self {
        Self {
            title: None,
            fail: false,
            calls: AtomicUsize::new(0),
            last_name: std::sync::Mutex::new(None),
        }
    }

    fn failing() -> Self {
        Self {
            title: None,
            fail: true,
            calls: AtomicUsize::new(0),
            last_name: std::sync::Mutex::new(None),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TitleExtractor for StubExtractor {
    async fn title_for(&self, hits: &[SearchHit], name: &str) -> ExtractionAttempt {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_name.lock().unwrap() = Some(name.to_string());

        let request = GlinerRequest {
            id: Uuid::new_v4(),
            model: "gliner-stub".to_string(),
            labels: vec!["job title".to_string(), "person".to_string()],
            threshold: 0.5,
            searched_name: name.to_string(),
            inputs: title_finder::ner::hits_to_sequences(hits),
            started_at: Utc::now(),
        };

        let result = if self.fail {
            Err(ExtractError("model exploded".to_string()))
        } else {
            // A person span plus, when finding something, a title attributed to them.
            let mut spans = vec![GlinerSpan {
                input: 0,
                start: 0,
                end: name.len(),
                text: name.to_string(),
                label: "person".to_string(),
                probability: 0.99,
                attribution: None,
            }];
            spans.extend(self.title.map(|text| GlinerSpan {
                input: 0,
                start: name.len() + 8,
                end: name.len() + 8 + text.len(),
                text: text.to_string(),
                label: "job title".to_string(),
                probability: 0.85,
                attribution: Some(Attribution::SearchedPerson),
            }));

            Ok(ExtractionOutput {
                spans,
                chosen: self.title.map(|text| ExtractedTitle {
                    text: text.to_string(),
                    probability: 0.85,
                }),
            })
        };

        ExtractionAttempt {
            request,
            duration_ms: 42,
            result,
        }
    }
}

fn state_with(search: Arc<StubSearch>) -> AppState {
    state_with_verifier(search, Arc::new(StubVerifier::scoring(0.9)))
}

fn state_with_verifier(search: Arc<StubSearch>, verifier: Arc<StubVerifier>) -> AppState {
    state_with_all(search, verifier, Arc::new(StubExtractor::finding_nothing()))
}

fn state_with_all(
    search: Arc<StubSearch>,
    verifier: Arc<StubVerifier>,
    extractor: Arc<StubExtractor>,
) -> AppState {
    AppState::new(Store::with_seed_data(), search, verifier, extractor)
}

async fn send_to(state: AppState, request: Request<Body>) -> (StatusCode, Value) {
    let response = app(state).oneshot(request).await.expect("request failed");

    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("failed to read body")
        .to_bytes();

    (
        status,
        serde_json::from_slice(&bytes).expect("invalid JSON"),
    )
}

async fn send(request: Request<Body>) -> (StatusCode, Value) {
    send_to(state_with(Arc::new(StubSearch::empty())), request).await
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().uri(uri).body(Body::empty()).unwrap()
}

#[tokio::test]
async fn health_reports_ok() {
    let (status, body) = send(get("/health")).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn finds_a_seeded_title_ignoring_case() {
    let search = Arc::new(StubSearch::empty());
    let (status, body) = send_to(
        state_with(search.clone()),
        get("/v1/titles/search?name=ada%20lovelace&company=ANALYTICAL%20ENGINE%20CO"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["title"], "Lead Programmer");
    assert_eq!(body["source"], "directory");
    // A curated record must not cost an API credit.
    assert_eq!(search.calls(), 0);
}

#[tokio::test]
async fn unknown_person_falls_back_to_search() {
    let search = Arc::new(StubSearch::returning(
        "Jane Doe is the Vice President of Engineering at Acme Corp and joined in 2019.",
    ));

    let state = state_with_all(
        search.clone(),
        Arc::new(StubVerifier::scoring(0.9)),
        Arc::new(StubExtractor::finding("Vice President of Engineering")),
    );
    let (status, body) = send_to(
        state,
        get("/v1/titles/search?name=Jane%20Doe&company=Acme%20Corp"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["source"], "serp_api");
    assert_eq!(body["title"], "Vice President of Engineering");
    assert_eq!(body["title_source"], "gliner");
    assert_eq!(
        body["query"],
        "What is the job title for Jane Doe at Acme Corp?"
    );
    assert_eq!(body["hits"][0]["link"], "https://example.com/team");
    assert_eq!(search.calls(), 1);
}

#[tokio::test]
async fn repeat_lookup_is_served_from_the_store() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with_all(
        search.clone(),
        Arc::new(StubVerifier::scoring(0.9)),
        Arc::new(StubExtractor::finding("Staff Engineer")),
    );

    let (first, _) = send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;
    assert_eq!(first, StatusCode::OK);

    // Different casing must hit the same stored entry.
    let (status, body) = send_to(
        state.clone(),
        get("/v1/titles/search?name=jane%20doe&company=ACME"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["source"], "serp_api_cached");
    assert_eq!(body["title"], "Staff Engineer");
    assert_eq!(
        search.calls(),
        1,
        "second lookup should not re-query SerpAPI"
    );
    assert_eq!(state.store.stored_search_count(), 1);
}

#[tokio::test]
async fn refresh_bypasses_the_stored_search() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with(search.clone());

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;

    let (status, body) = send_to(
        state,
        get("/v1/titles/search?name=Jane%20Doe&company=Acme&refresh=true"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["source"], "serp_api");
    assert_eq!(search.calls(), 2);
}

#[tokio::test]
async fn search_without_a_recognisable_title_returns_a_null_title() {
    let search = Arc::new(StubSearch::returning("No information is available here"));

    let (status, body) = send_to(
        state_with(search),
        get("/v1/titles/search?name=Nobody&company=Nowhere"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["title"].is_null());
    assert_eq!(body["title_source"], "not_found");
    assert_eq!(body["source"], "serp_api");
}

#[tokio::test]
async fn upstream_failure_is_a_502() {
    let search = Arc::new(StubSearch::failing("Invalid API key"));

    let (status, body) = send_to(
        state_with(search),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"], "upstream_error");
    assert!(
        body["message"]
            .as_str()
            .expect("message should be a string")
            .contains("Invalid API key")
    );
}

#[tokio::test]
async fn saves_the_serpapi_request() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with(search);

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;

    let requests = state.store.requests(None);
    assert_eq!(requests.len(), 1);

    let log = &requests[0];
    assert_eq!(log.request.engine, "google");
    assert_eq!(
        log.request.query,
        "What is the job title for Jane Doe at Acme?"
    );
    assert_eq!(log.http_status, Some(200));
    assert!(matches!(
        log.outcome,
        RequestOutcome::Succeeded { hit_count: 1 }
    ));

    // The stored search points back at the same request.
    let stored = state
        .store
        .stored_search("Jane Doe", "Acme")
        .expect("search should be stored");
    assert_eq!(stored.request.id, log.request.id);
}

#[tokio::test]
async fn saves_failed_serpapi_requests() {
    let search = Arc::new(StubSearch::failing("Invalid API key"));
    let state = state_with(search);

    let (status, _) = send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);

    let requests = state.store.requests(None);
    assert_eq!(requests.len(), 1, "a failed request must still be saved");
    assert_eq!(requests[0].http_status, Some(401));
    assert!(matches!(
        &requests[0].outcome,
        RequestOutcome::Failed { message } if message.contains("Invalid API key")
    ));

    // A failure must not be cached as an answer.
    assert_eq!(state.store.stored_search_count(), 0);
}

#[tokio::test]
async fn saved_request_json_never_contains_the_api_key() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with(search);

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;

    let json = serde_json::to_value(state.store.requests(None)).expect("serializable");
    assert!(json[0]["request"].get("api_key").is_none());
    assert_eq!(json[0]["outcome"]["status"], "succeeded");
}

#[tokio::test]
async fn cached_and_directory_lookups_send_no_request() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with(search);

    for uri in [
        "/v1/titles/search?name=Jane%20Doe&company=Acme",
        "/v1/titles/search?name=Jane%20Doe&company=Acme",
        "/v1/titles/search?name=Ada%20Lovelace&company=Analytical%20Engine%20Co",
    ] {
        send_to(state.clone(), get(uri)).await;
    }

    assert_eq!(state.store.requests(None).len(), 1);
}

#[tokio::test]
async fn lists_saved_requests_newest_first() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with(search);

    for uri in [
        "/v1/titles/search?name=First%20Person&company=Acme",
        "/v1/titles/search?name=Second%20Person&company=Acme",
    ] {
        send_to(state.clone(), get(uri)).await;
    }

    let (status, body) = send_to(state.clone(), get("/v1/requests")).await;
    assert_eq!(status, StatusCode::OK);

    let list = body.as_array().expect("expected an array");
    assert_eq!(list.len(), 2);
    assert_eq!(
        list[0]["request"]["query"],
        "What is the job title for Second Person at Acme?"
    );
    assert_eq!(list[0]["request"]["engine"], "google");
    assert_eq!(list[0]["http_status"], 200);
    assert_eq!(list[0]["outcome"]["status"], "succeeded");
    assert_eq!(list[0]["outcome"]["hit_count"], 1);

    let (_, limited) = send_to(state, get("/v1/requests?limit=1")).await;
    assert_eq!(limited.as_array().expect("expected an array").len(), 1);
}

#[tokio::test]
async fn fetches_one_saved_request_by_id() {
    let search = Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer"));
    let state = state_with(search);

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;
    let id = state.store.requests(None)[0].request.id;

    let (status, body) = send_to(state.clone(), get(&format!("/v1/requests/{id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["request"]["id"], id.to_string());

    let (missing, body) = send_to(
        state.clone(),
        get(&format!("/v1/requests/{}", Uuid::new_v4())),
    )
    .await;
    assert_eq!(missing, StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "not_found");

    let (bad_id, body) = send_to(state, get("/v1/requests/not-a-uuid")).await;
    assert_eq!(bad_id, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "bad_request");
}

#[tokio::test]
async fn saved_requests_survive_a_restart() {
    let path = std::env::temp_dir().join(format!("title-finder-{}.jsonl", Uuid::new_v4()));

    // First "run": one good request, one failed one.
    {
        let store = Store::with_seed_data()
            .with_request_log(&path)
            .expect("open log");

        send_to(
            AppState::new(
                store.clone(),
                Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
                Arc::new(StubVerifier::scoring(0.9)),
                Arc::new(StubExtractor::finding_nothing()),
            ),
            get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
        )
        .await;
        send_to(
            AppState::new(
                store,
                Arc::new(StubSearch::failing("Invalid API key")),
                Arc::new(StubVerifier::scoring(0.9)),
                Arc::new(StubExtractor::finding_nothing()),
            ),
            get("/v1/titles/search?name=Sam%20Lee&company=Acme"),
        )
        .await;
    }

    // Second "run": a fresh store reading the same file.
    let reloaded = Store::with_seed_data()
        .with_request_log(&path)
        .expect("reopen log");
    let saved = reloaded.requests(None);

    assert_eq!(saved.len(), 2);
    assert_eq!(
        saved[0].request.query,
        "What is the job title for Sam Lee at Acme?"
    );
    assert!(matches!(saved[0].outcome, RequestOutcome::Failed { .. }));
    assert!(matches!(
        saved[1].outcome,
        RequestOutcome::Succeeded { hit_count: 1 }
    ));

    let raw = std::fs::read_to_string(&path).expect("log file");
    // SerpAPI success + its Jev check + its GLiNER extraction + SerpAPI
    // failure (which neither Jev nor GLiNER ever sees).
    assert_eq!(raw.lines().count(), 4, "one JSON line per request");
    assert!(!raw.contains("api_key"));

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn corrupt_request_log_is_reported_at_startup() {
    let path = std::env::temp_dir().join(format!("title-finder-{}.jsonl", Uuid::new_v4()));
    std::fs::write(&path, "{not json}\n").expect("write");

    let error = Store::with_seed_data()
        .with_request_log(&path)
        .err()
        .expect("should fail to load");
    assert!(
        error.to_string().contains(":1:"),
        "names the bad line: {error}"
    );

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn search_results_are_checked_by_jev() {
    let verifier = Arc::new(StubVerifier::scoring(0.98));
    let state = state_with_verifier(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        verifier.clone(),
    );

    let (status, body) =
        send_to(state, get("/v1/titles/search?name=Jane%20Doe&company=Acme")).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verification"]["status"], "checked");
    assert_eq!(body["verification"]["score"], 0.98);
    assert_eq!(body["verification"]["answers_question"], true);
    assert_eq!(verifier.calls(), 1);

    // Jev reads the search hits, not the extracted title.
    let text = verifier
        .last_text
        .lock()
        .unwrap()
        .clone()
        .expect("text sent");
    assert!(text.contains("https://example.com/team"), "{text}");
    assert!(text.contains("Jane Doe is a Staff Engineer"), "{text}");
}

#[tokio::test]
async fn low_jev_score_marks_the_answer_as_unconfirmed() {
    let state = state_with_verifier(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        Arc::new(StubVerifier::scoring(0.01)),
    );

    let (_, body) = send_to(state, get("/v1/titles/search?name=Jane%20Doe&company=Acme")).await;

    assert_eq!(body["verification"]["answers_question"], false);
    // Results Jev doesn't trust aren't mined for a title.
    assert!(body["title"].is_null());
    assert_eq!(body["title_source"], "unverified");
}

#[tokio::test]
async fn jev_failure_does_not_fail_the_lookup() {
    let state = state_with_verifier(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        Arc::new(StubVerifier::failing()),
    );

    let (status, body) =
        send_to(state, get("/v1/titles/search?name=Jane%20Doe&company=Acme")).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["title"].is_null());
    assert_eq!(body["title_source"], "unverified");
    assert_eq!(body["verification"]["status"], "failed");
    assert!(
        body["verification"]["message"]
            .as_str()
            .unwrap()
            .contains("authentication_error")
    );
}

#[tokio::test]
async fn empty_search_skips_jev() {
    let verifier = Arc::new(StubVerifier::scoring(0.9));
    let state = state_with_verifier(Arc::new(StubSearch::empty()), verifier.clone());

    let (_, body) = send_to(state, get("/v1/titles/search?name=Nobody&company=Nowhere")).await;

    assert_eq!(body["verification"]["status"], "skipped");
    assert_eq!(verifier.calls(), 0);
}

#[tokio::test]
async fn cached_lookup_reuses_the_stored_verification() {
    let verifier = Arc::new(StubVerifier::scoring(0.98));
    let state = state_with_verifier(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        verifier.clone(),
    );

    for _ in 0..2 {
        send_to(
            state.clone(),
            get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
        )
        .await;
    }

    let (_, body) = send_to(state, get("/v1/titles/search?name=Jane%20Doe&company=Acme")).await;

    assert_eq!(body["source"], "serp_api_cached");
    assert_eq!(body["verification"]["score"], 0.98);
    assert_eq!(verifier.calls(), 1, "Jev should only be asked once");
}

#[tokio::test]
async fn directory_hits_are_not_sent_to_jev() {
    let verifier = Arc::new(StubVerifier::scoring(0.9));
    let state = state_with_verifier(Arc::new(StubSearch::empty()), verifier.clone());

    let (_, body) = send_to(
        state,
        get("/v1/titles/search?name=Ada%20Lovelace&company=Analytical%20Engine%20Co"),
    )
    .await;

    assert!(body.get("verification").is_none());
    assert_eq!(verifier.calls(), 0);
}

#[tokio::test]
async fn jev_request_is_saved_with_its_serp_request() {
    let state = state_with_verifier(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        Arc::new(StubVerifier::scoring(0.98)),
    );

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;

    let (status, body) = send_to(state.clone(), get("/v1/requests")).await;
    assert_eq!(status, StatusCode::OK);

    let entry = &body[0];
    let jev = &entry["jev"];
    assert_eq!(jev["serp_request_id"], entry["request"]["id"]);
    assert_eq!(jev["request"]["model"], "jev-latest");
    assert_eq!(
        jev["request"]["instructions"],
        "Does this text answer the question What is the job title of Jane Doe at Acme?"
    );
    // The exact text Jev judged, for comparison with the SerpAPI hits.
    assert!(
        jev["request"]["state"]
            .as_str()
            .unwrap()
            .contains("Jane Doe is a Staff Engineer")
    );
    assert_eq!(jev["http_status"], 200);
    assert_eq!(jev["outcome"]["status"], "succeeded");
    assert_eq!(jev["outcome"]["score"], 0.98);

    // Also reachable through the single-request endpoint.
    let id = entry["request"]["id"].as_str().unwrap();
    let (_, one) = send_to(state, get(&format!("/v1/requests/{id}"))).await;
    assert_eq!(one["jev"]["outcome"]["score"], 0.98);
}

#[tokio::test]
async fn failed_jev_request_is_saved() {
    let state = state_with_verifier(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        Arc::new(StubVerifier::failing()),
    );

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
    )
    .await;

    let jev = state.store.requests(None)[0]
        .jev
        .clone()
        .expect("failed Jev call must still be saved");
    assert_eq!(jev.http_status, Some(401));
    assert!(matches!(
        jev.outcome,
        JevOutcome::Failed { ref message } if message.contains("authentication_error")
    ));
}

#[tokio::test]
async fn no_jev_entry_when_jev_is_not_called() {
    let state = state_with_verifier(
        Arc::new(StubSearch::empty()),
        Arc::new(StubVerifier::scoring(0.9)),
    );

    send_to(
        state.clone(),
        get("/v1/titles/search?name=Nobody&company=Nowhere"),
    )
    .await;

    let (_, body) = send_to(state, get("/v1/requests")).await;
    assert_eq!(body.as_array().unwrap().len(), 1);
    assert!(body[0].get("jev").is_none());
}

#[tokio::test]
async fn jev_requests_survive_a_restart_and_old_logs_still_load() {
    let path = std::env::temp_dir().join(format!("title-finder-{}.jsonl", Uuid::new_v4()));

    // A line in the pre-Jev format, as written by earlier versions.
    std::fs::write(
        &path,
        concat!(
            r#"{"request":{"id":"6f1c2c1e-1111-4a4a-9b9b-000000000001","endpoint":"https://serpapi.com/search","engine":"google","query":"What is the job title for Old Entry at Acme?","num":10,"sent_at":"2026-09-21T10:00:00Z"},"http_status":200,"outcome":{"status":"succeeded","hit_count":3}}"#,
            "\n"
        ),
    )
    .expect("seed log");

    {
        let store = Store::with_seed_data()
            .with_request_log(&path)
            .expect("open log");
        send_to(
            AppState::new(
                store,
                Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
                Arc::new(StubVerifier::scoring(0.97)),
                Arc::new(StubExtractor::finding_nothing()),
            ),
            get("/v1/titles/search?name=Jane%20Doe&company=Acme"),
        )
        .await;
    }

    let raw = std::fs::read_to_string(&path).expect("log file");
    assert_eq!(
        raw.lines().count(),
        4,
        "old line + SerpAPI line + Jev line + GLiNER line"
    );

    let saved = Store::with_seed_data()
        .with_request_log(&path)
        .expect("reopen log")
        .requests(None);

    assert_eq!(
        saved.len(),
        2,
        "Jev lines nest rather than becoming entries"
    );
    let jev = saved[0].jev.as_ref().expect("Jev nested under its search");
    assert_eq!(jev.serp_request_id, saved[0].request.id);
    assert!(matches!(jev.outcome, JevOutcome::Succeeded { .. }));
    assert!(saved[1].jev.is_none(), "old entry has no Jev check");

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn orphaned_jev_line_is_reported_at_startup() {
    let path = std::env::temp_dir().join(format!("title-finder-{}.jsonl", Uuid::new_v4()));
    std::fs::write(
        &path,
        concat!(
            r#"{"serp_request_id":"6f1c2c1e-1111-4a4a-9b9b-00000000dead","request":{"id":"6f1c2c1e-2222-4a4a-9b9b-000000000002","endpoint":"https://api.typesafe.ai/v1/systemone","model":"jev-latest","question_key":"answers_job_title","instructions":"x","state":"x","sent_at":"2026-09-21T10:00:00Z"},"http_status":200,"outcome":{"status":"succeeded","score":0.9,"answers_question":true,"model":"jev-1.13.0"}}"#,
            "\n"
        ),
    )
    .expect("seed log");

    let error = Store::with_seed_data()
        .with_request_log(&path)
        .err()
        .expect("should fail to load");
    assert!(
        error.to_string().contains("unknown SerpAPI request"),
        "{error}"
    );

    std::fs::remove_file(&path).ok();
}

const JANE: &str = "/v1/titles/search?name=Jane%20Doe&company=Acme";

fn jane_state(score: f64, extractor: Arc<StubExtractor>) -> AppState {
    state_with_all(
        Arc::new(StubSearch::returning(
            "Jane Doe is the Vice President of Engineering at Acme Corp and joined in 2019.",
        )),
        Arc::new(StubVerifier::scoring(score)),
        extractor,
    )
}

#[tokio::test]
async fn gliner_extracts_the_title_when_jev_is_confident() {
    let extractor = Arc::new(StubExtractor::finding("Vice President of Engineering"));
    let (status, body) = send_to(jane_state(0.98, extractor.clone()), get(JANE)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["title"], "Vice President of Engineering");
    assert_eq!(body["title_source"], "gliner");
    assert_eq!(body["title_confidence"], 0.85);
    assert_eq!(extractor.calls(), 1);
    assert_eq!(
        extractor.last_name.lock().unwrap().as_deref(),
        Some("Jane Doe"),
        "GLiNER must be told whose title to find"
    );
}

#[tokio::test]
async fn gliner_needs_a_score_strictly_above_seventy_percent() {
    let extractor = Arc::new(StubExtractor::finding("Vice President of Engineering"));
    let (_, body) = send_to(jane_state(0.70, extractor.clone()), get(JANE)).await;

    assert_eq!(body["title_source"], "unverified");
    assert!(body["title"].is_null());
    assert!(body.get("title_confidence").is_none());
    assert_eq!(extractor.calls(), 0, "0.70 is not above the threshold");
}

#[tokio::test]
async fn low_jev_score_returns_no_title() {
    let extractor = Arc::new(StubExtractor::finding("Vice President of Engineering"));
    let (_, body) = send_to(jane_state(0.01, extractor.clone()), get(JANE)).await;

    assert_eq!(body["title_source"], "unverified");
    assert!(body["title"].is_null());
    assert_eq!(extractor.calls(), 0);
}

#[tokio::test]
async fn gliner_finding_nothing_returns_no_title() {
    let extractor = Arc::new(StubExtractor::finding_nothing());
    let (_, body) = send_to(jane_state(0.98, extractor.clone()), get(JANE)).await;

    assert_eq!(body["title_source"], "not_found");
    assert!(body["title"].is_null());
    assert_eq!(extractor.calls(), 1);
}

#[tokio::test]
async fn gliner_failure_returns_no_title() {
    let (status, body) = send_to(
        jane_state(0.98, Arc::new(StubExtractor::failing())),
        get(JANE),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["title_source"], "extraction_failed");
    assert!(body["title"].is_null());
}

#[tokio::test]
async fn failed_jev_check_skips_gliner() {
    let extractor = Arc::new(StubExtractor::finding("Vice President of Engineering"));
    let state = state_with_all(
        Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
        Arc::new(StubVerifier::failing()),
        extractor.clone(),
    );

    let (_, body) = send_to(state, get(JANE)).await;

    assert_eq!(body["title_source"], "unverified");
    assert!(body["title"].is_null());
    assert_eq!(extractor.calls(), 0);
}

#[tokio::test]
async fn cached_lookup_keeps_the_gliner_title_without_rerunning_it() {
    let extractor = Arc::new(StubExtractor::finding("Vice President of Engineering"));
    let state = jane_state(0.98, extractor.clone());

    send_to(state.clone(), get(JANE)).await;
    let (_, body) = send_to(state, get(JANE)).await;

    assert_eq!(body["source"], "serp_api_cached");
    assert_eq!(body["title_source"], "gliner");
    assert_eq!(body["title"], "Vice President of Engineering");
    assert_eq!(extractor.calls(), 1);
}

#[tokio::test]
async fn directory_titles_are_labelled_as_such() {
    let (_, body) = send(get(
        "/v1/titles/search?name=Ada%20Lovelace&company=Analytical%20Engine%20Co",
    ))
    .await;

    assert_eq!(body["title_source"], "directory");
}

#[tokio::test]
async fn gliner_extraction_is_saved_with_its_serp_request() {
    let state = jane_state(
        0.98,
        Arc::new(StubExtractor::finding("Vice President of Engineering")),
    );
    send_to(state.clone(), get(JANE)).await;

    let (status, body) = send_to(state.clone(), get("/v1/requests")).await;
    assert_eq!(status, StatusCode::OK);

    let entry = &body[0];
    let gliner = &entry["gliner"];
    assert_eq!(gliner["serp_request_id"], entry["request"]["id"]);
    assert_eq!(gliner["request"]["searched_name"], "Jane Doe");
    assert_eq!(gliner["request"]["labels"], json!(["job title", "person"]));
    assert_eq!(gliner["request"]["threshold"], 0.5);
    // The exact text GLiNER read, for comparison with the SerpAPI hits.
    assert!(
        gliner["request"]["inputs"][0]
            .as_str()
            .unwrap()
            .contains("Vice President of Engineering at Acme Corp")
    );
    assert_eq!(gliner["duration_ms"], 42);
    assert_eq!(gliner["outcome"]["status"], "succeeded");
    assert_eq!(
        gliner["outcome"]["chosen"]["text"],
        "Vice President of Engineering"
    );

    let spans = gliner["outcome"]["spans"].as_array().unwrap();
    assert_eq!(spans.len(), 2, "person and title spans are both kept");
    let title_span = spans
        .iter()
        .find(|span| span["label"] == "job title")
        .unwrap();
    assert_eq!(title_span["attribution"], "searched_person");
    let person_span = spans.iter().find(|span| span["label"] == "person").unwrap();
    assert!(person_span.get("attribution").is_none());

    // The Jev check sits alongside it on the same entry.
    assert_eq!(entry["jev"]["serp_request_id"], entry["request"]["id"]);

    // And the single-request endpoint returns it too.
    let id = entry["request"]["id"].as_str().unwrap();
    let (_, one) = send_to(state, get(&format!("/v1/requests/{id}"))).await;
    assert_eq!(
        one["gliner"]["outcome"]["chosen"]["text"],
        "Vice President of Engineering"
    );
}

#[tokio::test]
async fn gliner_finding_nothing_is_saved_with_no_choice() {
    let state = jane_state(0.98, Arc::new(StubExtractor::finding_nothing()));
    send_to(state.clone(), get(JANE)).await;

    let gliner = state.store.requests(None)[0]
        .gliner
        .clone()
        .expect("extraction saved");
    assert!(matches!(
        gliner.outcome,
        GlinerOutcome::Succeeded { chosen: None, .. }
    ));
}

#[tokio::test]
async fn failed_gliner_extraction_is_saved() {
    let state = jane_state(0.98, Arc::new(StubExtractor::failing()));
    send_to(state.clone(), get(JANE)).await;

    let gliner = state.store.requests(None)[0]
        .gliner
        .clone()
        .expect("a failed extraction must still be saved");
    assert!(matches!(
        gliner.outcome,
        GlinerOutcome::Failed { ref message } if message.contains("model exploded")
    ));
    assert_eq!(gliner.request.searched_name, "Jane Doe");
}

#[tokio::test]
async fn no_gliner_entry_when_gliner_does_not_run() {
    // Jev at exactly 0.70 is not confident enough to run GLiNER.
    let state = jane_state(
        0.70,
        Arc::new(StubExtractor::finding("Vice President of Engineering")),
    );
    send_to(state.clone(), get(JANE)).await;
    // A cached repeat runs nothing either.
    send_to(state.clone(), get(JANE)).await;

    let (_, body) = send_to(state, get("/v1/requests")).await;
    assert_eq!(body.as_array().unwrap().len(), 1);
    assert!(body[0].get("gliner").is_none());
    assert!(body[0].get("jev").is_some());
}

#[tokio::test]
async fn gliner_extractions_survive_a_restart() {
    let path = std::env::temp_dir().join(format!("title-finder-{}.jsonl", Uuid::new_v4()));

    {
        let store = Store::with_seed_data()
            .with_request_log(&path)
            .expect("open log");
        send_to(
            AppState::new(
                store,
                Arc::new(StubSearch::returning("Jane Doe is a Staff Engineer")),
                Arc::new(StubVerifier::scoring(0.97)),
                Arc::new(StubExtractor::finding("Staff Engineer")),
            ),
            get(JANE),
        )
        .await;
    }

    let raw = std::fs::read_to_string(&path).expect("log file");
    assert_eq!(raw.lines().count(), 3, "SerpAPI + Jev + GLiNER lines");

    let saved = Store::with_seed_data()
        .with_request_log(&path)
        .expect("reopen log")
        .requests(None);

    assert_eq!(
        saved.len(),
        1,
        "Jev and GLiNER lines nest under their search"
    );
    assert!(saved[0].jev.is_some());
    let gliner = saved[0]
        .gliner
        .as_ref()
        .expect("GLiNER nested under its search");
    assert_eq!(gliner.serp_request_id, saved[0].request.id);
    assert!(matches!(
        &gliner.outcome,
        GlinerOutcome::Succeeded { chosen: Some(title), spans }
            if title.text == "Staff Engineer" && spans.len() == 2
    ));

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn orphaned_gliner_line_is_reported_at_startup() {
    let path = std::env::temp_dir().join(format!("title-finder-{}.jsonl", Uuid::new_v4()));
    std::fs::write(
        &path,
        concat!(
            r#"{"serp_request_id":"6f1c2c1e-1111-4a4a-9b9b-00000000dead","request":{"id":"6f1c2c1e-3333-4a4a-9b9b-000000000003","model":"gliner_large-v2.1","labels":["job title","person"],"threshold":0.5,"searched_name":"Jane Doe","inputs":["x"],"started_at":"2026-09-21T10:00:00Z"},"duration_ms":500,"outcome":{"status":"succeeded","spans":[],"chosen":null}}"#,
            "\n"
        ),
    )
    .expect("seed log");

    let error = Store::with_seed_data()
        .with_request_log(&path)
        .err()
        .expect("should fail to load");
    assert!(error.to_string().contains("GLiNER extraction"), "{error}");

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn blank_query_is_a_400() {
    let search = Arc::new(StubSearch::empty());
    let (status, body) = send_to(
        state_with(search.clone()),
        get("/v1/titles/search?name=%20&company=Acme"),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "bad_request");
    assert_eq!(search.calls(), 0);
}

#[tokio::test]
async fn creates_a_title() {
    let payload = json!({
        "name": "Alan Turing",
        "company": "NPL",
        "title": "Principal Scientist"
    });

    let (status, body) = send(
        Request::builder()
            .method("POST")
            .uri("/v1/titles")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["title"], "Principal Scientist");
    assert!(body["id"].is_string());
}

#[tokio::test]
async fn lists_seeded_titles() {
    let (status, body) = send(get("/v1/titles")).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().expect("expected an array").len(), 2);
}
