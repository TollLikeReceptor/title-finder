//! Runs the real GLiNER model. Ignored by default because it needs the model
//! files (`./scripts/download-gliner.sh`); run with
//! `cargo test --test gliner_live -- --ignored`.

use title_finder::models::SearchHit;
use title_finder::ner::{GlinerExtractor, TitleExtractor};

fn extractor() -> GlinerExtractor {
    let dir = std::env::var("GLINER_MODEL_DIR")
        .unwrap_or_else(|_| "models/gliner_large-v2.1".to_string());
    GlinerExtractor::load(dir).expect("model loads — run ./scripts/download-gliner.sh")
}

fn hit(title: &str, snippet: &str) -> SearchHit {
    SearchHit {
        title: title.to_string(),
        link: "https://example.com".to_string(),
        snippet: snippet.to_string(),
    }
}

#[tokio::test]
#[ignore = "needs the GLiNER model files"]
async fn takes_the_most_confident_of_the_persons_titles() {
    let hits = vec![hit(
        "Satya Nadella - Microsoft",
        "Satya Nadella is the Chairman and Chief Executive Officer of Microsoft and joined the company in 1992.",
    )];

    let title = extractor()
        .title_for(&hits, "Satya Nadella")
        .await
        .result
        .expect("inference runs")
        .chosen
        .expect("a title is found");
    println!("{title:?}");

    // Chairman (0.72) comes first, but CEO (0.91) is more confident.
    assert_eq!(title.text, "Chief Executive Officer");
}

#[tokio::test]
#[ignore = "needs the GLiNER model files"]
async fn gives_each_person_their_own_title() {
    let hits = vec![hit(
        "Microsoft finance",
        "Former CFO Mark Chen left in 2021; Lisa Park now leads finance as Chief Financial Officer.",
    )];
    let extractor = extractor();

    let lisa = extractor
        .title_for(&hits, "Lisa Park")
        .await
        .result
        .expect("inference runs")
        .chosen;
    let mark = extractor
        .title_for(&hits, "Mark Chen")
        .await
        .result
        .expect("inference runs")
        .chosen;
    println!("Lisa: {lisa:?}\nMark: {mark:?}");

    assert_eq!(lisa.expect("Lisa's title").text, "Chief Financial Officer");

    // The full record keeps every span and whose title each one is.
    let attempt = extractor.title_for(&hits, "Lisa Park").await;
    let spans = attempt.result.expect("inference runs").spans;
    println!("{spans:#?}");
    let attribution_of = |text: &str| {
        spans
            .iter()
            .find(|span| span.text == text)
            .and_then(|span| span.attribution)
    };
    use title_finder::models::Attribution;
    assert_eq!(
        attribution_of("Chief Financial Officer"),
        Some(Attribution::SearchedPerson)
    );
    assert_eq!(attribution_of("Former CFO"), Some(Attribution::SomeoneElse));
    assert!(
        spans
            .iter()
            .any(|span| span.label == "person" && span.text == "Mark Chen")
    );
    assert_eq!(attempt.request.model, "gliner_large-v2.1");
    assert!(mark.expect("Mark's title").text.contains("CFO"));
}

#[tokio::test]
#[ignore = "needs the GLiNER model files"]
async fn does_not_take_someone_elses_title() {
    let hits = vec![hit(
        "Acme team",
        "Jane reports to John Smith, the VP of Sales at Acme.",
    )];

    let title = extractor()
        .title_for(&hits, "Jane Doe")
        .await
        .result
        .expect("inference runs")
        .chosen;
    println!("{title:?}");

    assert!(title.is_none(), "VP of Sales is John Smith's: {title:?}");
}

#[tokio::test]
#[ignore = "needs the GLiNER model files"]
async fn finds_nothing_in_unrelated_text() {
    let hits = vec![hit(
        "Acme Corp - About Us",
        "Acme Corp was founded in 1949 and makes anvils, rockets and giant magnets.",
    )];

    let title = extractor()
        .title_for(&hits, "Jane Doe")
        .await
        .result
        .expect("inference runs")
        .chosen;
    println!("{title:?}");

    assert!(title.is_none(), "{title:?}");
}

/// Full lookup through the router with the real Jev and real GLiNER; only
/// SerpAPI is stubbed, so no search credit is spent. Needs `JEV_API_KEY` too.
#[tokio::test]
#[ignore = "needs the GLiNER model files and JEV_API_KEY"]
async fn full_lookup_uses_gliner_when_jev_is_confident() {
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::Request;
    use chrono::Utc;
    use http_body_util::BodyExt;
    use title_finder::models::SerpRequest;
    use title_finder::search::{SearchAttempt, TitleSearch};
    use title_finder::store::Store;
    use title_finder::verify::JevClient;
    use title_finder::{AppState, app};
    use tower::ServiceExt;

    struct CannedSearch;

    #[async_trait]
    impl TitleSearch for CannedSearch {
        async fn search(&self, query: &str) -> SearchAttempt {
            SearchAttempt {
                request: SerpRequest {
                    id: uuid::Uuid::new_v4(),
                    endpoint: "https://serpapi.com/search".to_string(),
                    engine: "google".to_string(),
                    query: query.to_string(),
                    num: 10,
                    sent_at: Utc::now(),
                },
                http_status: Some(200),
                result: Ok(vec![hit(
                    "Satya Nadella - Microsoft",
                    "Satya Nadella is the Chairman and Chief Executive Officer of Microsoft and joined the company in 1992.",
                )]),
            }
        }
    }

    let jev = JevClient::new(std::env::var("JEV_API_KEY").expect("JEV_API_KEY")).unwrap();
    let log =
        std::env::temp_dir().join(format!("title-finder-live-{}.jsonl", uuid::Uuid::new_v4()));
    let state = AppState::new(
        Store::default().with_request_log(&log).expect("open log"),
        Arc::new(CannedSearch),
        Arc::new(jev),
        Arc::new(extractor()),
    );

    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/v1/titles/search?name=Satya%20Nadella&company=Microsoft")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    println!(
        "title={} title_source={} title_confidence={} jev_score={}",
        body["title"],
        body["title_source"],
        body["title_confidence"],
        body["verification"]["score"]
    );

    assert_eq!(body["title_source"], "gliner");
    assert!(body["verification"]["score"].as_f64().unwrap() > 0.70);
    assert_eq!(body["title"], "Chief Executive Officer");

    // The extraction is saved to disk and survives a reload.
    let saved = Store::default()
        .with_request_log(&log)
        .expect("reload log")
        .requests(None);
    let gliner = saved[0].gliner.as_ref().expect("GLiNER extraction saved");
    println!(
        "saved extraction: {}",
        serde_json::to_string_pretty(gliner).unwrap()
    );
    assert_eq!(gliner.serp_request_id, saved[0].request.id);
    assert_eq!(gliner.request.searched_name, "Satya Nadella");
    assert!(saved[0].jev.is_some());
    std::fs::remove_file(&log).ok();
}
