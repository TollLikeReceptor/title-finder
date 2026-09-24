//! Calls the real Jev API. Ignored by default because it needs `JEV_API_KEY`
//! and spends credits; run with `cargo test --test jev_live -- --ignored`.

use title_finder::models::{SearchHit, Verification};
use title_finder::verify::{AnswerCheck, JevClient, hits_to_text};

fn client() -> JevClient {
    let key = std::env::var("JEV_API_KEY").expect("JEV_API_KEY must be set for live tests");
    JevClient::new(key).expect("client builds")
}

fn hit(title: &str, snippet: &str) -> Vec<SearchHit> {
    vec![SearchHit {
        title: title.to_string(),
        link: "https://example.com".to_string(),
        snippet: snippet.to_string(),
    }]
}

#[tokio::test]
#[ignore = "calls the live Jev API"]
async fn jev_confirms_an_answering_snippet() {
    let text = hits_to_text(&hit(
        "Satya Nadella - Microsoft",
        "Satya Nadella is the Chairman and Chief Executive Officer of Microsoft.",
    ));

    let verification = client()
        .check(&text, "Satya Nadella", "Microsoft")
        .await
        .result
        .expect("Jev call succeeds");
    println!("{verification:?}");

    assert!(matches!(
        verification,
        Verification::Checked {
            answers_question: true,
            ..
        }
    ));
}

#[tokio::test]
#[ignore = "calls the live Jev API"]
async fn jev_rejects_an_unrelated_snippet() {
    let text = hits_to_text(&hit(
        "Acme Corp - About Us",
        "Acme Corp was founded in 1949 and makes anvils, rockets and giant magnets.",
    ));

    let verification = client()
        .check(&text, "Jane Doe", "Acme Corp")
        .await
        .result
        .expect("Jev call succeeds");
    println!("{verification:?}");

    assert!(matches!(
        verification,
        Verification::Checked {
            answers_question: false,
            ..
        }
    ));
}

#[tokio::test]
#[ignore = "calls the live Jev API"]
async fn jev_rejects_a_bad_key() {
    let error = JevClient::new("invalid".to_string())
        .unwrap()
        .check("x", "Jane Doe", "Acme")
        .await
        .result
        .expect_err("bad key must fail");
    println!("{error}");

    assert!(error.to_string().contains("401"));
}
