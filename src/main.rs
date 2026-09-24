use std::net::SocketAddr;
use std::sync::Arc;

use title_finder::ner::GlinerExtractor;
use title_finder::search::SerpApiClient;
use title_finder::verify::JevClient;
use title_finder::{AppState, app, store::Store};
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("title_finder=debug,tower_http=debug,info")),
        )
        .init();

    let api_key = std::env::var("SERP_API_KEY").map_err(|_| {
        "SERP_API_KEY is not set — get a key at https://serpapi.com and export it before starting"
    })?;

    let search = Arc::new(SerpApiClient::new(api_key)?);

    let jev_api_key = std::env::var("JEV_API_KEY")
        .map_err(|_| "JEV_API_KEY is not set — it is needed to check search results")?;
    let verifier = Arc::new(JevClient::new(jev_api_key)?);

    let request_log = match std::env::var("REQUEST_LOG") {
        Ok(path) => path,
        Err(_) => {
            migrate_legacy_request_log()?;
            DEFAULT_REQUEST_LOG.to_string()
        }
    };
    let store = Store::with_seed_data().with_request_log(&request_log)?;
    let gliner_dir = std::env::var("GLINER_MODEL_DIR")
        .unwrap_or_else(|_| "models/gliner_large-v2.1".to_string());
    tracing::info!("loading GLiNER model from {gliner_dir}");
    let extractor = Arc::new(GlinerExtractor::load(&gliner_dir)?);

    let state = AppState::new(store, search, verifier, extractor);

    //rebind for prod deploy
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|port| port.parse().ok())
        .unwrap_or(3000);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("listening on http://{addr}");

    axum::serve(listener, app(state)).await?;

    Ok(())
}

const DEFAULT_REQUEST_LOG: &str = "data/requests.jsonl";

/// Where the request log lived before it also held Jev and GLiNER entries.
const LEGACY_REQUEST_LOG: &str = "data/serp_requests.jsonl";

/// Moves a log at the old default path to the new one, once, so existing
/// history isn't silently left behind. Never overwrites an existing log.
fn migrate_legacy_request_log() -> std::io::Result<()> {
    let legacy = std::path::Path::new(LEGACY_REQUEST_LOG);
    let current = std::path::Path::new(DEFAULT_REQUEST_LOG);

    if legacy.is_file() && !current.exists() {
        std::fs::rename(legacy, current)?;
        tracing::info!("moved request log from {LEGACY_REQUEST_LOG} to {DEFAULT_REQUEST_LOG}");
    } else if legacy.is_file() {
        tracing::warn!(
            "both {LEGACY_REQUEST_LOG} and {DEFAULT_REQUEST_LOG} exist; using {DEFAULT_REQUEST_LOG} and leaving the old file alone"
        );
    }

    Ok(())
}
