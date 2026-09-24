pub mod error;
pub mod handlers;
pub mod models;
pub mod ner;
pub mod search;
pub mod store;
pub mod verify;

use std::sync::Arc;

use axum::Router;
use axum::routing::get;
use tower_http::trace::TraceLayer;

use crate::ner::TitleExtractor;
use crate::search::TitleSearch;
use crate::store::Store;
use crate::verify::AnswerCheck;

#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub search: Arc<dyn TitleSearch>,
    pub verifier: Arc<dyn AnswerCheck>,
    pub extractor: Arc<dyn TitleExtractor>,
}

impl AppState {
    pub fn new(
        store: Store,
        search: Arc<dyn TitleSearch>,
        verifier: Arc<dyn AnswerCheck>,
        extractor: Arc<dyn TitleExtractor>,
    ) -> Self {
        Self {
            store,
            search,
            verifier,
            extractor,
        }
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route(
            "/v1/titles",
            get(handlers::list_titles).post(handlers::create_title),
        )
        .route("/v1/titles/search", get(handlers::find_title))
        .route("/v1/requests", get(handlers::list_requests))
        .route("/v1/requests/{id}", get(handlers::get_request))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
