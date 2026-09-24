use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;

use crate::AppState;
use crate::error::ApiError;
use crate::models::{
    Health, NewTitle, RequestListQuery, SearchHit, SerpRequestLog, Source, TitleLookup, TitleQuery,
    TitleRecord, TitleSource, Verification,
};
use crate::ner::MIN_VERIFICATION_SCORE;
use crate::search::build_query;
use crate::store::ChosenTitle;
use crate::verify::hits_to_text;
use uuid::Uuid;

pub async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

//Check cached searches first, this will need to get updated to a db for prod
pub async fn find_title(
    State(state): State<AppState>,
    Query(query): Query<TitleQuery>,
) -> Result<Json<TitleLookup>, ApiError> {
    let name = query.name.trim();
    let company = query.company.trim();

    if name.is_empty() || company.is_empty() {
        return Err(ApiError::BadRequest(
            "`name` and `company` must not be empty".to_string(),
        ));
    }

    if let Some(record) = state.store.find(name, company) {
        return Ok(Json(TitleLookup {
            name: record.name,
            company: record.company,
            title: Some(record.title),
            title_source: TitleSource::Directory,
            title_confidence: None,
            source: Source::Directory,
            query: None,
            hits: Vec::new(),
            verification: None,
            retrieved_at: chrono::Utc::now(),
        }));
    }

    if !query.refresh
        && let Some(stored) = state.store.stored_search(name, company)
    {
        tracing::debug!("serving stored search for {name} at {company}");

        return Ok(Json(TitleLookup {
            name: name.to_string(),
            company: company.to_string(),
            title: stored.title,
            title_source: stored.title_source,
            title_confidence: stored.title_confidence,
            source: Source::SerpApiCached,
            query: Some(stored.request.query),
            hits: stored.hits,
            verification: Some(stored.verification),
            retrieved_at: stored.retrieved_at,
        }));
    }

    let search_query = build_query(name, company);
    tracing::info!("searching SerpAPI: {search_query}");

    let attempt = state.search.search(&search_query).await;

    if let Err(error) = state.store.record_request(&attempt).await {
        tracing::error!(
            "could not save SerpAPI request {} to disk: {error}",
            attempt.request.id
        );
    }

    let hits = attempt.result?;
    let verification = verify_hits(&state, attempt.request.id, &hits, name, company).await;
    let title = choose_title(&state, attempt.request.id, &hits, &verification, name).await;

    let stored = state
        .store
        .save_search(name, company, attempt.request, title, hits, verification);

    Ok(Json(TitleLookup {
        name: name.to_string(),
        company: company.to_string(),
        title: stored.title,
        title_source: stored.title_source,
        title_confidence: stored.title_confidence,
        source: Source::SerpApi,
        query: Some(stored.request.query),
        hits: stored.hits,
        verification: Some(stored.verification),
        retrieved_at: stored.retrieved_at,
    }))
}

async fn choose_title(
    state: &AppState,
    serp_request_id: Uuid,
    hits: &[SearchHit],
    verification: &Verification,
    name: &str,
) -> ChosenTitle {
    let no_title = |source| ChosenTitle {
        text: None,
        source,
        confidence: None,
    };

    let Verification::Checked { score, .. } = verification else {
        return no_title(TitleSource::Unverified);
    };

    if *score <= MIN_VERIFICATION_SCORE {
        return no_title(TitleSource::Unverified);
    }

    let attempt = state.extractor.title_for(hits, name).await;

    if let Err(error) = state
        .store
        .record_gliner_request(serp_request_id, &attempt)
        .await
    {
        tracing::error!(
            "could not save GLiNER extraction {} to disk: {error}",
            attempt.request.id
        );
    }

    match attempt.result.map(|output| output.chosen) {
        Ok(Some(title)) => ChosenTitle {
            text: Some(title.text),
            source: TitleSource::Gliner,
            confidence: Some(title.probability),
        },
        Ok(None) => {
            tracing::debug!("GLiNER found no job title for {name}");
            no_title(TitleSource::NotFound)
        }
        Err(error) => {
            tracing::error!("{error}");
            no_title(TitleSource::ExtractionFailed)
        }
    }
}

async fn verify_hits(
    state: &AppState,
    serp_request_id: Uuid,
    hits: &[SearchHit],
    name: &str,
    company: &str,
) -> Verification {
    let text = hits_to_text(hits);

    if text.is_empty() {
        return Verification::Skipped {
            reason: "search returned no text to check".to_string(),
        };
    }

    let attempt = state.verifier.check(&text, name, company).await;

    if let Err(error) = state
        .store
        .record_jev_request(serp_request_id, &attempt)
        .await
    {
        tracing::error!(
            "could not save Jev request {} to disk: {error}",
            attempt.request.id
        );
    }

    match attempt.result {
        Ok(verification) => verification,
        Err(error) => {
            tracing::error!("Jev check failed for {name} at {company}: {error}");
            Verification::Failed {
                message: error.to_string(),
            }
        }
    }
}

/// `GET /v1/requests?limit=...` — saved SerpAPI requests, newest first.
pub async fn list_requests(
    State(state): State<AppState>,
    Query(query): Query<RequestListQuery>,
) -> Json<Vec<SerpRequestLog>> {
    Json(state.store.requests(query.limit))
}

/// `GET /v1/requests/{id}`
pub async fn get_request(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SerpRequestLog>, ApiError> {
    let id = Uuid::parse_str(&id)
        .map_err(|_| ApiError::BadRequest(format!("`{id}` is not a valid request id")))?;

    state
        .store
        .request(id)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("no saved request with id {id}")))
}

/// `GET /v1/titles`
pub async fn list_titles(State(state): State<AppState>) -> Json<Vec<TitleRecord>> {
    Json(state.store.list())
}

/// `POST /v1/titles`
pub async fn create_title(
    State(state): State<AppState>,
    Json(payload): Json<NewTitle>,
) -> Result<(StatusCode, Json<TitleRecord>), ApiError> {
    if payload.name.trim().is_empty()
        || payload.company.trim().is_empty()
        || payload.title.trim().is_empty()
    {
        return Err(ApiError::BadRequest(
            "`name`, `company` and `title` must not be empty".to_string(),
        ));
    }

    let record = state.store.insert(
        payload.name.trim(),
        payload.company.trim(),
        payload.title.trim(),
    );

    Ok((StatusCode::CREATED, Json(record)))
}
