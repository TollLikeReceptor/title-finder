use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::search::SearchError;

#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    Upstream(String),
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    message: String,
}

impl From<SearchError> for ApiError {
    fn from(error: SearchError) -> Self {
        ApiError::Upstream(error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, message) = match self {
            ApiError::NotFound(message) => (StatusCode::NOT_FOUND, "not_found", message),
            ApiError::BadRequest(message) => (StatusCode::BAD_REQUEST, "bad_request", message),
            ApiError::Upstream(message) => (StatusCode::BAD_GATEWAY, "upstream_error", message),
        };

        if status.is_server_error() {
            tracing::error!("{error}: {message}");
        }

        let body = ErrorBody {
            error: error.to_string(),
            message,
        };

        (status, Json(body)).into_response()
    }
}
