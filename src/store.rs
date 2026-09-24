use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use serde::{Deserialize, Serialize};

use crate::models::{
    GlinerOutcome, GlinerRequestLog, JevOutcome, JevRequestLog, RequestOutcome, SearchHit,
    SerpRequest, SerpRequestLog, TitleRecord, TitleSource, Verification,
};
use crate::ner::ExtractionAttempt;
use crate::search::SearchAttempt;
use crate::verify::JevAttempt;

/// A SerpAPI search we have already run, kept so repeat lookups for the same
/// person don't spend another API credit.
#[derive(Debug, Clone)]
pub struct StoredSearch {
    /// The SerpAPI request that produced these hits.
    pub request: SerpRequest,
    pub title: Option<String>,
    pub title_source: TitleSource,
    pub title_confidence: Option<f32>,
    pub hits: Vec<SearchHit>,
    /// Jev's check of `hits`, kept so cached lookups don't pay for it again.
    pub verification: Verification,
    pub retrieved_at: DateTime<Utc>,
}

/// Storage: the curated title directory and stored SerpAPI results live in
/// memory; the log of every SerpAPI request sent is also written to a
/// JSON-lines file when one is configured, so it survives restarts.
/// Swap this for a real database later; the handlers only depend on the methods
/// below.
#[derive(Clone, Default)]
pub struct Store {
    records: Arc<RwLock<HashMap<Uuid, TitleRecord>>>,
    searches: Arc<RwLock<HashMap<String, StoredSearch>>>,
    requests: Arc<RwLock<Vec<SerpRequestLog>>>,
    request_log: Option<Arc<RequestLogFile>>,
}

/// The title picked for a search and how it was picked.
#[derive(Debug, Clone)]
pub struct ChosenTitle {
    pub text: Option<String>,
    pub source: TitleSource,
    pub confidence: Option<f32>,
}

/// One line of the request log file. Jev and GLiNER lines are written after
/// the SerpAPI line they belong to and are nested under it when loaded.
///
/// Untagged so log files written before Jev or GLiNER were added still load.
/// The variants are told apart by required fields: only Jev and GLiNER lines
/// have `serp_request_id`, and their `request`s share no required fields
/// (Jev's has `endpoint` and `state`, GLiNER's has `labels` and `inputs`).
/// SerpAPI is tried last because it is the one with no `serp_request_id`.
// Only ever one of these exists at a time, while a line is read or written,
// so the uneven variant sizes cost nothing worth boxing for.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum LogLine {
    Jev(JevRequestLog),
    Gliner(GlinerRequestLog),
    Serp(SerpRequestLog),
}

/// Append-only JSON-lines file of saved SerpAPI and Jev requests, one per line.
struct RequestLogFile {
    path: PathBuf,
    /// Serialises appends so concurrent requests can't interleave lines.
    write_lock: Mutex<()>,
}

impl Store {
    /// A store preloaded with a couple of records, so the API returns something
    /// useful on a fresh start.
    pub fn with_seed_data() -> Self {
        let store = Self::default();
        store.insert("Ada Lovelace", "Analytical Engine Co", "Lead Programmer");
        store.insert("Grace Hopper", "UNIVAC", "Senior Systems Engineer");
        store
    }

    /// Persists SerpAPI requests to `path`, loading any already saved there.
    /// The file and its parent directory are created on first write.
    pub fn with_request_log(mut self, path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let saved = Self::load_request_log(&path)?;

        tracing::info!(
            "loaded {} saved SerpAPI request(s) from {}",
            saved.len(),
            path.display()
        );

        *self.requests.write().expect("store lock poisoned") = saved;
        self.request_log = Some(Arc::new(RequestLogFile {
            path,
            write_lock: Mutex::new(()),
        }));

        Ok(self)
    }

    fn load_request_log(path: &Path) -> io::Result<Vec<SerpRequestLog>> {
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };

        let mut saved: Vec<SerpRequestLog> = Vec::new();

        for (index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let invalid = |message: String| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}:{}: {message}", path.display(), index + 1),
                )
            };

            match serde_json::from_str(line).map_err(|error| invalid(error.to_string()))? {
                LogLine::Serp(log) => saved.push(log),
                LogLine::Jev(jev) => {
                    let serp = saved
                        .iter_mut()
                        .rev()
                        .find(|log| log.request.id == jev.serp_request_id)
                        .ok_or_else(|| {
                            invalid(format!(
                                "Jev request {} refers to unknown SerpAPI request {}",
                                jev.request.id, jev.serp_request_id
                            ))
                        })?;
                    serp.jev = Some(jev);
                }
                LogLine::Gliner(gliner) => {
                    let serp = saved
                        .iter_mut()
                        .rev()
                        .find(|log| log.request.id == gliner.serp_request_id)
                        .ok_or_else(|| {
                            invalid(format!(
                                "GLiNER extraction {} refers to unknown SerpAPI request {}",
                                gliner.request.id, gliner.serp_request_id
                            ))
                        })?;
                    serp.gliner = Some(gliner);
                }
            }
        }

        Ok(saved)
    }

    /// Cache key for a person at a company; case- and whitespace-insensitive so
    /// "ada lovelace" and "Ada Lovelace" share one entry.
    fn search_key(name: &str, company: &str) -> String {
        format!(
            "{}@{}",
            name.trim().to_lowercase(),
            company.trim().to_lowercase()
        )
    }

    pub fn insert(&self, name: &str, company: &str, title: &str) -> TitleRecord {
        let record = TitleRecord {
            id: Uuid::new_v4(),
            name: name.to_string(),
            company: company.to_string(),
            title: title.to_string(),
        };

        self.records
            .write()
            .expect("store lock poisoned")
            .insert(record.id, record.clone());

        record
    }

    /// Case-insensitive lookup of a person at a company.
    pub fn find(&self, name: &str, company: &str) -> Option<TitleRecord> {
        self.records
            .read()
            .expect("store lock poisoned")
            .values()
            .find(|record| {
                record.name.eq_ignore_ascii_case(name)
                    && record.company.eq_ignore_ascii_case(company)
            })
            .cloned()
    }

    pub fn list(&self) -> Vec<TitleRecord> {
        self.records
            .read()
            .expect("store lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// A previously stored SerpAPI search, if we ran one for this person.
    pub fn stored_search(&self, name: &str, company: &str) -> Option<StoredSearch> {
        self.searches
            .read()
            .expect("store lock poisoned")
            .get(&Self::search_key(name, company))
            .cloned()
    }

    /// Stores a SerpAPI result, replacing any earlier search for the same person.
    pub fn save_search(
        &self,
        name: &str,
        company: &str,
        request: SerpRequest,
        title: ChosenTitle,
        hits: Vec<SearchHit>,
        verification: Verification,
    ) -> StoredSearch {
        let stored = StoredSearch {
            request,
            title: title.text,
            title_source: title.source,
            title_confidence: title.confidence,
            hits,
            verification,
            retrieved_at: Utc::now(),
        };

        self.searches
            .write()
            .expect("store lock poisoned")
            .insert(Self::search_key(name, company), stored.clone());

        stored
    }

    /// Saves a SerpAPI request and its outcome. Called for every request sent,
    /// including failed ones, which never reach `save_search`.
    ///
    /// The in-memory log is always updated. If the file write fails the error is
    /// returned, but the entry stays in memory so it is still served until
    /// restart.
    pub async fn record_request(&self, attempt: &SearchAttempt) -> io::Result<SerpRequestLog> {
        let outcome = match &attempt.result {
            Ok(hits) => RequestOutcome::Succeeded {
                hit_count: hits.len(),
            },
            Err(error) => RequestOutcome::Failed {
                message: error.to_string(),
            },
        };

        let log = SerpRequestLog {
            request: attempt.request.clone(),
            http_status: attempt.http_status,
            outcome,
            jev: None,
            gliner: None,
        };

        self.requests
            .write()
            .expect("store lock poisoned")
            .push(log.clone());

        if let Some(file) = &self.request_log {
            file.append(&LogLine::Serp(log.clone())).await?;
        }

        Ok(log)
    }

    /// Saves a GLiNER extraction under the SerpAPI request whose results it
    /// read. Same failure behaviour as `record_request`.
    pub async fn record_gliner_request(
        &self,
        serp_request_id: Uuid,
        attempt: &ExtractionAttempt,
    ) -> io::Result<GlinerRequestLog> {
        let outcome = match &attempt.result {
            Ok(output) => GlinerOutcome::Succeeded {
                spans: output.spans.clone(),
                chosen: output.chosen.clone(),
            },
            Err(error) => GlinerOutcome::Failed {
                message: error.to_string(),
            },
        };

        let log = GlinerRequestLog {
            serp_request_id,
            request: attempt.request.clone(),
            duration_ms: attempt.duration_ms,
            outcome,
        };

        {
            let mut requests = self.requests.write().expect("store lock poisoned");
            match requests
                .iter_mut()
                .rev()
                .find(|serp| serp.request.id == serp_request_id)
            {
                Some(serp) => serp.gliner = Some(log.clone()),
                // Handlers always record the SerpAPI request first, so this is a bug.
                None => tracing::warn!(
                    "GLiNER extraction {} has no SerpAPI request {serp_request_id} to attach to",
                    log.request.id
                ),
            }
        }

        if let Some(file) = &self.request_log {
            file.append(&LogLine::Gliner(log.clone())).await?;
        }

        Ok(log)
    }

    /// Saves a Jev request and its outcome under the SerpAPI request whose
    /// results it judged. Same failure behaviour as `record_request`.
    pub async fn record_jev_request(
        &self,
        serp_request_id: Uuid,
        attempt: &JevAttempt,
    ) -> io::Result<JevRequestLog> {
        let outcome = match &attempt.result {
            Ok(Verification::Checked {
                score,
                answers_question,
                model,
                ..
            }) => JevOutcome::Succeeded {
                score: *score,
                answers_question: *answers_question,
                model: model.clone(),
            },
            Ok(other) => JevOutcome::Failed {
                message: format!("unexpected verification {other:?}"),
            },
            Err(error) => JevOutcome::Failed {
                message: error.to_string(),
            },
        };

        let log = JevRequestLog {
            serp_request_id,
            request: attempt.request.clone(),
            http_status: attempt.http_status,
            outcome,
        };

        {
            let mut requests = self.requests.write().expect("store lock poisoned");
            match requests
                .iter_mut()
                .rev()
                .find(|serp| serp.request.id == serp_request_id)
            {
                Some(serp) => serp.jev = Some(log.clone()),
                // Handlers always record the SerpAPI request first, so this is a bug.
                None => tracing::warn!(
                    "Jev request {} has no SerpAPI request {serp_request_id} to attach to",
                    log.request.id
                ),
            }
        }

        if let Some(file) = &self.request_log {
            file.append(&LogLine::Jev(log.clone())).await?;
        }

        Ok(log)
    }

    /// Saved SerpAPI requests, newest first, optionally capped at `limit`.
    pub fn requests(&self, limit: Option<usize>) -> Vec<SerpRequestLog> {
        self.requests
            .read()
            .expect("store lock poisoned")
            .iter()
            .rev()
            .take(limit.unwrap_or(usize::MAX))
            .cloned()
            .collect()
    }

    /// One saved SerpAPI request by its id.
    pub fn request(&self, id: Uuid) -> Option<SerpRequestLog> {
        self.requests
            .read()
            .expect("store lock poisoned")
            .iter()
            .find(|log| log.request.id == id)
            .cloned()
    }

    /// Number of stored searches. Useful for diagnostics and tests.
    pub fn stored_search_count(&self) -> usize {
        self.searches.read().expect("store lock poisoned").len()
    }
}

impl RequestLogFile {
    async fn append(&self, log: &LogLine) -> io::Result<()> {
        let mut line = serde_json::to_string(log)?;
        line.push('\n');

        let _guard = self.write_lock.lock().await;

        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;

        file.write_all(line.as_bytes()).await?;
        file.flush().await
    }
}
