# title-finder

A simple API for finding a job title based on name and company.

Built with [axum](https://github.com/tokio-rs/axum) on Tokio. Unknown people are
looked up through [SerpAPI](https://serpapi.com) by asking Google
*"What is the job title for &lt;name&gt; at &lt;company&gt;?"*, and the results are kept
in memory so the same question is never paid for twice.

## Running

Job titles are extracted with [GLiNER](https://github.com/urchade/GLiNER), run
locally through [gline-rs](https://github.com/fbilhaut/gline-rs). Fetch the model
once (it goes in `models/`, which is gitignored):

```sh
./scripts/download-gliner.sh              # gliner_large-v2.1, ~1.8 GB 
```

A SerpAPI key is required — the server refuses to start without one.

```sh
export SERP_API_KEY=your-key
export JEV_API_KEY=your-jev-key
cargo run           # listens on http://127.0.0.1:3000
PORT=8080 cargo run # or pick a port
```

| Variable           | Required | Default                | Purpose                       |
| ------------------ | -------- | ---------------------- | ----------------------------- |
| `SERP_API_KEY`     | yes      | —                      | SerpAPI credential            |
| `JEV_API_KEY`      | yes      | —                      | Jev credential, for checking search results |
| `GLINER_MODEL_DIR` | no       | `models/gliner_small-v2.1` | GLiNER `tokenizer.json` + `onnx/model.onnx` |
| `REQUEST_LOG`      | no       | `data/requests.jsonl`  | Where SerpAPI, Jev and GLiNER requests are saved |
| `PORT`             | no       | `3000`                 | Listen port                   |
| `RUST_LOG`         | no       | `title_finder=debug,…` | Log filter                    |

## Endpoints

| Method | Path                 | Description                                  |
| ------ | -------------------- | -------------------------------------------- |
| GET    | `/health`            | Liveness check                                |
| GET    | `/v1/titles`         | List all known records                        |
| GET    | `/v1/titles/search`  | Find a title by `name` and `company` params   |
| POST   | `/v1/titles`         | Add a record                                  |
| GET    | `/v1/requests`       | Saved SerpAPI requests with their Jev checks and GLiNER extractions, newest first (`?limit=N`) |
| GET    | `/v1/requests/{id}`  | One saved SerpAPI request, its Jev check and GLiNER extraction |

### How a lookup resolves

`GET /v1/titles/search` tries three sources in order, and the `source` field in
the response says which one answered:

1. `directory` — a curated record added via `POST /v1/titles`. Never costs a credit.
2. `serp_api_cached` — a SerpAPI search already run for this person and stored.
3. `serp_api` — a fresh SerpAPI search. Pass `refresh=true` to force this.

For searches, `title` is extracted by GLiNER, and only when Jev confirms the
results answer the question (see below). Otherwise it is `null`, and
`title_source` says why. The raw `hits` come back alongside it so callers can
check the reasoning.

### Checking the search with Jev

After a fresh SerpAPI search, the hits are sent to Jev (typesafe.ai `systemone`)
with the question *"Does this text answer the question What is the job title of
&lt;name&gt; at &lt;company&gt;?"*. Its verdict comes back in `verification`:

```json
"verification": {
  "status": "checked",
  "score": 0.98,
  "answers_question": true,
  "model": "jev-1.13.0",
  "checked_at": "2026-09-22T04:02:32.499934Z"
}
```

`answers_question` is `score >= 0.5`. The verdict is stored with the search, so
cached lookups reuse it without calling Jev again. If Jev is down the lookup
still succeeds with `"status": "failed"`, and a search with no text to judge gets
`"status": "skipped"`. Directory hits have no `verification` field.

### How the title is picked

The `title_source` field says which path produced `title`:

| `title_source` | When |
| -------------- | ---- |
| `gliner`       | Jev's score is **above 0.70**: GLiNER returns the most confident job title belonging to the searched person (see below). `title_confidence` is GLiNER's probability. |
| `unverified`   | `title` is `null`: Jev's score was 0.70 or below, or Jev failed or was skipped, so the results weren't mined for a title. |
| `not_found`    | `title` is `null`: GLiNER ran but found no title belonging to the searched person. |
| `extraction_failed` | `title` is `null`: GLiNER errored. Retry with `refresh=true`. |
| `directory`    | A curated record from `POST /v1/titles`. |

**Whose title is it?** GLiNER tags both `job title` and `person` spans in each
hit, and each title is attributed to the nearest person mentioned in that same
hit. The searched person is recognised from GLiNER's person spans ("Nadella",
"Mr. Nadella" and "Satya Nadella" all match) and from literal occurrences of
their full name or surname. Then:

1. Titles nearest the searched person are preferred; the most confident wins.
2. If there are none, titles from hits that mention nobody (e.g. a bare answer
   box) are used as a fallback.
3. Titles nearest someone else are never used — in "Jane reports to John Smith,
   the VP of Sales", Jane does not get "VP of Sales".

Nearest-person is a proxy for grammar, so it can misattribute: in "Nadella, who
succeeded Steve Ballmer as CEO", "CEO" is nearer Ballmer. That case is pinned in
the unit tests as a known limitation.

### Saved SerpAPI requests

Every request sent to SerpAPI — successful or not — is appended as one JSON line
to `REQUEST_LOG` and reloaded on startup, so the history survives restarts.
The API key is never written. Lookups answered from the directory or from a
stored search send no request and add nothing to the log.

```json
{
  "request": {
    "id": "6f1c2c1e-1111-4a4a-9b9b-000000000001",
    "endpoint": "https://serpapi.com/search",
    "engine": "google",
    "query": "What is the job title for Jane Doe at Acme?",
    "num": 10,
    "sent_at": "2026-09-21T10:00:00Z"
  },
  "http_status": 200,
  "outcome": { "status": "succeeded", "hit_count": 3 }
}
```

A failed request has `"outcome": { "status": "failed", "message": "..." }`, and
`http_status` is `null` when no response arrived at all.

When the search's hits were checked by Jev, that request is nested under the
SerpAPI request it judged, so the two can be compared directly. `state` is the
exact text Jev was given; the bearer token is never written.

```json
"jev": {
  "serp_request_id": "6f1c2c1e-1111-4a4a-9b9b-000000000001",
  "request": {
    "id": "9b2d4e7a-...",
    "endpoint": "https://api.typesafe.ai/v1/systemone",
    "model": "jev-latest",
    "question_key": "answers_job_title",
    "instructions": "Does this text answer the question What is the job title of Jane Doe at Acme?",
    "state": "Jane Doe - Acme\nhttps://example.com/team\nJane Doe is a Staff Engineer...",
    "sent_at": "2026-09-21T10:00:01Z"
  },
  "http_status": 200,
  "outcome": { "status": "succeeded", "score": 0.97, "answers_question": true, "model": "jev-1.13.0" }
}
```

When GLiNER ran on the search's hits, that extraction is nested alongside as
`gliner`: the model, labels and threshold, the exact `inputs` it read (one per
hit), every span it found — people and titles, with whose title each one was
judged to be — and the title it `chose`.

```json
"gliner": {
  "serp_request_id": "6f1c2c1e-1111-4a4a-9b9b-000000000001",
  "request": {
    "model": "gliner_large-v2.1",
    "labels": ["job title", "person"],
    "threshold": 0.5,
    "searched_name": "Satya Nadella",
    "inputs": ["Satya Nadella - Microsoft\nSatya Nadella is the Chairman and Chief Executive Officer of Microsoft..."],
    "started_at": "2026-09-22T05:04:22.202324Z"
  },
  "duration_ms": 114,
  "outcome": {
    "status": "succeeded",
    "spans": [
      { "input": 0, "start": 0,  "end": 13, "text": "Satya Nadella", "label": "person", "probability": 0.997 },
      { "input": 0, "start": 47, "end": 55, "text": "Chairman", "label": "job title", "probability": 0.723, "attribution": "searched_person" },
      { "input": 0, "start": 60, "end": 83, "text": "Chief Executive Officer", "label": "job title", "probability": 0.905, "attribution": "searched_person" }
    ],
    "chosen": { "text": "Chief Executive Officer", "probability": 0.905 }
  }
}
```

`attribution` is `searched_person`, `nobody` or `someone_else`; `start`/`end`
are byte offsets into `inputs[input]`. A failed extraction is saved with
`"outcome": { "status": "failed", "message": "..." }`. There is no `gliner`
entry when GLiNER didn't run (Jev at 0.70 or below, or a cached lookup).

On disk each Jev request and GLiNER extraction is its own line, written after
its SerpAPI line and linked by `serp_request_id`; they are joined up on load.
Log files from before Jev or GLiNER were added still load unchanged. `data/` is gitignored
because the log contains the names people looked up.

### Examples

```sh
curl localhost:3000/health

curl "localhost:3000/v1/titles/search?name=Ada%20Lovelace&company=Analytical%20Engine%20Co"

# Someone not in the directory: goes out to SerpAPI
curl "localhost:3000/v1/titles/search?name=Satya%20Nadella&company=Microsoft"

# Ignore the stored result and search again
curl "localhost:3000/v1/titles/search?name=Satya%20Nadella&company=Microsoft&refresh=true"

# What we've asked SerpAPI
curl "localhost:3000/v1/requests?limit=20"

curl -X POST localhost:3000/v1/titles \
  -H 'content-type: application/json' \
  -d '{"name":"Alan Turing","company":"NPL","title":"Principal Scientist"}'
```

A SerpAPI-backed lookup answers like this:

```json
{
  "name": "Satya Nadella",
  "company": "Microsoft",
  "title": "Chairman and Chief Executive Officer of Microsoft",
  "source": "serp_api",
  "query": "What is the job title for Satya Nadella at Microsoft?",
  "hits": [
    {
      "title": "Satya Nadella - Microsoft",
      "link": "https://example.com/satya",
      "snippet": "Satya Nadella is the Chairman and Chief Executive Officer of Microsoft..."
    }
  ],
  "retrieved_at": "2026-09-20T04:05:45.281678Z"
}
```

Lookups ignore case. If SerpAPI itself fails — bad key, exhausted quota, network
trouble — the response is `502` with a JSON error body:

```json
{ "error": "upstream_error", "message": "search provider returned 200: Invalid API key..." }
```

## Layout

```
src/
  main.rs      # binary entrypoint: config, logging, port, serve
  lib.rs       # AppState + builds the Router (shared with tests)
  handlers.rs  # one function per endpoint
  models.rs    # request/response types
  search.rs    # TitleSearch trait + SerpAPI client
  verify.rs    # AnswerCheck trait + Jev client
  ner.rs       # TitleExtractor trait + GLiNER extractor
  store.rs     # records, stored SerpAPI results, persisted request log
  error.rs     # ApiError -> HTTP response
tests/
  api.rs       # integration tests over the full router
```

Search sits behind the `TitleSearch` trait, so swapping SerpAPI for another
provider means adding one implementation and changing one line in `main.rs`.

## Tests

```sh
cargo test
```

Tests substitute stubs for `TitleSearch` and `AnswerCheck`, so the suite makes
no network calls and needs no API key. Live checks against the real Jev API are
opt-in:

```sh
cargo test --test jev_live -- --ignored      # real Jev API
cargo test --test gliner_live -- --ignored   # real GLiNER model (+ Jev for the full-chain test)
```
