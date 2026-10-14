# Satark — Observability Platforms, Pipelines, Production Evaluation and Drift

*Satark (सतर्क) means "vigilant" or "alert." Nidan taught you to diagnose a
single trace by hand; Satark assumes you now have thousands of them per
hour and asks the next question — how do you stay vigilant over an agent
you can no longer read one trace at a time?*

## 1. What Changed Since Nidan (the Previous Session)

| | Nidan | Satark |
|---|---|---|
| Core question | Given a trace and nothing else, what broke? | Given thousands of traces an hour, is the agent still healthy, and would you even notice if it weren't? |
| Signal type | Traces only | Traces (unchanged) **+ metrics** — a MeterProvider and real instruments (`metrics.py`), not just span attributes |
| Backends | Jaeger only | Jaeger (trace shape) **+ Prometheus/Grafana** (aggregate health) **+ an optional Langfuse profile** (GenAI-aware trace views, same spans, zero extra instrumentation) |
| Evaluation | Offline only — `trajectory_eval.py`'s keyword/tool check, run on demand via `/grade` | Offline (unchanged) **+ online** — `online_eval.py` samples LIVE `/ask`/`/chat` traffic with a judge-model call, continuously |
| Drift | Not tracked | `drift.py` — embedding-centroid drift (inputs/outputs) and tool-mix distribution drift, against a frozen baseline |
| Alerting | None | `alerting.py` — four rules (model-version mismatch, eval-score drop, embedding drift, cost spike), firing without a human reading every trace |
| Human feedback | None | `feedback.py` — thumbs up/down capture plus an annotation queue merging negative feedback with low-scoring sampled turns |
| Failure classes | 4, seeded blind | **5** — incidents 1-4 carried over unchanged; incident 5 (`silent_model_swap`) is new and deliberately NOT diagnosable from a single trace |
| New endpoints | `/delegate`, `/replay`, `/diff`, `/grade`, `/seed-incident/{n}`, `/incident-status`, `/reveal-incident/{n}` | `/eval/recent`, `/drift/status`, `/drift/reset-baseline`, `/alerts`, `/alerts/clear`, `/feedback`, `/feedback/recent`, `/annotation-queue`, plus `/seed-incident/5` and `/reveal-incident/5` |
| Reused unchanged | — | `loaders.py`, `pii.py`, `chunking.py`, `context.py`, `vectorstore.py`, `tools.py`, `delegation.py`, `memory_store.py`, `replay.py`, `trajectory_eval.py`, `make_documents.py`, `build_vectorstore.py`, incidents 1-4 |

Same 5 documents, same domain, same underlying reference agent. Satark adds
a second signal type (metrics), a second evaluation mode (online, sampled,
continuous), and the statistical layer (drift + alerting) that a
trace-by-trace workflow like Nidan's structurally cannot provide.

## 2. The Five Documents, Five Formats

Unchanged from Nirikshan/Smriti/Drishti/Nidan. Regenerate with
`python make_documents.py`.

## 3. The Platform Landscape — What Each Piece Is Actually Optimising For

| Tool | Optimised for | Not optimised for |
|---|---|---|
| **Jaeger** | "What happened on THIS request" — deep, single-trace, span-shape debugging (Nidan's whole exercise) | Aggregating thousands of requests into one number |
| **Prometheus + Grafana** | "What's the p95 latency / cost / failure rate over the last hour across every request" — cheap aggregation, bad at showing you any ONE request's story | A specific customer's specific bad answer |
| **Langfuse** (optional profile) | GenAI-aware views on the SAME spans Jaeger shows you — cost-by-model, latency-by-tool, a built-in eval UI — without a second instrumentation pass | Being light to self-host: 4 extra containers (Postgres, ClickHouse, Redis, MinIO) for this one UI layer |
| **A managed backend** (Arize Phoenix, LangSmith, Datadog) | Zero infrastructure, usually better default dashboards, a support contract | Data residency/retention control, and the per-seat or per-GB cost model self-hosting avoids |

`docker-compose.yml`'s `langfuse` Compose **profile** is the self-hosted-
vs-managed tradeoff made literal: the core stack (Jaeger, Prometheus,
Grafana) is always on because it's cheap to run; Langfuse's four extra
containers are opt-in, because standing them up and keeping them up is a
real, ongoing cost — the "real cost of telemetry" M4 asks you to notice is
not the OTLP bandwidth, it's the containers downstream of it.

Try it: `docker compose up` for the core stack; `docker compose --profile
langfuse up` to add Langfuse on top.

## 4. Dashboard Design for Agent Health — What a Duty Engineer Needs on One Screen

`grafana/dashboards/satark-agent-health.json` (auto-provisioned) has eight
panels, each answering ONE question a duty engineer would actually ask
first, deliberately not a wall of secondary metrics:

1. Request rate — is traffic flowing at all?
2. Outcome / trajectory failure rate — is the agent getting things wrong?
3. Latency p50/p95 — is it slow?
4. Cost per hour — is spend normal?
5. Mean cost per call — the FASTEST incident-5 signal (moves on the first swapped call, before eval/drift windows have enough data)
6. Online-eval score — is live traffic's quality holding up?
7. Embedding drift — are answers subtly changing shape?
8. Guardrail/refusal trigger rate — is the agent declining more than usual?

Try it: run `python agent.py` to generate ~20 requests, then open
`http://localhost:3000` (admin/admin) and watch the panels move.

**Verify metric names before trusting the dashboard queries** — run `curl
http://localhost:8889/metrics` after a few requests and confirm the
Collector's Prometheus exporter named things the way the dashboard JSON
expects (dots become underscores; unit suffixes can shift between
Collector versions). This is a real step, not a formality — "assessed
rather than asserted" cuts both ways.

## 5. Online Evaluation — Sampled Model-Graded Scoring on Live Traffic

`online_eval.py`'s `maybe_grade()` is wired into `app.py`'s
`_run_traced_agent_turn()`, so it runs on real `/ask` and `/chat` traffic
at `SATARK_EVAL_SAMPLE_RATE` (default 0.3), not a separate endpoint you
have to remember to call. A second `chat_raw()` call (the "judge") scores
the answer 1-5 against the actual question — no reference answer needed,
which is the whole point: offline grading (`trajectory_eval.py`, unchanged
from Nidan) only ever covers questions someone wrote a keyword list for;
online grading covers the long tail real customers actually ask.

**Calibrate the grader before you trust it**: `GET /eval/recent` shows
every sampled judge verdict with its rationale — read a sample of these
yourself before assuming the judge model's scores mean what you think they
mean. A miscalibrated judge is a second failure mode layered on top of
whatever the agent itself is doing wrong.

Try it: `GET /eval/recent`.

## 6. Guardrail, Refusal and Fallback Telemetry as Quality Signals

`app.py`'s `_detect_guardrail()` is a deliberately simple phrase match
(standing in for a real moderation/guardrail service) that tags every turn
with `satark.guardrail.triggered` and `satark.refusal.detected`, and feeds
`metrics.guardrail_counter`. The point isn't the detector's sophistication —
it's that a refusal rate is itself a quality signal worth a dashboard panel
(panel 8), not just something you'd notice by reading transcripts.

## 7. Drift Detection Across Inputs, Outputs and Embeddings

`drift.py` tracks two independent kinds of drift, because they catch
different failures:

- **Embedding drift** — the first `SATARK_DRIFT_BASELINE_SIZE` answers
  freeze a baseline centroid; every answer after that is scored by cosine
  distance from it, over a rolling window. Catches a model swap whose
  individual answers still look correct (incident 5).
- **Tool-mix drift** — the same idea applied to the categorical
  distribution of which tools fire, via total-variation distance. Catches
  a routing/prompt regression that shifts the agent toward the wrong tool
  more often, even when each answer still reads fine (connects back to
  Nidan's incident 2, but as a trend instead of one misrouted call).

Try it: `GET /drift/status`; `POST /drift/reset-baseline` before a fresh
exercise.

## 8. Regression Detection Across Model Versions — the Failure That Arrives Without a Deployment

This is incident 5, `silent_model_swap` (see `incidents.py`): `chat_raw()`
silently serves `gpt-4o` instead of the requested `gpt-4o-mini`, with no
error, no log line, and — this is the point — no single trace that looks
wrong. `gen_ai.request.model` still reads correctly, because that's what
the CALLER asked for; only `gen_ai.response.model` and the per-call cost
actually move. Nidan's whole workflow (read one trace, diagnose it) cannot
catch this. Three independent layers can, each noticing something
different:

1. **Cost** — panel 5 / `alerting.py`'s `cost_spike` rule, the fastest to fire.
2. **Online eval** — `alerting.py`'s `eval_score_drop` rule, slower, but catches a QUALITY shift cost alone wouldn't.
3. **The attribute itself** — `alerting.py`'s `model_version_mismatch` rule, comparing `gen_ai.request.model` against `gen_ai.response.model` directly. This is the one `incidents.py`'s own answer key calls out as ground-truth: alert on the disagreement itself, not only on its downstream effects.

## 9. Human Feedback Capture and Annotation Queues

`feedback.py`'s `record_feedback()` takes a thumbs up/down plus an optional
note against a `trace_id` — the cheapest, most direct quality signal
there is, no judge model required. `annotation_queue()` merges that
explicit negative feedback with `online_eval.py`'s own low-scoring sampled
turns into ONE reviewer-facing list, so a human reviewer isn't stuck
choosing between "only look at complaints" and "only look at what the
judge model already flagged."

Try it: `POST /feedback?trace_id=...&rating=down&note=...`, then
`GET /annotation-queue`.

## 10. Architecture — the Full Flow

```
STEP 1 -- make_documents.py, then build_vectorstore.py (run once)

STEP 2 -- docker compose up                  (OTel Collector, Jaeger,
          (Terminal 1)                        Prometheus, Grafana)
          optionally: docker compose --profile langfuse up
          Jaeger UI:     http://localhost:16686
          Grafana UI:    http://localhost:3000   (admin/admin)
          Prometheus UI: http://localhost:9090
          Langfuse UI:   http://localhost:3001   (langfuse profile only)

STEP 3 -- app.py  (Terminal 2)

    GET  /ask?q=...                 GET  /eval/recent
    POST /chat?...                  GET  /drift/status
    GET  /memory/{session_id}       POST /drift/reset-baseline
    GET  /delegate?amount=...       GET  /alerts
    GET  /replay?q=...&n=...        POST /alerts/clear
    GET  /diff?q=...                POST /feedback
    GET  /grade?q=...               GET  /feedback/recent
    POST /seed-incident/{n}         GET  /annotation-queue
    GET  /incident-status           (n = 1..5)
    GET  /reveal-incident/{n}

STEP 4 -- agent.py  (Terminal 3)

    Domain questions (looped, so the dashboard has something to show),
    then M4's dashboard walkthrough, M5's eval/drift/feedback demos, then
    the incident-5 exercise.
```

### Where each new file sits in the request path

Every `/ask` or `/chat` call still runs Nidan's exact loop inside
`app.py`'s `_run_traced_agent_turn()`. What's new happens in a fixed order
AFTER the loop produces an answer, every single time, not as a separate
path someone has to remember to call:

```
agent_turn finishes (answer, tool_sequence, cost, duration known)
  -> metrics.record_agent_turn()      (M4: counters + histograms -> Collector -> Prometheus)
  -> _detect_guardrail()              (refusal/guardrail telemetry on the span)
  -> online_eval.maybe_grade()        (M5: sampled judge-model call, SATARK_EVAL_SAMPLE_RATE)
  -> alerting.observe_and_alert()     (M5: 4 alert rules, including...)
       -> drift.observe()             (M5: embedding + tool-mix drift, updates rolling windows)
```

## 11. Setup — Step by Step

1. Clone/copy this folder.
2. `cp .env.example .env` and fill in `OPENAI_API_KEY`.
3. `python -m venv .venv && source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `python make_documents.py`
6. `python build_vectorstore.py`
7. **Terminal 1:** `docker compose up` (add `--profile langfuse` for the optional Langfuse stack)
8. **Terminal 2:** `uvicorn app:app --reload --port 8000`
9. **Terminal 3:** `python agent.py`
10. Open Jaeger (`:16686`), Grafana (`:3000`, admin/admin), Prometheus (`:9090`), and — if the `langfuse` profile is running — Langfuse (`:3001`).

## 12. What to Notice While Running This

- **A trace backend and a metrics backend answer different questions.** Jaeger tells you about one request; Grafana tells you whether the agent, in aggregate, is still healthy. Needing both is not redundancy.
- **Incident 5 cannot be diagnosed from a single trace, on purpose.** If you're staring at one `/ask` response trying to spot the model swap, you've picked the wrong tool for this incident — generate traffic and watch the dashboard instead.
- **The cheapest signal (cost) is often the fastest one.** Panel 5 moves on the very first swapped call; the eval-score and drift signals need several requests before their rolling windows mean anything. A real alerting setup wants both the fast, cheap signal and the slower, more diagnostic ones.
- **A judge model can be wrong too.** `/eval/recent`'s rationale field exists so you can sanity-check the judge, not just trust its score.

## 13. Troubleshooting

| Problem | Likely cause |
|---|---|
| `/ask` or `/chat` fails with "OPENAI_API_KEY is not set" | Copy `.env.example` to `.env` and fill in your key — see `llm_provider.py`'s and `vectorstore.py`'s error messages, which now name the exact `.env` path they checked |
| Grafana panels show "No data" | Confirm `docker compose up` is running, then generate traffic with `python agent.py` — panels need at least one scrape interval (10s) of real requests |
| Dashboard queries return nothing even with traffic flowing | Run `curl http://localhost:8889/metrics` and confirm the metric names match what the dashboard JSON expects (see section 4) |
| Collector logs show repeated connection errors to `langfuse-web:3000` | Expected if the `langfuse` Compose profile isn't running — that exporter is optional; see section 3 |
| Incident 5 doesn't seem to do anything | It never breaks a single request — generate at least 10-15 requests with it seeded before expecting `/alerts`, `/eval/recent`, or the cost panel to show anything |
| No traces in Jaeger | Confirm `docker compose up` is running; see Nidan's README for the same checklist |

## 14. Where This Goes Next

Satark's alerting is a list a human has to poll (`/alerts`), and its
drift/eval baselines live in process memory, reset on every restart. A
production system pushes these to an actual paging system, persists
baselines somewhere durable, and — the real next step — closes the loop
from "an alert fired" back to "a fix shipped and the regression stopped
recurring," which is a CI/CD and release-process problem as much as an
observability one.
