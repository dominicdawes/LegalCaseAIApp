# case_brief — Agent Map

Two pipelines live in this package. `CASE_BRIEF_COMPACT` (default **`true`**) selects
between them in `graph.py::_build_graph`.

Provider env var: `BRIEF_AGENT_PROVIDER` (default **`deepseek`**, `worker_config.py:25`).
`BRIEF_RESEARCH_PROVIDER` overrides just the compact research stage.

---

## Compact pipeline — `compact_nodes.py` (DEFAULT)

**5 graph nodes · ~14 LLM calls** (vs 14 nodes / ~31 calls in legacy).

```
START ──┬─▶ plan_agent ─────┐
        └─▶ research_agent ─┴─▶ sync_barrier ─▶ [Send×N_units] section_generator
                                                   → final_formatter → END
                               └──(no dossier)──────→ final_formatter
```

| Node | Function | Fan-out | Worker Class | LLM |
|---|---|---|---|---|
| `plan_agent` | Job control: `brief_mode`, `target_length`, `retrieval_depth` | 1 | `worker_mid` | 1 call, 640 tok |
| `research_agent` | Bounded tool-calling ReAct loop → lean case dossier | 1 | `orchestrator` | ≤ `BRIEF_RESEARCH_MAX_TURNS` (5) + 1 emission |
| `sync_barrier` | No-op join of the two parallel branches | 1 | — | none |
| `section_generator` | Writes ONE brief unit in FINAL template form, self-grounded | × N_units (≤8) | `orchestrator` | ≤ `BRIEF_UNIT_MAX_TURNS` (2) + 1 |
| `final_formatter` | Deterministic assembly | 1 | — | **none** |

### What each stage absorbed

- **`plan_agent`** ← `head_orchestrator`. Keeps mode/length/depth calibration; drops its tool call (the source list is pre-fetched).
- **`research_agent`** ← `source_profiler` (×N) + `retrieval_planner` (2 calls) + `planned_retriever` (×T) + `evidence_card_builder` (×B) + `legal_artifact_extractor` (×4) + `doctrinal_synthesizer`. Their intents are folded into `_research_system()`: pick the primary opinion and classify other sources; case-name-anchored retrieval over generic semantic search; materiality test; holding ≠ disposition; standalone reusable rule; the 7-type reasoning taxonomy; the dissent hallucination guard; the pedagogy payload.
- **`section_generator`** ← `section_writer` (×10) + `section_grounder` + `critic` + `brief_revision_agent` + the formatter's per-section synthesis. Self-grounds instead of a downstream critic pass.
- **`final_formatter`** ← `brief_assembler` + presentation. Zero LLM.

### Brief units (`BRIEF_UNITS`, the fan-out table)

Case-brief sections are known a priori, so the fan-out is a constant table — nothing is discovered.

| unit | template sections | evidence roles | ~words |
|---|---|---|---|
| `identity` | I. One-Sentence Rule, II. Posture | citation, posture, holding | 180 |
| `facts` | III. Facts (A/B/C) | facts | 260 |
| `issue_holding` | IV. Framework, V. Issues, VI. Holdings | issue, holding, citation | 220 |
| `rule` | VII. Rule & Test, X. Black Letter | rule, holding | 240 |
| `reasoning` | VIII. Reasoning, IX. Arguments | reasoning, rule | 280 |
| `dissent` *(only if `has_dissent`)* | XII. Dissent | dissent | 160 |
| `pedagogy` | XI. Significance, XIII. Pedagogy | pedagogy, reasoning | 200 |
| `exam` | XIV. Exam Translation, XV. If/Then, XVI. Cold Call | exam_trigger, pedagogy, holding | 320 |

Student-facing output is the same canonical 16-section template as legacy; targets **hold** the legacy ~1,600-word length rather than cutting it.

### Key design points

- **Units emit FINAL markdown, not drafts.** Legacy wrote 10 sections, kept only 6 flat `SectionDraft` fields, then spent a **12,000-token** LLM call in `final_formatter` re-deriving the structure it had discarded. That call is gone.
- **Lean dossier.** `research_agent` emits identity + anchors + an `evidence_by_role` chunk-id index only. Section writers derive their own prose. (A fat dossier is what made attack_outline's serial emission its dominant cost.)
- **Deterministic pruning + renumbering.** `[OMIT-IF-ABSENT]` sections and self-omitting units are dropped in Python, then `_renumber_roman_headings` closes the numbering gaps. Legacy left this to the LLM while `_prune_and_renumber_sections` sat dead in `nodes.py`.
- **`sync_barrier` keeps conditional edges off fan-out nodes.** Legacy hung routers directly on Send-parallel nodes (`planned_retriever`, `evidence_card_builder`, `section_writer`), where the branch re-evaluates per task over an accumulating list.
- **Chunk-id citations stripped at the presentation boundary.** Raw markdown keeps them in `state["brief_units"]` and the ledger for audit.
- **`verify_claim` is available to section writers** (unlike attack_outline, which excludes it for latency): a wrong holding is fatal in a brief, and units run in parallel, so its nested LLM call costs one call's latency rather than one per unit.
- No critic / revision loop. Self-grounding inside each unit replaces it.

### State (compact fields, `state.py`)

| Field | Reducer | Carries |
|---|---|---|
| `case_dossier` | replace | case identity, anchors, `evidence_by_role` index |
| `evidence_store` | replace | every harvested chunk, keyed by `chunk_id` |
| `brief_units` | `operator.add` | one entry per generated unit |

### Env vars

`CASE_BRIEF_COMPACT` (true) · `BRIEF_RESEARCH_MAX_TURNS` (5) · `BRIEF_UNIT_MAX_TURNS` (2) · `BRIEF_DOSSIER_TARGET_WORDS` (700) · `BRIEF_HARD_OUTPUT_TOKENS` (14000) · `BRIEF_PREFETCH_MAX_OUTLINES` (10) · `BRIEF_PREFETCH_MAX_SECTIONS` (40) · `BRIEF_RESEARCH_PROVIDER`

---

## Legacy pipeline — `nodes.py` (`CASE_BRIEF_COMPACT=false`)

Retained for rollback. **14 nodes · ~31 LLM calls** (3 sources / 9 bundles).

```
head_orchestrator       → [Send×N_docs] source_profiler
source_profiler         → retrieval_planner                (fan-in; inlines corpus orientation)
retrieval_planner       → [Send×N_targets] planned_retriever
planned_retriever       → [Send×N_bundles] evidence_card_builder
evidence_card_builder   → [Send×4] legal_artifact_extractor (4 artifact TYPES total, not per card)
legal_artifact_extractor → doctrinal_synthesizer           (fan-in)
doctrinal_synthesizer   → brief_drafter
brief_drafter           → [Send×10] section_writer
section_writer          → [Send×N_sections] section_grounder
section_grounder        → brief_assembler                  (fan-in)
brief_assembler         → critic
critic → (should_revise?)
  → brief_revision_agent → critic                          (loop ≤ 1, MAX_REVISIONS)
  → final_formatter → END
```

| Node | Function | Fan-out | Worker Class |
|---|---|---|---|
| `head_orchestrator` | Job plan: mode, length, depth | 1 | `orchestrator` (640 tok) |
| `source_profiler` | Profile one source doc | × N_docs | `worker_mid` (1200) |
| `retrieval_planner` | Corpus orientation **+** per-target plans — **2 LLM calls** | 1 | `worker_mid` (768) + `orchestrator` (5000) |
| `planned_retriever` | Execute + rerank one plan | × N_targets | `tool_only` |
| `evidence_card_builder` | Role-tagged evidence cards from one bundle | × N_bundles | `worker_low` (3000) |
| `legal_artifact_extractor` | One of 4 artifact types (`facts_posture`, `issue_holding`, `rule_reasoning`, `dissent`) | ×4 | `worker_mid` (2500) |
| `doctrinal_synthesizer` | Pedagogy payload | 1 | `worker_mid` (2000) |
| `brief_drafter` | Section manifest + word budgets | 1 | **no LLM** (pure Python) |
| `section_writer` | Write one of 10 sections | ×≤10 | `worker_mid` (1500) |
| `section_grounder` | `verify_claim` sweep per section | × N_sections | `tool_only` (`worker_low` 200 fallback) |
| `brief_assembler` | Stitch sections | 1 | `tool_only` |
| `critic` | Score 4 axes, name revision targets | 1 | `orchestrator` (1500) |
| `brief_revision_agent` | Repair ≤4 flagged sections | ≤1 pass | `worker_mid` (1200) ×≤4 |
| `final_formatter` | Re-synthesise into the 16-section template | 1 | `worker_mid` (**12000**) |

### Known issues in the legacy path

- `_add_budget` (`nodes.py:267`) is never called, so the footer always reports 0 tokens / $0.0000.
- `_prune_and_renumber_sections` + `_ROMAN_NUMERALS` (`nodes.py:1847`) are defined but never invoked; `[omit if absent]` is left to the LLM, so numbering gaps survive.
- `section_writer` prompts request rich per-section JSON, but only the six `SectionDraft` fields reach state — `final_formatter` re-derives the rest from `draft_text`.
- Conditional edges attached to Send fan-out nodes risk T×B / S² task multiplication (no join node).

---

## Contract (both pipelines)

- `run_case_brief_agent` / `run_case_brief_agent_stream` — 8 params: `request, project_id, source_ids, use_voyage, thread_id, job_id, run_id, user_id`.
- Caller reads exactly one key: `final_state["final_output"]` (Markdown) → `notes.content_markdown` (`tasks/note_tasks.py:1392`).
- `job_id == ""` disables all ledger artifact writes.
- The agent performs **no** direct DB writes of its own — persistence is the caller's `notes` UPDATE plus ledger artifacts.
- Activation gate: `USE_CASE_BRIEF_AGENT` (read in `tasks/note_tasks.py:86`).
