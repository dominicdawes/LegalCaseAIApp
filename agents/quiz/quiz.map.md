# quiz — Agent Map

**13 nodes** · Provider env var: `QUIZ_AGENT_PROVIDER` (default: `anthropic`)

```
Phase 1 — Source Understanding:
  head_orchestrator → [Send×N_docs] source_profiler
  source_profiler → case_rule_extractor          (fan-in: all docs)
  case_rule_extractor → cross_doc_concepts_synthesis
  cross_doc_concepts_synthesis → quiz_blueprint_planner

Phase 2 — Batch Loop (repeats per batch until num_questions reached):
  quiz_blueprint_planner → question_drafter ◄────────────────────────┐
  question_drafter → false_trap_red_herring_generator                 │
  false_trap_red_herring_generator → question_evaluator               │
  question_evaluator → (should_revise_batch?)                         │
    → reviser → question_evaluator               (loop ≤ 2)           │
    → grounder → batch_commit ─────────────────────────────────────── ┘
      → critic  (when all batches done)

Phase 3 — Final QA:
  critic → final_formatter → END
```

| Node Name | Node Function | Fan-out Quantity | Worker Class | Default LLM Model |
|---|---|---|---|---|
| `head_orchestrator` | Plan quiz structure, validate sources, set batch params | 1 | `worker_mid` | claude-sonnet-4-6 |
| `source_profiler` | Profile one source doc | × N_docs | `worker_low` | claude-haiku-4-5-20251001 |
| `case_rule_extractor` | Extract case rules and holdings from all merged profiles | 1 | `orchestrator` | claude-opus-4-7 |
| `cross_doc_concepts_synthesis` | Synthesize cross-doc testable concepts | 1 | `worker_mid` | claude-sonnet-4-6 |
| `quiz_blueprint_planner` | Plan question type distribution and batching strategy | 1 | `worker_mid` | claude-sonnet-4-6 |
| `question_drafter` | Draft one batch of MCQ question stems | 1 (batch loop) | `orchestrator` | claude-opus-4-7 |
| `false_trap_red_herring_generator` | Generate MCQ distractors using distractor taxonomy | 1 (batch loop) | `orchestrator` | claude-opus-4-7 |
| `question_evaluator` | Evaluate current batch quality; route to revise or accept | 1 (batch loop) | `worker_mid` | claude-sonnet-4-6 |
| `reviser` | Revise failing batch based on evaluator feedback | 1 (≤ 2 loops) | `orchestrator` | claude-opus-4-7 |
| `grounder` | Ground accepted batch questions against source chunks | 1 (batch loop) | `worker_low` | claude-haiku-4-5-20251001 |
| `batch_commit` | Write grounded batch to quiz_questions + quiz_answers DB; advance batch index | 1 | `tool_only` | N/A |
| `critic` | Final quality review across all accepted questions | 1 | `worker_low` | claude-haiku-4-5-20251001 |
| `final_formatter` | Format and emit final quiz markdown | 1 | `worker_low` | claude-haiku-4-5-20251001 |

## Worker class → model (anthropic default)

| Worker Class | Model | Role |
|---|---|---|
| `tool_only` | N/A | Deterministic DB writes — no LLM call |
| `worker_low` | claude-haiku-4-5-20251001 | Source profiling, grounding, critic scoring, formatting |
| `worker_mid` | claude-sonnet-4-6 | Planning, concept synthesis, batch evaluation |
| `orchestrator` | claude-opus-4-7 | Rule extraction, MCQ drafting, distractor generation, revision |
