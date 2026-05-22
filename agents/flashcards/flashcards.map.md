# flashcards — Agent Map

**11 nodes** · Provider env var: `FLASHCARD_AGENT_PROVIDER` (default: `anthropic`)

```
Phase 1 — Source Understanding:
  head_orchestrator → [Send×N_docs] source_profiler
  source_profiler → concept_extractor            (fan-in: all docs)
  concept_extractor → card_blueprint_planner

Phase 2 — Batch Loop (repeats per batch until num_cards reached):
  card_blueprint_planner → flashcard_drafter ◄───────────────────────┐
  flashcard_drafter → answer_backside_enricher                        │
  answer_backside_enricher → local_card_critic                        │
  local_card_critic → (should_repair_batch?)                          │
    → card_repair_agent → local_card_critic      (loop ≤ 2)           │
    → batch_commit ─────────────────────────────────────────────────── ┘
      → global_deck_critic  (when all batches done)

Phase 3 — Final QA + Persistence:
  global_deck_critic → deterministic_formatter_persister → END
```

| Node Name | Node Function | Fan-out Quantity | Worker Class | Default LLM Model |
|---|---|---|---|---|
| `head_orchestrator` | Plan card targets, validate sources, set batch params | 1 | `worker_mid` | claude-sonnet-4-6 |
| `source_profiler` | Profile one source doc | × N_docs | `worker_low` | claude-haiku-4-5-20251001 |
| `concept_extractor` | Extract 33-type card concepts from all merged profiles | 1 | `orchestrator` | claude-opus-4-7 |
| `card_blueprint_planner` | Plan card type distribution and batching strategy | 1 | `worker_mid` | claude-sonnet-4-6 |
| `flashcard_drafter` | Draft one batch of flashcards (front + back) | 1 (batch loop) | `orchestrator` | claude-opus-4-7 |
| `answer_backside_enricher` | Enrich back-side with memory hooks, examples, mnemonics | 1 (batch loop) | `worker_mid` | claude-sonnet-4-6 |
| `local_card_critic` | Evaluate batch quality (clarity, accuracy); route repair or commit | 1 (batch loop) | `worker_mid` | claude-sonnet-4-6 |
| `card_repair_agent` | Repair flagged cards in current batch | 1 (≤ 2 loops) | `orchestrator` | claude-opus-4-7 |
| `batch_commit` | Write accepted batch to individual_cards DB; advance batch index | 1 | `tool_only` | N/A |
| `global_deck_critic` | Review final deck for coverage and consistency gaps | 1 | `worker_low` | claude-haiku-4-5-20251001 |
| `deterministic_formatter_persister` | Format deck summary and finalise DB records | 1 | `tool_only` | N/A |

## Worker class → model (anthropic default)

| Worker Class | Model | Role |
|---|---|---|
| `tool_only` | N/A | Deterministic DB writes — no LLM call |
| `worker_low` | claude-haiku-4-5-20251001 | Source profiling, global deck critique |
| `worker_mid` | claude-sonnet-4-6 | Blueprint planning, back-side enrichment, local critic |
| `orchestrator` | claude-opus-4-7 | Concept extraction, card drafting, repair |
