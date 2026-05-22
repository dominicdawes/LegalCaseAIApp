# cold_call — Agent Map

**14 nodes** · Provider env var: `COLD_CALL_AGENT_PROVIDER` (default: `anthropic`)

```
Phase 1 — Source Understanding:
  head_orchestrator → [Send×N_docs] source_profiler
  source_profiler   → corpus_synthesizer                  (fan-in: all docs)
  corpus_synthesizer → [Send×N_cases] case_rule_extractor
  case_rule_extractor → doctrine_mapper                   (fan-in: all cases)
  doctrine_mapper → compare_distinguish_mapper
  compare_distinguish_mapper → question_type_bank_selector

Phase 2 — Seed Generation with Diversity Retry:
  question_type_bank_selector → [Send×N_cases] cold_call_seed_generator
  cold_call_seed_generator → seed_diversity_agent         (fan-in: all seeds)
  seed_diversity_agent → (seeds_routing?)
    → [Send×N_cases] cold_call_seed_generator             (retry ≤ 2)
    → [Send×N_seeds] socratic_thread_builder

Phase 3 — Answer + QA:
  socratic_thread_builder → [Send×N_threads] socratic_answer_agent
  socratic_answer_agent   → [Send×N_answers] grounder_agent
  grounder_agent → critic_coverage_agent                  (fan-in: all answers)
  critic_coverage_agent → formatter_export_agent → END
```

| Node Name | Node Function | Fan-out Quantity | Worker Class | Default LLM Model |
|---|---|---|---|---|
| `head_orchestrator` | Plan job, validate sources, set difficulty/sequence count | 1 | `orchestrator` | claude-opus-4-7 |
| `source_profiler` | Profile one source doc (TOC, holdings, rules) | × N_docs | `worker_mid` | claude-sonnet-4-6 |
| `corpus_synthesizer` | Cross-doc synthesis; identify cold-callable cases | 1 | `worker_mid` | claude-sonnet-4-6 |
| `case_rule_extractor` | Extract rules, holdings, and elements for one case | × N_cases | `orchestrator` | claude-opus-4-7 |
| `doctrine_mapper` | Map extracted rules into a doctrinal taxonomy | 1 | `orchestrator` | claude-opus-4-7 |
| `compare_distinguish_mapper` | Build compare/distinguish argument map across cases | 1 | `worker_mid` | claude-sonnet-4-6 |
| `question_type_bank_selector` | Select cold-call question types from taxonomy per case | 1 | `worker_mid` | claude-sonnet-4-6 |
| `cold_call_seed_generator` | Generate seed cold-call questions for one case | × N_cases | `orchestrator` | claude-opus-4-7 |
| `seed_diversity_agent` | Review seed diversity; approve or route back for retry | 1 (≤ 2 retries) | `worker_mid` | claude-sonnet-4-6 |
| `socratic_thread_builder` | Build full Socratic Q→A→follow-up thread from one seed | × N_seeds | `orchestrator` | claude-opus-4-7 |
| `socratic_answer_agent` | Generate model answers for one Socratic thread | × N_threads | `orchestrator` | claude-opus-4-7 |
| `grounder_agent` | Ground one answer sequence against source chunks | × N_answers | `worker_mid` | claude-sonnet-4-6 |
| `critic_coverage_agent` | Score coverage; flag gaps across all sequences | 1 | `orchestrator` | claude-opus-4-7 |
| `formatter_export_agent` | Format and write to cold_call_sequences DB table | 1 | `worker_low` | claude-haiku-4-5-20251001 |

## Worker class → model (anthropic default)

| Worker Class | Model | Role |
|---|---|---|
| `tool_only` | N/A | Deterministic retrieval / DB writes — no LLM call |
| `worker_low` | claude-haiku-4-5-20251001 | Final export formatting |
| `worker_mid` | claude-sonnet-4-6 | Synthesis, diversity checks, grounding |
| `orchestrator` | claude-opus-4-7 | Orchestration, rule extraction, thread building, critique |
