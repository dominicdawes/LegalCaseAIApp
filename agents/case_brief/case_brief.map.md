# case_brief — Agent Map

**17 nodes** · Provider env var: `BRIEF_AGENT_PROVIDER` (default: `anthropic`)

```
head_orchestrator       → [Send×N_docs] source_profiler
source_profiler         → corpus_orientation_synthesizer   (fan-in: all docs)
corpus_orientation_synthesizer → retrieval_planner
retrieval_planner       → [Send×N_targets] planned_retriever
planned_retriever       → [Send×N_bundles] evidence_card_builder
evidence_card_builder   → [Send×4] legal_artifact_extractor (4 artifact types per card)
legal_artifact_extractor → doctrinal_synthesizer           (fan-in: all types)
doctrinal_synthesizer   → brief_drafter
brief_drafter           → [Send×10] section_writer         (10 brief section types)
section_writer          → [Send×N_sections] section_grounder
section_grounder        → section_reviser                  (fan-in: all sections)
section_reviser         → brief_assembler
brief_assembler         → global_coherence_editor
global_coherence_editor → critic
critic → (should_revise?)
  → brief_revision_agent → critic                          (loop ≤ 2)
  → final_formatter → END
```

| Node Name | Node Function | Fan-out Quantity | Worker Class | Default LLM Model |
|---|---|---|---|---|
| `head_orchestrator` | Plan brief structure, validate sources | 1 | `orchestrator` | claude-opus-4-7 |
| `source_profiler` | Profile one source doc | × N_docs | `worker_mid` | claude-sonnet-4-6 |
| `corpus_orientation_synthesizer` | Synthesise cross-doc orientation for brief scoping | 1 | `worker_mid` | claude-sonnet-4-6 |
| `retrieval_planner` | Generate per-target retrieval plans | 1 | `orchestrator` | claude-opus-4-7 |
| `planned_retriever` | Execute + rerank retrieval for one retrieval target | × N_targets | `tool_only` | N/A |
| `evidence_card_builder` | Build structured evidence cards from one retrieval bundle | × N_bundles | `worker_low` | claude-haiku-4-5-20251001 |
| `legal_artifact_extractor` | Extract one artifact type from one evidence card (4 types per card) | 4 per card | `worker_mid` | claude-sonnet-4-6 |
| `doctrinal_synthesizer` | Synthesise all extracted artifacts into unified doctrinal map | 1 | `worker_mid` | claude-sonnet-4-6 |
| `brief_drafter` | Plan brief section structure and content scope | 1 | `orchestrator` | claude-opus-4-7 |
| `section_writer` | Write one brief section (facts / issue / rule / analysis / conclusion / etc.) | 10 (section types) | `worker_mid` | claude-sonnet-4-6 |
| `section_grounder` | Ground one written section against source citations | × N_sections | `worker_mid` | claude-sonnet-4-6 |
| `section_reviser` | Sequentially revise all grounded sections | 1 | `worker_mid` | claude-sonnet-4-6 |
| `brief_assembler` | Merge all revised sections into draft brief | 1 | `tool_only` | N/A |
| `global_coherence_editor` | Edit brief for cross-section coherence and flow | 1 | `worker_mid` | claude-sonnet-4-6 |
| `critic` | Score brief quality; issue revision targets | 1 | `orchestrator` | claude-opus-4-7 |
| `brief_revision_agent` | Revise flagged sections based on critic feedback | 1 (≤ 2 loops) | `worker_mid` | claude-sonnet-4-6 |
| `final_formatter` | Final polish and student-ready formatting | 1 | `worker_low` | claude-haiku-4-5-20251001 |

## Worker class → model (anthropic default)

| Worker Class | Model | Role |
|---|---|---|
| `tool_only` | N/A | Deterministic retrieval / DB writes — no LLM call |
| `worker_low` | claude-haiku-4-5-20251001 | Evidence card building, final formatting |
| `worker_mid` | claude-sonnet-4-6 | Profiling, synthesis, section writing, grounding, revision |
| `orchestrator` | claude-opus-4-7 | Planning, drafter structuring, critique |
