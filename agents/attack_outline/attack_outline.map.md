# attack_outline — Agent Map

**15 nodes** · Provider env var: `ATTACK_AGENT_PROVIDER` (default: `anthropic`)

```
head_orchestrator → [Send×N_docs] source_profiler
source_profiler   → corpus_topic_mapper           (fan-in: all docs)
corpus_topic_mapper → retrieval_planner
retrieval_planner → [Send×N_plans] planned_retriever
planned_retriever → [Send×N_bundles] legal_artifact_extractor
legal_artifact_extractor → artifact_normalizer    (fan-in: all bundles)
artifact_normalizer → concept_clusterer
concept_clusterer → doctrine_graph_builder
doctrine_graph_builder → [Send×N_clusters] attack_block_builder
attack_block_builder → attack_outline_assembler   (fan-in: all clusters)
attack_outline_assembler → [Send×N_blocks] grounding_verifier
grounding_verifier → attack_outline_critic        (fan-in: all blocks)
attack_outline_critic → (should_revise?)
  → revision_agent → attack_outline_critic        (loop ≤ 2)
  → final_compressor_formatter → END
```

| Node Name | Node Function | Fan-out Quantity | Worker Class | Default LLM Model |
|---|---|---|---|---|
| `head_orchestrator` | Plan job, validate sources, set execution strategy | 1 | `orchestrator` | claude-opus-4-7 |
| `source_profiler` | Profile one source doc (doctrine list, TOC, key concepts) | × N_docs | `worker_mid` | claude-sonnet-4-6 |
| `corpus_topic_mapper` | Cross-doc topic map from all merged profiles | 1 | `worker_mid` | claude-sonnet-4-6 |
| `retrieval_planner` | Generate per-concept retrieval plans (BM25, vector, regex) | 1 | `orchestrator` | claude-opus-4-7 |
| `planned_retriever` | Execute + rerank retrieval for one concept plan | × N_plans | `tool_only` | N/A |
| `legal_artifact_extractor` | Extract typed legal artifacts from one retrieval bundle | × N_bundles | `worker_mid` | claude-sonnet-4-6 |
| `artifact_normalizer` | Deduplicate and normalise all raw artifacts | 1 | `worker_low` | claude-haiku-4-5-20251001 |
| `concept_clusterer` | Group normalised artifacts into doctrine modules | 1 | `worker_low` | claude-haiku-4-5-20251001 |
| `doctrine_graph_builder` | Build if/then doctrine graph from clusters | 1 | `orchestrator` | claude-opus-4-7 |
| `attack_block_builder` | Build one attack block per doctrine cluster | × N_clusters | `worker_mid` | claude-sonnet-4-6 |
| `attack_outline_assembler` | Order and assemble all blocks into outline markdown | 1 | `tool_only` | N/A |
| `grounding_verifier` | Verify source citations for one attack block | × N_blocks | `worker_mid` | claude-sonnet-4-6 |
| `attack_outline_critic` | Score outline quality; issue revision targets | 1 | `orchestrator` | claude-opus-4-7 |
| `revision_agent` | Revise flagged blocks and re-assemble outline | 1 (≤ 2 loops) | `worker_mid` | claude-sonnet-4-6 |
| `final_compressor_formatter` | Final polish and student-ready formatting | 1 | `worker_low` | claude-haiku-4-5-20251001 |

## Worker class → model (anthropic default)

| Worker Class | Model | Role |
|---|---|---|
| `tool_only` | N/A | Deterministic retrieval / DB writes — no LLM call |
| `worker_low` | claude-haiku-4-5-20251001 | Deduplication, clustering, final formatting |
| `worker_mid` | claude-sonnet-4-6 | Core extraction, normalisation, block building |
| `orchestrator` | claude-opus-4-7 | Planning, critique, graph building |
