# exam_questions — Agent Map

**12 nodes** · Provider env var: `EXAM_AGENT_PROVIDER` (default: `anthropic`)

```
planner → [Send×N_docs] source_profiler
source_profiler → concept_synthesizer            (fan-in: all docs)
concept_synthesizer → issue_clusterer
issue_clusterer → [Send×N_issues] retriever
retriever → [Send×N_bundles] question_drafter
question_drafter → [Send×N_drafts] answer_key_builder
answer_key_builder → [Send×N_qa_pairs] grounder
grounder → critic                                (fan-in: all Q&A pairs)
critic → (should_revise?)
  → reviser → critic                             (loop ≤ 2)
  → final_drafter
final_drafter → [Send×N_questions] exam_card_writer → END
```

| Node Name | Node Function | Fan-out Quantity | Worker Class | Default LLM Model |
|---|---|---|---|---|
| `planner` | Plan exam structure; extract concepts from source profiles | 1 | `worker_mid` | claude-sonnet-4-6 |
| `source_profiler` | Profile one source doc (metadata, TOC, key concepts) | × N_docs | `tool_only` | N/A |
| `concept_synthesizer` | Synthesize concepts across all merged profiles | 1 | `worker_mid` | claude-sonnet-4-6 |
| `issue_clusterer` | Cluster concepts into discrete testable legal issues | 1 | `worker_mid` | claude-sonnet-4-6 |
| `retriever` | Execute retrieval for one clustered issue | × N_issues | `tool_only` | N/A |
| `question_drafter` | Draft exam questions for one retrieval bundle | × N_bundles | `orchestrator` | claude-opus-4-7 |
| `answer_key_builder` | Build model answer key for one drafted question set | × N_drafts | `orchestrator` | claude-opus-4-7 |
| `grounder` | Ground claims for one Q&A pair against source chunks | × N_qa_pairs | `worker_low` | claude-haiku-4-5-20251001 |
| `critic` | Evaluate overall exam quality; decide revision | 1 | `worker_low` | claude-haiku-4-5-20251001 |
| `reviser` | Revise questions based on critic feedback | 1 (≤ 2 loops) | `worker_mid` | claude-sonnet-4-6 |
| `final_drafter` | Finalise and format all questions for output | 1 | `orchestrator` | claude-opus-4-7 |
| `exam_card_writer` | Write one exam Q&A card row to exam_questions + exam_answers DB | × N_questions | `tool_only` | N/A |

## Worker class → model (anthropic default)

| Worker Class | Model | Role |
|---|---|---|
| `tool_only` | N/A | Deterministic retrieval / DB writes — no LLM call |
| `worker_low` | claude-haiku-4-5-20251001 | Grounding, critic scoring |
| `worker_mid` | claude-sonnet-4-6 | Planning, clustering, revision |
| `orchestrator` | claude-opus-4-7 | Question drafting, answer keys, final output |
