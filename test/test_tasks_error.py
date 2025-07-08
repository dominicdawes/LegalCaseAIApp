import os
import sys
import types

promptic_stub = types.ModuleType("promptic")


def _llm(*args, **kwargs):
    def wrapper(f):
        return f

    return wrapper


promptic_stub.llm = _llm
sys.modules.setdefault("promptic", promptic_stub)
sys.modules.setdefault("textract", types.ModuleType("textract"))
sys.modules.setdefault("docx", types.ModuleType("docx"))
sys.modules.setdefault("fitz", types.ModuleType("fitz"))
sys.modules.setdefault("ebooklib", types.ModuleType("ebooklib"))
ebooklib_stub = sys.modules["ebooklib"]
ebooklib_stub.epub = types.SimpleNamespace()
ebooklib_stub.ITEM_DOCUMENT = object()
bs4_stub = types.ModuleType("bs4")
bs4_stub.BeautifulSoup = object
sys.modules.setdefault("bs4", bs4_stub)
sys.modules.setdefault("tasks.test_tasks", types.ModuleType("tasks.test_tasks"))
sys.modules["tasks.test_tasks"].addition_task = lambda x, y: x + y
sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))
sys.modules["anthropic"].Anthropic = object
sys.modules.setdefault("vertexai", types.ModuleType("vertexai"))
vertexai_stub = sys.modules["vertexai"]
gm_mod = types.ModuleType("generative_models")
gm_mod.GenerativeModel = object
vertexai_stub.generative_models = gm_mod
vertexai_stub.__path__ = ["dummy"]
sys.modules["vertexai.generative_models"] = gm_mod
vertexai_stub.init = lambda **kwargs: None
sys.modules.setdefault("google", types.ModuleType("google"))
google_stub = sys.modules["google"]
oauth2_mod = types.ModuleType("oauth2")
oauth2_mod.service_account = object
google_stub.oauth2 = oauth2_mod
sys.modules["google.oauth2"] = oauth2_mod
google_stub.__path__ = ["dummy"]
os.environ.setdefault("SUPABASE_PROJECT_URL", "http://localhost")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "key")
os.environ.setdefault("REDIS_PASSWORD", "")
os.environ.setdefault("REDIS_PUBLIC_ENDPOINT", "localhost:6379")
os.environ.setdefault("REDIS_LABS_URL_AND_PASS", "redis://localhost:6379/0")
os.environ.setdefault("DEEPSEEK_API_KEY", "x")
os.environ.setdefault("DEEPSEEK_BASE_URL", "http://localhost")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("GEMINI_PROJECT_ID", "dummy")

import pytest
from unittest.mock import patch, MagicMock

import tasks.chat_tasks as chat_tasks
import utils.llm_factory as llm_factory
import tasks.note_tasks as note_tasks
import tasks.upload_tasks as upload_tasks


class DummyResponse:
    def __init__(self, data=None):
        self.data = data or [{"id": "1"}]


def test_rag_chat_task_logs_error():
    dummy_llm = types.SimpleNamespace(
        model_name="gpt", chat=lambda prompt: (_ for _ in ()).throw(Exception("fail"))
    )
    dummy_supabase = MagicMock()
    dummy_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = (
        None
    )
    with patch.object(chat_tasks, "supabase_client", dummy_supabase), patch.object(
        chat_tasks, "fetch_relevant_chunks", return_value=[{"content": "x"}]
    ), patch.object(
        chat_tasks, "insert_chat_message_supabase_record", return_value=DummyResponse()
    ), patch(
        "utils.supabase_utils.log_llm_error"
    ) as log_error, patch.object(
        llm_factory.LLMFactory,
        "get_client",
        return_value=dummy_llm,
    ):
        with pytest.raises(Exception):
            chat_tasks.rag_chat_task.run("m", "u", "c", "q", "p", "openai", "gpt")


def test_rag_note_task_logs_error():
    dummy_llm = types.SimpleNamespace(
        chat=lambda prompt: (_ for _ in ()).throw(Exception("fail"))
    )
    dummy_supabase = MagicMock()
    with patch.object(note_tasks, "supabase_client", dummy_supabase), patch.object(
        note_tasks, "fetch_relevant_chunks", return_value=[{"content": "x"}]
    ), patch("utils.supabase_utils.log_llm_error") as log_error, patch.object(
        llm_factory.LLMFactory,
        "get_client",
        return_value=dummy_llm,
    ), patch.object(
        note_tasks, "save_note", return_value=None
    ):
        with pytest.raises(Exception):
            note_tasks.rag_note_task.run(
                "u", "outline", "p", "title", "openai", "gpt", 0.7, {}
            )


def test_finalize_workflow_triggers_note_task():
    dummy_supabase = MagicMock()
    dummy_supabase.table.return_value.select.return_value.in_.return_value.execute.side_effect = [
        types.SimpleNamespace(data=[{"id": "s1", "vector_embed_status": "COMPLETE"}]),
        types.SimpleNamespace(count=1),
    ]
    with patch.object(upload_tasks, "supabase_client", dummy_supabase), patch.object(
        upload_tasks, "rag_note_task"
    ) as note_task:
        upload_tasks.finalize_document_processing_workflow.run(
            results=[None],
            user_id="u",
            source_ids=["s1"],
            project_id="p",
            note_title="t",
            note_type="outline",
            provider="openai",
            model_name="gpt",
            temperature=0.7,
            addtl_params={},
        )
        note_task.apply_async.assert_called_once()
