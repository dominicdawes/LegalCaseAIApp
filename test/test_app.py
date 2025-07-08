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

from fastapi.testclient import TestClient
from app.main import app


def test_root_endpoint():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"success": "Hello Server LegalNoteAI FastAPI App"}
