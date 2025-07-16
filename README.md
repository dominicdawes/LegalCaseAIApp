# LegalCaseAIApp
pdf rag and generative podcast 

Conda envirionment (legalnote python=3.12)


```
$ git config --global http.postBuffer 524288000
git clone https://github.com/dominicdawes/LegalCaseAIApp.git
```

## Rough Project Structure

```

root
├── app/
│   ├── __init__.py
│   └── main.py
│
├── tasks/
│   ├── __init__.py
│   ├── celery_app.py               # Inits Celery client
│   └── chat_tasks.py               # (Minimal changes to call into factory)
│
├── utils/
│   ├── __init__.py
│   ├── prompt_utils.py
│   ├── supabase_utils.py
│   ├── llm_factory.py              # Factory/Dispatcher
│   └── llm_clients/
│       ├── __init__.py
│       ├── openai_client.py
│       ├── deepseek_client.py
│       ├── anthropic_client.py
│       ├── gemini_client.py
│       ├── sonnet_client.py
│       └── qwen_client.py
│
├── docs/
│   └── documentation.md
├── .env
└── requirements.txt


```