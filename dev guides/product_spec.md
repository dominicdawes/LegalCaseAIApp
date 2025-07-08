# Legalnote.io Product Specification

## Overview
Legalnote.io is a verticalized B2C SaaS platform tailored to first year law students (1Ls) across the United States. Modeled after Google NotebookLM, it acts as an AI-driven education partner that streamlines case prep, test prep, and note management.

## Key Features
- **Case Repository**: A curated database of essential cases for core 1L courses.
- **Note-Taking Workspace**: Integrated note editor that links student notes with uploaded cases.
- **Flashcards & Outlines**: Automatically generate flashcards, case briefs, and course outlines from uploaded materials.
- **Exam Question Generation**: Create hypothetical questions and answers for self-assessment.
- **Chat-Based Retrieval**: Use RAG to answer questions about uploaded documents and existing notes.
- **Podcast Creation**: Convert outlines or chat dialogues into audio files for review on the go.

## Technical Architecture
- **FastAPI Backend**: Hosts REST endpoints for uploading documents, creating RAG pipelines, generating notes, and producing audio.
- **Celery Workers**: Handle long-running tasks such as PDF processing, embedding with Supabase/S3, and generating audio via OpenAI TTS.
- **Supabase & S3**: Store user documents and embeddings with scalable storage and CDN delivery through CloudFront.
- **LLM Providers**: Modular factory pattern supports OpenAI, DeepSeek, Anthropic, Gemini, and more for chat and generation tasks.
- **LangChain RAG**: Implements retrieval-augmented generation workflows using vector stores and structured prompts.
- **Audio Utilities**: Transform dialogue into MP3 files, enabling “study by listening” podcasts.

## User Workflow
1. **Upload** PDF cases or class notes through the web interface.
2. **Processing**: Backend tasks convert documents into text, generate embeddings, and store them.
3. **Interact**: Students chat with the system to receive case briefs, flashcards, or clarifying explanations.
4. **Study**: Create outlines or exam-style questions and optionally convert them into audio podcasts for mobile review.

## Goals & Future Work
- Provide comprehensive legal study support in a single platform.
- Expand document parsing to handle a variety of file types.
- Integrate additional AI models as they become available.
- Build collaborative features so students can share notes and flashcards securely.

