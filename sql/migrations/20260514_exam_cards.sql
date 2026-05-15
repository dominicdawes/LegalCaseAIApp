-- =============================================================================
-- Migration: exam_questions + exam_answers tables with RLS
--
-- exam_questions  — one row per exam question; parent: public.notes (exam_id)
-- exam_answers    — one answer key per question; dual FK to notes + exam_questions
--
-- RLS policies
--   service_role  → ALL (unrestricted; used by Celery workers / backend)
--   authenticated → ALL on own rows (user_id = auth.uid() on exam_questions;
--                   join-check on exam_answers since it has no user_id column)
--
-- Safe to run multiple times — uses IF NOT EXISTS throughout.
-- Apply in the Supabase SQL editor.
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 0. Drop existing tables (answers first — it holds the FK to questions)
-- ─────────────────────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS public.exam_answers   CASCADE;
DROP TABLE IF EXISTS public.exam_questions CASCADE;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. exam_questions
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE public.exam_questions (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id           UUID        NOT NULL REFERENCES public.notes(id)   ON DELETE CASCADE,
    user_id           UUID                 REFERENCES auth.users(id)     ON DELETE SET NULL,
    question_index    INTEGER     NOT NULL,
    issue_label       TEXT,
    fact_pattern      TEXT        NOT NULL,
    call_of_question  TEXT        NOT NULL,
    grounding_verdict TEXT        NOT NULL DEFAULT 'pass'
                                  CHECK (grounding_verdict IN ('pass', 'warn', 'fail')),
    revised           BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_exam_questions_exam_id  ON public.exam_questions(exam_id);
CREATE INDEX idx_exam_questions_user_id  ON public.exam_questions(user_id);

-- Enable RLS
ALTER TABLE public.exam_questions ENABLE ROW LEVEL SECURITY;

-- service_role: unrestricted (backend Celery workers use this key)
-- Note: service_role bypasses RLS by default in Supabase; this policy makes
-- the intent explicit and survives any future RLS setting changes.
CREATE POLICY "service_role_all_exam_questions"
    ON public.exam_questions
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- authenticated users: full CRUD on their own rows only
CREATE POLICY "authenticated_select_own_exam_questions"
    ON public.exam_questions
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "authenticated_insert_own_exam_questions"
    ON public.exam_questions
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "authenticated_update_own_exam_questions"
    ON public.exam_questions
    FOR UPDATE
    TO authenticated
    USING      (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "authenticated_delete_own_exam_questions"
    ON public.exam_questions
    FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. exam_answers
-- ─────────────────────────────────────────────────────────────────────────────
-- No user_id column here — ownership is determined by joining through
-- exam_questions.  This keeps the table normalised and the check is
-- identical whether you're selecting, inserting, updating or deleting.

CREATE TABLE public.exam_answers (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id     UUID        NOT NULL REFERENCES public.notes(id)          ON DELETE CASCADE,
    question_id UUID        NOT NULL REFERENCES public.exam_questions(id) ON DELETE CASCADE,
    answer_key  TEXT        NOT NULL,
    citations   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_exam_answers_exam_id     ON public.exam_answers(exam_id);
CREATE INDEX idx_exam_answers_question_id ON public.exam_answers(question_id);

-- Enable RLS
ALTER TABLE public.exam_answers ENABLE ROW LEVEL SECURITY;

-- service_role: unrestricted
CREATE POLICY "service_role_all_exam_answers"
    ON public.exam_answers
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Reusable ownership check: answer belongs to user if its parent question does.
-- Defined once here and referenced in each policy below to keep it DRY.
-- (Supabase does not support CREATE POLICY with a named "helper", so the
--  subquery is repeated — but it's index-covered by idx_exam_questions_user_id.)

CREATE POLICY "authenticated_select_own_exam_answers"
    ON public.exam_answers
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.exam_questions eq
            WHERE eq.id = exam_answers.question_id
              AND eq.user_id = auth.uid()
        )
    );

CREATE POLICY "authenticated_insert_own_exam_answers"
    ON public.exam_answers
    FOR INSERT
    TO authenticated
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.exam_questions eq
            WHERE eq.id = exam_answers.question_id
              AND eq.user_id = auth.uid()
        )
    );

CREATE POLICY "authenticated_update_own_exam_answers"
    ON public.exam_answers
    FOR UPDATE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.exam_questions eq
            WHERE eq.id = exam_answers.question_id
              AND eq.user_id = auth.uid()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.exam_questions eq
            WHERE eq.id = exam_answers.question_id
              AND eq.user_id = auth.uid()
        )
    );

CREATE POLICY "authenticated_delete_own_exam_answers"
    ON public.exam_answers
    FOR DELETE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.exam_questions eq
            WHERE eq.id = exam_answers.question_id
              AND eq.user_id = auth.uid()
        )
    );
