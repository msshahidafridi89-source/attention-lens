# AttentionLens

A browser-based prototype that estimates learner attention in real time during a short lesson, adapts the session when attention drifts, and produces an attention and metacognition report.

Built as a prototype for research on **multimodal learning analytics and adaptive learning environments**.

## What it does

1. The learner watches a ~2 min video (or reads the transcript).
2. Webcam frames are processed **in the browser** (MediaPipe Face Landmarker). Only numeric features leave the device; no video is recorded or uploaded.
3. Every 5 s the app computes a feature window: face presence, blink rate, gaze off-screen ratio, gaze variability, head yaw/pitch, head motion, tab visibility, and pause/seek events.
4. The backend model returns a mind-wandering risk. When smoothed risk stays high, the lesson pauses and offers a replay (adaptive support).
5. Two thought probes ("Was your mind wandering?") give self-report labels.
6. After the lesson, an LLM (Groq) generates 5 questions from the transcript and grades short answers. Learners rate confidence per question.
7. The report shows an attention timeline, probes, per-question attention at the relevant lesson segment, confidence vs accuracy, and an LLM-written summary.

The LLM never judges attention. It only writes questions, grades answers, and summarises the report.

## Architecture

```
frontend (HTML/CSS/JS, MediaPipe in browser)
   | numeric features every 5 s                | quiz, grading, report
   v                                           v
backend (FastAPI) --> attention_model.py     Groq (openai/gpt-oss-120b)
   (trained model or heuristic fallback)
```

## Run locally

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY
uvicorn main:app --reload   # http://localhost:8000

# frontend (any static server; camera needs localhost or https)
cd frontend
python -m http.server 5500  # open http://localhost:5500
```

Add your lesson video as `frontend/assets/lesson.mp4` and keep `backend/content/transcript.txt` in sync (format: `[mm:ss] text`). The included transcript is a sample on the spacing effect; record or generate a matching 2 min video, or replace both.

## Training and evaluation

```bash
cd backend
python train.py                          # synthetic data: tests the pipeline only
python train.py --real sessions/*.json   # real sessions exported from the report page
```

- Subject-wise cross-validation (GroupKFold), logistic regression and gradient boosting baselines.
- Leave-one-modality-out ablations (gaze, head, face/blink, behaviour).
- Real-data labels: windows in the 10 s before each thought probe get that probe's answer.
- Until a model is trained, a transparent heuristic is used (`/api/health` shows which one is active).

## Limitations (stated openly)

- Webcam gaze from blendshapes is noisy and not a substitute for an eye tracker.
- Synthetic-data scores only show the pipeline works; they are not evidence of validity.
- Two probes per session give very few labels. A real study needs more probes and participants.
- Probe responses are self-reports and can themselves be inaccurate.

## Proposed extensions

- Validate webcam features against an eye tracker (e.g. fixation and saccade measures).
- Run a user study (n ≥ 30), then personalise models per learner.
- Add video, audio and log-data modalities, and learner-state models for metacognitive monitoring.
- Compare adaptive strategies (replay, retrieval prompt, break) in a randomised experiment.

## Deploy

- Frontend: Vercel or Netlify (static). Set `frontend/config.js` to your backend URL.
- Backend: Render or Railway (`uvicorn main:app --host 0.0.0.0 --port $PORT`). Set `GROQ_API_KEY` and `CORS_ORIGINS`.
