import hashlib, json, os, re, threading
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from attention_model import AttentionModel

load_dotenv()
BASE = Path(__file__).parent
TRANSCRIPT = BASE / "content" / "transcript.txt"
QPROMPT = BASE / "content" / "question_prompt.txt"
FALLBACK = BASE / "content" / "questions_fallback.json"
CACHE = BASE / "cache"

# Primary model first, then fallbacks if it is unavailable or returns unusable output.
_primary = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MODELS = list(dict.fromkeys([_primary, "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]))

app = FastAPI(title="AttentionLens API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)
attention = AttentionModel()
_client = None
_lock = threading.Lock()


def groq_client():
    global _client
    if _client is None:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise HTTPException(500, "GROQ_API_KEY is not set in backend/.env")
        _client = Groq(api_key=key, timeout=30.0, max_retries=1)
    return _client


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        elif ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON in the reply")


def llm_json(system: str, user: str, max_tokens: int = 3000, check=None):
    """Try each model (with and without JSON mode) until one returns valid, checked JSON."""
    c = groq_client()
    last = None
    for model in MODELS:
        for use_format in (True, False):
            kwargs = dict(model=model, temperature=0.2, max_tokens=max_tokens,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if use_format:
                kwargs["response_format"] = {"type": "json_object"}
            if model.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            try:
                resp = c.chat.completions.create(**kwargs)
                data = extract_json(resp.choices[0].message.content or "")
                if check and not check(data):
                    raise ValueError("reply failed validation")
                return data, model
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"[llm] {model} json_mode={use_format} failed: {type(e).__name__}: {str(e)[:160]}")
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                    raise HTTPException(502, "Groq rejected the API key. Check GROQ_API_KEY in backend/.env.")
    raise HTTPException(502, f"The language model is unavailable: {str(last)[:200]}")


def load_segments():
    segs = []
    for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[(\d+):(\d{2})\]\s*(.+)$", line.strip())
        if m:
            segs.append({"t": int(m.group(1)) * 60 + int(m.group(2)), "text": m.group(3)})
    return segs


# ---------------- questions: validated, cached on disk, pre-generated, with a fallback set ----------------
def clean_questions(raw, segs):
    valid_ts = [s["t"] for s in segs]
    out = []
    for q in raw if isinstance(raw, list) else []:
        if not isinstance(q, dict) or not q.get("question"):
            continue
        t = q.get("type")
        item = {"type": t, "question": str(q["question"])}
        if t == "mcq":
            opts = q.get("options")
            try:
                ai = int(q.get("answer_index"))
            except (TypeError, ValueError):
                continue
            if not (isinstance(opts, list) and len(opts) == 4 and 0 <= ai < 4):
                continue
            item.update(options=[str(o) for o in opts], answer_index=ai)
        elif t == "short":
            if not q.get("rubric"):
                continue
            item["rubric"] = str(q["rubric"])
        else:
            continue
        try:
            st = int(q.get("segment_t"))
        except (TypeError, ValueError):
            st = 0
        item["segment_t"] = min(valid_ts, key=lambda x: abs(x - st)) if valid_ts else st
        out.append(item)
    out.sort(key=lambda q: q["segment_t"])
    for i, q in enumerate(out, 1):
        q["id"] = i
    return out


def questions_cache_file():
    segs = load_segments()
    transcript = "\n".join(f"[{s['t']}s] {s['text']}" for s in segs)
    instructions = QPROMPT.read_text(encoding="utf-8").strip()
    key = hashlib.sha256((transcript + "|" + instructions + "|v2").encode()).hexdigest()[:16]
    return CACHE / f"questions_{key}.json", segs, transcript, instructions


def build_questions():
    cache_file, segs, transcript, instructions = questions_cache_file()
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8")), "cache"
    system = (
        "You write comprehension questions for a short lesson. Return ONLY JSON: "
        '{"questions":[{"type":"mcq"|"short","question":str,'
        '"options":[str,str,str,str] (mcq only),"answer_index":int 0-3 (mcq only),'
        '"rubric":str (short only: what a correct answer must contain),'
        '"segment_t":int (start second of the transcript line the question is based on)}]}'
    )
    user = (instructions + "\n\nUse only timestamps that appear in the transcript for segment_t."
            + "\n\nTranscript:\n" + transcript)
    data, model = llm_json(system, user, 3000, check=lambda d: len(clean_questions(d.get("questions"), segs)) >= 3)
    qs = clean_questions(data.get("questions"), segs)
    CACHE.mkdir(exist_ok=True)
    result = {"questions": qs, "model": model}
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    return result, "llm"


def _warm():
    try:
        with _lock:
            build_questions()
        print("[warm-up] questions ready")
    except Exception as e:  # noqa: BLE001
        print("[warm-up] skipped:", e)


@app.on_event("startup")
def startup():
    threading.Thread(target=_warm, daemon=True).start()


class AttentionIn(BaseModel):
    features: dict[str, float]


class GradeIn(BaseModel):
    questions: list[dict]
    answers: list[dict]


class ReportIn(BaseModel):
    summary: dict


@app.get("/api/health")
def health():
    ready = False
    try:
        ready = questions_cache_file()[0].exists()
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "attention_model": attention.source, "llm_models": MODELS,
            "key_set": bool(os.getenv("GROQ_API_KEY")), "questions_ready": ready}


@app.get("/api/selftest")
def selftest():
    data, model = llm_json('Return ONLY JSON: {"ok": true}', "ping", 200)
    return {"ok": True, "model": model}


@app.get("/api/lesson")
def lesson():
    return {"segments": load_segments()}


@app.post("/api/attention")
def attention_endpoint(body: AttentionIn):
    risk, source = attention.predict(body.features)
    return {"risk": risk, "source": source}


@app.post("/api/questions")
def questions():
    with _lock:
        try:
            data, source = build_questions()
            return {"questions": data["questions"], "source": source}
        except Exception as e:  # noqa: BLE001
            print("[questions] generation failed:", e)
            if FALLBACK.exists():
                fb = json.loads(FALLBACK.read_text(encoding="utf-8"))
                return {"questions": fb["questions"], "source": "fallback", "error": str(e)[:200]}
            raise


# ---------------- grading ----------------
def keyword_score(answer: str, rubric: str) -> float:
    toks = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3}  # noqa: E731
    r, a = toks(rubric), toks(answer)
    return min(1.0, 1.5 * len(r & a) / len(r)) if r else 0.0


@app.post("/api/grade")
def grade(body: GradeIn):
    ans = {a["id"]: a for a in body.answers}
    results, short_items = [], []
    for q in body.questions:
        a = ans.get(q["id"], {})
        conf = a.get("confidence")
        if q["type"] == "mcq":
            ok = a.get("answer") == q.get("answer_index")
            results.append({"id": q["id"], "correct": bool(ok), "score": 1.0 if ok else 0.0,
                            "confidence": conf, "feedback": f"Correct answer: {q['options'][q['answer_index']]}"})
        else:
            results.append({"id": q["id"], "correct": False, "score": 0.0, "confidence": conf, "feedback": ""})
            short_items.append({"id": q["id"], "question": q["question"], "rubric": q.get("rubric", ""),
                                "answer": (a.get("answer") or "").strip()})
    grader = "none"
    if short_items:
        by_id = {}
        try:
            system = ("You grade short answers strictly against the rubric. Return ONLY JSON: "
                      '{"grades":[{"id":int,"score":number between 0 and 1,"feedback":str (one sentence)}]}')
            data, _ = llm_json(system, json.dumps(short_items), 1500, check=lambda d: isinstance(d.get("grades"), list))
            by_id = {g["id"]: g for g in data["grades"] if isinstance(g, dict) and "id" in g}
            grader = "llm"
        except Exception as e:  # noqa: BLE001
            print("[grade] LLM grading failed, using keyword match:", e)
            grader = "keyword"
        items = {i["id"]: i for i in short_items}
        for r in results:
            if r["id"] not in items:
                continue
            if r["id"] in by_id:
                g = by_id[r["id"]]
                r["score"] = max(0.0, min(1.0, float(g.get("score", 0))))
                r["feedback"] = g.get("feedback", "")
            else:
                r["score"] = keyword_score(items[r["id"]]["answer"], items[r["id"]]["rubric"])
                r["feedback"] = "Graded by keyword match (the AI grader was unavailable)."
            r["correct"] = r["score"] >= 0.6
    total = sum(r["score"] for r in results) / max(len(results), 1)
    return {"results": results, "score_pct": round(100 * total), "grader": grader}


# ---------------- report ----------------
def fallback_report(s: dict) -> dict:
    q, qs = s.get("quiz", {}), s.get("questions", [])
    dq, snaps = s.get("data_quality", {}), s.get("snapshots", [])
    conf_wrong = [x["id"] for x in qs if not x.get("correct") and (x.get("confidence") or 0) >= 4]
    parts = [f"You scored {q.get('score_pct')}% on the quiz.",
             f"Estimated attention risk during the lesson averaged {s.get('mean_risk')} (0 = focused, 1 = distracted), "
             f"with {s.get('pct_high_risk_windows')}% of the time in the high-risk range."]
    parts.append(f"{len(snaps)} snapshot(s) were taken at moments of visible distraction." if snaps
                 else "No clear drops in attention were detected.")
    if conf_wrong:
        parts.append("Question(s) " + ", ".join(map(str, conf_wrong)) + " were answered wrongly with high confidence, which suggests overconfidence.")
    if dq.get("warnings"):
        parts.append("Some measures were less reliable in this session; see the data quality notes.")
    recs = ["Review the questions you missed, starting with any answered with high confidence."]
    recs.append("Compare the snapshot moments with the lesson content at those times." if snaps
                else "Repeat the session with more participants to see whether this pattern holds.")
    recs.append("Treat results from a single session as anecdotal.")
    return {"summary": " ".join(parts), "recommendations": recs, "source": "fallback"}


@app.post("/api/report")
def report(body: ReportIn):
    system = (
        "You write a short, plain-language learning report from webcam-based attention estimates, "
        "self-reported thought probes, and quiz results. Rules:\n"
        "- The attention signal is approximate, not a diagnosis. Make no medical or psychological claims.\n"
        "- Read data_quality first. If blink detection is unreliable, gaze is relative, the frame rate is low, or ad time was excluded, "
        "say which numbers to treat with caution. Never interpret blink numbers when blink detection is unreliable.\n"
        "- Do not read meaning into small differences: gaze percentages within 10 points of each other, or risk values within 0.1.\n"
        "- Lesson time (seconds into the video) and quiz time (seconds into the quiz) are different clocks. Never mix them: "
        "say 'at 0:40 of the lesson' or 'during question 3', not a bare timestamp.\n"
        "- Point out confident-but-wrong answers (confidence 4 or 5 and correct=false) as a metacognitive finding, and note under-confident correct answers.\n"
        "- If snapshots exist, mention their lesson/quiz context and visible reasons. If there are none, do not invent lapses.\n"
        "- If the data shows no attention problems, say so plainly and do not give generic advice about breaks.\n"
        "- Recommendations must follow from this session's data.\n"
        'Return ONLY JSON: {"summary": str (max 150 words), "recommendations": [str, str, str]}'
    )
    try:
        data, _ = llm_json(system, json.dumps(body.summary), 2000,
                           check=lambda d: isinstance(d.get("summary"), str) and isinstance(d.get("recommendations"), list))
        data["source"] = "llm"
        return data
    except Exception as e:  # noqa: BLE001
        print("[report] LLM failed, using rule-based summary:", e)
        return fallback_report(body.summary)