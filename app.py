import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()

VAPI_API_KEY = os.environ.get("VAPI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# In-memory store: call_id -> score dict
scores: dict[str, Any] = {}


# ── MODELS ───────────────────────────────────────────────────────────────────

class QAPair(BaseModel):
    question: str
    answer: str


class Tip(BaseModel):
    type: str  # "good" | "improve"
    tip: str
    explanation: str = ""


class Section(BaseModel):
    score: int = 0
    tips: list[Tip] = []


class ResumeAnalysis(BaseModel):
    """Matches the `data` object returned by the resume analysis service."""
    overallScore: int = 0
    ATS: Section | None = None
    toneAndStyle: Section | None = None
    content: Section | None = None
    structure: Section | None = None
    skills: Section | None = None


class StartInterviewRequest(BaseModel):
    resume_analysis: ResumeAnalysis  # analysis JSON (the `data` object)
    previous_qa: list[QAPair] = []


class StartInterviewResponse(BaseModel):
    assistant: dict[str, Any]
    vapi_public_key: str


class ScoreRequest(BaseModel):
    transcript: str


# ── PROMPT / ASSISTANT BUILDERS ───────────────────────────────────────────────

# Only these sections help a *spoken* interview. ATS / structure are formatting.
INTERVIEW_SECTIONS = ["content", "skills", "toneAndStyle"]


def build_analysis_context(analysis: ResumeAnalysis | None) -> str:
    """Turn the analysis into a short internal briefing for the interviewer."""
    if not analysis:
        return ""

    sections = [
        getattr(analysis, name)
        for name in INTERVIEW_SECTIONS
        if getattr(analysis, name)
    ]
    # weakest sections first, so the biggest gaps get probed first
    sections.sort(key=lambda s: s.score)

    strengths: list[str] = []
    gaps: list[str] = []
    for section in sections:
        for t in section.tips:
            (strengths if t.type == "good" else gaps).append(t.tip)

    strengths, gaps = strengths[:4], gaps[:6]  # keep the prompt short
    if not strengths and not gaps:
        return ""

    ctx = "\n\nResume analysis (INTERNAL - never mention this to the candidate):\n"
    if strengths:
        ctx += (
            "Claimed strengths to VERIFY "
            "(ask them to explain their own ownership and decisions):\n"
        )
        ctx += "\n".join(f"- {s}" for s in strengths) + "\n"
    if gaps:
        ctx += (
            "Gaps / vague claims to PROBE "
            "(ask about metrics, datasets, trade-offs, or real usage):\n"
        )
        ctx += "\n".join(f"- {g}" for g in gaps) + "\n"
    return ctx


def build_system_prompt(
    qa_pairs: list[QAPair],
    analysis: ResumeAnalysis | None = None,
) -> str:
    qa_context = ""
    if qa_pairs:
        qa_context = "\n\nPrevious Q&A with this candidate:\n"
        for i, qa in enumerate(qa_pairs, 1):
            qa_context += f"Q{i}: {qa.question}\nA{i}: {qa.answer}\n"

    analysis_context = build_analysis_context(analysis)

    return f"""You are Jiya, a senior technical interviewer at a top tech company.
You are conducting a live voice interview with a candidate.
{analysis_context}{qa_context}

Your interview structure:
1. WARM-UP (1 question): a light intro, e.g. "Tell me a bit about yourself."
2. VERIFY (1-2 questions): pick a claimed strength above and ask the candidate to walk you through what THEY personally built and decided.
3. PROBE (2-3 questions): pick different gaps above. Ask about datasets, metrics, evaluation, trade-offs, or whether they have actually used a tool in practice.
If the analysis above is empty, cover general technical depth, system design, problem-solving, and past impact.

Your interview style:
- Ask ONE question at a time. Wait for the full answer before proceeding. Never ask multi-part questions.
- After each answer, ask AT MOST 1-2 follow-ups if the answer is vague or worth probing, then move on.
- NEVER say the resume was analysed, scored, or has weaknesses. Phrase probes naturally, e.g. "How did you measure how well that model worked?"
- Be warm and conversational. Use natural transitions like "That's interesting..." or "Let's shift to..."
- After 5-6 questions total (including warm-up), wrap up: "Those are all my questions for today. Thank you so much!"
- Do NOT repeat questions already covered in the Previous Q&A above.
- Keep responses SHORT.

Start by greeting the candidate warmly and asking your first warm-up question."""


def build_assistant(
    qa_pairs: list[QAPair],
    analysis: ResumeAnalysis | None = None,
) -> dict[str, Any]:
    return {
        "firstMessage": (
            "Hello! I'm Jiya, your interviewer today. "
            "Are you in a quiet place and ready to begin?"
        ),
        "model": {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(qa_pairs, analysis),
                }
            ],
            "temperature": 0.7,
            "maxTokens": 150,
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "S3F8rLt9v7twQC170pA5",
        },
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en",
        },
        "startSpeakingPlan": {"waitSeconds": 1.2},
        "stopSpeakingPlan": {
            "numWords": 5,
            "voiceSeconds": 0.5,
            "backoffSeconds": 2,
        },
        "endCallPhrases": [
            "those are all my questions for today",
            "thank you so much for your time",
            "best of luck",
        ],
        "recordingEnabled": False,
        "silenceTimeoutSeconds": 30,
        "maxDurationSeconds": 600,
        "backgroundDenoisingEnabled": False,
    }


# ── SCORING ───────────────────────────────────────────────────────────────────

SCORE_PROMPT = """You are an expert technical interview evaluator.
Given the transcript of a voice interview, score the candidate on exactly these 5 factors.
Each score is an integer from 1 (very poor) to 10 (excellent).
Also write a 1-sentence detail for each factor explaining the score.

Factors:
1. technical_knowledge   - depth and accuracy of technical answers
2. communication_skills  - clarity, structure, and conciseness of explanations
3. problem_solving       - ability to reason through challenges and find solutions
4. system_design         - understanding of scalability, architecture, trade-offs
5. confidence            - how assured and composed the candidate sounded

Respond with ONLY a valid JSON object in this exact format (no extra text):
{
  "technical_knowledge":  { "score": <1-10>, "detail": "<1 sentence>" },
  "communication_skills": { "score": <1-10>, "detail": "<1 sentence>" },
  "problem_solving":      { "score": <1-10>, "detail": "<1 sentence>" },
  "system_design":        { "score": <1-10>, "detail": "<1 sentence>" },
  "confidence":           { "score": <1-10>, "detail": "<1 sentence>" },
  "overall":              <average of the 5 scores, rounded to 1 decimal>,
  "summary":              "<2-3 sentence hiring recommendation>"
}"""


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse model output as JSON, tolerating ```json fences or stray text."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def score_interview(transcript: str) -> dict[str, Any]:
    if not groq_client:
        return {"error": "GROQ_API_KEY not configured"}

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SCORE_PROMPT},
            {"role": "user", "content": f"Interview transcript:\n\n{transcript}"},
        ],
        temperature=0.2,
        # gpt-oss is a reasoning model; reasoning tokens count toward this limit
        max_tokens=1500,
    )

    return parse_json_response(response.choices[0].message.content or "")


# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Interview Bot API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/interview/start", response_model=StartInterviewResponse)
def start_interview(req: StartInterviewRequest):
    if not VAPI_API_KEY:
        raise HTTPException(status_code=500, detail="VAPI_API_KEY not configured on server")

    assistant = build_assistant(req.previous_qa, req.resume_analysis)
    return StartInterviewResponse(assistant=assistant, vapi_public_key=VAPI_API_KEY)


@app.post("/api/interview/score")
def score_from_transcript(req: ScoreRequest):
    """Score a candidate directly from a transcript (for local/frontend use)."""
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    try:
        return score_interview(req.transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interview/{call_id}/score")
def get_score(call_id: str):
    """Poll this after the call ends to get the candidate's score (webhook flow)."""
    if call_id not in scores:
        raise HTTPException(status_code=404, detail="Score not ready yet")
    return scores[call_id]


@app.post("/api/webhook")
async def vapi_webhook(request: Request):
    """VAPI sends events here. Point your VAPI dashboard webhook URL at /api/webhook."""
    payload: dict[str, Any] = await request.json()
    message = payload.get("message", {})
    event_type = message.get("type")

    if event_type == "transcript":
        role = message.get("role", "unknown")
        text = message.get("transcript", "")
        print(f"[TRANSCRIPT] {role}: {text}")

    elif event_type == "end-of-call-report":
        call_id = message.get("call", {}).get("id", "unknown")
        transcript = message.get("transcript", "")
        print(f"[CALL ENDED] id={call_id} - scoring...")

        try:
            result = score_interview(transcript)
            scores[call_id] = result
            print(f"[SCORE] {result}")
        except Exception as e:
            print(f"[SCORE ERROR] {e}")
            scores[call_id] = {"error": str(e)}

    elif event_type == "status-update":
        print(f"[STATUS] {message.get('status')}")

    return {"received": True}