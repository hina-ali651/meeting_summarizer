# main.py
import os
import dotenv
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
import asyncio
from agents import AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api
# Load environment variables
dotenv.load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)


# ------------------------------------------------------------------
# 0. FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="Meeting Summarizer (single-file)",
    version="0.1.0",
    description="Demo backend for portfolio using openai-agents SDK"
)

# ------------------------------------------------------------------
# 1. In-mem "DB" (job store)
# ------------------------------------------------------------------
jobs: dict[str, dict] = {}

# ------------------------------------------------------------------
# 2. A tiny custom tool we’ll give to the summarizer
# ------------------------------------------------------------------
@function_tool
def send_email(recipients: List[str], subject: str, body: str) -> str:
    """Send the summary via email (stub)."""
    print(f"[EMAIL] To: {recipients}\nSubject: {subject}\n{body}")
    return "email_sent"

# ------------------------------------------------------------------
# 3. Define the 3 agents inline
# ------------------------------------------------------------------
transcriber = Agent(
    name="Transcriber",
    instructions="You turn raw meeting audio transcripts into clean, speaker-labelled text.",
    model="gemini-1.5-flash",
)

clarifier = Agent(
    name="Clarifier",
    instructions=(
        "Read the transcript and ask the summarizer to resolve any ambiguous "
        "points before final summary is produced.",
    

    ),
    model="gemini-1.5-flash",
)

summarizer = Agent(
    name="Summarizer",
    instructions=(
        "Produce a concise 1-page summary plus a Mermaid timeline. "
        "Then call send_email to deliver it."
    ),
    model="gemini-1.5-flash",
    tools=[send_email]
)

# ------------------------------------------------------------------
# 4. Endpoints
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    transcript: str

@app.post("/summarize")
async def summarize_job(req: SummarizeRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "started"}
    asyncio.create_task(_run_agents(job_id, req.transcript))
    return {"job_id": job_id, "status": "queued"}
# ------------------------------------------------------------------
# 5. Background runner
# ------------------------------------------------------------------
async def _run_agents(job_id: str, transcript: str):
    try:
        # 1. Transcriber (no-op if text already provided)
        result1 = await Runner.run(transcriber, transcript)
        clean_text = result1.final_output

        # 2. Clarifier asks questions (we skip real Q&A for brevity)
        result2 = await Runner.run(clarifier, clean_text)
        clarified = result2.final_output or clean_text

        # 3. Summarizer → summary + timeline + email stub
        await Runner.run(
            summarizer,
            clarified,
            context={
                "recipients": ["team@example.com"],
                "subject": f"Summary #{job_id}"
            }
        )

        jobs[job_id]["status"] = "completed"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# ------------------------------------------------------------------
# 6. Simple status check
# ------------------------------------------------------------------
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return jobs[job_id]