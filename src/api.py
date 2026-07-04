"""FastAPI wrapper: POST /analyze -> fixed JSON schema."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .infer import analyze, describe_mode

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Mr. Spiky — Intuition Compiler", version="0.1.0")

# CORS: the frontend origins that the demo runs on. Localhost for dev,
# crnicholson.com for the deployed demo. Widen this list if you host the
# frontend somewhere else.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mr-spiky.crnicholson.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_LANGUAGES = {"python"}

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze")
    language: str = Field(default="python", description="Source language; only 'python' is supported")
    axis_weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional per-team axis weights. Multiplies suspicion score by a "
            "weighted mean of these across the axes firing on each line. "
            "Missing axes default to 1.0. Example: "
            '{"exception_surface": 1.5, "naming": 0.6}'
        ),
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        **describe_mode(),
    }


@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> dict:
    lang = req.language.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"language {req.language!r} not supported. "
                f"Mr. Spiky's AST features are Python-only. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
            ),
        )
    return analyze(req.code, axis_weights=req.axis_weights)