"""FastAPI wrapper: POST /analyze -> fixed JSON schema."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .infer import analyze, describe_mode

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Mr. Spiky — Intuition Compiler", version="0.1.0")

SUPPORTED_LANGUAGES = {"python"}


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze")
    language: str = Field(default="python", description="Source language; only 'python' is supported")


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
    return analyze(req.code)