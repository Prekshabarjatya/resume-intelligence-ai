"""Groq-backed LLM clients.

Two tiers, per the cost-management principle in the PRD: a small/fast model
for cheap classification, and a stronger model for structured extraction and
reasoning-heavy steps (gap analysis).
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import settings


@lru_cache(maxsize=1)
def get_classifier_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_classifier_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_reasoning_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_reasoning_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )
