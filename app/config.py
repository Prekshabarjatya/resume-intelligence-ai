from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_classifier_model: str = "llama-3.1-8b-instant"
    groq_reasoning_model: str = "llama-3.3-70b-versatile"

    max_optimization_iterations: int = 3
    max_upload_mb: int = 10

    # Minimum LLM-reported confidence to accept a skill-synonym judgment as
    # a match (see app/agents/skill_matching.py).
    semantic_match_threshold: float = 0.6

    # Weights for the overall match score. Must sum to 1.0.
    weight_skill_match: float = 0.30
    weight_experience_match: float = 0.20
    weight_responsibility_match: float = 0.15
    weight_keyword_coverage: float = 0.15
    weight_qualification_match: float = 0.10
    weight_ats_compatibility: float = 0.10


settings = Settings()
