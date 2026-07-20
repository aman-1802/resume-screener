"""Shared dataclasses used across the pipeline."""

from dataclasses import dataclass, field


@dataclass
class JobDescription:
    id: int
    role_title: str
    extracted_text: str
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    candidate_name: str
    resume_filename: str
    resume_path: str
    score: float
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
