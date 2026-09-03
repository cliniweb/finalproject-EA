"""Structural chunking for RAG (mirrors the estimator's JSONStructuralChunker idea).

Two document types feed the knowledge base:

1. **Doctor profile JSON** — chunked *structurally*: each top-level section
   (specialties, locations, education, insurance, schedules, ...) becomes one
   chunk. Structural boundaries beat fixed-size splitting here because the
   profile is a shallow, heterogeneous document where a section = a topic.

2. **FAQ / policy free text** — chunked by paragraph with a max-char guard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

MAX_CHUNK_CHARS = 1_500


@dataclass
class Chunk:
    """One retrievable unit of knowledge."""
    doctor_id: str
    source: str          # e.g. "profile:especialidades" or "faq:paragraph-3"
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Doctor profile — structural chunking
# ---------------------------------------------------------------------------

# Human-readable Spanish labels so the chunk text reads naturally to the LLM.
_SECTION_LABELS = {
    "nombrePersona": "Nombre del médico",
    "especialidades": "Especialidades",
    "localidades": "Localidades y direcciones",
    "educacion": "Educación y formación",
    "seguros": "Seguros aceptados",
    "idiomas": "Idiomas",
    "servicios": "Servicios ofrecidos",
    "experiencia": "Experiencia profesional",
    "horarios": "Horarios de atención",
    "descripcion": "Descripción general",
    "sexo": "Sexo",
}


def _render_value(value) -> str:
    """Render a JSON value as compact human-readable Spanish text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_render_value(v) for v in value]
        return "; ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = [f"{k}: {_render_value(v)}" for k, v in value.items() if v not in (None, "", [], {})]
        return ", ".join(parts)
    return ""


def chunk_doctor_profile(doctor_id: str, doctor_data: dict) -> list[Chunk]:
    """One chunk per top-level profile section, plus a summary chunk."""
    chunks: list[Chunk] = []

    for key, value in doctor_data.items():
        if value in (None, "", [], {}):
            continue
        label = _SECTION_LABELS.get(key, key)
        text = f"{label}: {_render_value(value)}"
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS]
        chunks.append(
            Chunk(
                doctor_id=doctor_id,
                source=f"profile:{key}",
                text=text,
                metadata={"section": key},
            )
        )

    # A summary chunk helps broad questions ("háblame del doctor").
    name = doctor_data.get("nombrePersona", "")
    summary_bits = [f"Médico: {name}."] if name else []
    for key in ("especialidades", "localidades"):
        if doctor_data.get(key):
            summary_bits.append(f"{_SECTION_LABELS[key]}: {_render_value(doctor_data[key])}."[:400])
    if summary_bits:
        chunks.append(
            Chunk(
                doctor_id=doctor_id,
                source="profile:summary",
                text=" ".join(summary_bits),
                metadata={"section": "summary"},
            )
        )

    log.info("profile_chunked", doctor_id=doctor_id, chunks=len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# FAQ / free text — paragraph chunking
# ---------------------------------------------------------------------------

def chunk_faq_text(doctor_id: str, text: str, source_name: str = "faq") -> list[Chunk]:
    """Split free text on blank lines; merge tiny paragraphs; cap chunk size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= MAX_CHUNK_CHARS:
            buffer = candidate
        else:
            if buffer:
                chunks.append(
                    Chunk(
                        doctor_id=doctor_id,
                        source=f"{source_name}:paragraph-{len(chunks)}",
                        text=buffer,
                    )
                )
            buffer = para[:MAX_CHUNK_CHARS]

    if buffer:
        chunks.append(
            Chunk(
                doctor_id=doctor_id,
                source=f"{source_name}:paragraph-{len(chunks)}",
                text=buffer,
            )
        )

    log.info("faq_chunked", doctor_id=doctor_id, chunks=len(chunks))
    return chunks
