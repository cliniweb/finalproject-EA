"""Logs de consola en español — mensajes legibles para seguir el flujo en vivo.

Complementa a structlog (que emite eventos estructurados) con líneas humanas
directas a stdout, p. ej.:

    [CLINIAI] 🎯 LLAMADA IGUAL!! se usó la caché semántica (similitud=0.9432)
"""

from __future__ import annotations


def say(msg: str) -> None:
    """Imprime un mensaje legible en la línea de comandos."""
    print(f"[CLINIAI] {msg}", flush=True)
