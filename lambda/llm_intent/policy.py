# -*- coding: utf-8 -*-
"""Detecção conservadora de perguntas que podem exigir dados atuais."""

import re
import unicodedata
from datetime import datetime


CURRENT_MARKERS = (
    "hoje", "agora", "atual", "atualmente", "recente", "recentes", "últim",
    "notícia", "notícias", "novidade", "novidades", "esta semana", "este mês",
    "este ano", "ontem", "amanhã", "ao vivo", "tempo", "previsão", "cotação",
    "preço", "valor", "dólar", "euro", "placar", "resultado", "eleição", "eleições",
    "versão mais nova", "versão atual", "lançamento", "aconteceu", "está acontecendo",
    "pesquise", "pesquisar", "procure", "buscar", "busque", "na internet", "online",
)

ROLE_PATTERNS = (
    r"\bquem (é|são|está|ocupa|governa)\b",
    r"\bqual (é|o|a) (atual|novo|nova|mais recente)?\s*(presidente|primeiro|ministro|governador|prefeito|ceo|diretor|papa|líder)\b",
    r"\bpresidente (dos|da|do)\b",
    r"\bceo (da|do|de)\b",
    r"\b(quem|qual) (ganhou|venceu|está liderando)\b",
)

TIME_SENSITIVE_PATTERNS = (
    r"\bquanto (está|custa|vale)\b",
    r"\bqual a cotação\b",
    r"\b(placar|resultado)\b",
    r"\b(agenda|horário|programação)\b",
    r"\bquando (é|será)\b",
)

ATEMPORAL_STARTS = (
    "o que é", "explique", "como funciona", "por que", "porque", "qual é a diferença",
    "quanto é", "calcule", "resuma", "traduza", "escreva", "crie", "conte uma piada",
    "quem foi", "quem era", "quando nasceu", "quando morreu",
)

BIOGRAPHICAL_PATTERNS = (
    r"\bquando( que)? (ele|ela|essa pessoa|esse homem|essa mulher) (nasceu|morreu|faleceu)\b",
    r"\bonde (ele|ela|essa pessoa) (nasceu|morreu|estudou|viveu)\b",
    r"\bqual (era|foi) (a|o)\b",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()


def needs_web_search(query: str) -> bool:
    """Prefere buscar quando um erro factual seria pior que uma pequena latência."""
    q = normalize(query)
    if not q:
        return False
    if any(marker in q for marker in (normalize(x) for x in CURRENT_MARKERS)):
        return True
    patterns = tuple(normalize(pattern) for pattern in ROLE_PATTERNS + TIME_SENSITIVE_PATTERNS)
    if any(re.search(pattern, q) for pattern in patterns):
        return True
    if any(q.startswith(normalize(prefix)) for prefix in ATEMPORAL_STARTS):
        return False
    if any(re.search(normalize(pattern), q) for pattern in BIOGRAPHICAL_PATTERNS):
        return False
    return False


def has_current_claim(response: str) -> bool:
    """Detecta indícios de resposta atual para decidir se uma busca é necessária."""
    year = datetime.now().year
    return bool(re.search(r"\b20\d{2}\b", response or "")) or any(
        x in normalize(response) for x in ("atualmente", "hoje", "agora", "presidente atual")
    ) and year >= 2024
