# -*- coding: utf-8 -*-
import html
import json
import os
import random
import re


def load_config():
    defaults = {
        "llm_provider": os.environ.get("LLM_PROVIDER", "ollama"),
        "llm_url": os.environ.get("OLLAMA_URL", "https://ollama.com/api/chat"),
        "llm_key": os.environ.get("OLLAMA_API_KEY", ""),
        "llm_model": os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
        "web_search_url": "https://ollama.com/api/web_search",
        "web_search_key": "",
        "tool_calling_enabled": False,
    }
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), "config.json")
    if not os.path.exists(path):
        return defaults
    with open(path, "r", encoding="utf-8") as handle:
        return {**defaults, **json.load(handle)}


def safe_speech(text, max_chars=750):
    text = re.sub(r"```.*?```", "", text or "", flags=re.S)
    text = re.sub(r"[\*_#`\[\]{}<>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.escape(text[:max_chars], quote=False)


def format_search_results(results, max_chars=4200):
    chunks = []
    for item in (results or [])[:4]:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()[:800]
        if content:
            chunks.append(f"Fonte: {title}. {content} URL: {url}")
    return "\n".join(chunks)[:max_chars]


def search_instructions(reference_time, timezone, context):
    """Texto que acompanha o contexto de busca (ou a ausência dele) para o modelo."""
    if not context:
        return "A busca não retornou nenhum resultado confiável. Diga claramente que não há confirmação atual, sem inventar."
    return (f"Data e hora de referência: {reference_time}, fuso {timezone}. "
            "Use somente fatos confirmados pelas fontes abaixo. Não transforme evento futuro em acontecimento passado. "
            "Para notícias, escolha no máximo três itens distintos e confirme que cada um corresponde à data pedida. "
            "Não cite URLs nem diga que recebeu fontes. Se as fontes não confirmarem a resposta, diga isso claramente.\n\n"
            f"{context}")


class CannedResponse:
    DATA = {
        "launch": ["Pode falar.", "Estou ouvindo.", "Pronto. Qual é a sua pergunta?"],
        "reprompt": ["Quer saber mais alguma coisa?", "Pode continuar.", "O que mais você gostaria de saber?"],
        "help": [
            "Você pode me perguntar sobre notícias, pessoas, tecnologia ou qualquer outro assunto. "
            "Por exemplo: me fale sobre a Segunda Guerra Mundial, ou o que é buraco negro."
        ],
        "fallback": [
            "Não entendi. Tente começar com 'me fale sobre', 'o que é' ou 'quem foi', seguido do assunto.",
            "Não captei. Pode tentar de novo começando com 'me fale sobre' ou 'quem foi'?",
        ],
        "goodbye": ["Até mais!", "Tudo bem. Até a próxima!"],
        "error": ["Não consegui consultar o serviço agora. Tente novamente em alguns segundos."],
        "search_error": ["Não consegui verificar informações atualizadas agora. Tente novamente daqui a pouco."],
        "thinking": ["Só um momento, deixa eu verificar.", "Peraí, vou conferir isso.", "Um instante, estou checando."],
        "no_ack": ["Tudo bem.", "Ok, sem problemas.", "Entendido."],
        "nothing_to_repeat": ["Ainda não falei nada para repetir. Pode fazer sua pergunta."],
    }

    def get(self, key):
        return random.choice(self.DATA[key])
