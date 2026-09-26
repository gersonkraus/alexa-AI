# -*- coding: utf-8 -*-
"""Adaptadores de provedor de LLM.

Cada provedor sabe montar o corpo da requisição (com ou sem tools) e
normalizar a resposta em texto e/ou chamadas de ferramenta, para que o
LLMClient trabalhe com um formato único independente de ser Ollama,
uma API compatível com OpenAI (Grok/xAI, etc.) ou a API da Anthropic.
"""

import json

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_SEARCH_DESCRIPTION = (
    "Busca informação atual na internet. Use somente quando a pergunta depender de dados "
    "recentes ou que mudam com o tempo: notícias, preços, clima, cargos/pessoas atuais, "
    "versões de software, resultados de jogos ou eleições."
)


def _web_search_parameters():
    return {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Termo de busca em português"}},
        "required": ["query"],
    }


class ToolCall:
    def __init__(self, call_id, name, arguments):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments or {}


class ChatTurn:
    """Resultado normalizado de uma chamada ao provedor."""

    def __init__(self, text=None, tool_calls=None, raw_assistant_message=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_assistant_message = raw_assistant_message


class OllamaProvider:
    """Formato nativo do Ollama (Cloud ou self-hosted): /api/chat."""

    name = "ollama"

    def headers(self, api_key):
        return {"Authorization": f"Bearer {api_key}"}

    def build_body(self, model, system_prompt, messages, tools, max_tokens):
        body = {
            "model": model, "messages": [{"role": "system", "content": system_prompt}] + messages, "stream": False,
            # gpt-oss (e outros modelos de raciocínio no Ollama) gastam parte do
            # orçamento de tokens "pensando" antes de escrever a resposta visível
            # — para voz, isso já cortava respostas no meio da frase mesmo com
            # max_tokens generoso. Desligado porque não usamos o raciocínio.
            "think": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": WEB_SEARCH_TOOL_NAME, "description": WEB_SEARCH_DESCRIPTION, "parameters": _web_search_parameters(),
            }}]
        return body

    def parse(self, body):
        message = body.get("message", {}) or {}
        tool_calls = [
            ToolCall(call.get("id"), call.get("function", {}).get("name"), call.get("function", {}).get("arguments"))
            for call in (message.get("tool_calls") or [])
        ]
        text = (message.get("content") or "").strip() or None
        return ChatTurn(text=text, tool_calls=tool_calls, raw_assistant_message=message)

    def tool_result_messages(self, turn, tool_call, result_text):
        return [turn.raw_assistant_message, {"role": "tool", "content": result_text}]


class OpenAICompatibleProvider:
    """APIs compatíveis com a Chat Completions da OpenAI: Grok/xAI, OpenRouter, etc."""

    name = "openai"

    def headers(self, api_key):
        return {"Authorization": f"Bearer {api_key}"}

    def build_body(self, model, system_prompt, messages, tools, max_tokens):
        body = {"model": model, "messages": [{"role": "system", "content": system_prompt}] + messages, "stream": False}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": WEB_SEARCH_TOOL_NAME, "description": WEB_SEARCH_DESCRIPTION, "parameters": _web_search_parameters(),
            }}]
        return body

    def parse(self, body):
        message = ((body.get("choices") or [{}])[0]).get("message", {}) or {}
        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {}) or {}
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except ValueError:
                    arguments = {}
            tool_calls.append(ToolCall(call.get("id"), fn.get("name"), arguments))
        text = (message.get("content") or "").strip() or None
        return ChatTurn(text=text, tool_calls=tool_calls, raw_assistant_message=message)

    def tool_result_messages(self, turn, tool_call, result_text):
        return [turn.raw_assistant_message, {"role": "tool", "tool_call_id": tool_call.call_id, "content": result_text}]


class AnthropicProvider:
    """API Messages da Anthropic (Claude, incl. Haiku): formato de content blocks."""

    name = "anthropic"

    def headers(self, api_key):
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def build_body(self, model, system_prompt, messages, tools, max_tokens):
        body = {"model": model, "system": system_prompt, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = [{
                "name": WEB_SEARCH_TOOL_NAME, "description": WEB_SEARCH_DESCRIPTION, "input_schema": _web_search_parameters(),
            }]
        return body

    def parse(self, body):
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip() or None
        tool_calls = [ToolCall(b.get("id"), b.get("name"), b.get("input")) for b in blocks if b.get("type") == "tool_use"]
        return ChatTurn(text=text, tool_calls=tool_calls, raw_assistant_message={"role": "assistant", "content": blocks})

    def tool_result_messages(self, turn, tool_call, result_text):
        return [
            turn.raw_assistant_message,
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_call.call_id, "content": result_text}]},
        ]


PROVIDERS = {"ollama": OllamaProvider(), "openai": OpenAICompatibleProvider(), "anthropic": AnthropicProvider()}


def get_provider(name):
    key = (name or "ollama").strip().lower()
    if key not in PROVIDERS:
        raise ValueError(f"llm_provider desconhecido: {name!r}. Use um de {sorted(PROVIDERS)}.")
    return PROVIDERS[key]
