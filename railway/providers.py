# -*- coding: utf-8 -*-
"""Adaptadores de provedor de LLM para o gateway.

Mesmo desenho de lambda/llm_intent/providers.py (Lambda e gateway não
compartilham pacote entre si), com um adicional: o gateway expõe
temperatura/limite de tokens configuráveis pelo painel, então build_body
também recebe `temperature`.
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
    def __init__(self, text=None, tool_calls=None, raw_assistant_message=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_assistant_message = raw_assistant_message


class OllamaProvider:
    name = "ollama"

    def headers(self, api_key):
        return {"Authorization": f"Bearer {api_key}"}

    def build_body(self, model, system_prompt, messages, tools, max_tokens, temperature):
        body = {
            "model": model, "messages": [{"role": "system", "content": system_prompt}] + messages, "stream": False,
            # gpt-oss (e outros modelos de raciocínio no Ollama) gastam parte do
            # orçamento de tokens "pensando" antes de escrever a resposta visível
            # — para voz, isso cortava respostas no meio da frase mesmo com
            # max_tokens generoso. Desligado porque não usamos o raciocínio.
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
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
    name = "openai"

    def headers(self, api_key):
        return {"Authorization": f"Bearer {api_key}"}

    def build_body(self, model, system_prompt, messages, tools, max_tokens, temperature):
        body = {
            "model": model, "messages": [{"role": "system", "content": system_prompt}] + messages, "stream": False,
            "temperature": temperature, "max_tokens": max_tokens,
        }
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
    name = "anthropic"

    def headers(self, api_key):
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def build_body(self, model, system_prompt, messages, tools, max_tokens, temperature):
        body = {"model": model, "system": system_prompt, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
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
