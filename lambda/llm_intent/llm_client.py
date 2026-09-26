# -*- coding: utf-8 -*-
import json
import logging
import socket
import urllib.error
import urllib.request

from llm_intent.providers import get_provider
from llm_intent.utils import format_search_results, search_instructions

logger = logging.getLogger(__name__)

# No Alexa-hosted Lambda (Python 3.8), socket.timeout ainda é uma classe própria
# — só virou alias de TimeoutError a partir do Python 3.10. Sem isso aqui, um
# timeout de leitura (ex.: resposta demorada do modelo) escapa como exceção
# crua e cai no handler de erro genérico em vez da mensagem específica.
NETWORK_TIMEOUT_ERRORS = (TimeoutError, socket.timeout)


class UpstreamError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider, url, api_key, model, search_url, search_key,
                 timeout=7.0, search_timeout=3.0, max_tokens=300, max_search_chars=4200, tool_calling_enabled=False):
        self.provider = get_provider(provider)
        self.url = url
        self.api_key = api_key
        self.model = model
        self.search_url = search_url
        # Chave própria para a busca: fica separada da chave do provedor de chat,
        # já que a busca web hoje só existe na API do Ollama, mesmo trocando de
        # provedor de chat para Grok/Anthropic/etc.
        self.search_key = search_key or api_key
        self.timeout = float(timeout)
        self.search_timeout = float(search_timeout)
        self.max_tokens = int(max_tokens)
        self.max_search_chars = int(max_search_chars)
        self.tool_calling_enabled = bool(tool_calling_enabled)

    def _post(self, url, payload, timeout, headers):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise UpstreamError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, *NETWORK_TIMEOUT_ERRORS) as exc:
            raise UpstreamError(str(exc)) from exc
        except ValueError as exc:
            # Corpo com status 200 mas não-JSON (proxy quebrado, resposta truncada, etc.).
            raise UpstreamError(f"invalid response body: {exc}") from exc

    def _call_provider(self, system_prompt, messages, tools):
        body = self.provider.build_body(self.model, system_prompt, messages, tools, self.max_tokens)
        response = self._post(self.url, body, self.timeout, self.provider.headers(self.api_key))
        return self.provider.parse(response)

    def chat(self, system_prompt, messages):
        """Um único turno, sem tools — comportamento clássico usado com a política de busca por regex."""
        turn = self._call_provider(system_prompt, messages, tools=False)
        if not turn.text:
            raise UpstreamError("empty model response")
        return turn.text

    def chat_with_tools(self, system_prompt, messages, reference_time, timezone):
        """Deixa o próprio modelo decidir se precisa buscar, em exatamente duas
        chamadas no máximo: uma oferecendo a ferramenta de busca, e — só se ela
        for usada — uma segunda sem tools que força uma resposta final em texto.

        Retorna (texto, usou_busca). Se a busca falhar ou vier vazia, o modelo
        recebe isso como resultado da ferramenta em vez de travar a resposta —
        quem evita a alucinação aqui é a instrução do system prompt para não
        inventar fatos atuais, já que o protocolo de tool calling exige uma
        resposta para toda chamada de ferramenta.
        """
        conversation = list(messages)
        turn = self._call_provider(system_prompt, conversation, tools=True)
        if not turn.tool_calls:
            if not turn.text:
                raise UpstreamError("empty model response")
            return turn.text, False
        call = turn.tool_calls[0]
        query = (call.arguments or {}).get("query") or ""
        results = self.web_search(query) if query else []
        context = format_search_results(results, self.max_search_chars)
        result_text = search_instructions(reference_time, timezone, context)
        conversation.extend(self.provider.tool_result_messages(turn, call, result_text))
        final_turn = self._call_provider(system_prompt, conversation, tools=False)
        if not final_turn.text:
            raise UpstreamError("empty model response")
        return final_turn.text, True

    def web_search(self, query, max_results=4):
        if not self.search_url:
            logger.info("web search disabled: no search URL configured")
            return []
        try:
            body = self._post(
                self.search_url,
                {"query": query, "max_results": min(max_results, 10)},
                self.search_timeout,
                {"Authorization": f"Bearer {self.search_key}"},
            )
            results = body.get("results", []) or []
            logger.info("web search results=%d query=%s", len(results), query)
            return results
        except UpstreamError as exc:
            logger.warning("web search failed query=%s error=%s", query, exc)
            return []

    def gateway_chat(self, gateway_url, gateway_token, conversation_id, message, voice_rules):
        """Retorna (texto, status). status="search_unavailable" quando o gateway
        sinaliza (HTTP 503 com detail="search_unavailable") que a pergunta
        precisava de busca e ela falhou — mesma falha segura da política local,
        só que decidida do lado do gateway agora."""
        request = urllib.request.Request(
            gateway_url.rstrip("/") + "/v1/chat",
            data=json.dumps({"conversation_id": conversation_id, "message": message, "voice_rules": voice_rules}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Gateway-Token": gateway_token or ""},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
            except ValueError:
                parsed = {}
            if exc.code == 503 and parsed.get("detail") == "search_unavailable":
                return None, "search_unavailable"
            raise UpstreamError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, *NETWORK_TIMEOUT_ERRORS) as exc:
            raise UpstreamError(str(exc)) from exc
        except ValueError as exc:
            raise UpstreamError(f"invalid gateway response body: {exc}") from exc
        content = body.get("message", "").strip()
        if not content:
            raise UpstreamError("empty gateway response")
        return content, None
