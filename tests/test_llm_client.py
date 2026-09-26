import socket
import sys
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "lambda"))
from llm_intent.llm_client import LLMClient, UpstreamError


def make_client(provider="ollama", tool_calling_enabled=True):
    return LLMClient(
        provider, "https://example.invalid/chat", "chat-key", "test-model",
        "https://example.invalid/search", "search-key",
        timeout=1.0, search_timeout=1.0, max_tokens=200, max_search_chars=4200,
        tool_calling_enabled=tool_calling_enabled,
    )


def test_chat_with_tools_returns_text_without_search_when_model_does_not_call_tool():
    client = make_client()
    with patch.object(client, "_post", return_value={"message": {"role": "assistant", "content": "Dois mais dois são quatro."}}) as mock_post:
        text, used_search = client.chat_with_tools("system", [{"role": "user", "content": "quanto é 2+2"}], "26/09/2026 10:00", "America/Sao_Paulo")
    assert text == "Dois mais dois são quatro."
    assert used_search is False
    assert mock_post.call_count == 1


def test_chat_with_tools_calls_search_then_forces_final_text():
    client = make_client()
    tool_call_response = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "web_search", "arguments": {"query": "presidente atual dos eua"}}}
    ]}}
    search_response = {"results": [{"title": "Fonte", "url": "http://x", "content": "Informação de teste."}]}
    final_response = {"message": {"role": "assistant", "content": "O presidente atual é fulano."}}

    responses = iter([tool_call_response, search_response, final_response])
    with patch.object(client, "_post", side_effect=lambda *a, **k: next(responses)) as mock_post:
        text, used_search = client.chat_with_tools(
            "system", [{"role": "user", "content": "quem é o presidente"}], "26/09/2026 10:00", "America/Sao_Paulo"
        )
    assert text == "O presidente atual é fulano."
    assert used_search is True
    assert mock_post.call_count == 3


def test_chat_with_tools_handles_empty_search_without_crashing():
    client = make_client()
    tool_call_response = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "web_search", "arguments": {"query": "algo bem obscuro"}}}
    ]}}
    empty_search_response = {"results": []}
    final_response = {"message": {"role": "assistant", "content": "Não encontrei confirmação atual sobre isso."}}
    responses = iter([tool_call_response, empty_search_response, final_response])
    with patch.object(client, "_post", side_effect=lambda *a, **k: next(responses)):
        text, used_search = client.chat_with_tools(
            "system", [{"role": "user", "content": "pergunta obscura"}], "26/09/2026 10:00", "America/Sao_Paulo"
        )
    assert "não encontrei" in text.lower() or "não" in text.lower()
    assert used_search is True


def test_chat_without_tools_raises_on_empty_response():
    client = make_client(tool_calling_enabled=False)
    with patch.object(client, "_post", return_value={"message": {"role": "assistant", "content": ""}}):
        try:
            client.chat("system", [{"role": "user", "content": "oi"}])
            assert False, "esperava UpstreamError"
        except UpstreamError:
            pass


def test_search_key_defaults_to_chat_key_when_not_set():
    client = LLMClient("ollama", "https://x/chat", "chat-key", "m", "https://x/search", "", 1.0, 1.0, 200, 4200, False)
    assert client.search_key == "chat-key"


def test_search_key_can_be_separate_from_chat_key():
    client = LLMClient("openai", "https://api.x.ai/v1/chat/completions", "grok-key", "grok-4.6",
                        "https://ollama.com/api/web_search", "ollama-search-key", 1.0, 1.0, 200, 4200, False)
    assert client.api_key == "grok-key"
    assert client.search_key == "ollama-search-key"


def test_socket_timeout_becomes_upstream_error_not_a_raw_exception():
    # Regressão: no Alexa-hosted Lambda (Python 3.8), socket.timeout não é
    # TimeoutError (isso só passou a valer a partir do 3.10). Sem capturar as
    # duas, um timeout de leitura escapava como exceção crua e caía no handler
    # de erro genérico em vez da mensagem específica de timeout.
    client = make_client(tool_calling_enabled=False)
    with patch.object(urllib.request, "urlopen", side_effect=socket.timeout("timed out")):
        try:
            client.chat("system", [{"role": "user", "content": "oi"}])
            assert False, "esperava UpstreamError"
        except UpstreamError:
            pass
        except Exception as exc:  # noqa: BLE001 - é exatamente isso que não pode escapar
            assert False, f"socket.timeout vazou como {type(exc).__name__}, não UpstreamError: {exc}"


def test_invalid_json_body_becomes_upstream_error_not_a_raw_exception():
    from unittest.mock import MagicMock

    client = make_client(tool_calling_enabled=False)
    fake_response = MagicMock()
    fake_response.read.return_value = b"<html>not json</html>"
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    with patch.object(urllib.request, "urlopen", return_value=fake_response):
        try:
            client.chat("system", [{"role": "user", "content": "oi"}])
            assert False, "esperava UpstreamError"
        except UpstreamError:
            pass
