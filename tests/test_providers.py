import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "lambda"))
from llm_intent.providers import get_provider


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_provider("bedrock")


def test_ollama_build_body_without_tools():
    provider = get_provider("ollama")
    body = provider.build_body("gpt-oss:20b", "seja breve", [{"role": "user", "content": "oi"}], tools=False, max_tokens=300)
    assert body["model"] == "gpt-oss:20b"
    assert body["messages"][0] == {"role": "system", "content": "seja breve"}
    assert "tools" not in body
    # Regressão: sem isso, gpt-oss gastava parte do orçamento "pensando" e
    # cortava a resposta de voz no meio da frase.
    assert body["think"] is False
    assert body["options"]["num_predict"] == 300


def test_ollama_build_body_with_tools_exposes_web_search():
    provider = get_provider("ollama")
    body = provider.build_body("gpt-oss:20b", "seja breve", [], tools=True, max_tokens=300)
    assert body["tools"][0]["function"]["name"] == "web_search"


def test_ollama_parse_plain_text():
    provider = get_provider("ollama")
    turn = provider.parse({"message": {"role": "assistant", "content": "Olá!"}})
    assert turn.text == "Olá!"
    assert turn.tool_calls == []


def test_ollama_parse_tool_call_arguments_stay_as_dict():
    provider = get_provider("ollama")
    body = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "web_search", "arguments": {"query": "clima em Palhoça"}}}
    ]}}
    turn = provider.parse(body)
    assert turn.text is None
    assert turn.tool_calls[0].name == "web_search"
    assert turn.tool_calls[0].arguments == {"query": "clima em Palhoça"}


def test_openai_compatible_parses_json_encoded_arguments():
    provider = get_provider("openai")
    body = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "presidente atual"}'}}
    ]}}]}
    turn = provider.parse(body)
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.tool_calls[0].arguments == {"query": "presidente atual"}


def test_openai_compatible_parses_plain_text():
    provider = get_provider("openai")
    body = {"choices": [{"message": {"role": "assistant", "content": "42"}}]}
    turn = provider.parse(body)
    assert turn.text == "42"


def test_anthropic_build_body_uses_top_level_system_and_max_tokens():
    provider = get_provider("anthropic")
    body = provider.build_body("claude-haiku-4-5", "seja breve", [{"role": "user", "content": "oi"}], tools=False, max_tokens=250)
    assert body["system"] == "seja breve"
    assert body["max_tokens"] == 250
    assert all(m["role"] != "system" for m in body["messages"])


def test_anthropic_build_body_with_tools_uses_input_schema():
    provider = get_provider("anthropic")
    body = provider.build_body("claude-haiku-4-5", "seja breve", [], tools=True, max_tokens=250)
    assert body["tools"][0]["name"] == "web_search"
    assert "input_schema" in body["tools"][0]


def test_anthropic_parse_text_block():
    provider = get_provider("anthropic")
    turn = provider.parse({"content": [{"type": "text", "text": "Olá!"}]})
    assert turn.text == "Olá!"


def test_anthropic_parse_tool_use_block():
    provider = get_provider("anthropic")
    body = {"content": [{"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "cotação do dólar"}}]}
    turn = provider.parse(body)
    assert turn.text is None
    assert turn.tool_calls[0].call_id == "toolu_1"
    assert turn.tool_calls[0].arguments == {"query": "cotação do dólar"}


def test_anthropic_tool_result_messages_shape():
    provider = get_provider("anthropic")
    body = {"content": [{"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "x"}}]}
    turn = provider.parse(body)
    messages = provider.tool_result_messages(turn, turn.tool_calls[0], "resultado da busca")
    assert messages[0] == turn.raw_assistant_message
    assert messages[1]["content"][0]["tool_use_id"] == "toolu_1"
    assert messages[1]["content"][0]["content"] == "resultado da busca"
