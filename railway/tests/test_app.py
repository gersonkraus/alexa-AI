import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

# As constantes de app.py (tokens, DB_PATH, chave de criptografia) são lidas do
# ambiente na importação do módulo — precisam existir antes do import.
os.environ.setdefault("GATEWAY_TOKEN", "test-gateway-token")
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "test.sqlite3"))

sys.path.insert(0, str(Path(__file__).parents[1]))
import app as gateway_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

gateway_app.init_db()  # TestClient sem "with" não dispara o evento de startup
client = TestClient(gateway_app.app)
ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}
GATEWAY_HEADERS = {"X-Gateway-Token": "test-gateway-token"}


def setup_function(_):
    # Isola cada teste: limpa configurações salvas e histórico de conversa.
    with gateway_app.closing(gateway_app.connect()) as conn:
        conn.execute("delete from settings")
        conn.execute("delete from messages")
        conn.commit()


def test_health_reports_defaults():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-oss:20b"
    assert resp.json()["provider"] == "ollama"


def test_admin_settings_requires_token():
    resp = client.get("/admin/settings")
    assert resp.status_code == 401


def test_admin_settings_rejects_unknown_provider():
    payload = {
        "llm_provider": "bedrock", "llm_url": "https://x", "llm_model": "m",
        "web_search_url": "", "tool_calling_enabled": False,
        "system_prompt": "responda de forma breve e natural para voz, sempre em portugues",
        "temperature": 0.2, "max_tokens": 120, "timeout_seconds": 3.5, "max_messages": 16,
    }
    resp = client.put("/admin/settings", json=payload, headers=ADMIN_HEADERS)
    assert resp.status_code == 422


def test_admin_settings_roundtrip_and_key_encryption():
    with patch.object(gateway_app, "ENCRYPTION_KEY", _fernet_key()):
        payload = {
            "llm_provider": "openai", "llm_url": "https://api.x.ai/v1/chat/completions", "llm_model": "grok-4.6",
            "api_key": "grok-secret", "web_search_url": "https://ollama.com/api/web_search",
            "web_search_key": "ollama-secret", "tool_calling_enabled": True,
            "system_prompt": "responda de forma breve e natural para voz, sempre em portugues",
            "temperature": 0.3, "max_tokens": 150, "timeout_seconds": 4.0, "max_messages": 10,
        }
        put_resp = client.put("/admin/settings", json=payload, headers=ADMIN_HEADERS)
        assert put_resp.status_code == 200, put_resp.text

        get_resp = client.get("/admin/settings", headers=ADMIN_HEADERS)
        body = get_resp.json()
        assert body["llm_provider"] == "openai"
        assert body["api_key_configured"] is True
        assert body["web_search_key_configured"] is True
        assert "api_key" not in body and "web_search_key" not in body  # nunca devolve em texto plano
        # Regressão: o blob cifrado não pode sair da API, mesmo cifrado.
        assert "api_key_encrypted" not in body
        assert "web_search_key_encrypted" not in body

        assert gateway_app.encrypted_api_key() == "grok-secret"
        assert gateway_app.encrypted_web_search_key() == "ollama-secret"


def _fernet_key():
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def test_chat_requires_gateway_token():
    resp = client.post("/v1/chat", json={"conversation_id": "abc12345", "message": "oi"})
    assert resp.status_code == 401


def test_chat_without_api_key_returns_503():
    resp = client.post("/v1/chat", json={"conversation_id": "abc12345", "message": "oi"}, headers=GATEWAY_HEADERS)
    assert resp.status_code == 503


def test_chat_accepts_real_length_alexa_user_id_as_conversation_id():
    # Regressão: amzn1.ask.account.* passa de 250 caracteres. max_length=160
    # rejeitava com 422 toda requisição real vinda da Lambda (só um userId de
    # verdade, não um mock curto de teste, revelava isso).
    real_length_user_id = "amzn1.ask.account." + "A" * 235
    assert len(real_length_user_id) > 160
    with patch.object(gateway_app, "ENCRYPTION_KEY", ""), patch.dict(os.environ, {}, clear=False):
        resp = client.post("/v1/chat", json={"conversation_id": real_length_user_id, "message": "oi"}, headers=GATEWAY_HEADERS)
    assert resp.status_code != 422, resp.text


def _mock_response(json_body, status_code=200):
    mock = AsyncMock()
    mock.status_code = status_code
    mock.json = lambda: json_body
    mock.raise_for_status = lambda: None
    return mock


def test_chat_policy_path_answers_without_search_for_stable_question():
    with patch.object(gateway_app, "ENCRYPTION_KEY", ""), \
         patch.dict(os.environ, {"LLM_API_KEY": "ollama-key"}), \
         patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_mock_response(
             {"message": {"role": "assistant", "content": "Fotossíntese é o processo pelo qual plantas produzem energia."}}
         ))) as mock_post:
        resp = client.post("/v1/chat", json={
            "conversation_id": "abc12345", "message": "explique como funciona a fotossíntese", "voice_rules": "seja breve",
        }, headers=GATEWAY_HEADERS)
        assert resp.status_code == 200, resp.text
        assert "Fotossíntese" in resp.json()["message"]
        assert mock_post.call_count == 1  # só o chat, sem chamada de busca


def test_chat_policy_path_returns_503_when_search_required_and_unavailable():
    with patch.object(gateway_app, "ENCRYPTION_KEY", ""), \
         patch.dict(os.environ, {"LLM_API_KEY": "ollama-key"}), \
         patch("app.run_web_search", new=AsyncMock(return_value=[])):
        resp = client.post("/v1/chat", json={
            "conversation_id": "abc12345", "message": "quem é o presidente dos Estados Unidos", "voice_rules": "seja breve",
        }, headers=GATEWAY_HEADERS)
        assert resp.status_code == 503
        assert resp.json()["detail"] == "search_unavailable"


def test_chat_tool_calling_path_calls_search_then_final_answer():
    with patch.object(gateway_app, "ENCRYPTION_KEY", ""), \
         patch.dict(os.environ, {"LLM_API_KEY": "ollama-key"}):
        # liga tool calling só para este teste
        gateway_app.write_settings({"tool_calling_enabled": True})
        tool_call_body = {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "web_search", "arguments": {"query": "presidente atual dos eua"}}}
        ]}}
        final_body = {"message": {"role": "assistant", "content": "O presidente atual é fulano."}}
        responses = iter([_mock_response(tool_call_body), _mock_response(final_body)])
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=lambda *a, **k: next(responses))), \
             patch("app.run_web_search", new=AsyncMock(return_value=[{"title": "Fonte", "url": "http://x", "content": "Info de teste"}])):
            resp = client.post("/v1/chat", json={
                "conversation_id": "abc12345", "message": "quem é o presidente dos Estados Unidos", "voice_rules": "seja breve",
            }, headers=GATEWAY_HEADERS)
            assert resp.status_code == 200, resp.text
            assert "fulano" in resp.json()["message"].lower()


def test_chat_persists_history_across_calls():
    with patch.object(gateway_app, "ENCRYPTION_KEY", ""), \
         patch.dict(os.environ, {"LLM_API_KEY": "ollama-key"}):
        captured_bodies = []

        async def fake_post(self, url, headers=None, json=None):
            captured_bodies.append(json)
            return _mock_response({"message": {"role": "assistant", "content": f"resposta {len(captured_bodies)}"}})

        with patch("httpx.AsyncClient.post", new=fake_post):
            client.post("/v1/chat", json={"conversation_id": "conv-hist-1", "message": "oi, tudo bem"}, headers=GATEWAY_HEADERS)
            client.post("/v1/chat", json={"conversation_id": "conv-hist-1", "message": "e você"}, headers=GATEWAY_HEADERS)
        # a segunda chamada deve incluir a primeira troca no histórico enviado ao modelo
        second_call_messages = captured_bodies[1]["messages"]
        contents = [m["content"] for m in second_call_messages]
        assert "oi, tudo bem" in contents
        assert "resposta 1" in contents
