import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from policy import needs_web_search
from providers import get_provider

logger = logging.getLogger("gateway")
logger.setLevel(logging.INFO)

app = FastAPI(title="Alexa Ollama V2 gateway", version="2.2.0")
DB_PATH = os.getenv("DB_PATH", "/data/alexa_memory.sqlite3")
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", GATEWAY_TOKEN)
ENCRYPTION_KEY = os.getenv("SETTINGS_ENCRYPTION_KEY", "")
ADMIN_HTML = Path(__file__).with_name("admin.html")
TIMEZONE = os.getenv("TIMEZONE", "America/Sao_Paulo")

DEFAULTS = {
    "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
    "llm_url": os.getenv("LLM_URL", "https://ollama.com/api/chat"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-oss:20b"),
    "web_search_url": os.getenv("WEB_SEARCH_URL", "https://ollama.com/api/web_search"),
    "tool_calling_enabled": os.getenv("TOOL_CALLING_ENABLED", "false").strip().lower() == "true",
    "system_prompt": os.getenv("SYSTEM_PROMPT", "Você é um assistente de voz em português do Brasil. Responda de forma breve, natural e honesta."),
    "temperature": float(os.getenv("TEMPERATURE", "0.2")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "120")),
    # Deve ficar abaixo do request_timeout_seconds da Lambda (config.example.json),
    # senão a Lambda desiste antes da Railway, e o erro vira timeout genérico em vez
    # da mensagem específica que o gateway devolveria.
    "timeout_seconds": float(os.getenv("TIMEOUT_SECONDS", "3.5")),
    "max_messages": int(os.getenv("MAX_MESSAGES", "16")),
}


class ChatRequest(BaseModel):
    # amzn1.ask.account.* (o userId que a Lambda usa como conversation_id) passa
    # tranquilamente de 250 caracteres — 160 rejeitava toda requisição real com
    # 422, algo que só um teste com um userId de verdade revela.
    conversation_id: str = Field(min_length=8, max_length=400)
    # Pergunta crua do usuário. A decisão de buscar (e a busca em si) é toda
    # daqui pra frente — a Lambda não faz mais nenhum pré-processamento quando
    # o gateway está configurado.
    message: str = Field(min_length=1, max_length=5000)
    # Regras obrigatórias de voz vindas da Lambda (sem markdown, poucas frases, etc.).
    # A personalidade continua vindo de settings["system_prompt"], editável pelo painel.
    voice_rules: str | None = Field(default=None, max_length=2000)


class SettingsUpdate(BaseModel):
    llm_provider: str = Field(pattern="^(ollama|openai|anthropic)$")
    llm_url: str = Field(min_length=8, max_length=300)
    llm_model: str = Field(min_length=2, max_length=120)
    api_key: str | None = Field(default=None, max_length=500)
    web_search_url: str = Field(default="", max_length=300)
    web_search_key: str | None = Field(default=None, max_length=500)
    tool_calling_enabled: bool = False
    system_prompt: str = Field(min_length=20, max_length=5000)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(ge=40, le=1000)
    timeout_seconds: float = Field(ge=2, le=30)
    max_messages: int = Field(ge=2, le=50)


def connect():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with closing(connect()) as conn:
        conn.execute("create table if not exists messages (conversation_id text, role text, content text, created real)")
        conn.execute("create table if not exists settings (key text primary key, value text not null, updated real not null)")
        conn.commit()


@app.on_event("startup")
def startup():
    init_db()


def token_matches(received, expected):
    return bool(expected and received and secrets.compare_digest(received, expected))


def require_gateway(token):
    if not token_matches(token, GATEWAY_TOKEN):
        raise HTTPException(401, "invalid gateway token")


def require_admin(token):
    if not token_matches(token, ADMIN_TOKEN):
        raise HTTPException(401, "invalid admin token")


def read_stored_settings():
    with closing(connect()) as conn:
        rows = conn.execute("select key, value from settings").fetchall()
    return {key: json.loads(value) for key, value in rows}


def runtime_settings():
    return {**DEFAULTS, **read_stored_settings()}


def write_settings(values):
    now = time.time()
    with closing(connect()) as conn:
        conn.executemany(
            "insert into settings(key, value, updated) values (?, ?, ?) on conflict(key) do update set value=excluded.value, updated=excluded.updated",
            [(key, json.dumps(value, ensure_ascii=False), now) for key, value in values.items()],
        )
        conn.commit()


def fernet():
    if not ENCRYPTION_KEY:
        return None
    try:
        return Fernet(ENCRYPTION_KEY.encode("utf-8"))
    except ValueError as exc:
        raise RuntimeError("SETTINGS_ENCRYPTION_KEY is invalid") from exc


def _decrypt(stored, what):
    cipher = fernet()
    if not cipher:
        raise RuntimeError(f"SETTINGS_ENCRYPTION_KEY is required to read the saved {what}")
    try:
        return cipher.decrypt(stored.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError(f"Unable to decrypt saved {what}") from exc


def encrypted_api_key():
    stored = read_stored_settings().get("api_key_encrypted")
    if stored:
        return _decrypt(stored, "API key")
    return os.getenv("LLM_API_KEY", "")


def encrypted_web_search_key():
    stored = read_stored_settings().get("web_search_key_encrypted")
    if stored:
        return _decrypt(stored, "web search key")
    # Sem chave própria: se o provedor de chat for o Ollama, a mesma conta
    # também vale para a busca. Para outros provedores, a busca fica desligada
    # até uma chave da Ollama Cloud ser configurada separadamente.
    if runtime_settings().get("llm_provider", "ollama") == "ollama":
        return encrypted_api_key()
    return os.getenv("WEB_SEARCH_KEY", "")


def format_search_results(results, max_chars=4200):
    chunks = []
    for item in (results or [])[:4]:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()[:800]
        if content:
            chunks.append(f"Fonte: {title}. {content} URL: {url}")
    return "\n".join(chunks)[:max_chars]


def search_instructions(now, timezone, context):
    """`now` é o datetime de referência: "amanhã"/"hoje" são calculados aqui em
    Python e entregues prontos ao modelo. Pedir pro gpt-oss calcular a data por
    conta própria (mesmo com "think": "low") produzia datas erradas de forma
    consistente em teste real — ex.: "amanhã" virava um dia antes de "hoje".
    """
    if not context:
        return "A busca não retornou nenhum resultado confiável. Diga claramente que não há confirmação atual, sem inventar."
    today = now.strftime("%d/%m/%Y")
    tomorrow = (now + timedelta(days=1)).strftime("%d/%m/%Y")
    return (f"Data e hora de referência: {now.strftime('%d/%m/%Y %H:%M')}, fuso {timezone}. Hoje é {today}. "
            f"Se a pergunta mencionar \"amanhã\", a data é {tomorrow} — use esse valor exato, não calcule por conta própria. "
            "Use somente fatos confirmados pelas fontes abaixo. Não transforme evento futuro em acontecimento passado. "
            "Para notícias, escolha no máximo três itens distintos e confirme que cada um corresponde à data pedida. "
            "Não cite URLs nem diga que recebeu fontes. Se as fontes não confirmarem a resposta, diga isso claramente.\n\n"
            f"{context}")


def reference_time_now():
    return datetime.now(ZoneInfo(TIMEZONE))


async def run_web_search(settings, query):
    search_url = settings.get("web_search_url") or ""
    search_key = encrypted_web_search_key()
    if not search_url or not search_key or not query:
        return []
    try:
        async with httpx.AsyncClient(timeout=settings["timeout_seconds"]) as client:
            response = await client.post(search_url, headers={"Authorization": f"Bearer {search_key}"},
                                          json={"query": query, "max_results": 4})
        response.raise_for_status()
        return response.json().get("results", []) or []
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("web search failed query=%s error=%s", query, exc)
        return []


async def call_provider(provider, settings, api_key, system_prompt, messages, tools):
    body = provider.build_body(settings["llm_model"], system_prompt, messages, tools, settings["max_tokens"], settings["temperature"])
    try:
        async with httpx.AsyncClient(timeout=settings["timeout_seconds"]) as client:
            response = await client.post(settings["llm_url"], headers=provider.headers(api_key), json=body)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        logger.warning("llm request timed out: url=%s timeout=%s", settings["llm_url"], settings["timeout_seconds"])
        raise HTTPException(504, "LLM request timed out") from exc
    except httpx.HTTPStatusError as exc:
        # Sem isso, o log só mostrava "502 Bad Gateway" na linha de acesso do
        # uvicorn — nenhuma pista do que o provedor respondeu de verdade.
        logger.warning("llm request failed: status=%s body=%s", exc.response.status_code, exc.response.text[:500])
        raise HTTPException(502, "LLM request failed") from exc
    except httpx.HTTPError as exc:
        logger.warning("llm request failed: %s", exc)
        raise HTTPException(502, "LLM request failed") from exc
    try:
        parsed_body = response.json()
        turn = provider.parse(parsed_body)
    except ValueError as exc:
        logger.warning("invalid llm response body: %s", exc)
        raise HTTPException(502, "invalid LLM response body") from exc
    if not turn.text and not turn.tool_calls:
        # Diagnóstico temporário: texto vazio sem tool_calls é inesperado e o
        # corpo cru ajuda a distinguir corte por token de outra causa (ex.:
        # done_reason, canal de raciocínio separado do content).
        logger.warning("empty parsed response body=%s", json.dumps(parsed_body, ensure_ascii=False)[:800])
    return turn


async def chat_via_policy(provider, settings, api_key, system_prompt, messages, query):
    """Sem tool calling: a mesma política de regex da Lambda decide se busca."""
    if needs_web_search(query):
        context = format_search_results(await run_web_search(settings, query))
        if not context:
            # Falha segura: não deixa o modelo responder com conhecimento velho
            # quando a pergunta claramente depende de dado atual.
            raise HTTPException(503, "search_unavailable")
        instructions = search_instructions(reference_time_now(), TIMEZONE, context)
        messages = messages[:-1] + [{"role": "user", "content": f"{instructions}\n\nPergunta: {query}"}]
    turn = await call_provider(provider, settings, api_key, system_prompt, messages, tools=False)
    if not turn.text:
        raise HTTPException(502, "empty model response")
    return turn.text


async def chat_via_tools(provider, settings, api_key, system_prompt, messages):
    """O próprio modelo decide se busca, em no máximo duas chamadas.

    Sem corte duro: se a busca falhar ou vier vazia, o modelo recebe essa
    informação como resultado da ferramenta e é instruído a admitir isso,
    em vez de travar a resposta.
    """
    turn = await call_provider(provider, settings, api_key, system_prompt, messages, tools=True)
    if not turn.tool_calls:
        if not turn.text:
            raise HTTPException(502, "empty model response")
        return turn.text
    call = turn.tool_calls[0]
    query = (call.arguments or {}).get("query") or ""
    context = format_search_results(await run_web_search(settings, query))
    result_text = search_instructions(reference_time_now(), TIMEZONE, context)
    conversation = messages + provider.tool_result_messages(turn, call, result_text)
    final_turn = await call_provider(provider, settings, api_key, system_prompt, conversation, tools=False)
    if not final_turn.text:
        raise HTTPException(502, "empty model response")
    return final_turn.text


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(ADMIN_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    settings = runtime_settings()
    return {"ok": True, "service": "alexa-ollama-v2", "version": "2.2.0", "provider": settings["llm_provider"], "model": settings["llm_model"]}


@app.get("/admin/settings")
def get_settings(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    # Nunca devolver os blobs criptografados: não são a chave em texto puro,
    # mas não têm por que sair da API — se SETTINGS_ENCRYPTION_KEY vazar um dia,
    # essas respostas já capturadas (proxy, logs) não podem virar chaves.
    settings = {k: v for k, v in runtime_settings().items() if not k.endswith("_encrypted")}
    return {
        **settings,
        "api_key_configured": bool(encrypted_api_key()),
        "api_key_editable": bool(ENCRYPTION_KEY),
        "web_search_key_configured": bool(encrypted_web_search_key()),
        "web_search_key_editable": bool(ENCRYPTION_KEY),
    }


@app.put("/admin/settings")
def update_settings(payload: SettingsUpdate, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    get_provider(payload.llm_provider)  # valida antes de salvar
    values = payload.model_dump(exclude={"api_key", "web_search_key"})
    if payload.api_key:
        cipher = fernet()
        if not cipher:
            raise HTTPException(400, "Configure SETTINGS_ENCRYPTION_KEY before changing the API key")
        values["api_key_encrypted"] = cipher.encrypt(payload.api_key.encode("utf-8")).decode("utf-8")
    if payload.web_search_key:
        cipher = fernet()
        if not cipher:
            raise HTTPException(400, "Configure SETTINGS_ENCRYPTION_KEY before changing the web search key")
        values["web_search_key_encrypted"] = cipher.encrypt(payload.web_search_key.encode("utf-8")).decode("utf-8")
    write_settings(values)
    return {"ok": True, "message": "Configurações salvas. As próximas respostas já usarão os novos valores."}


@app.post("/v1/chat")
async def chat(payload: ChatRequest, x_gateway_token: str | None = Header(default=None)):
    require_gateway(x_gateway_token)
    settings = runtime_settings()
    provider = get_provider(settings["llm_provider"])
    api_key = encrypted_api_key()
    if not api_key:
        raise HTTPException(503, "LLM API key is not configured")

    with closing(connect()) as conn:
        rows = conn.execute(
            "select role, content from messages where conversation_id=? order by created desc limit ?",
            (payload.conversation_id, settings["max_messages"]),
        ).fetchall()
    history = [{"role": role, "content": content} for role, content in reversed(rows)]
    messages = history + [{"role": "user", "content": payload.message}]

    system_prompt = settings["system_prompt"]
    if payload.voice_rules:
        system_prompt = f"{system_prompt} {payload.voice_rules}"

    started = time.perf_counter()
    if settings["tool_calling_enabled"]:
        answer = await chat_via_tools(provider, settings, api_key, system_prompt, messages)
    else:
        answer = await chat_via_policy(provider, settings, api_key, system_prompt, messages, payload.message)

    with closing(connect()) as conn:
        now = time.time()
        conn.executemany("insert into messages values (?, ?, ?, ?)", [(payload.conversation_id, "user", payload.message, now), (payload.conversation_id, "assistant", answer, now + 0.001)])
        conn.commit()
    return {"message": answer, "latency_ms": round((time.perf_counter() - started) * 1000), "model": settings["llm_model"]}
