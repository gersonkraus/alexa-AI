# -*- coding: utf-8 -*-
import html
import logging
import time
from datetime import datetime, timedelta, timezone

from ask_sdk_core import utils as ask_utils
from ask_sdk_core.dispatch_components import AbstractExceptionHandler, AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_model.ui import SimpleCard
from llm_intent.directives import send_progressive_response
from llm_intent.llm_client import LLMClient, UpstreamError
from llm_intent.policy import needs_web_search
from llm_intent.utils import CannedResponse, format_search_results, load_config, safe_speech, search_instructions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
CONFIG = load_config()
CANNED = CannedResponse()
MAX_HISTORY = int(CONFIG.get("max_history_messages", 12))
MAX_SEARCH_CHARS = int(CONFIG.get("max_search_context_chars", 4200))
CLIENT = LLMClient(
    CONFIG.get("llm_provider", "ollama"), CONFIG["llm_url"], CONFIG["llm_key"], CONFIG["llm_model"],
    CONFIG.get("web_search_url", ""), CONFIG.get("web_search_key", ""),
    CONFIG.get("request_timeout_seconds", 4.0), CONFIG.get("search_timeout_seconds", 2.5),
    CONFIG.get("llm_max_tokens", 300), MAX_SEARCH_CHARS, CONFIG.get("tool_calling_enabled", False),
)
PERSONALITY_PROMPT = CONFIG.get("llm_system_prompt", "Você é um assistente de voz em português do Brasil. Seja breve e natural.")
# Regras que sempre valem para voz, independente da personalidade configurada.
# Enviadas também ao gateway Railway (voice_rules), que as soma à personalidade
# guardada no painel — assim o painel continua editável sem perder essas garantias.
VOICE_RULES = (
    "Regras obrigatórias: use no máximo três frases curtas; não use markdown nem URLs; "
    "para fatos atuais, responda somente com informações confirmadas no contexto fornecido; "
    "use datas concretas e nunca diga expressões como futuro não definido."
)
SYSTEM_PROMPT = f"{PERSONALITY_PROMPT} {VOICE_RULES}"
TIMEZONE = CONFIG.get("timezone", "America/Sao_Paulo")
TIMEZONE_UTC_OFFSET_HOURS = int(CONFIG.get("timezone_utc_offset_hours", -3))
GATEWAY_URL = CONFIG.get("gateway_url", "").strip()

WEEKDAYS_PT = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo")
MONTHS_PT = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro")


def upstream_error_message(exception):
    detail = str(exception).lower()
    if "401" in detail or "403" in detail:
        return "A chave do Ollama foi recusada. Verifique a chave no arquivo de configuração."
    if "404" in detail:
        return "O endereço da API ou o nome do modelo não foi encontrado. Verifique a configuração."
    if "429" in detail:
        return "O limite de uso do Ollama foi atingido. Tente novamente daqui a pouco."
    if "timed out" in detail or "timeout" in detail:
        return "O Ollama demorou mais que o permitido para responder. Tente novamente."
    if "http 503" in detail:
        return "O gateway não tem uma chave de API configurada. Entre no painel da Railway e configure uma."
    if "http 502" in detail:
        return "O serviço de IA está indisponível no momento. Tente novamente em alguns segundos."
    return CANNED.get("error")


def local_now():
    # O Brasil não adota horário de verão desde 2019. O offset configurável
    # evita depender de zoneinfo, ausente em alguns runtimes Alexa Python 3.8.
    return datetime.now(timezone(timedelta(hours=TIMEZONE_UTC_OFFSET_HOURS)))


def spoken_date(moment):
    return f"Hoje é {WEEKDAYS_PT[moment.weekday()]}, {moment.day} de {MONTHS_PT[moment.month - 1]} de {moment.year}."


def spoken_time(moment):
    if moment.minute == 0:
        return f"Agora são {moment.hour} horas."
    return f"Agora são {moment.hour} horas e {moment.minute} minutos."


def attr(handler_input):
    return handler_input.attributes_manager.session_attributes


def history(handler_input):
    return attr(handler_input).get("history", [])


def remember(handler_input, role, content):
    items = history(handler_input)
    items.append({"role": role, "content": content})
    attr(handler_input)["history"] = items[-MAX_HISTORY:]


def clean_query(text):
    return " ".join((text or "").replace("  ", " ").split()).strip()


def _answer_via_gateway(handler_input, query):
    # O gateway (railway/app.py) agora decide sozinho se e como buscar — regex
    # ou tool calling, conforme configurado no painel — e fala com o provedor
    # que estiver escolhido lá. A Lambda só encaminha a pergunta crua e recebe
    # de volta o texto pronto ou o sinal de busca indisponível.
    user = getattr(getattr(handler_input.request_envelope, "session", None), "user", None)
    conversation_id = getattr(user, "user_id", None) or "alexa-anonymous"
    response, status = CLIENT.gateway_chat(GATEWAY_URL, CONFIG.get("gateway_token", ""), conversation_id, query, VOICE_RULES)
    return response, status, "gateway"


def _answer_via_tool_calling(handler_input, query):
    # O próprio modelo decide se precisa buscar (ver LLMClient.chat_with_tools).
    # Sem o corte duro de "search_unavailable": se a busca falhar, o modelo é
    # instruído a admitir isso em vez de responder com conhecimento desatualizado.
    messages = list(history(handler_input))
    messages.append({"role": "user", "content": query})
    reference_time = local_now().strftime("%d/%m/%Y %H:%M")
    response, used_search = CLIENT.chat_with_tools(SYSTEM_PROMPT, messages, reference_time, TIMEZONE)
    remember(handler_input, "user", query)
    remember(handler_input, "assistant", response)
    return response, None, used_search


def _answer_via_policy(handler_input, query, force_search):
    should_search = force_search or needs_web_search(query)
    prompt = query
    if should_search:
        context = format_search_results(CLIENT.web_search(query), MAX_SEARCH_CHARS)
        if not context:
            # Do not let stale model knowledge answer a request explicitly/currently requiring search.
            logger.warning("required search unavailable query=%s", query)
            return None, "search_unavailable", should_search
        reference_time = local_now().strftime("%d/%m/%Y %H:%M")
        prompt = f"{search_instructions(reference_time, TIMEZONE, context)}\n\nPergunta: {query}"
    messages = list(history(handler_input))
    messages.append({"role": "user", "content": prompt})
    response = CLIENT.chat(SYSTEM_PROMPT, messages)
    remember(handler_input, "user", query)
    remember(handler_input, "assistant", response)
    return response, None, should_search


def answer(handler_input, user_text, force_search=False):
    start = time.perf_counter()
    query = clean_query(user_text)
    if GATEWAY_URL:
        response, status, used_search = _answer_via_gateway(handler_input, query)
    elif CLIENT.tool_calling_enabled:
        response, status, used_search = _answer_via_tool_calling(handler_input, query)
    else:
        response, status, used_search = _answer_via_policy(handler_input, query, force_search)
    if status:
        return None, status
    logger.info("answer_ms=%d search=%s history=%d", int((time.perf_counter() - start) * 1000), used_search, len(history(handler_input)))
    return safe_speech(response), None


def respond(handler_input, speech, keep_open=True, card_title="Assistente IA"):
    attr(handler_input)["last_speech"] = speech
    builder = handler_input.response_builder.speak(speech)
    if keep_open:
        builder.ask(CANNED.get("reprompt"))
    if card_title:
        builder.set_card(SimpleCard(card_title, html.unescape(speech)))
    return builder.response


class LaunchHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        return respond(handler_input, CANNED.get("launch"))


class QuestionHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("QuestionIntent")(handler_input)

    def handle(self, handler_input):
        request = handler_input.request_envelope.request
        slots = getattr(request, "intent", None).slots if getattr(request, "intent", None) else {}
        slot = slots.get("question") or slots.get("searchQuery") if slots else None
        text = getattr(slot, "value", None) if slot else None
        if not text:
            return respond(handler_input, "Não captei a pergunta. Pode repetir?")
        # Preenche o silêncio da busca + LLM, que juntos podem se aproximar dos 8
        # segundos que a Alexa tolera. Isso não estende esse limite, só melhora a
        # percepção de resposta enquanto ele ainda não estourou. Com gateway ou
        # tool calling ligados não dá para prever de antemão se vai haver busca
        # (a decisão sai da Lambda), e o gateway já é um salto de rede a mais
        # por si só, então avisamos sempre nesses dois casos.
        if GATEWAY_URL or CLIENT.tool_calling_enabled or needs_web_search(clean_query(text)):
            send_progressive_response(handler_input, CANNED.get("thinking"))
        try:
            speech, status = answer(handler_input, text)
            if status == "search_unavailable":
                return respond(handler_input, CANNED.get("search_error"))
            return respond(handler_input, speech or CANNED.get("error"))
        except UpstreamError as exc:
            logger.exception("upstream failure")
            return respond(handler_input, upstream_error_message(exc))


class DateHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("DateIntent")(handler_input)

    def handle(self, handler_input):
        return respond(handler_input, spoken_date(local_now()))


class TimeHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("TimeIntent")(handler_input)

    def handle(self, handler_input):
        return respond(handler_input, spoken_time(local_now()))


class YesHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.YesIntent")(handler_input)

    def handle(self, handler_input):
        try:
            speech, status = answer(handler_input, "Pode me contar mais sobre isso.")
            return respond(handler_input, CANNED.get("search_error") if status else speech)
        except UpstreamError as exc:
            logger.exception("upstream failure")
            return respond(handler_input, upstream_error_message(exc))


class NoHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.NoIntent")(handler_input)

    def handle(self, handler_input):
        return respond(handler_input, CANNED.get("no_ack"), card_title=None)


class RepeatHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.RepeatIntent")(handler_input)

    def handle(self, handler_input):
        last_speech = attr(handler_input).get("last_speech")
        if not last_speech:
            return respond(handler_input, CANNED.get("nothing_to_repeat"), card_title=None)
        return respond(handler_input, last_speech)


class HelpHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        return respond(handler_input, CANNED.get("help"))


class StopHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input) or ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.speak(CANNED.get("goodbye")).response


class FallbackHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # Perguntas de acompanhamento sem frase-gatilho (ex.: "que dia ele nasceu"
        # após já estar no meio de uma conversa) caem aqui — a Alexa não entrega o
        # texto bruto no FallbackIntent, então não dá para reencaminhar a pergunta
        # para o modelo. O melhor possível é ensinar uma frase que de fato funciona.
        return respond(handler_input, CANNED.get("fallback"))


class SessionEndedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response


class Errors(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.exception("unhandled request error")
        return handler_input.response_builder.speak(CANNED.get("error")).response


sb = SkillBuilder()
for handler in (
    LaunchHandler(), DateHandler(), TimeHandler(), QuestionHandler(), YesHandler(), NoHandler(),
    RepeatHandler(), HelpHandler(), StopHandler(), FallbackHandler(), SessionEndedHandler(),
):
    sb.add_request_handler(handler)
sb.add_exception_handler(Errors())
lambda_handler = sb.lambda_handler()
