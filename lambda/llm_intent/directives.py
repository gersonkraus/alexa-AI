# -*- coding: utf-8 -*-
"""Progressive Response: preenche o silêncio enquanto a busca/LLM demoram.

Não amplia a janela de 8 segundos que a Alexa dá para a resposta final, então
falhas aqui nunca podem interromper o fluxo principal — são só melhor esforço.
"""

import json
import logging
import socket
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DIRECTIVE_TIMEOUT_SECONDS = 1.5


def send_progressive_response(handler_input, speech_text):
    try:
        envelope = handler_input.request_envelope
        system = envelope.context.system
        api_endpoint = system.api_endpoint
        api_access_token = system.api_access_token
        request_id = envelope.request.request_id
        if not (api_endpoint and api_access_token and request_id):
            return
        body = {
            "header": {"requestId": request_id},
            "directive": {"type": "VoicePlayer.Speak", "speech": f"<speak>{speech_text}</speak>"},
        }
        request = urllib.request.Request(
            api_endpoint.rstrip("/") + "/v1/directives",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_access_token}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=DIRECTIVE_TIMEOUT_SECONDS):
            pass
    except (urllib.error.URLError, urllib.error.HTTPError, AttributeError, TimeoutError, socket.timeout) as exc:
        logger.warning("progressive response failed: %s", exc)
