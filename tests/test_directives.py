import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "lambda"))
from llm_intent.directives import send_progressive_response


class _ExplodingSystem:
    @property
    def api_endpoint(self):
        # Regressão: um tipo de exceção fora de (URLError, HTTPError,
        # AttributeError, TimeoutError, socket.timeout) escapava daqui sem
        # ser capturado — e como send_progressive_response roda antes do
        # try/except do QuestionHandler, derrubava a resposta inteira no
        # handler de erro genérico, mesmo com o gateway respondendo certo.
        raise RuntimeError("boom")


class _Context:
    system = _ExplodingSystem()


class _Envelope:
    context = _Context()

    class request:
        request_id = "req-1"


class _HandlerInput:
    request_envelope = _Envelope()


def test_send_progressive_response_never_raises_on_unexpected_exception():
    send_progressive_response(_HandlerInput(), "Só um momento, deixa eu verificar.")
