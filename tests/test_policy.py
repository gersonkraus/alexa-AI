import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "lambda"))
from llm_intent.policy import needs_web_search


def test_current_leader_requires_search():
    assert needs_web_search("Quem é o presidente dos Estados Unidos?")


def test_explicit_current_question_requires_search():
    assert needs_web_search("Qual é a versão atual do Python?")


def test_stable_explanation_does_not_require_search():
    assert not needs_web_search("Explique como funciona a fotossíntese")


def test_biographical_follow_up_does_not_require_search():
    assert not needs_web_search("Quando que ele nasceu?")


def test_weather_and_news_require_search():
    assert needs_web_search("Qual a previsão do tempo para Palhoça amanhã?")
    assert needs_web_search("Pesquise as principais notícias de tecnologia de hoje")
