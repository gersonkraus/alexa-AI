# Alexa Ollama V2 Final — assistente de voz em pt-BR

Esta é a pasta consolidada da versão final, já contendo as correções de Web Search, follow-up biográfico, data e hora em pt-BR, modelo de interação sem amostras inválidas e painel administrativo opcional para Railway.

Implementação pronta para Alexa Custom Skill + Ollama Cloud. O modo padrão é direto e rápido: Alexa-hosted Lambda chama `ollama.com/api/chat` e usa `ollama.com/api/web_search` somente quando a pergunta pode depender de informação atual. O modo opcional usa o gateway em `railway/` para guardar memória entre sessões e criar uma base para ferramentas externas.

## O que mudou na V2

- Detecção conservadora de atualidade: presidente, CEO, preços, clima, notícias, versões e perguntas explicitamente atuais entram em busca; explicações estáveis e cálculos não.
- Falha segura: se uma pergunta atual exigir busca e a busca falhar, a skill não responde com conhecimento potencialmente velho.
- Multi-turn dentro da sessão com histórico limitado e follow-up natural.
- Resposta própria para voz: curta, sem markdown, URLs ou símbolos difíceis de pronunciar.
- Timeouts separados para busca e LLM, calibrados para caber com folga na janela de 8 segundos que a Alexa dá para a resposta, logs de latência e erros sem expor chaves.
- Gateway Railway opcional com SQLite persistente, health check, autenticação por token e histórico entre sessões.
- Resposta progressiva ("Só um momento, deixa eu verificar...") quando a pergunta exige busca, para não deixar o usuário no silêncio enquanto a skill trabalha.
- "Alexa, repita" repete a última fala; "não" tem reconhecimento próprio em vez de cair no erro genérico.
- Card visual (título e texto da resposta) no app Alexa e em dispositivos com tela, como Echo Show.
- No modo gateway, as regras obrigatórias de voz (sem markdown, poucas frases) agora chegam de fato até o Ollama — antes eram calculadas na Lambda mas descartadas pela Railway.
- Suporte a outros provedores de LLM além do Ollama (Grok/xAI, Anthropic/Claude), com chave e URL de busca web separadas da chave de chat.
- Tool calling opcional: em vez da política de regex decidir quando buscar, o próprio modelo decide (desligado por padrão).

## Trocando de provedor de LLM (Ollama, Grok/xAI, Anthropic)

O campo `llm_provider` em `lambda/config.json` decide como a Lambda fala com o provedor de chat. A busca web continua sempre na API do Ollama, com sua própria credencial — assim dá para usar Grok ou Claude para conversar e manter a busca do Ollama funcionando.

| `llm_provider` | `llm_url` | `llm_key` | `llm_model` (exemplo) |
|---|---|---|---|
| `ollama` (padrão) | `https://ollama.com/api/chat` | chave da Ollama Cloud | `gpt-oss:20b` |
| `openai` (qualquer API compatível com Chat Completions: Grok/xAI, OpenRouter, etc.) | `https://api.x.ai/v1/chat/completions` | chave da xAI | `grok-4.6` |
| `anthropic` | `https://api.anthropic.com/v1/messages` | chave da Anthropic | `claude-haiku-4-5` |

Se `llm_provider` for diferente de `ollama`, preencha também `web_search_key` com uma chave válida da Ollama Cloud — sem ela, a busca falha silenciosamente (a skill continua respondendo, só sem contexto atual) porque a chave de chat de outro provedor não é aceita pela API de busca do Ollama.

`llm_max_tokens` limita o tamanho da resposta e é obrigatório para a Anthropic (a API exige esse campo); para os demais provedores ele só ajuda a manter a resposta curta e a latência baixa.

### Tool calling (opcional)

Com `"tool_calling_enabled": true`, a decisão de buscar ou não deixa de ser a política de regex em português (`llm_intent/policy.py`) e passa a ser do próprio modelo: ele recebe uma ferramenta `web_search` e decide se e quando chamá-la. Isso corrige casos que a regex não prevê, mas muda dois comportamentos:

- **Latência**: perguntas que buscam fazem duas idas e vindas ao modelo em vez de uma. Reduza `request_timeout_seconds` e `search_timeout_seconds` proporcionalmente para continuar cabendo na janela de 8 segundos da Alexa (por exemplo, 3.0 e 1.5).
- **Falha segura mais branda**: se a busca falhar ou vier vazia, a skill não bloqueia mais a resposta com "search_unavailable" — o modelo recebe essa informação e é instruído a admitir que não há confirmação atual, em vez de inventar. Teste esse caminho antes de confiar nele: há relatos de instabilidade de tool calling com `gpt-oss` em alguns clientes.

Fica desligado por padrão porque a política de regex já tem testes cobrindo os casos que motivaram a V2; ligue como experimento e compare os dois antes de assumir como padrão.

### E no modo gateway?

O gateway (`railway/`) tem a mesma paridade: provedor, chave de busca separada e tool calling são configurados pelo painel web, não pelo `config.json` da Lambda. Quando `gateway_url` está preenchido, a Lambda não decide mais nada sozinha — só encaminha a pergunta crua para a Railway, que decide se busca (regex ou tool calling, conforme o painel), fala com o provedor escolhido e devolve o texto pronto. Isso significa que, em modo gateway, os campos `llm_provider`, `llm_key`, `web_search_key` e `tool_calling_enabled` do `config.json` da Lambda ficam sem uso — tudo isso mora no painel da Railway a partir de agora.

## Deploy rápido sem Railway

1. Crie uma skill Custom no Alexa Developer Console, locale Português (Brasil).
2. Em Build, JSON Editor, importe [`skill-package/interactionModels/custom/pt-BR.json`](skill-package/interactionModels/custom/pt-BR.json), salve e construa o modelo.
3. Em Code, substitua `lambda_function.py`, crie a pasta `llm_intent` com os três arquivos, e adicione `requirements.txt`.
4. Preserve o seu `lambda/config.json` atual. Em uma instalação nova, copie `lambda/config.example.json` para `lambda/config.json` e preencha `llm_key`. Não publique esse arquivo nem compartilhe a chave. O arquivo `config.json` não vem preenchido nesta pasta porque contém sua chave privada.
5. Clique em Deploy. Em Test, selecione Development e teste: “Alexa, abrir assistente ia”, “quem é o presidente dos Estados Unidos?” e depois “me fale mais”.

O campo `llm_model` pode ser trocado por outro modelo disponível na sua conta Ollama Cloud. A API oficial usa `https://ollama.com/api/chat` e autenticação Bearer; a busca usa `https://ollama.com/api/web_search`.

## Deploy opcional na Railway

Use este modo se você quer memória entre sessões ou pretende adicionar ferramentas domésticas depois. Na Railway, crie um serviço a partir da pasta `railway/` (Dockerfile detectado automaticamente), adicione um Volume montado em `/data`, e configure as variáveis de [`railway/.env.example`](railway/.env.example). Gere um domínio público e teste `GET /health`.

Depois, em `lambda/config.json`, configure:

```json
{
  "gateway_url": "https://seu-servico.up.railway.app",
  "gateway_token": "o-mesmo-token-forte-da-railway"
}
```

A partir daqui, o gateway é quem decide o provedor, a busca e o tool calling — tudo pelo painel web (veja abaixo). O `llm_key`/`llm_provider`/etc. do `config.json` da Lambda não são mais usados quando `gateway_url` está preenchido. O gateway recebe a pergunta crua e as regras obrigatórias de voz, decide se precisa buscar, chama o provedor configurado e mantém o histórico persistente. Para produção, use um volume Railway; SQLite sem volume é temporário.

Mantenha `TIMEOUT_SECONDS` da Railway sempre menor que `request_timeout_seconds` da Lambda (o padrão de ambos já vem calibrado assim). Se a Railway demorar mais que a Lambda espera, você perde a mensagem de erro específica do gateway e cai no erro genérico de timeout. Se ligar o tool calling do gateway, reduza ainda mais essa margem (até duas chamadas ao modelo em sequência).

### Painel de configurações

Depois do deploy, abra o domínio público da Railway. O painel permite alterar sem novo deploy:

- provedor de chat (Ollama, compatível com OpenAI — Grok/xAI, etc. — ou Anthropic), URL e modelo;
- chave da API de chat, sempre mascarada e criptografada no banco;
- URL e chave da busca web, independentes da chave de chat;
- tool calling: liga/desliga o modelo decidindo sozinho quando buscar;
- system prompt (personalidade — as regras obrigatórias de voz continuam vindo da Lambda e são somadas a isso automaticamente);
- temperatura e limite da resposta;
- timeout e quantidade de mensagens mantidas na memória.

Entre usando o valor de `ADMIN_TOKEN`. Defina também `SETTINGS_ENCRYPTION_KEY` para liberar a alteração segura das chaves pelo painel. Gere essa chave uma vez com:

```text
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Não troque `SETTINGS_ENCRYPTION_KEY` depois de salvar uma chave pelo painel, pois ela é necessária para descriptografá-la. As configurações ficam no volume `/data` e passam a valer na chamada seguinte.

## Testes locais

Na raiz deste pacote:

```text
python -m pytest -q
python -m compileall lambda railway
```

Os testes unitários verificam a decisão de busca por regex (`test_policy.py`), os adaptadores de provedor — Ollama, compatível com OpenAI e Anthropic (`test_providers.py`) — e o ciclo de tool calling com rede mockada (`test_llm_client.py`). O teste contra o provedor de verdade deve ser feito somente depois de configurar a chave.

O gateway tem sua própria suíte, separada por depender de FastAPI/httpx (a Lambda não pode ter essas dependências pesadas). Rode com Python 3.9–3.12 (o `pydantic` pinado não compila em versões mais novas):

```text
cd railway
pip install -r requirements.txt pytest
python -m pytest tests/ -q
```

### Se perguntas atuais falharem, mas perguntas gerais funcionarem

Isso indica que o chat está acessível, mas a etapa de Web Search falhou. Confirme que `web_search_url` é `https://ollama.com/api/web_search`, que `web_search_key` (ou `llm_key`, se `web_search_key` estiver vazio e `llm_provider` for `ollama`) tem acesso à busca, e consulte os logs da Lambda por `web search failed`. Se `llm_provider` não for `ollama`, `web_search_key` precisa estar preenchida — a chave do outro provedor não funciona na busca do Ollama. A skill não usa conhecimento antigo como fallback para clima, notícias, cotações ou cargos atuais.

## Limites e próximos passos

A skill não controla dispositivos da casa ainda: isso exige APIs específicas e permissões próprias. O gateway foi isolado para permitir adicionar Home Assistant, calendário ou clima sem colocar credenciais dessas integrações no código da Alexa. Para uso pessoal, comece sem Railway; habilite-a quando memória entre sessões realmente fizer diferença.
