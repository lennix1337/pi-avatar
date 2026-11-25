# Companion

Cérebro de uma companheira de desktop em pt-BR: conversa por voz (ou por texto), tem personalidade, lembra do usuário, fala sozinha de vez em quando e comanda o pet (`rune-pet`) pelo protocolo de `PROTOCOL.md` (TCP `127.0.0.1:47800`, JSON por linha; o companion é o servidor).

```
voz/texto ──► Companion (app.py) ──► Pi (--mode rpc, processo persistente, sem ferramentas)
                 │  ▲                     │ tokens em streaming
                 │  └── memória, iniciativa, persona
                 ▼
            PetServer ──► rune-pet   (state, say, mouth, emotion, gesture, end)
```

## Rodar

```
cd companion
uv run python -m companion --text            # conversa digitada, sem áudio (precisa do Pi logado)
uv run python -m companion --text --fake-llm # sem rede, respostas de mentira
uv run python -m companion --state-file C:\caminho\harness.json   # eventos do harness (ciRed, merged, workers, waitingOperator, idleMinutes)
uv run pytest
```

`/sair` encerra o modo texto. A voz real é o pacote `companion.voice` (função `create_voice()`); sem ele só o modo `--text` funciona.

## Configuração

`%APPDATA%\pi-avatar\companion.json` (Linux: `~/.config/pi-avatar/`), todos os campos opcionais:

| campo | padrão | |
|---|---|---|
| `name` | `Jujuba` | nome da personagem |
| `user_name` | vazio | como ela chama o usuário (vazio: "ele") |
| `model` | `anthropic/claude-sonnet-5-5` | modelo do Pi (`provider/id`); `summarizer_model` usa outro para os resumos |
| `pi_command` | `pi` | executável do Pi; a conta/OAuth é a do próprio Pi, o companion nunca mexe nelas |
| `persona_path` | embutida | arquivo de persona próprio (`{name}` e `{user}` são substituídos) |
| `filler_after_s` | `1.2` | só enche o silêncio ("hmm", "peraí") se o primeiro token demorar mais que isso |
| `history_window`, `summarize_after`, `reset_session_after` | 30, 16, 60 | janela de histórico, mensagens novas até resumir, turnos até reiniciar a sessão do Pi |
| `initiative` | ver `config.py` | `silence_s`, `backoff`, `max_interval_s`, `max_unanswered`, `quiet_start_hour`/`quiet_end_hour`, cooldowns… |

Memória e config ficam na pasta de perfil (`memory/default.json`), nunca no repositório. `--profile-dir` troca a pasta.

## Como funciona

- **LLM** (`brain/llm.py`): um `pi --mode rpc --no-session --no-tools --no-extensions ... --system-prompt <persona>` persistente. Só o Pi (OAuth) fala com o modelo. O texto chega em deltas; `SentenceChunker` corta em frases e a primeira frase já é falada enquanto o resto ainda é gerado.
- **Persona** (`brain/persona.md`): personagem original, sarcástica e carinhosa, com exemplos de tom, regras de fala e etiquetas `[smug]`, `[gesture:tilt]` que `brain/tags.py` converte em mensagens `emotion`/`gesture` e remove do texto falado.
- **Memória** (`brain/memory.py`): janela deslizante + resumo periódico (fatos, piadas internas, resumo) feito pelo LLM fora do caminho quente e injetado no começo de cada sessão do Pi.
- **Iniciativa** (`brain/initiative.py`): cumprimento ao subir, fala após silêncio com recuo exponencial e jitter, calada de madrugada (a menos que ele esteja falando), comentário de "ainda acordado", reação a eventos do harness (CI vermelho, merge, esperando por ele) e a toques do pet.
- **Turnos** (`app.py`): estados `listening/thinking/speaking` para o pet, barge-in (corta a fala e a geração, mantém o trecho interrompido no histórico), filler só se o 1º token passar de 1,2 s, resposta de falha falada se o Pi cair.
- **Segurança**: o Pi roda sem ferramentas, sem extensões e sem MCP; o companion não executa comandos nem envia mensagens a ninguém.

## Voz

O orquestrador conversa com a voz por `voiceapi.VoiceIO` (`listen()`, `speak(text, emotion, on_mouth)`, `cancel()`, callback `on_speech_start`). `FakeVoice` implementa isso em console para `--text` e testes.

## Medidas

`uv run python scripts/bench.py --models anthropic/claude-haiku-5-5 ... --turns 13` mede, com o Pi real, o tempo do fim do texto até a primeira frase pronta. `scripts/sample_conversation.py` gera o transcrito de `docs/`.

### Latência (Pi real, 13 turnos, fim do texto → primeira frase pronta)

| modelo | 1º token p50 | 1ª frase p50 | p90 | máx |
|---|---|---|---|---|
| anthropic/claude-haiku-4-5 | 0.37 s | 0.54 s | 0.87 s | 0.93 s |
| anthropic/claude-haiku-5-5 | 0.50 s | 0.63 s | 0.83 s | 0.96 s |
| anthropic/claude-sonnet-5-5 | 0.96 s | 1.33 s | 2.11 s | 4.94 s |
| anthropic/claude-sonnet-4-6 | 1.13 s | 1.73 s | 2.17 s | 2.25 s |

(Numa segunda rodada anterior, sonnet-5-5 deu 0.76 s de p50; há variação de rede.) O padrão é `claude-sonnet-5-5`: no transcrito, o haiku fala em parágrafos e faz pergunta em toda fala; o sonnet responde curto e com opinião, mais humano. Quem preferir latência mínima troca para `anthropic/claude-haiku-5-5` em `companion.json` (p50 0.63 s).

Conversa de exemplo: `docs/sample-conversation.md`.
