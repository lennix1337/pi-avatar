# Voz da companheira

Camada de voz (`companion/companion/voice/`): ela **ouve** sem apertar botão, **fala** com voz feminina brasileira
local e **mexe a boca** do avatar (mensagens `mouth` do `PROTOCOL.md`, ~30 Hz). Tudo roda no PC, sem API paga.

```python
from companion.voice import Voice, Utterance

voice = Voice(config, on_speech_start=brain.interrupt)   # config = dict parcial; o resto vem de voice/config.py
async for utt in voice.listen():                          # Utterance(text, started_at, ended_at)
    ...
await voice.speak("Oi!", "happy", on_mouth=pet.send_mouth)   # uma chamada por frase
voice.cancel()                                                # barge-in: para já e descarta a fila
await voice.close()
```

`speak()` espera a frase **tocar**; o cérebro pode disparar `speak()` da frase 2 sem esperar a 1 (as chamadas
entram numa fila): enquanto a frase N toca, a N+1 já está sendo sintetizada e vai para o mesmo buffer de saída,
sem buraco. `on_mouth` recebe `{"aa","ih","ou","ee","oh"}` em 0..1 a 30 Hz e um último frame zerado ao terminar
(o pet zera a boca sozinho depois de 150 ms sem mensagem).

## Como funciona

| Peça | Arquivo | Decisão |
|---|---|---|
| Microfone | `mic.py` | `sounddevice` 16 kHz, frames de 32 ms. `file:x.wav` e `ArrayMic` leem WAV (testes, sem microfone). |
| VAD | `vad.py` | Silero (neural) com limiar duplo; cai para VAD por energia sem Silero. `Segmenter`: confirma fala em 96 ms (dispara `on_speech_start`), ignora estalos < 250 ms, fecha o turno após 550 ms de silêncio, pré-roll de 320 ms, corta em 30 s. |
| Anti-eco | `echo.py` | Ver abaixo. |
| STT | `stt.py` | `auto`: **Canary 1B v2** do Pi Voice (transcribe.cpp, GGUF Q8_0) por um sidecar Node (`canary_worker.mjs`) se o modelo e o `transcribe-cpp` existirem na máquina; senão `faster-whisper large-v3-turbo` (CUDA). |
| TTS | `tts/` | Uma interface (`stream()` devolve pedaços float32). Backends medidos abaixo. Padrão: **Supertonic 3**. |
| Saída | `sink.py`, `speaker.py` | Um buffer com relógio de reprodução. Uma thread sintetiza as frases em ordem e escreve no buffer; a boca é calculada do áudio e emitida pelo **relógio da reprodução** (não da síntese), então não adianta nem atrasa. |
| Boca | `lipsync.py` | Do áudio, sem fonemas: LPC → formantes F1/F2 → vogal mais próxima (aa, ih, ou, ee, oh) → suavização (ataque rápido, soltura lenta) × volume. Funciona igual para qualquer TTS. |

### Anti-eco (ela não se interrompe com a própria voz)

O microfone capta a caixa de som. `EchoGate` conhece o volume que está tocando (`sink.recent_rms()`) e aprende
online quanto vaza para o microfone (`coupling`; nos primeiros 300 ms de cada fala tudo é eco e ele calibra).
Enquanto ela fala (+250 ms de cauda), o microfone só conta como "usuário" se for **≥ 2,5× o eco esperado**, acima de
-42 dBFS e por 4 frames seguidos (um pico isolado não vale); depois de reconhecer o usuário o portão fica aberto
por ~320 ms para atravessar pausas entre sílabas. O limiar do VAD também sobe (0,5 → 0,8) enquanto ela fala. Ao
confirmar fala do usuário: `cancel()` (o áudio para) e `on_speech_start()` (o cérebro interrompe).
Modos em `echo.mode`: `auto` (padrão), `mute` (surda enquanto fala; com fone de ouvido), `off`.

### Push-to-talk

`listen.mode = "ptt"` e `voice.ptt(True/False)` (o cérebro/pet liga isso a uma tecla): o turno vale do apertar ao soltar.

## Configuração (`voice/config.py`)

```python
{"tts": {"backend": "supertonic", "options": {}},      # kokoro | supertonic | voxcpm | omnivoice | qwen3
 "stt": {"backend": "auto", "language": "pt"},         # auto | canary | faster-whisper
 "listen": {"mode": "vad", "input_device": None, "vad": {"end_silence_ms": 550, ...}},
 "echo": {"mode": "auto", "margin": 2.5, ...},
 "output": {"device": None}, "lipsync": {"fps": 30, "gain": 1.0}}
```

Canary precisa do modelo `canary-1b-v2-Q8_0.gguf` no cache do Hugging Face e do `transcribe-cpp` em
`~/.pi/agent/npm/node_modules` (onde o Pi Voice o instala); outros caminhos: `COMPANION_CANARY_MODEL` e
`COMPANION_TRANSCRIBE_MODULES`. Dependências Python em `companion/voice-requirements.txt`.

## Identidade da voz e licenças

Voz **desenhada, não clonada**: nenhum clipe de pessoa real, nenhum clone de streamer. O padrão (Supertonic, voz
predefinida `F1`) é uma voz feminina de fábrica do modelo. Nenhuma referência de áudio é usada em nenhum backend
(o VoxCPM gera a própria voz-persona a partir de uma descrição e a reutiliza).

| Modelo | Licença (card do Hugging Face) |
|---|---|
| Supertonic 3 (`Supertone/supertonic-3`) | OpenRAIL |
| Kokoro-82M | Apache-2.0 |
| VoxCPM2 (`openbmb/VoxCPM2`) | Apache-2.0 |
| Qwen3-TTS 1.7B VoiceDesign | Apache-2.0 |
| OmniVoice (`k2-fsa/OmniVoice`) | pesos CC-BY-NC segundo o README: **uso não comercial**, por isso não é o padrão |
| Canary 1B v2 (STT) | modelo do Pi Voice, já instalado |

## Medições (PC, RTX 4070 SUPER, 12 GB, base de 2050 MiB já usados por outros programas)

Dez frases PT de ~12 palavras (`voice/bench/sentences.py`), uma emoção por frase. WER/UTMOS vêm do áudio gerado.
**TTFA** = tempo até o primeiro áudio, mediana (máx.). **VRAM** = pico acima da base, só o TTS.
WER = ida e volta pelo `faster-whisper large-v3-turbo` (mesmo juiz para todos). UTMOS = `utmos22_strong` (SpeechMOS), 1-5.

| TTS | TTFA p50 (máx.) | RTF | VRAM TTS | WER | UTMOS | Observação |
|---|---|---|---|---|---|---|
| **supertonic** (padrão) | 528 ms (606) | 0,10 | 0 (CPU/ONNX) | 2,5 % | **4,25** | melhor nota; emoção só muda o ritmo |
| kokoro (`pf_dora`) | **74 ms** (78) | 0,017 | 979 MiB | 2,5 % | 3,78 | mais rápido; emoção só muda o ritmo |
| omnivoice | 343 ms (378) | 0,08 | 2315 MiB | 2,7 % | 2,80 | pesos não comerciais |
| voxcpm | 118 ms (129) | 0,71 | 6133 MiB | 4,3 % | 1,29 | emoção por prompt; estoura 7 GB com o STT; saída bem baixa (pico 0,12-0,22) |
| qwen3 | 7084 ms (9015) | 1,49 | 4510 MiB | 0,0 % | 2,24 | não é streaming: inviável |

(O TTFA do Supertonic é a síntese inteira de uma frase na CPU; no `demo` com uma frase curta deu 390 ms. O limite era ≤ 600 ms.)

**STT** (`companion.voice.bench.stt_bench`, mesmas 10 frases, áudio com 0,3 s antes e 0,4 s depois): latência = tempo de `transcribe()`
depois do fim da fala.

| STT | p50 | máx. | WER | carga |
|---|---|---|---|---|
| **canary** (Q8_0, Vulkan, via Node) | **74 ms** | 921 ms (2 de 10 frases passaram de 500 ms) | 0,7 % | 8,6 s |
| faster-whisper large-v3-turbo fp16 | 128 ms | 132 ms | 2,5 % | 33 s |

O atraso total que o usuário sente do fim da fala até o texto = `end_silence_ms` (550 ms, configurável) + ~75 ms.

**VRAM total, STT + TTS carregados e usados** (`bench.vram_total`, pico acima da base; limite 7 GB = 7168 MiB):

| Combinação | Pico |
|---|---|
| supertonic + canary | **1363 MiB** |
| kokoro + canary | 2327 MiB |
| omnivoice + canary | 3590 MiB |
| voxcpm + canary | 7450 MiB (acima do limite); com faster-whisper, 8289 MiB |

Escolha: **Supertonic + Canary** (melhor naturalidade medida, 1,4 GB de VRAM, STT mais rápido e preciso).
Kokoro é a opção de menor latência (`tts.backend = "kokoro"`).

### Comandos

```
# Windows, venv do bench (C:\projetos\pi-avatar-voice-bench\venvs\<nome>), PYTHONPATH=<worktree>\companion
python -m companion.voice.bench.tts_bench --backend kokoro --out out\kokoro      # TTFA, RTF, VRAM, WAVs
python -m companion.voice.bench.score --dir out\kokoro                           # WER ida e volta + UTMOS
python -m companion.voice.bench.stt_bench --dir out\kokoro --stt canary          # latência do STT
python -m companion.voice.bench.vram_total --tts supertonic --stt canary         # VRAM total
python -m companion.voice.demo --say "Oi, tudo bem?" --emotion happy --out out.wav
python -m companion.voice.demo --transcribe out.wav [--stt faster-whisper]
COMPANION_VOICE_E2E=1 python -m pytest voice-tests -q                            # inclui TTS -> WAV -> listen() -> texto
python -m pytest voice-tests -q                                                  # sem modelos (qualquer máquina)
```

## Testes (`companion/voice-tests/`)

Sem modelos, sem microfone, sem som: `FakeTts` (tom por palavra), `VirtualSink` (placa de som simulada com relógio
próprio, conta o silêncio forçado entre frases), `ArrayMic`/`FileMic` e WAVs sintéticos. Cobrem: mapeamento de
visemas (vogais sintéticas → boca certa, 30 Hz, suavização), frases em pipeline sem buraco (síntese de N+1 durante N, `starved_s` ≈ 0),
ordem e continuidade na linha do tempo, `cancel()` < 100 ms, portão de eco (nível de eco bloqueado, usuário por cima passa,
calibração do vazamento, eco da própria voz não dispara barge-in, usuário falando por cima dispara e para o áudio),
segmentação do VAD (dois turnos, pausa curta, estalo, tosse, ruído, corte longo, latência de fim de turno), push-to-talk,
WAV → `listen()` → `Utterance`. Um teste opcional (`COMPANION_VOICE_E2E=1`) faz a ida e volta real com os modelos.

## Limites conhecidos

- **Não testado com microfone e caixa de som reais** (rodar só com WAV e placa de som simulada): `SoundDeviceMic`/`SoundDeviceSink`
  e a calibração do anti-eco em sala real precisam de um teste ao vivo com fone e com caixa. Se houver falso barge-in, subir
  `echo.margin` ou usar `echo.mode = "mute"`.
- Emoção: Supertonic e Kokoro só mudam o ritmo (não há controle de estilo no modelo); VoxCPM e OmniVoice recebem a emoção por texto
  (instrução) mas não passaram no orçamento/qualidade. Nenhum TTS foi avaliado por ouvido humano; as notas são UTMOS e WER.
- O UTMOS 1,29 do VoxCPM é suspeito (áudio de volume baixo); não investiguei, pois ele já estoura os 7 GB.
- Canary teve 2 frases de 10 com 0,5-0,9 s (o resto 70-80 ms); a causa não foi medida.
- O Chatterbox foi cortado (a instalação falhou no `pkuseg`).
