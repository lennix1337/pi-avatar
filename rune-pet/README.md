# Pi Avatar

Desktop pet nativo em Rust (sem WebView e sem Electron) renderizando o avatar 3D VRM 1.0 (`../models/avatar.vrm`) com o framework **Rune** (`anomalyco/rune`, rev `2312d60bf07a300a3142dace9d067b24f2454488`).

## Arquitetura e notas sobre o Rune

- **Janela nativa transparente e always-on-top**: o Rune oferece `WindowOptions { transparent: true, borderless: true, always_on_top: true, resizable: false, min_size: Some((0.0, 0.0)), ... }` com DirectComposition (`WS_EX_NOREDIRECTIONBITMAP`) no Windows, além de `App::open_inactive()` e `WindowHandle::set_position((x, y))`.
- **Fora da barra de tarefas e do Alt+Tab no Windows**: o pet ganha `WS_EX_TOOLWINDOW` e perde `WS_EX_APPWINDOW` antes de aparecer. Esses dois bits são preservados quando o click-through muda.
- **Renderização 3D VRM 1.0**: o renderizador padrão do Rune (`uber-quad` SDF + Taffy + Parley) é 2D e não inclui cena 3D nem loader VRM. Em vez de criar um segundo dispositivo GPU ou fazer readback para CPU, o `rune-pet` implementa um `rune::compose::Compositor<PetAppState>` (`src/renderer.rs`) que desenha diretamente no `wgpu::Device`, `wgpu::Queue` e `TextureView` do próprio Rune, com:
  - Loader GLB/VRM 1.0 nativo (`src/vrm.rs`): ossos humanoides normalizados (fórmula exata do `@pixiv/three-vrm`), `VRMC_node_constraint` (`aim`, `roll`, `rotation` para mangas/ombros/coxas) e morph targets esparsos/densos para expressões (`happy`, `blink`, etc.).
  - Shader MToon (`src/vrm.wgsl`) com skinning linear na GPU, MSAA 4× em alvo transparente e texturas HD até `1024×1024` com mipmaps em streaming (buffers CPU liberados imediatamente após upload para VRAM).
  - Pacing a ~60 FPS durante movimento e 0 FPS em `idle` estabilizado (máquina de estados a 4 Hz com ~0% CPU).
- **Tray e interação**: bandeja do sistema nativa via `rune::Tray` (`Pausar / Retomar` e `Sair`), arraste pela tela (`El::on_drag`) e clique duplo (`El::on_double_press_tx`) alternando poses `cheer` (braços levantados) e `wave`.

## Como preparar e compilar

No Windows, use o Rust com toolchain MSVC e o Windows SDK (inclui `rc.exe` para embutir o ícone e os metadados do EXE).

```powershell
cd rune-pet
pwsh scripts/setup-rune.ps1   # clone esparso pinado de anomalyco/rune em .rune-cache/
cargo test
cargo build --release
```

### Testes a partir da VM Linux

A VM não tem `cargo`. `npm test` na raiz (e a receita `.pi/checks.json`) roda `scripts/pc-test.cmd` no PC por `pc-task`: faz fast-forward de `origin/main` lá e executa `cargo test --quiet` em `rune-pet`. A saída completa fica em `~/.pi/agent/pc-task-logs/pet-test-*.log`; confira a linha `test result:` no log, não só o código de saída do comando.

## Execução e capturas headless

```powershell
# Desktop pet ao vivo na taskbar (bandeja: Pausar / Sair)
.\target\release\pi-avatar.exe
.\target\release\pi-avatar.exe --state-file caminho\harness-state.json   # reações do harness (JSON local)

# Medição solo: imprime avg/min/p5 FPS e working set (RAM) após N segundos e sai
.\target\release\pi-avatar.exe --force-walk --measure-secs 25 --stats-file stats.json

# Capturas PNG headless usando o tooling nativo do Rune (--snap + --scene)
.\target\release\pi-avatar.exe --snap docs\rune-shot-idle.png --scene idle --scale 2
# cenas: idle, walk, landing, drowsy, arms-up, mood-worried, mood-celebrate, mood-working, mood-wave, mood-sleep, delight-pat,
#        talk-aa, talk-oh, emo-smug, emo-embarrassed, emo-excited, emo-confused, listening, thinking, subtitle-long
```

## Companion (conversa)

O pet é cliente TCP do companion (`companion/PROTOCOL.md`, linhas JSON). Por padrão conecta em
`127.0.0.1:47800` e tenta de novo a cada 2 s; sem companion funciona como antes.

```powershell
.\target\release\pi-avatar.exe --companion 127.0.0.1:47800   # padrão
.\target\release\pi-avatar.exe --no-companion
.\target\release\pi-avatar.exe --companion-replay scripts\talk-20s.jsonl   # toca um fluxo gravado (t em segundos)

python scripts\fake_companion.py serve          # companion falso, conversa roteirizada
python scripts\fake_companion.py emit > x.jsonl  # gera o fluxo de replay
pwsh scripts\measure-companion.ps1              # FPS/RAM: ocioso, sem companion, replay de 20 s e replay + cauda
```

Lábios (aa/ih/ou/ee/oh) com ataque/soltura e queda a zero em 150 ms sem `mouth`; emoções (inclusive
smug, teasing, embarrassed, bored, excited, confused) viram mistura de expressões VRM + postura sutil;
gestos são animações aditivas curtas; a fala sobrepõe os humores do harness e aparece como legenda
no balão (quebra de linha, até 3 linhas visíveis, some ~1,5 s após `end`). Durante a conversa o
pet fica parado; o loop só é acordado pelo companion quando a conversa está inativa, porque o Rune
renderiza a cada wake e um wake por mensagem `mouth` somaria quadros ao pacer de 60 FPS.

A versão release do Windows abre sem console e fica acessível pelo ícone **Pi Avatar**
na bandeja. Clique com o botão direito no ícone e escolha **Sair** para fechar.
Para gravar a saída de `--measure-secs`, use `--stats-file stats.json` ou, no cmd:

```cmd
target\release\pi-avatar.exe --measure-secs 25 > stats.json
```

No modo ocioso, o pet alterna olhares laterais e pequenos suspiros. Ao aproximar o cursor,
os olhos reagem primeiro e a cabeça acompanha; carícias lentas também contam em telas com DPI alto.
O piscar varia de ritmo, às vezes vem em dobro, e dá lugar aos olhos fechados no sono.
Ao mudar o humor do harness, o pet faz um gesto breve: abaixa a cabeça com o CI vermelho,
se anima com um PR mesclado ou acena com a cabeça quando está trabalhando ou precisa da sua atenção.

Em modo automático, a pose ociosa fica sonolenta após um período de inatividade no Windows, mais cedo à noite. O avatar consulta apenas o tempo agregado desde a última atividade e a hora local; não recebe eventos de teclado nem guarda ou envia esses sinais. A expressão cede ao olhar, ao carinho e aos humores do harness. As cenas headless e os modos forçados não consultam esses sinais.

## Métricas medidas (Windows, NVIDIA GeForce RTX 4070 SUPER, DX12, pet sozinho, 25 s)

| | Antes (v1) | Depois (v2) |
| --- | ---: | ---: |
| FPS médio (idle) | 27,7 | **60,33** |
| FPS médio (walk) | não medido | **60,33** |
| FPS médio (auto) | não medido | **60,33** |
| FPS mínimo / p5 (walk) | — | 54,13 / 58,17 |
| RAM idle (working set) | 113 MB | **129,1 MB** |
| RAM walk | 117 MB | **117,4 MB** |
| RAM auto | — | 109,3 MB |
| Binário release | 22,05 MB | 23,4 MB |

O pacer (`src/pacing.rs`) ancora o próximo frame no instante atual quando o timer dispara cedo,
evitando que ticks de 15,6 ms do Windows derrubem 1 frame a cada ~16 (teste de regressão
`jittered_15_6ms_windows_ticks_stay_above_55_fps`).

Capturas em `docs/rune-shot-*.png` (idle, walk, arms-up, cada humor do harness e o toque extra, `delight-pat`: faíscas ao fazer carinho na cabeça / duplo clique).
