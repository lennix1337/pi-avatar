# Rune Desktop Pet (`rune-pet`)

Desktop pet nativo em Rust (sem WebView e sem Electron) renderizando o avatar 3D VRM 1.0 (`../models/avatar.vrm`) com o framework **Rune** (`anomalyco/rune`, rev `2312d60bf07a300a3142dace9d067b24f2454488`).

## Arquitetura e notas sobre o Rune

- **Janela nativa transparente e always-on-top**: o Rune oferece `WindowOptions { transparent: true, borderless: true, always_on_top: true, resizable: false, min_size: Some((0.0, 0.0)), ... }` com DirectComposition (`WS_EX_NOREDIRECTIONBITMAP`) no Windows, além de `App::open_inactive()` e `WindowHandle::set_position((x, y))`.
- **Ocultar da barra de tarefas no Windows**: o `WindowSpec` padrão do Rune não expõe `skip_taskbar`, então `src/win32.rs` aplica `WS_EX_TOOLWINDOW` (removendo `WS_EX_APPWINDOW`) usando o hook oficial `WindowHandle::with_raw_handle`.
- **Renderização 3D VRM 1.0**: o renderizador padrão do Rune (`uber-quad` SDF + Taffy + Parley) é 2D e não inclui cena 3D nem loader VRM. Em vez de criar um segundo dispositivo GPU ou fazer readback para CPU, o `rune-pet` implementa um `rune::compose::Compositor<PetAppState>` (`src/renderer.rs`) que desenha diretamente no `wgpu::Device`, `wgpu::Queue` e `TextureView` do próprio Rune, com:
  - Loader GLB/VRM 1.0 nativo (`src/vrm.rs`): ossos humanoides normalizados (fórmula exata do `@pixiv/three-vrm`), `VRMC_node_constraint` (`aim`, `roll`, `rotation` para mangas/ombros/coxas) e morph targets esparsos/densos para expressões (`happy`, `blink`, etc.).
  - Shader MToon (`src/vrm.wgsl`) com skinning linear na GPU, MSAA 4× em alvo transparente e texturas HD até `1024×1024` com mipmaps em streaming (buffers CPU liberados imediatamente após upload para VRAM).
  - Pacing a ~60 FPS durante movimento e 0 FPS em `idle` estabilizado (máquina de estados a 4 Hz com ~0% CPU).
- **Tray e interação**: bandeja do sistema nativa via `rune::Tray` (`Pausar / Retomar` e `Sair`), arraste pela tela (`El::on_drag`) e clique duplo (`El::on_double_press_tx`) alternando poses `cheer` (braços levantados) e `wave`.

## Como preparar e compilar

```powershell
cd rune-pet
pwsh scripts/setup-rune.ps1   # clone esparso pinado de anomalyco/rune em .rune-cache/
cargo test
cargo build --release
```

## Execução e capturas headless

```powershell
# Desktop pet ao vivo na taskbar (bandeja: Pausar / Sair)
.\target\release\rune-pet.exe
.\target\release\rune-pet.exe --state-file caminho\harness-state.json   # reações do harness (JSON local)

# Medição solo: imprime avg/min/p5 FPS e working set (RAM) após N segundos e sai
.\target\release\rune-pet.exe --force-walk --measure-secs 25 --stats-file stats.json

# Capturas PNG headless usando o tooling nativo do Rune (--snap + --scene)
.\target\release\rune-pet.exe --snap docs\rune-shot-idle.png --scene idle --scale 2
# cenas: idle, walk, arms-up, mood-worried, mood-celebrate, mood-working, mood-wave, mood-sleep, delight-pat
```

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
