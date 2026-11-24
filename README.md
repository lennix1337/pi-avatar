# pi-avatar

Pet de mesa nativo em Rust: um avatar 3D VRM que anda pela barra de tarefas do Windows, reage ao estado do seu harness de agentes (CI vermelho, PR mesclado, workers ativos, sua vez) e não usa WebView nem Electron.

- `rune-pet/`: o pet, feito com o framework [Rune](https://github.com/anomalyco/rune) e um compositor 3D próprio (VRM 1.0, MToon, skinning na GPU). Build, flags e métricas no [README do pet](rune-pet/README.md).
- `models/avatar.vrm`: modelo de exemplo (licença MIT da pixiv, em `models/avatar.vrm.LICENSE.txt`).
- `poses.json`: poses e expressões usadas pelo pet.

## Início rápido (Windows)

```powershell
cd rune-pet
pwsh scripts/setup-rune.ps1
cargo build --release
.\target\release\rune-pet.exe
```

Para ligar o humor do pet a um harness, passe `--state-file caminho\estado.json` com os campos `ciRed`, `merged`, `workers`, `waitingOperator` e `idleMinutes`.

## Medido (RTX 4070 SUPER, 25 s, pet sozinho)

60 FPS médios andando e parado (mínimo 54, p5 58) e 109 a 129 MB de RAM.

![andando](rune-pet/docs/rune-shot-walk.png) ![PR mesclado](rune-pet/docs/rune-shot-mood-celebrate.png)
