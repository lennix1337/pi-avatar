# Protocolo companion ↔ pet

O `companion/` (processo Python: cérebro, voz e servidor) fala com o pet (`rune-pet`, Rust) por TCP em `127.0.0.1:47800` (configurável), uma mensagem JSON por linha (UTF-8, `\n`). O companion é o servidor; o pet conecta ao iniciar e tenta de novo a cada 2 s quando a conexão cai. Sem companion, o pet funciona como hoje.

Campos desconhecidos são ignorados pelos dois lados; mensagem inválida é descartada sem derrubar a conexão.

## companion → pet

| `t` | Campos | Efeito no pet |
| --- | --- | --- |
| `state` | `listening`, `thinking`, `speaking` (bool) | indicador discreto: ouvindo (olhar atento), pensando (olhar para cima/lado), falando |
| `say` | `id`, `text`, `emotion` | começa a legenda (balão) com `text`; troca a expressão para `emotion` |
| `mouth` | `aa`, `ih`, `ou`, `ee`, `oh` (0..1) | pesos de visema do quadro atual, ~30 Hz enquanto o áudio toca; zero tudo se nada chegar por 150 ms |
| `emotion` | `name`, `intensity` (0..1) | expressão VRM: `happy`, `angry`, `sad`, `relaxed`, `surprised`, `neutral`; nomes extras (`smug`, `teasing`, `embarrassed`, `bored`, `excited`, `confused`) são mapeados pelo pet para combinações das expressões VRM e da pose |
| `gesture` | `name` | gesto curto: `wave`, `cheer`, `nod`, `shake`, `tilt`, `laugh`, `shrug`, `lean_in`, `look_away` |
| `end` | `id` | fim da fala: a legenda some depois de ~1,5 s, boca fecha |

## pet → companion

| `t` | Campos | Significado |
| --- | --- | --- |
| `hello` | `version` | pet conectado |
| `poke` | `kind`: `pat`, `double`, `drag`, `drop` | o usuário mexeu no avatar; o cérebro pode reagir |

## Exemplo

```
{"t":"state","listening":false,"thinking":true,"speaking":false}
{"t":"say","id":"u7","text":"Você de novo? Achei que tinha ido dormir.","emotion":"smug"}
{"t":"gesture","name":"tilt"}
{"t":"mouth","aa":0.62,"ih":0.0,"ou":0.08,"ee":0.0,"oh":0.12}
{"t":"end","id":"u7"}
```
