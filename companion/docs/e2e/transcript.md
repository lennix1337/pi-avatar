# Conversa de ponta a ponta (voz real, Pi real, pet por TCP)

Modelo: `anthropic/claude-sonnet-5-5`. Entrada: Supertonic M1 → microfone injetado em tempo real → VAD Silero → Canary. Saída: Supertonic F1 → placa virtual (relógio real, sem alto-falante). PC com RTX 4070 SUPER.

## Turno 1 (normal)
- Ele disse: Oi, tudo bem? Acordei agora e já quero conversar com você.
- Ouvido (STT): Oi, tudo bem? Acordei agora e já quero conversar com você.
- Ela: Acordou agora? São quase três da manhã, criatura. Isso não é acordar, é uma falha no sistema. Perdeu o sono ou teve sonho esquisito?

## Turno 2 (normal)
- Ele disse: Hoje eu passei o dia inteiro em reunião, tô exausto.
- Ouvido (STT): Hoje eu passei o dia inteiro em reunião, estou exausto.
- Ela: Que crueldade. Dia inteiro de reunião e ainda acorda de madrugada, o corpo nem descansa de graça. Pelo menos me diz que teve café e alguém falou "vamos alinhar" umas vinte vezes.

## Turno 3 (normal)
- Ele disse: Você acha que eu devo jantar pizza ou lámen?
- Ouvido (STT): Você acha que eu devo jantar pizza ou lâmina?
- Ela: Lâmina? Peraí, isso foi o reconhecimento de voz ou você tá planejando algo sinistro? Enfim, pizza, sem discussão. Lâmina não tem queijo.

## Turno 4 (normal)
- Ele disse: Me conta uma curiosidade aleatória sobre o espaço.
- Ouvido (STT): Me conta uma curiosidade aleatória sobre o espaço.
- Enrolação: ó...
- Ela: Em Vênus o dia é mais longo que o ano. Ele demora mais pra dar uma volta em si mesmo do que pra dar a volta no Sol. Ou seja, lá você nunca chega no fim da segunda-feira.

## Turno 5 (normal)
- Ele disse: Valeu, você é bem divertida. Agora preciso dormir, boa noite!
- Ouvido (STT): Valeu, você é bem divertida. Agora preciso dormir. Boa noite.
- Ela: Valeu você, vai lá. E se sonhar com reunião, eu não tenho nada a ver com isso. Boa noite, criatura.

## Turno 6 (story-interrupted)
- Ele disse: Me conta uma história bem comprida sobre um dragão que tinha medo de fogo.
- Ouvido (STT): Me conta uma história bem concomprida sobre um dragão que tinha medo de fogo.
- Ela: Olha, ele disse que ia dormir e já tá pedindo história. Tá, mas é uma só e bem baixinho.

## Turno 7 (interruption)
- Ele disse: Espera, para um segundo, deixa eu te falar uma coisa.
- Ouvido (STT): Espera, para um segundo, deixa eu te falar uma coisa.
- Ela: Hm, "preciso dormir" durou quanto, uns dois minutos? Pode falar, tô ouvindo. O dragão tá esperando de braços cruzados.

## Interrupção (barge-in)
- Ele começou a falar → fala confirmada pelo VAD: 0.344 s
- Ele começou a falar → áudio dela cortado: 0.344 s
