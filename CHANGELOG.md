# Changelog

## 1.0.11 - 2026-11-17

### English interface and retained table data

- While waiting for the Hero to act, keep the last valid cards, board, pot, stack, player stats, and reasoning on screen. The action area shows `Waiting for your turn`.
- Translate the interface, agent recommendations, solver notes, and capture errors into English.
- Remove the capture information line above the Auto and Capture controls.

## 1.0.10 — 2026-11-17

### Estado de espera

- Quando a mesa está visível e a fila de ações confirma que não é a vez do Hero, a aplicação mostra `Waiting for your turn`, mesmo que o OCR tenha ambiguidades.
- Cartas, pote, stack, HUD e recomendações anteriores são limpos enquanto se aguarda a vez.
- Leituras com estado de turno desconhecido continuam a aparecer como falha, sem serem confundidas com espera.

## 1.0.9 — 2026-11-17

### Interface e estilos de jogo

- Inclui na versão os estilos já integrados no agente: Equilibrado, Passivo, Agressivo e Criativo.
- O título da janela passa a ser `Lugon Poker Lab v1.0.9`, sem “Assistente”.
- O processo Python recebe uma identidade própria no Windows; a janela usa o ícone de poker em `assets/lugon_poker_lab.ico` em vez do ícone genérico do Python.

## 1.0.8 — 2026-11-17

### Ajuste de leitura das cartas

- O prompt distingue explicitamente a Dama do 2 pela forma do símbolo. Quando o modelo lê 2 de paus no board, um recorte de centro confirma essa carta em vez de repetir a leitura em todos os frames.
- Nas 60 capturas da 1.0.6, a vez do Hero foi classificada corretamente em 60/60 frames; os 44 frames ativos ficaram prontos e os cinco campos de extração acertaram 220/220 (100%). Relatório: `data/captures/manual_benchmark_v1.0.8_on_v1.0.6.json`.

## 1.0.7 — 2026-11-17

### Corrigido após rever as novas capturas da 1.0.6

- Os recortes de correção do Hero, do board e do pote são chamados apenas quando as cartas extraídas entram em conflito; frames fora da vez continuam sem chamadas extra ao modelo.
- Cartas duplicadas são verificadas depois de confirmar os naipes pela cor. Um board secundário truncado ou malformado é tentado novamente.
- A leitura do Hero evita a área inferior das cartas, que pode ficar tapada pelos botões de ação. O rótulo do pote recebe um recorte curto próprio quando a leitura das cartas já exigiu revisão.
- Benchmark das novas capturas da 1.0.6 com a 1.0.7, após rever as cartas do Hero na imagem original: 218/220 campos (99,1%) nos 44 frames da vez; a correção de Q/2 e confirmação focada na 1.0.8 recuperam os dois erros.

Cada versão vive em `lugon_poker/__init__.py` (`__version__`), aparece no título da janela (`v<versão>`) e dá nome à
pasta onde as capturas de diagnóstico dessa versão são arquivadas: `data/captures/v<versão>/`. Cada `*_meta.json`
guarda também `app_version`, `game_type` e (desde 0.3.0) a leitura de pixéis do Mercury (`mercury_vision`).

**Ao lançar uma versão nova:**

1. Subir `__version__` e acrescentar a entrada aqui.
   Cada correção funcional recebe uma versão nova; validar a extração com as capturas arquivadas nessa mesma versão.
2. `py -m unittest discover -s tests`.
3. Reanalisar as capturas da versão anterior com o código novo, para provar que o que falhou ficou resolvido:
   `py -m lugon_poker.vision_cli replay data/captures/v<anterior> --vlm`
4. Jogar. As novas capturas ficam sozinhas em `data/captures/v<nova>/` — nunca misturadas com as anteriores.

## 1.0.6 — 2026-11-17

### Interface, velocidade e extração

- O botão **Capturar** fica ao lado do Auto e guarda o frame em segundo plano sem alterar o texto do botão nem mostrar confirmação.
- O Auto passa a esperar 2 segundos entre leituras. O subtítulo de sugestão e o tipo de jogo saem de cima da ação recomendada.
- O Companion passa a usar `qwen3-vl:2b-instruct`, com menor uso de memória que o modelo 4B.
- O recorte composto amplia as cartas do Hero e o centro da mesa. Se a leitura do board ou do rótulo do pote for inconsistente, um segundo recorte curto relê só o centro; naipes visíveis do 888 são confirmados pela cor da carta.
- Quando **PASSAR/CHECK** aparece junto de um botão de call contraditório, prevalece o check.
- O Auto arquiva automaticamente apenas estados em que a fila retangular de ações confirma a vez do Hero. Guarda a imagem efetivamente analisada, deduplica estados repetidos e mantém as leituras ambíguas dessa vez para correção.
- A stack ilegível deixa de ser substituída por 100 fichas. As respostas locais e externas são limitadas às ações legais; apostas/subidas que comprometem a stack aparecem como **ALL-IN · SHOVE**, e calls que usam a stack toda como **CALL · ALL-IN**.
- O motor heurístico usa o SPR para escolher shove quando a stack restante é igual ou inferior ao pote (SPR ≤ 1) e os limiares de equity para apostar/subir foram atingidos.
- Revisão visual e replay das 26 capturas históricas: as 6 fora da vez foram excluídas; nas 20 da vez, cartas, board, pote, call e stack ficaram corretos em **20/20 frames (100/100 campos)**. Os botões retangulares separaram também os 20 estados ativos dos 6 estados de espera, sem falsos positivos nem omissões. Relatório: `data/captures/manual_benchmark_v1.0.6.json`.
- Replay das duas capturas arquivadas da 1.0.5 com o 2B: **2/2 estados prontos**, com cartas, board, pote e valor a pagar reconhecidos.

## 1.0.5 — 2026-11-17

### Corrigido após rever as capturas da 1.0.2 e 1.0.4

- O recorte das cartas do Hero fica maior no painel composto enviado ao Qwen; o replay voltou a ler as cartas nas capturas que falhavam.
- **PASSAR** só dá `to_call=0` quando aparece como botão real. A cápsula **PASSAR/DESISTIR** com círculo é uma preferência de auto-check/fold e não indica a vez do Hero.
- O pote continua a vir apenas do rótulo central **Pote total:**, que já inclui as antes. O valor a pagar vem do botão **IGUALAR**, de **PASSAR** ou da stack restante num all-in curto; fichas e antes junto aos lugares nunca são usadas como pote ou call.
- O replay guarda e usa o preset/modo da captura e recorta a janela da mesa antes de a reanalisar, como no fluxo em direto.

### Simplificação

- A área de captura tem só o botão **Capturar**; o Auto continua a ler e o botão guarda um frame sem parar a leitura.

## 1.0.4 — 2026-11-17

### Corrigido após rever as capturas da 1.0.2

- O 888 falhava repetidamente ao ler as cartas do Hero. O prompt passa a usar o baralho de quatro cores do layout (♦ azul, ♣ verde, ♥ vermelho, ♠ preto) e a imagem enviada ao Qwen inclui um recorte ampliado das cartas e dos botões.
- O `pot` só pode vir do valor junto de **Pote total:**; esse total já inclui blinds, antes e apostas. Contribuições/antes visíveis junto aos lugares nunca são o pote nem o `to_call`.
- Se a mesa só mostrar **Desistir** e **All-in**, o valor do botão All-in é tratado como aposta total do Hero, não como o valor a pagar. Se não for possível calcular o call com segurança, a leitura é marcada como ambígua.
- Posição desconhecida deixa de ser convertida silenciosamente em BTN.
- O replay usa o preset e o modo guardados nos metadados e recorta a janela da mesa antes de a reanalisar, como no fluxo em direto.

## 1.0.3 — 2026-11-17

### Novo

- O botão **Capturar frame para correção** guarda o ecrã completo e os metadados na pasta desta versão mesmo com o Auto ligado; não chama o modelo nem interrompe a leitura contínua.

## 1.0.2 — 2026-11-17

### Corrigido após rever capturas reais da 1.0.1

- O prompt do 888 só aceita cartas visíveis no lugar do Hero; cartas de adversários expostas em all-in/showdown não podem preencher a mão do Hero.
- `hero_to_act` exige a fila real de botões de ação; etiquetas de ação sobre o avatar e ecrãs `WINNER` não contam.
- Quando o Hero está à espera, a interface limpa cartas, pote, stack e posição antigos para não mostrar dados da mão anterior.
- O Auto mantém o Qwen dedicado à leitura e usa a recomendação imediata do motor local, evitando chamadas simultâneas ao mesmo modelo.
- O timeout visual do Companion passa a 15 segundos. Ligar o Auto não força nem arquiva o primeiro frame; só **Analisar agora** e **F8** guardam capturas de diagnóstico.

## 1.0.1 — 2026-11-17

### Alterado

- A janela Companion usa apenas `qwen3-vl:4b-instruct` para leitura visual e explicação estratégica, evitando trocar entre modelos locais.
- Quando o Solver ICM é ativado na entrada manual, a solução calculada é enviada ao mesmo modelo para explicação; a ação e o montante continuam definidos pelo solver.
- As opções de jogo, layout, atalho F8 e gravação ficam recolhidas por defeito para libertar espaço à mesa e ao assistente.
- As capturas novas ficam separadas em `data/captures/v1.0.1/`, sem serem misturadas com os arquivos anteriores.
- A explicação estratégica tem limite de 20 segundos; se o modelo não responder, a recomendação local ou do solver continua visível.

## 0.3.0 — 2026-11-16

### Corrigido (falhas observadas nas capturas da 0.2.0)

- **Banner do prémio lido como board.** O VLM lia `€5 ♠♠♠♠` (topo da mesa Spin & Go) como `5s 5s 5s 5s 5s`, o que
  gerava "cartas duplicadas" e `LEITURA FALHADA` em quase todas as mãos pré-flop. Agora (a) o recorte enviado ao VLM
  começa abaixo do banner, (b) o prompt manda ignorá-lo, e (c) o número de cartas no board vem da contagem de pixéis
  da mesa, nunca do VLM.
- **Rank 9 nunca lido pelo leitor determinístico.** O template do 9 pontuava 0.92 contra 0.90 do 8 (margem 0.03 < 0.04)
  e a carta era rejeitada, caindo no VLM, que falhava o naipe. Acrescentada a variante medida em capturas reais.
- Cartas legíveis por pixéis (`hero_complete`/`board_complete`) passam a prevalecer sobre o VLM, que trocava naipes
  (ex.: `4♠`→`4c`, `A♣`→`Ah`, `K♦ 6♦`→`Kd 6c`, `Q♠`→`Qd`).

### Novo

- **Seletor "Jogo": Spin & Go / Torneio (MTT)** no cabeçalho do Assistente (guardado em `data/settings.json`).
  Afeta o advisor (limiares de aposta/raise e margem de ICM em calls grandes no torneio), o prompt do LLM de
  estratégia, o prompt de visão (3-max com oponentes no topo vs. todos os lugares) e a deteção de lugares do Mercury.
- **Versionamento de capturas**: `data/captures/v<versão>/`, `app_version` no meta, versão visível na janela.
- **`vision_cli replay <pasta>`**: reanalisa capturas arquivadas com o código atual e compara com a leitura original.
- O meta das capturas inclui agora `game_type`, `companion_rect` (para o replay mascarar a mesma área) e
  `mercury_vision` com score/margem por carta, para diagnosticar leituras falhadas sem adivinhar.

### Resultado medido

Replay das últimas 25 capturas da 0.2.0 com Ollama (`qwen3-vl:4b-instruct`): `failed` 17 → 0.

## 0.2.0 — 2026-11-15/16 (retroativo)

- Primeira versão com captura de ecrã em direto, preset PokerStars Mercury 4 cores e Vision-LLM local.
- Capturas desta versão arquivadas em `data/captures/v0.2.0/` (84 capturas). Não tinham `app_version`; a pasta é a
  referência. O código mudou várias vezes durante o dia sem subir a versão — foi isso que motivou o versionamento.
