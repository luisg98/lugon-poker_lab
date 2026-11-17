# Lugon Poker Lab — PoC de uma mesa

PoC local para análise posterior e estudo de uma mão de No-Limit Texas Hold'em. O Milestone 1 importa um estado em JSON, valida-o, avalia mãos, estima equity e calcula pot odds/EV de call. O Milestone 2 inclui um baseline visual sobre capturas sintéticas de layout calibrado. O Milestone 3 adiciona reconstrução e validação de históricos de mãos, deteção de transições entre capturas e perfis estatísticos persistentes por nickname com intervalos de incerteza explícitos. O Milestone 4 integra uma mesa virtual interativa com um fluxo de assistência estratégica desacoplado que suporta um advisor heurístico local ou LLMs locais/cloud (Ollama, OpenAI-compatible, Gemini) alimentados exclusivamente pelos factos validados pelo motor.

## Documentação e Estrutura
- [Mapa de Diretórios e Componentes](docs/PROJECT_MAP.md) — Organização das pastas, convenções e responsabilidades de cada ficheiro.
- [Arquitetura do Sistema](docs/ARCHITECTURE.md) — Princípios de design, fluxo de dados e desacoplamento do motor.
- [Guias de Fluxo de Trabalho (Workflows)](docs/WORKFLOWS.md) — Exemplos práticos para Companion GUI, simulador headless e solvers.
- [Roadmap Estratégico](ROADMAP.md) e [Registo de Alterações](CHANGELOG.md).

## Investigação técnica e decisões

| Área | Opções consideradas | Decisão para a PoC | Razão |
|---|---|---|---|
| Avaliação de mãos | PokerKit, Treys/avaliador C++, evaluator próprio | Evaluator Python pequeno, coberto por testes | Sem instalação, dependências ou downloads; ranking de 5–7 cartas é explícito e auditável. PokerKit é a alternativa recomendada para um motor de regras mais amplo; Treys/OMPEval são opções quando throughput dominar. |
| Equity | Enumeração exata, Monte Carlo | Monte Carlo com seed fixa; enumeração exata no river heads-up | Resultados reproduzíveis e custo controlável na maioria das streets; no river heads-up todas as combinações legais são enumeradas. Ranges são uniformes por combinação física. |
| Visão | ROIs/templates; detector treinado; OCR | Baseline atual: ROIs normalizadas e reconhecimento bitmap local para um layout sintético 16:9. Próxima evolução: calibrar com capturas autorizadas e comparar detector de cartas com OCR de texto/fichas. | O baseline é imediato, reprodutível e sem modelos. Templates/glyphs são frágeis fora do layout conhecido. Detector tolera variação, mas exige dataset rotulado, treino e medição; OCR geral serve melhor nicknames/valores do que símbolos de cartas. |
| Solver/GTO | regras e odds; CFR/MCCFR e abstrações; solver local comercial | Não incluir solver no M1; usar CFR/MCCFR num simulador limitado no futuro; tratar solver comercial apenas como possível referência de validação | CFR/MCCFR em No-Limit exige abstração de cartas/apostas e uma árvore de ações definida. OpenSpiel é útil para protótipos e medir exploitability em jogos menores. PioSOLVER documenta execução dependente de CPU/RAM; a variante Edge pré-flop pede pelo menos 64 GB RAM, acima dos 32 GB disponíveis. Equity não é uma estratégia GTO. |
| Opponent modeling | frequências diretas; Bayes; modelos supervisionados | Preparar dados explícitos agora; milestone próprio com priors e intervalos de incerteza | Frequências simples têm alta variância com pouca amostra. Atualização Bayesiana permite representar incerteza, mas exige histórico de ações com contexto e denominadores de oportunidade. |
| ML, RL e LLM | classificador/detector; self-play/RL; LLM para explicação | ML apenas onde há imagens/dados suficientes; RL fica para simulador próprio; LLM não calcula cartas, odds nem EV | DeepStack combina procura/solução recursiva com valores aproximados aprendidos em self-play. LLMs podem gerar explicações, mas não substituem as rotinas matemáticas verificáveis. |

### Latência, precisão e execução local

- O Milestone 1 não precisa da RTX: o avaliador e as simulações correm em CPU e usam apenas a biblioteca padrão do Python. O número de amostras limita diretamente o tempo e a variância da equity Monte Carlo.
- Para visão, pré-processamento/OpenCV e ROIs podem começar em CPU. Só faz sentido escolher YOLO/PaddleOCR e usar a RTX depois de medir um conjunto autorizado, incluindo resolução, taxa de erro por campo e latência na máquina alvo.
- Uma imagem ambígua deve resultar em campo desconhecido para revisão manual, não numa carta ou stack inferidos sem evidência.

### Arquitetura mínima

```text
screen_capture.py (Ecrã/Região/Janela) ──> live_vision.py (Calibrado | Vision-LLM) ──┐
simulator.py (Mesa Virtual) ──> synthetic.py ──> vision.py ────────────────────────────┼──> state.py ──> cards.py / equity.py / strategy.py
JSON manual ───────────────────────────────────────────────────────────────────────────┘
Históricos JSON / diffs ──> history.py ──> profiles.py <──> persistence.py (SQLite)          │
                                                └──────────────────────────────┬─────────────┘
                                                                               ▼
                                                 assistant.py (Heuristic | Ollama | OpenAI-compatible | Gemini)
                                                 cli.py / vision_cli.py / profiles_cli.py / sim_cli.py / companion_gui.py
```

O adaptador visual produz o mesmo `HandState`; não está acoplado à matemática. `ranges.py` expande classes de mãos para combinações físicas. `history.py` valida sequências de ações e deteta transições entre capturas; `profiles.py` calcula estatísticas por oportunidade e intervalos de incerteza; `persistence.py` guarda mãos deduplicadas por `hand_id` em SQLite local. `simulator.py` e `assistant.py` orquestram mãos completas e entregam o contexto validado (`DecisionContext`) a um assistente heurístico ou a um LLM local/cloud, validando que qualquer ação recomendada respeita os limites legais da mesa.

### Alternativas com fontes primárias

- [PokerKit](https://github.com/uoftcprg/pokerkit) fornece simulação, avaliação e análise estatística em Python. A documentação inclui benchmarks e observa que soluções C++ como OMPEval podem ser mais rápidas: [avaliação e benchmarks](https://pokerkit.readthedocs.io/en/stable/evaluation.html).
- [DeepStack](https://arxiv.org/abs/1701.01724) descreve decomposição, raciocínio recursivo e redes treinadas em self-play para HUNL; [CFR/MCCFR em abstrações](https://cdn.aaai.org/ocs/ws/ws0081/10137-45954-1-PB.pdf) mostra por que um solver NLHE não é uma extensão pequena de equity.
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) disponibiliza jogos e algoritmos como CFR e exploitability para investigação; [PioSOLVER](https://site.piosolver.com/docs/technical_details/) é uma referência local proprietária, com dimensionamento documentado por CPU/RAM. Nenhum solver foi instalado nem usado para gerar estes resultados.
- [OpenCV template matching](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html), [Ultralytics predict](https://docs.ultralytics.com/modes/predict) e [PaddleOCR](https://www.paddleocr.ai/latest/en/quick_start.html) representam as opções para templates, detecção e OCR. PaddleOCR tem configuração de runtime/modelos e requisitos de CUDA que devem ser fechados antes de instalar.
- [Bayes’ Bluff](https://webdocs.cs.ualberta.ca/~mbowling/papers/05uai.pdf) modela separadamente incerteza do jogo e da estratégia adversária; é uma base mais adequada para perfis com amostras pequenas do que rótulos categóricos imediatos. A recomendação é estimar VPIP/PFR/3-bet por oportunidades observadas e guardar amostra e incerteza, não atribuir tipos fixos cedo.
- Estudos recentes de poker com LLM, como [PokerBench](https://arxiv.org/abs/2501.08328), usam avaliação controlada e encontram limites em supervised fine-tuning simples; isso reforça o papel de um motor de regras/solver externo para os números e de um LLM, se usado, apenas para explicar.

## Milestone 1 — critérios de aceitação

1. Importar um estado JSON e rejeitar cartas repetidas, board incompleto, valores inválidos e ranges sem combinações legais.
2. Classificar corretamente mãos de 5–7 cartas, incluindo wheel, flush, full house e kickers.
3. Enumerar exatamente a equity heads-up no river; nas restantes situações produzir equity Monte Carlo com seed, amostras e frequências win/tie/loss visíveis.
4. Expandir ranges comuns (`AA`, `AKs`, `AKo`, `QQ+`, `AJs+`) e respeitar blockers conhecidos.
5. Calcular pot odds e EV incremental de call com premissas apresentadas; testar casos conhecidos e repetibilidade.
6. Executar a suíte de testes usando apenas `unittest`, sem baixar pacotes ou modelos.

## Executar

Requer Python 3.10 ou superior. Não há dependências externas.

```powershell
py -m unittest discover -s tests -v
py -m lugon_poker.cli examples/flop_call.json --iterations 20000 --seed 7
py -m lugon_poker.cli examples/flop_call.json --iterations 20000 --seed 7 --json
py -m lugon_poker.vision_cli scan examples/vision/flop_960.png --json
py -m lugon_poker.vision_cli scan examples/vision/flop_960.png --analyze --iterations 10000 --seed 7
py -m lugon_poker.vision_cli evaluate --scene examples/vision/flop_table.json
py -m lugon_poker.profiles_cli show --history examples/histories/sample_hands.json
py -m lugon_poker.profiles_cli show --history examples/histories/sample_hands.json --player Vitor --json
py -m lugon_poker.sim_cli play --iterations 500 --seed 7 --show-prompt
py -m lugon_poker.sim_cli evaluate --hands 5 --iterations 500 --seed 7
```

Edite `examples/flop_call.json` para introduzir outra mão. Cartas usam `As`, `Td`, `2c`; naipes `c` paus, `d` ouros, `h` copas, `s` espadas. Ranges aceitam combinações separadas por vírgulas, como `KK,AKs,AKo`; `*` ou range omitido significa todas as combinações. As combinações são uniformemente ponderadas, sem pesos.

## Milestone 2 — baseline visual sintético

`vision.py` lê PNGs RGB/RGBA, grayscale ou palette de 8 bits, procura a assinatura de uma mesa calibrada, lê cartas, board, pot, blinds, nomes, posições, stacks, apostas e marcadores de herói/jogador ativo. A saída `hand_state` é validada pelo modelo do Milestone 1. Campos ilegíveis ficam vazios e são descritos em `ambiguities`; uma carta ausente no meio do board impede a criação do estado.

O gerador local cria capturas de referência em [960×540](examples/vision/flop_960.png) e [1280×720](examples/vision/flop_1280.png). O fixture obteve 8/8 campos exatos nas resoluções 640×360, 800×450, 960×540, 1280×720 e 1600×900. Este número mede apenas a imagem que o próprio gerador produz, não a precisão em capturas reais. O pipeline está limitado a proporção 16:9, seis posições fixas, tema verde e texto bitmap ASCII. A range dos adversários permanece desconhecida; se `--analyze` for usado, o motor M1 assume combinações uniformes e reporta isso como pressuposto.

Para renderizar outra resolução 16:9 a partir do mesmo fixture:

```powershell
py -m lugon_poker.vision_cli render --scene examples/vision/flop_table.json --output examples/vision/custom.png --width 1600 --height 900
py -m lugon_poker.vision_cli scan examples/vision/custom.png --json
```

Este baseline valida a separação visão→estado→matemática. A próxima etapa de visão é reunir capturas autorizadas de cada layout/resolução-alvo, anotar campos e medir erros por carta, valor, nickname e marcador antes de escolher OpenCV, detector ou OCR de produção.

## Milestone 3 — Player Intelligence e histórico de mãos

- `history.py` valida históricos completos de mãos (`HandHistory`): ordem de streets (`preflop` → `flop` → `turn` → `river`), consistência do board, stacks, apostas e regras de ação (`check` proibido perante aposta, impossibilidade de agir após `fold`, etc.). Inclui também `diff_table_captures` para detetar alterações entre duas capturas da mesa sem inventar ações intermédias não observadas.
- `profiles.py` agrega estatísticas por oportunidade real:
  - **VPIP** e **PFR** pré-flop (excluindo walks na BB onde não houve oportunidade de ação voluntária).
  - **3-Bet** pré-flop medido apenas quando o jogador enfrentou exatamente um open-raise antes de agir.
  - **Agressividade**: `AFq = (bets + raises) / (bets + raises + calls + folds)` (excluindo `check`) e `AF = (bets + raises) / calls` (`null` quando `calls = 0`).
  - **Tendências por posição** (`BTN`, `SB`, `BB`, `UTG`, `MP`, `CO`, etc.) e **padrões de dimensionamento de apostas** em função do pote anterior (`amount / pot_before`).
  - **Incerteza estatística**: separa a frequência empírica `k/n` e o respetivo intervalo de confiança de Wilson a 95% da estimativa Bayesiana `Beta(1, 1)` (média e desvio-padrão posteriores explicitamente assinalados como dependentes do prior).
  - **Proteção contra amostras pequenas**: classifica a adequação da amostra (`no_observations`, `insufficient_sample`, `low_sample`, `moderate_sample`, `adequate_sample`) e recusa atribuir arquétipos definitivos com poucas observações.
- `persistence.py` guarda mãos em SQLite (`sqlite3` da biblioteca padrão) com deduplicação idempotente por `hand_id`.

Exemplos de utilização com base persistente:

```powershell
py -m lugon_poker.profiles_cli import examples/histories/sample_hands.json --db data/profiles.sqlite
py -m lugon_poker.profiles_cli show --db data/profiles.sqlite --player Vitor
py -m lugon_poker.profiles_cli diff examples/vision/flop_960.png examples/vision/flop_960.png --json
```

## Milestone 4 — Simulador interativo e fluxo de assistência (Baseline / LLM local ou cloud)

- `simulator.py` implementa uma mesa virtual de No-Limit Texas Hold'em (2 a 6 lugares) que simula mãos completas (`preflop` → `flop` → `turn` → `river` → `showdown`), conserva stacks/potes e exporta cada mão terminada como `HandHistory` validada para `ProfileStore`.
- Em cada decisão do herói, o pipeline executa e cronometra quatro etapas:
  1. **Visão**: renderiza a mesa atual (`synthetic.py`) e lê os píxeis via `vision.py`, verificando a correspondência com o estado real da mesa (`vision_verified`).
  2. **Motor matemático**: constrói o `HandState`, calcula a equity (`equity.py`), as pot odds e o EV de call (`strategy.py`), bem como stack efetivo e SPR (`effective_stack / pot`).
  3. **Inteligência de jogadores**: consulta os perfis acumulados em `ProfileStore` (VPIP, PFR, 3-Bet, AFq e intervalos de Wilson a 95%).
  4. **Fluxo de assistência estratégica (`assistant.py`)**: empacota todos os valores calculados num `DecisionContext` e consulta o fornecedor configurado:
     - `heuristic` (default offline): advisor determinístico local que avalia `check`, `call`, `fold`, `bet` e `raise` com base em equity, pot odds e SPR.
     - `ollama`: envia o prompt estruturado para um LLM local via HTTP (`http://127.0.0.1:11434/api/chat`, ex.: `--provider ollama --model llama3.1`).
     - `openai_compatible`: liga a servidores locais (LM Studio, vLLM, llama.cpp) ou cloud compatíveis com `/v1/chat/completions` (`OPENAI_BASE_URL` / `--endpoint` e `OPENAI_API_KEY`).
     - `gemini`: liga à REST API Gemini (`GEMINI_API_KEY`).
- O LLM recebe os números já calculados pelo motor e está proibido pelo prompt de fabricar probabilidades; a resposta JSON é validada por `validate_and_clamp_action` para garantir que a ação e o montante pertencem a `legal_actions`. Se o endpoint LLM estiver indisponível, o sistema ativa automaticamente o *fallback* heurístico local sem interromper a simulação.
- A entrada manual inclui o painel recolhido **Solver ICM** para uma decisão 3-max: ativa **Usar nesta análise**, introduz as stacks iniciais BTN/SB/BB em BB, escolhe um dos seis nós push/fold e analisa uma mão pré-flop. A app executa `solve_three_max_push_fold_stratified` em segundo plano (250 deals por classe, até 100 iterações), com ICM 50/30/20, blinds 0,5/1 BB e sem ante; passa frequências, nó, classe de mão e diagnósticos ao LLM em `three_max_solver`. O solver fixa a ação e o tamanho legal; o LLM explica a estratégia. Se não houver LLM local ativo, a recomendação numérica do solver continua disponível. No modo solver, Pote/A pagar/Stack também têm de estar em BB; o campo Stack representa a stack atual do Hero. O primeiro cálculo para uma configuração demora cerca de 15–20 segundos nesta máquina; configurações repetidas usam cache.
- Esta ligação é uma versão experimental de push/fold pré-flop. Não cobre limps, pós-flop, antes, outros payouts ou estados de ação fora dos seis nós disponíveis. A estratégia é aproximada por classes de mão e o diagnóstico indica se convergiu na amostra.
- `companion_gui.py` abre o Assistente com captura em direto (`screen_capture.py` + `live_vision.py`) e entrada manual. Deteta automaticamente a janela 888 BLAST, encaixa a mesa à esquerda e o Assistente numa coluna mais estreita à direita; envia só a janela da mesa para análise e repõe o tamanho anterior ao fechar. Inicia por `iniciar_companion.bat` ou `py -m lugon_poker.companion_gui`; `LugonPokerLab.exe` é o lançador. A versão 1.0.6 usa um único modelo local, `qwen3-vl:2b-instruct`. Em **Auto**, o Qwen lê a mesa e o motor local dá a recomendação sem uma segunda chamada ao modelo; frames só são arquivados quando os botões confirmam a vez do Hero. O botão **Capturar** guarda manualmente o ecrã atual mesmo com Auto ligado, sem chamar o modelo nem interromper a leitura. Respostas externas são limitadas às ações legais e as apostas que usam a stack toda aparecem como **ALL-IN · SHOVE**. O Qwen lê o pote apenas do rótulo central **Pote total:**, que já inclui as antes; fichas/antes junto aos lugares não são usadas como pote nem como valor a pagar. Com **Auto desligado**, F8 continua disponível nas opções para analisar um frame; se o Solver ICM for ativado na entrada manual, recebe a solução e explica-a, mantendo a ação e o valor definidos pelo solver. As opções de jogo, layout, F8 e gravação estão recolhidas por defeito. As capturas da versão atual ficam em `data/captures/v1.0.6/`.
- O Assistente não importa mãos de demonstração para o perfil do utilizador. Para alimentar o HUD, importa um histórico explicitamente com `py -m lugon_poker.profiles_cli import caminho/do/historico.json --db data/profiles.sqlite`.
- Para simular mãos completas da Mesa Virtual (headless) e comparar baselines/latência, utiliza-se o `sim_cli.py` (`simulator.py`).

Exemplos de execução do simulador, captura em direto e assistente:

```powershell
# Abrir a versão atual do Assistente
.\iniciar_companion.bat
py -m lugon_poker.companion_gui

# Listar janelas visíveis ou capturar ecrã/região/janela em direto pelo terminal
py -m lugon_poker.vision_cli windows
py -m lugon_poker.vision_cli capture --mode hybrid --analyze
py -m lugon_poker.vision_cli capture --region 100,100,960,540 --mode hybrid --json

# Ver o estado da ponte em direto (Antigravity <-> Companion App) e modelos locais detetados
py -m lugon_poker.bridge_cli status

# Enviar uma resposta estratégica diretamente do agente (Antigravity) para a janela Companion aberta
py -m lugon_poker.bridge_cli reply --action raise --amount 14 --reasoning "Overpair forte num flop seco; extrair valor de top pair."

# Jogar/simular uma mão e inspecionar o prompt estruturado enviado ao LLM
py -m lugon_poker.sim_cli play --iterations 500 --seed 7 --show-prompt

# Modo interativo no terminal (recebe recomendação e permite escolher a ação)
py -m lugon_poker.sim_cli play --interactive --iterations 1000 --seed 7

# Usar um LLM local via Ollama (ex.: gpt-oss:20b, com fallback seguro se o servidor não estiver ativo)
py -m lugon_poker.sim_cli play --provider ollama --model gpt-oss:20b --iterations 1000

# Usar um endpoint OpenAI-compatible local (ex.: LM Studio / vLLM) ou cloud
py -m lugon_poker.sim_cli play --provider openai_compatible --endpoint http://127.0.0.1:1234/v1 --model gpt-oss-20b

# Avaliar sessão de várias mãos, comparar com baselines e medir latência por componente
py -m lugon_poker.sim_cli evaluate --hands 5 --iterations 500 --seed 7 --db data/sim_profiles.sqlite
```

## Formato de jogo (Spin & Go / Torneio)

O seletor **Jogo** no cabeçalho do Assistente escolhe o formato; a escolha fica guardada em `data/settings.json`.

| | Spin & Go | Torneio (MTT) |
|---|---|---|
| Advisor local | limiares mais agressivos (aposta com ~56% de equity, raise com ~66%) | mais conservador (60% / 70%) e margem extra de ICM (+2% a +3,5% sobre as pot odds) quando o call compromete ≥35% do stack (ou ≥20% com ≤20 BB) |
| Prompt do LLM de estratégia | 3-max hyper-turbo, winner-take-all | pressão de ICM e sobrevivência |
| Prompt de visão | oponentes nos lugares do topo (esq./dir.) | oponentes em qualquer lugar da mesa |
| Mercury (pixéis) | deteta lugares/dealer 3-max e a posição do Hero | só lê cartas e botões; lugares e posição vêm do VLM |

Na linha de comandos: `vision_cli scan|capture --game-type tournament`. A leitura de cartas por pixéis é igual nos dois formatos.

## Versionamento e capturas

Cada versão (`lugon_poker/__init__.py`, ver [CHANGELOG.md](CHANGELOG.md)) aparece na janela e arquiva as capturas de
diagnóstico em `data/captures/v<versão>/`. A versão 1.0.6 começa com uma pasta própria; as novas capturas guardam
`app_version` e a posição da janela do Assistente para que possam ser reanalisadas sem misturar dados antigos:

```powershell
py -m lugon_poker.vision_cli replay data/captures/v1.0.6 --vlm
```

## Cálculos e limites

- A avaliação da mão usa ranking lexicográfico padrão, com 5 cartas escolhidas entre até 7.
- Equity é a fração esperada do pote atribuída à mão: vitória vale 1, empate divide a parte por igual. O Monte Carlo rejeita distribuições que não respeitem ranges/blockers. O seed e número de amostras são apresentados; erro de amostragem não é um intervalo de confiança.
- No river contra um oponente, a equity é exata para as combinações permitidas pela range. Nos restantes casos é estimativa Monte Carlo.
- O EV de call usa `equity × (pot + call) − call`; o pot inserido deve excluir a contribuição ainda por pagar pelo herói. É uma comparação incremental contra fold e assume showdown sem apostas futuras, rake ou side pots.
- A decisão não é GTO nem considera fold equity exata, side pots ou equilíbrio de Nash multi-street. Quando um LLM local ou cloud é usado na assistência ao jogo, os números matemáticos (`equity`, `pot_odds`, `call_ev`, `spr`, intervalos de Wilson) provêm sempre do motor determinístico; o LLM atua como camada de síntese e recomendação tática sujeita a validação de legalidade das ações.
- A visão não foi validada em screenshots reais; não suporta layouts, temas, proporções ou fontes fora do fixture calibrado.
- Perfis estatísticos requerem amostras substanciais para convergir; as frequências observadas não substituem automaticamente a range manual atribuída no motor de equity.

## Solver heads-up push/fold (chip-EV v1)

`push_fold.py` calcula estratégias heads-up de shove/fold/call com stacks iniciais iguais, sem rake e ante opcional por jogador. Usa CFR+ sobre as 169 classes e a matriz de equity pré-flop. A saída inclui a frequência de shove e call para cada classe, o gap de Nash em BB e um indicador explícito de convergência. A abstração usa pesos pelo número de combos, mas ignora blockers entre as classes.

```powershell
py -m lugon_poker.push_fold_cli --stack-bb 10 --output data/charts/hu_10bb_no_ante.json
py -m lugon_poker.push_fold_cli --stack-bb 10 --ante-bb 0.1 --output data/charts/hu_10bb_ante.json
```

Com a matriz atual, o caso de 10 BB sem ante produz shove da SB em 58,2% e call da BB em 37,7% das combinações. Isto fica próximo do chart heads-up chip-EV publicado (58,4% / 37,3%) ([Poker Math Coach](https://pokermathcoach.com/tools/push-fold-chart/)); as diferenças incluem a aproximação por classes. A [ferramenta HUNE da HoldemResources](https://www.holdemresources.net/hune) documenta o mesmo jogo simplificado de push-or-fold heads-up.

A versão heads-up não cobre stacks desiguais, multiway ou ICM. O solver 3-max abaixo acrescenta stacks por posição, blockers e utilidade ICM.

## Solver 3-max push/fold com ICM e modelos locais experimentais

`icm.py` calcula a equity Malmuth-Harville exata. `three_max.py` aplica-a a sete resultados terminais de uma árvore simplificada BTN → SB → BB. As seis frequências de shove/call são representadas pelas 169 classes de mãos iniciais. Os deals usam cartas físicas, preservam blockers e resultados multiway, devolvem fichas não pagas e liquidam side pots antes do ICM.

O solver tem dois modos de amostragem. `fixed` usa uma amostra comum de deals físicos; `stratified` garante cobertura equilibrada de todas as classes em cada lugar. Ambos usam regret matching+ simultâneo. Em jogos de três jogadores isto **não garante** convergência para Nash. A validação estratificada usa cross-fit: escolhe a melhor resposta numa metade dos deals e mede o ganho na outra, reduzindo o viés de escolher e medir a ação na mesma amostra. Os resultados continuam a ser aproximações, não soluções exatas.

```powershell
# Resolver um perfil com amostragem estratificada
py -m lugon_poker.three_max_cli --stacks 10,10,10 --payouts 0.5,0.3,0.2 --sampling-mode stratified --samples-per-hand-class 500 --output data/charts/3max_10bb_50_30_20.json

# Construir etiquetas 50/30/20 com quatro seeds e validar antes do treino
py tools/build_three_max_charts.py --sampling-mode stratified --stacks 6,8,10,12,15 --payout 0.5,0.3,0.2 --seeds 7,19,53,71 --samples-per-hand-class 1000 --max-iterations 100 --chart-dir data/charts/three_max_stratified_3paid_v2 --dataset data/training/three_max_push_fold_3paid_stratified_v2.jsonl --summary data/charts/three_max_stratified_3paid_v2_summary.json
py tools/validate_three_max_stratified_sweep.py --summary data/charts/three_max_stratified_3paid_v2_summary.json --chart-dir data/charts/three_max_stratified_3paid_v2 --dataset data/training/three_max_push_fold_3paid_stratified_v2.jsonl --output data/charts/three_max_stratified_3paid_v2_validation.json --samples-per-class 1000 --seed 12021

# Treinar e consultar modelos por estrutura de prémios
py tools/train_three_max_model.py
py tools/predict_three_max_model.py --hand AKo --stack-bb 9

py tools/train_three_max_model.py --dataset data/training/three_max_push_fold_7030_stratified.jsonl --payouts 0.7 0.3 0
py tools/predict_three_max_model.py --model data/models/three_max_70_30_0_linear.json --hand AKo --stack-bb 9

py tools/train_three_max_model.py --dataset data/training/three_max_push_fold_wta_stratified.jsonl --payouts 1 0 0
```

A amostragem estratificada melhorou o perfil 8 BB winner-take-all: no mesmo holdout equilibrado, o ganho cross-fit médio caiu de 0,00718 no solver de amostra fixa para cerca de 0,00005 no solver estratificado. Para 50/30/20, a validação independente nos cinco stacks de treino também ficou perto de zero ganho cross-fit.

O pipeline treinou três regressões lineares de frequências por classe de mão e profundidade de stack. Cada uma usa 845 labels (5 stacks × 169 classes), dá menos peso a labels instáveis entre seeds e tem validação externa em 7, 9, 11 e 13,5 BB com duas seeds novas.

| Prémios | MAE leave-one-stack-out / baseline constante | MAE externo / desacordo entre seeds | Ganho cross-fit externo: modelo / solver médio |
|---|---:|---:|---:|
| 50/30/20 v1 (arquivo) | 0,0392 / 0,0521 | 0,0907 / 0,1196 | 0,00029 / 0,00007 |
| 50/30/20 v2 (atual) | 0,0256 / 0,0449 | 0,0681 / 0,1196 | −0,00013 / 0,00007 |
| 70/30/0 | 0,0305 / 0,0432 | 0,0885 / 0,1117 | 0,00072 / −0,00023 |
| 100/0/0 | 0,0641 / 0,0919 | 0,1247 / 0,1550 | 0,00067 / 0,00047 |

Os modelos desta tabela assumem 3-max, stacks iguais de 6–15 BB, uma estrutura de prémios fixa por ficheiro, sem ante, blinds 0,5/1 BB e jogo pré-flop push/fold.

Os números 50/30/20 são médias de dois holdouts nos stacks 7, 9, 11 e 13,5 BB; o primeiro usou duas seeds do solver e o segundo quatro. A v2 usa quatro seeds e 1 000 deals/classe no treino. No conjunto externo, o NashConv plug-in médio da v2 foi 0,00096 contra 0,00112 do perfil médio do solver; o ganho cross-fit médio foi −0,00013 contra +0,00007. A melhoria de frequência e de NashConv apareceu nos dois holdouts; a v2 é agora o modelo padrão, embora continue experimental.

Nas três estruturas, o erro externo de frequência é inferior ao desacordo entre seeds, mas as frequências exatas não são um alvo estável. Os ganhos cross-fit estratégicos são pequenos e próximos do ruído. Os modelos locais não substituem o solver.

O modelo predefinido 50/30/20 está em `data/models/three_max_50_30_20_linear.json`; a versão anterior ficou arquivada em `data/models/three_max_50_30_20_linear_v1.json`.

O comando de previsão inclui no JSON a classe de mão, o stack, os prémios, o domínio e a proveniência de treino do ficheiro carregado, além das seis frequências como probabilidades entre 0 e 1.

### Modelo experimental para stacks desiguais

O modelo principal mantém stacks iguais. O primeiro protótipo desigual usa uma inclinação independente por lugar (BTN, SB e BB) e foi treinado com 1 859 labels em 11 vetores, quatro seeds e 500 deals por classe. No leave-one-vector-out, o MAE combo-weighted foi 0,0585 contra 0,0659 do baseline constante. No primeiro holdout externo de 8/10/12 BB, o MAE foi 0,0827 contra 0,1327 de desacordo médio entre seeds.

A grelha v2 expandiu o treino para 23 vetores e 3 887 labels. Na mesma grelha, o MAE leave-one-vector-out foi 0,04775 no modelo aditivo e 0,04692 com interações BTN×SB, BTN×BB e SB×BB; o baseline constante foi 0,05857. Ambos foram comparados em dois holdouts externos, cada um com seis permutações de stacks e quatro seeds novas. A validação cross-fit usou 3 000 amostras por classe e duas seeds independentes:

| Holdout | Modelo | MAE / desacordo entre seeds | Ganho cross-fit: modelo / solver médio |
|---|---|---:|---:|
| Permutações de 8/10/12 BB | Aditivo v2 | 0,08166 / 0,13271 | 0,000475 / 0,000586 |
| Permutações de 8/10/12 BB | Interações v2 | 0,08111 / 0,13271 | 0,000461 / 0,000586 |
| Permutações de 9/10/11 BB | Aditivo v2 | 0,08480 / 0,13169 | 0,000759 / 0,000684 |
| Permutações de 9/10/11 BB | Interações v2 | 0,08473 / 0,13169 | 0,000770 / 0,000684 |

As interações reduzem ligeiramente o erro de frequência, mas os resultados estratégicos variam entre holdouts. Os erros de frequência continuam abaixo do desacordo entre seeds; os ganhos cross-fit ficam próximos dos do solver médio.

O erro concentra-se em `sb_push_after_btn_fold` (cerca de 0,15), `btn_push` (0,10–0,12) e `bb_call_multiway` (cerca de 0,10). Os calls heads-up ficam entre 0,03 e 0,06, por isso a próxima melhoria deve focar as frequências de shove e o call multiway.

A candidata v3 acrescentou as seis permutações de 9/10/11 BB: 29 vetores e 4 901 labels. O leave-one-vector-out caiu para 0,04459 no aditivo e 0,04350 com interações, contra baseline 0,05335. No holdout preservado de 8/10/12 BB, numa validação com 3 000 amostras por classe e seed 99265, o MAE foi 0,08194 no aditivo e 0,08149 com interações; o ganho cross-fit foi 0,000424 e 0,000414, contra 0,000517 do solver médio. Duas validações independentes de 1 000 amostras também deram ganho cross-fit médio inferior ao solver. O MAE externo da v3 foi ligeiramente pior que o da v2 nesse holdout. O holdout 9/10/11 passou a fazer parte do treino v3 e não é usado para validar essa versão.

O segundo holdout externo da v3 usa as seis permutações de 7/10/13 BB, quatro seeds novas e 3 000 amostras por classe (seed de validação 99271). O MAE ficou em 0,09131 no modelo aditivo e 0,09210 com interações, abaixo do desacordo entre seeds (0,13398). O ganho cross-fit foi, porém, maior que o do solver médio nas duas versões:

| Holdout | Modelo | MAE / desacordo entre seeds | Ganho cross-fit: modelo / solver médio | NashConv plug-in: modelo / solver médio |
|---|---|---:|---:|---:|
| Permutações de 7/10/13 BB | Aditivo v3 | 0,09131 / 0,13398 | 0,000694 / 0,000506 | 0,001004 / 0,000778 |
| Permutações de 7/10/13 BB | Interações v3 | 0,09210 / 0,13398 | 0,000674 / 0,000506 | 0,000987 / 0,000778 |

Juntando os dois holdouts externos com M=3 000, o ganho cross-fit médio da v3 fica em cerca de 0,000559 no aditivo e 0,000544 com interações, acima dos 0,000511 do solver médio. As interações não melhoram o MAE de frequência neste novo holdout. O maior erro continua em `sb_push_after_btn_fold` (~0,152), seguido de `bb_call_multiway` (~0,116) e `btn_push` (~0,110). Os resultados completos estão em `data/charts/three_max_unequal_additive_v3_holdout_2_m3000_seed99271.json` e `data/charts/three_max_unequal_interactions_v3_holdout_2_m3000_seed99271.json`. A v3 permanece experimental e desligada do Assistente: o próximo ciclo deve recolher mais labels para essas decisões e para stacks perto dos limites do domínio, preservando vetores independentes para validação.

Para repetir a grelha de treino, validar, treinar e consultar:

```powershell
py tools/build_three_max_unequal_charts.py
py tools/validate_three_max_unequal_sweep.py
py tools/train_three_max_unequal_model.py
py tools/predict_three_max_unequal_model.py --hand AKo --stacks-bb 8,10,12
```

O validador recusa vetores ou seeds do solver que já existam no treino e confirma que os charts correspondem aos stacks, prémios, amostragem e tolerância declarados no resumo.

Para treinar os candidatos expandidos já gravados:

```powershell
py tools/train_three_max_unequal_model.py --dataset data/training/three_max_unequal_train_v2.jsonl --output data/models/three_max_unequal_50_30_20_additive_v2.json
py tools/train_three_max_unequal_model.py --dataset data/training/three_max_unequal_train_v2.jsonl --output data/models/three_max_unequal_50_30_20_interactions_v2.json --interactions
py tools/train_three_max_unequal_model.py --dataset data/training/three_max_unequal_train_v3.jsonl --output data/models/three_max_unequal_50_30_20_additive_v3.json
py tools/train_three_max_unequal_model.py --dataset data/training/three_max_unequal_train_v3.jsonl --output data/models/three_max_unequal_50_30_20_interactions_v3.json --interactions
```

O modelo desigual também só suporta 50/30/20, stacks por lugar entre 6 e 15 BB, sem ante, blinds 0,5/1 BB e jogo pré-flop push/fold. A árvore não inclui limps, raises não all-in nem jogo pós-flop; o CFR simultâneo multi-jogador não garante convergência de Nash. Antes de o usar em decisões, faltam validar mais combinações de stacks, antes e mãos reais.
