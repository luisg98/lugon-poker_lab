# Lugon Poker Lab — Mapa do Projeto e Estrutura de Diretórios

Este documento fornece a visão completa da organização de diretórios do **Lugon Poker Lab**, detalhando a função de cada componente, convenções de ficheiros e fluxo de dados.

---

## 1. Visão Geral da Árvore de Diretórios

```text
lugon-poker_lab/
├── .editorconfig                  # Padrões de formatação de código (espaçamento, finais de linha)
├── .env.example                   # Modelo de variáveis de ambiente (chaves de API LLM, endpoints)
├── .gitignore                     # Regras abrangentes de exclusão para Git
├── CHANGELOG.md                   # Histórico de alterações e marcos por versão
├── iniciar_companion.bat          # Script de arranque rápido do Assistente GUI no Windows
├── LugonPokerLab.exe              # Executável compilado C# (lançador rápido sem janela de consola)
├── pyproject.toml                 # Configuração moderna de empacotamento Python (PEP 517/621)
├── README.md                      # Documentação técnica e guia principal do projeto
├── requirements.txt               # Documentação de dependências (stdlib estrita; 0 dependências pip)
├── requirements-dev.txt           # Ferramentas opcionais de desenvolvimento (pytest, mypy, ruff)
├── ROADMAP.md                     # Planeamento estratégico (Player Intelligence, ICM e Modelos)
├── run_tests.bat                  # Script de execução rápida da suíte de testes
│
├── assets/                        # Ficheiros visuais e recursos estáticos da aplicação
├── data/                          # Dados persistentes, modelos treinados, charts e diagnósticos
├── docs/                          # Documentação detalhada de arquitetura e guias
├── examples/                      # Fixtures, históricos de demonstração e capturas de teste
├── lugon_poker/                   # Código-fonte principal da biblioteca e aplicação
├── tests/                         # Suíte completa de testes unitários (unittest)
└── tools/                         # Utilitários de linha de comandos (solvers, treino de modelos, validação)
```

---

## 2. Descrição Detalhada dos Diretórios

### `lugon_poker/` — Núcleo da Aplicação
Biblioteca Python com zero dependências externas (apenas biblioteca padrão Python 3.10+ e ctypes no Windows).

| Módulo / Ficheiro | Responsabilidade |
|---|---|
| `__init__.py` | Versão global (`__version__`), usada para arquivamento e rastreabilidade de capturas. |
| `cards.py` | Estruturas de cartas (`Card`), ranks, naipes e avaliador de mãos de 5 a 7 cartas. |
| `equity.py` | Cálculo Monte Carlo e enumeração exata heads-up no river. |
| `ranges.py` | Parser e expansão de ranges em texto (`AA,AKs,AKo`) para combinações físicas. |
| `state.py` | Representação imutável e validação do estado da mesa (`HandState`). |
| `strategy.py` | Pot odds, SPR e EV incremental de call contra a range assumida. |
| `icm.py` | Cálculo exato de equidades de torneio Malmuth-Harville (ICM). |
| `push_fold.py` | Solver heads-up push/fold baseado em CFR+ para 169 classes pré-flop. |
| `three_max.py` | Solver 3-max push/fold com amostragem estratificada, blockers e ICM. |
| `three_max_model.py` | Modelos de regressão linear e com termos de interação para frequências 3-max. |
| `synthetic.py` | Renderizador sintético procedural de mesas 16:9 em pixéis RGB limpos. |
| `vision.py` | Leitor ótico determinístico para o layout calibrado sintético (sem OCR pesado). |
| `screen_capture.py` | Captura de ecrã/janela com suporte DPI aware via Win32 GDI/ctypes. |
| `live_vision.py` | Pipeline de visão híbrida (calibrada local + VLM sob pedido via Ollama/OpenAI). |
| `mercury_vision.py` | Pipeline de visão calibrado para o layout PokerStars Mercury (baralho 4 cores). |
| `replay.py` | Reanálise e validação em lote de capturas de diagnóstico gravadas. |
| `history.py` | Validação de históricos completos (`HandHistory`) e deteção de transições. |
| `profiles.py` | Cálculo de VPIP, PFR, 3-Bet, AFq, intervalos de Wilson a 95% e priors Bayesianos. |
| `persistence.py` | Armazenamento persistente de mãos e perfis em base de dados SQLite (`profiles.sqlite`). |
| `assistant.py` | Motor de assistência estratégica (advisor heurístico local e integrações LLM). |
| `agent_bridge.py` | Ponte de comunicação assíncrona entre o assistente gráfico e agentes de IA externos. |
| `companion_gui.py` | Interface gráfica Tkinter do Assistente Companion (captura em direto, HUD e advisor). |
| `cli.py`, `*_cli.py` | Pontos de entrada CLI para cada subsistema (vision, profiles, sim, bridge, etc.). |

---

### `data/` — Armazenamento e Resultados
Todo o armazenamento estruturado da aplicação fica centralizado nesta pasta.

- **`charts/`**: Saídas JSON dos solvers push/fold e resumos de convergência Nash (separados por estrutura de prémios e stack).
- **`training/`**: Ficheiros `.jsonl` gerados pelos solvers com rótulos estratificados para treino de modelos supervisionados.
- **`models/`**: Pesos e metadados dos modelos de regressão treinados (`.json`).
- **`captures/`**: Novas capturas de diagnóstico da versão atual em `v1.0.6/`; arquivos anteriores permanecem separados em subpastas de versões anteriores e não são usados pelo fluxo atual.
- **`live_bridge/`**: Ficheiros de comunicação IPC entre a GUI do Assistente e agentes externos (`request.json`, `response.json`, `bridge_history.jsonl`).
- **Ficheiros na raiz de `data/`**:
  - `preflop_equity_169.json`: Matriz pré-calculada de equidades heads-up all-in para as 169 classes.
  - `preflop_build_progress.json`: Estado de progresso do cálculo da matriz pré-flop.
  - `profiles.sqlite`: Base de dados persistente de perfis de jogadores.

---

### `tools/` — Solvers, Treino e Validação
Scripts executáveis de linha de comandos para tarefas de dados pesadas ou compilação.

- **Geração de Charts**: `build_three_max_charts.py`, `build_three_max_unequal_charts.py`, `build_preflop_matrix.py`.
- **Treino de Modelos**: `train_three_max_model.py`, `train_three_max_unequal_model.py`.
- **Validação de Modelos e Sweeps**: `validate_three_max_charts.py`, `validate_three_max_model.py`, `validate_three_max_stratified_sweep.py`, `validate_three_max_unequal_model.py`, `validate_three_max_unequal_sweep.py`.
- **Inferência Rápida**: `predict_three_max_model.py`, `predict_three_max_unequal_model.py`.
- **Compilação e Utilitários de Sistema**: `launcher.cs` (código-fonte C# do executável), `build_icon.ps1` (gerador do ícone `.ico`).

---

### `tests/` — Testes Automatizados
Suíte com 110+ testes baseados exclusivamente no `unittest` padrão:
- Cobrem avaliação de cartas, Monte Carlo, ICM, solvers, visão sintética, captura Win32, HUD e persistência SQLite.
- Executáveis via `py -m unittest discover -s tests -v` ou através de `run_tests.bat`.

---

### `examples/` — Fixtures e Demonstrações
Ficheiros de exemplo para execução e validação rápida sem necessidade de sessões reais:
- `flop_call.json`: Cenário representativo de decisão pós-flop.
- `histories/sample_hands.json`: Histórico de mãos com múltiplos jogadores para testar estatísticas.
- `vision/`: Imagens sintéticas de referência (640×360 até 1600×900) e fixtures de mesa (`flop_table.json`).

---

### `assets/` — Recursos Visuais
Recursos gráficos estáticos e identidade visual da aplicação:
- `lugon_poker_lab.ico`: Ícone da aplicação compilado em múltiplos tamanhos (16x16 até 256x256).
- `lugon_poker_lab.png`: Logótipo em alta resolução (1024x1024).
- `lugon_poker_lab_64.png`: Versão reduzida para a interface Tkinter e barra de tarefas.
