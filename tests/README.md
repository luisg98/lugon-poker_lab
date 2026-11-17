# Lugon Poker Lab — Suíte de Testes (`tests/`)

Esta diretoria contém os testes automatizados do projeto, desenvolvidos para correr com o framework padrão `unittest` do Python sem necessidade de instalar pacotes adicionais.

---

## Como Executar

Podes executar toda a suíte a partir da raiz do projeto através de qualquer um dos seguintes métodos:

```powershell
# Execução direta com py
py -m unittest discover -s tests -v

# Ou através do script batch incluído
.\run_tests.bat
```

Para correr apenas um módulo de teste específico:

```powershell
py -m unittest tests.test_engine -v
py -m unittest tests.test_icm -v
py -m unittest tests.test_three_max -v
```

---

## Módulos de Teste

| Ficheiro | O que testa |
|---|---|
| `test_engine.py` | Avaliação de mãos de 5 a 7 cartas, Monte Carlo de equity heads-up e multiway, expansão de ranges e blockers. |
| `test_icm.py` | Cálculo exato de equidades de torneio pelo modelo Malmuth-Harville. |
| `test_push_fold.py` | Algoritmo CFR+ heads-up, gap de Nash e convergência de estratégias all-in/fold. |
| `test_three_max.py` | Solver 3-max com amostragem estratificada, liquidação de potes multiway e treino de modelos lineares. |
| `test_game_type.py` | Alternância de formatos (Spin & Go vs Torneio MTT) e ajuste de limiares de decisão. |
| `test_vision.py` | Leitura determinística de píxeis em mesas sintéticas calibradas (640×360 até 1600×900). |
| `test_screen_capture.py` | Funções Win32 GDI de enumeração de janelas, recorte e captura com compensação de DPI. |
| `test_live_vision.py` | Pipeline de visão híbrida (calibração local vs Vision-LLM). |
| `test_simulator.py` | Simulação headless de mesas com rotação de lugares, regras de apostas e limites legais. |
| `test_profiles.py` | Métricas VPIP, PFR, 3-Bet, AFq, intervalos de confiança de Wilson e deduplicação em SQLite. |
| `test_bridge.py` | Mecanismo de IPC assíncrono entre a interface Companion e agentes externos. |
| `test_preflop.py` | Carregamento da matriz pré-flop de 169 classes e coerência de equidades. |
