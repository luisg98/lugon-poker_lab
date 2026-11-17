# Lugon Poker Lab — Arquitetura do Sistema

Este documento descreve os princípios de design, o fluxo de dados e os subsistemas do **Lugon Poker Lab**.

---

## 1. Princípios de Design

1. **Dependências Zero no Core:**
   A aplicação foi construída para rodar estritamente com a biblioteca padrão do Python 3.10+ e chamadas Win32 via `ctypes`. Não requer bibliotecas de terceiros como OpenCV, NumPy, PyTorch ou PIL para funcionamento padrão.
2. **Separação Rígida entre Matemática e Visão:**
   A camada de visão apenas produz um estado estruturado (`HandState`). Se houver incerteza ou oclusão, o campo permanece vazio (`ambiguities`), impedindo suposições incorretas.
3. **Determinação Rápida (<30ms) vs. LLM para Síntese:**
   O motor matemático local avalia equity, pot odds e EV em menos de 30 milissegundos na CPU. LLMs locais (Ollama, vLLM) ou cloud (Gemini, OpenAI) são usados exclusivamente para explicar e sugerir estratégias táticas, recebendo números pré-calculados e sendo validados por `validate_and_clamp_action`.
4. **Respeito às Políticas de Jogo Responsável:**
   A arquitetura distingue claramente o modo com cliente aberto (apenas HUD e cálculo de odds) do modo com cliente fechado (estudo offline, solvers CFR+, ICM e treino de modelos).

---

## 2. Diagrama de Fluxo de Dados

```text
Captura de Ecrã / Janela (Win32 GDI)
            │
            ▼
    Visão (Mercury / Sintética) ──► HandState (Validação Estrita)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         Motor Matemático                                Base de Perfis (SQLite)
      (Equity, Pot Odds, ICM)                           (VPIP, PFR, 3-Bet, Wilson)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                                  DecisionContext
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
           Advisor Heurístico Local                    LLM / Agent Bridge
              (<30ms, offline)                     (Ollama, LM Studio, Gemini)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                                Companion GUI / HUD
```

---

## 3. Subsistemas Principais

### A. Visão Computacional e Captura de Ecrã
- **`screen_capture.py`**: Interage com a API do Windows (GDI/User32) com compensação de DPI Awareness para capturar regiões ou janelas específicas (ex.: PokerStars) em memória RGB pura sem ficheiros temporários em disco.
- **`mercury_vision.py`**: Reconhecimento baseado em assinaturas de pixéis e histogramas calibrados para o tema Mercury do PokerStars com baralho de 4 cores (`vermelho=h, azul=d, verde=c, cinza=s`).
- **`live_vision.py`**: Suporta pipeline híbrido com fallback para modelos multimodais (Vision-LLM via Ollama) para mesas não calibradas.

### B. Motor Matemático e Teoria dos Jogos
- **`cards.py`**: Avaliação de 5 a 7 cartas por ranking lexicográfico com identificação de wheel, flush e kickers.
- **`equity.py`**: Simulação Monte Carlo heads-up e multiway com seed fixa; enumeração combinatória completa no river.
- **`icm.py`**: Algoritmo recursivo exato Malmuth-Harville para cálculo de equity de prémios de torneio.
- **`push_fold.py` & `three_max.py`**: Algoritmos baseados em CFR+ (Counterfactual Regret Minimization) para equilíbrio pré-flop com amostragem estratificada por classes de mãos.

### C. Modelos Supervisionados de Frequências
- **`three_max_model.py`**: Modela as 6 frequências de ação push/fold (shove BTN, shove SB, call BB vs SB, call BB vs BTN, call multiway, etc.) utilizando regressões lineares por classe de mão com termos de interação entre stacks.

### D. Inteligência de Jogadores (Player Intelligence)
- **`history.py`**: Validação determinística de sequências de ação e ordem de streets.
- **`profiles.py`**: Cálculo de taxas empíricas (`k/n`), intervalos de confiança de Wilson a 95% e estimadores Bayesianos com prior `Beta(1, 1)`.
- **`persistence.py`**: Persistência idempotente por `hand_id` em base de dados SQLite (`profiles.sqlite`).

### E. Assistente Companion e Bridge
- **`companion_gui.py`**: Janela flutuante leve em Tkinter com suporte para visualização transparente, seleção interativa de região e preview em tempo real.
- **`agent_bridge.py`**: Mecanismo IPC via ficheiros JSON em `data/live_bridge/` para sincronizar estados da mesa com agentes inteligentes externos.
