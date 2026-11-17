# Lugon Poker Lab — Guias de Fluxo de Trabalho (Workflows)

Este guia orienta a utilização dos quatro principais fluxos operacionais da aplicação.

---

## 1. Assistente em Tempo Real (Companion GUI)

Para utilizar o assistente durante o jogo ou revisão de ecrã:

```powershell
# Iniciar o Assistente Companion
.\iniciar_companion.bat
# ou
py -m lugon_poker.companion_gui
```

### Funcionalidades:
- **Layout 888 BLAST**: ao abrir, a app procura a mesa BLAST, coloca-a à esquerda e ocupa a altura útil do ecrã; o Assistente fica numa coluna mais estreita à direita. A captura passa a apontar apenas à janela da mesa e, ao fechar o Assistente, a mesa recupera o tamanho anterior.
- **Modelo único (1.0.6)**: usa `qwen3-vl:2b-instruct` local via Ollama. Com **Auto desligado**, F8 analisa um frame; quando o Solver ICM está ativo, recebe a solução calculada e mantém a ação e o valor definidos pelo solver. Em **Auto**, o Qwen fica dedicado à leitura e o motor local mostra a recomendação sem esperar por uma segunda chamada.
- **Auto**: ativa a leitura contínua, com intervalo de 2 segundos entre leituras. Usa o Qwen para ler e o motor local para recomendar; arquiva apenas frames com a fila de ações do Hero confirmada, deduplicando estados repetidos. O botão **Capturar**, junto do Auto, guarda manualmente o ecrã atual sem parar a leitura nem chamar o modelo. F8 faz uma análise pontual quando necessário.
- **Pote e ante no 888 BLAST**: o pote vem apenas do valor junto a **Pote total:**. O total já inclui blinds, antes e apostas; valores junto a lugares são contribuições, nunca o total nem o montante a pagar. Se não houver um botão de call legível, a app não inventa o `to_call` a partir das fichas.
- **Opções recolhidas**: o botão **Opções** abre o tipo de jogo, o layout, o atalho F8 e a gravação de capturas de diagnóstico.
- **Jogo e Solver ICM**: o tipo de jogo selecionado é usado pelo motor; na entrada manual, o Solver ICM calcula push/fold 3-max e fornece o resultado ao mesmo Qwen para explicação.

---

## 2. Simulação Headless e Benchmarking (`sim_cli.py`)

Para testar estratégias, medir latências ou avaliar mãos sem interface gráfica:

```powershell
# Simular uma única mão com apresentação do prompt estruturado enviado ao LLM:
py -m lugon_poker.sim_cli play --iterations 500 --seed 7 --show-prompt

# Modo interativo no terminal (recebe sugestões e escolhe ações manuais):
py -m lugon_poker.sim_cli play --interactive --iterations 1000 --seed 7

# Avaliar sessão de várias mãos e guardar resultados na base de perfis:
py -m lugon_poker.sim_cli evaluate --hands 10 --iterations 500 --seed 7 --db data/sim_profiles.sqlite
```

---

## 3. Solver 3-Max Push/Fold e Treino de Modelos

Para construir tabelas de decisão ótima e treinar modelos preditivos:

```powershell
# Passo 1: Construir grelha de charts estratificados com solvers CFR+
py tools/build_three_max_unequal_charts.py `
  --stack-vectors 7,10,13;13,10,7;10,7,13 `
  --seeds 251,257,263,269 `
  --samples-per-hand-class 500 `
  --chart-dir data/charts/meu_experimento `
  --dataset data/training/meu_experimento.jsonl `
  --summary data/charts/meu_experimento_summary.json

# Passo 2: Validar o conjunto de soluções geradas
py tools/validate_three_max_unequal_sweep.py `
  --chart-dir data/charts/meu_experimento `
  --dataset data/training/meu_experimento.jsonl `
  --summary data/charts/meu_experimento_summary.json

# Passo 3: Treinar o modelo de regressão com termos de interação
py tools/train_three_max_unequal_model.py `
  --dataset data/training/meu_experimento.jsonl `
  --output data/models/meu_modelo.json `
  --interactions

# Passo 4: Fazer predições imediatas para qualquer mão e stacks
py tools/predict_three_max_unequal_model.py `
  --model data/models/meu_modelo.json `
  --hand AKo `
  --stacks-bb 7,10,13
```

---

## 4. Importação de Históricos e Estatísticas de Jogadores

Para gerir perfis e estatísticas de oponentes em SQLite:

```powershell
# Importar ficheiro JSON de mãos para a base persistente:
py -m lugon_poker.profiles_cli import examples/histories/sample_hands.json --db data/profiles.sqlite

# Inspecionar estatísticas detalhadas de um jogador (VPIP, PFR, 3-Bet, AFq e intervalos Wilson):
py -m lugon_poker.profiles_cli show --db data/profiles.sqlite --player Vitor

# Exportar em formato JSON para processamento externo:
py -m lugon_poker.profiles_cli show --db data/profiles.sqlite --player Vitor --json
```
