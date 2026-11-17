# Lugon Poker Lab — Utilitários e Ferramentas (`tools/`)

Esta pasta reúne scripts de linha de comandos para tarefas analíticas pesadas, solvers de teoria de jogos (CFR+), pipelines de treino e validação supervisionada, e scripts de build de sistema.

---

## 1. Geração de Soluções e Datasets (Solvers)

| Script | Descrição |
|---|---|
| `build_preflop_matrix.py` | Pré-computa a matriz completa de equidades heads-up all-in entre as 169 classes de mãos (resultado em `data/preflop_equity_169.json`). |
| `build_three_max_charts.py` | Executa o solver CFR+ 3-max com amostragem estratificada para vetores de stacks iguais, exportando charts individuais e datasets `.jsonl`. |
| `build_three_max_unequal_charts.py` | Executa o solver CFR+ 3-max com stacks assimétricos (BTN, SB e BB independentes), exportando charts e datasets `.jsonl` para treino e holdout. |

---

## 2. Treino de Modelos de Regressão

| Script | Descrição |
|---|---|
| `train_three_max_model.py` | Ajusta modelos lineares de frequência pré-flop por classe de mão para stacks iguais sob diferentes distribuições de prémios. |
| `train_three_max_unequal_model.py` | Treina modelos de frequências com características de stacks independentes por posição (aditivos ou com termos de interação entre posições). |

---

## 3. Validação e Benchmarking

| Script | Descrição |
|---|---|
| `validate_three_max_charts.py` | Verifica a consistência matemática dos ficheiros de chart exportados. |
| `validate_three_max_model.py` | Avalia o modelo treinado de stacks iguais contra holdouts com validação cruzada independente. |
| `validate_three_max_stratified.py` | Executa testes pontuais de cross-fit estratégico para uma configuração específica. |
| `validate_three_max_stratified_sweep.py` | Avalia uma grelha completa de soluções estratificadas para medir convergência e estabilidade entre sementes. |
| `validate_three_max_unequal_model.py` | Valida modelos com stacks desiguais contra holdouts preservados (MAE, desacordo entre sementes e ganho cross-fit). |
| `validate_three_max_unequal_sweep.py` | Confirma integridade dos charts de stacks desiguais antes de iniciar o processo de treino. |

---

## 4. Inferência e Consulta Rápida

| Script | Descrição |
|---|---|
| `predict_three_max_model.py` | Consulta instantânea das 6 frequências de ação para uma classe de mão e stack igual dado (ex.: `py tools/predict_three_max_model.py --hand AKo --stack-bb 9`). |
| `predict_three_max_unequal_model.py` | Consulta de frequências para combinações com stacks desiguais (ex.: `py tools/predict_three_max_unequal_model.py --hand AKo --stacks-bb 8,10,12`). |

---

## 5. Compilação e Artefactos de Sistema

| Ficheiro | Descrição |
|---|---|
| `launcher.cs` | Código-fonte C# para compilar o executável `LugonPokerLab.exe`, permitindo lançar a aplicação via `pyw.exe` sem abrir janela de consola preta. |
| `build_icon.ps1` | Script PowerShell que gera o ficheiro multi-camada `assets/lugon_poker_lab.ico` a partir da imagem PNG original. |
