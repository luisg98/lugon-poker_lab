# Lugon Poker Lab — Estrutura da Diretoria `data/`

Esta pasta contém todos os dados persistentes, ficheiros de diagnóstico, modelos treinados e saídas de solvers.

---

## Estrutura de Subdiretórios

```text
data/
├── captures/                      # Capturas de ecrã completas organizadas por versão da aplicação
│   ├── v0.2.0/                    # Arquivo histórico
│   ├── v0.3.0/                    # Arquivo histórico
│   ├── v1.0.1/                    # Capturas de diagnóstico usadas para ajustar a extração
│   ├── v1.0.2/                    # Capturas analisadas para corrigir a leitura
│   ├── v1.0.3/                    # Arquivo histórico
│   ├── v1.0.4/                    # Capturas diagnósticas da versão anterior
│   └── v1.0.5/                    # Capturas novas da versão atual
│
├── charts/                        # Soluções push/fold e resumos de convergência Nash
│   ├── three_max_holdout*/        # Conjuntos de validação externa para 3-max (stacks iguais e desiguais)
│   ├── three_max_stratified*/     # Grelhas de treino estratificadas por estrutura de prémios (50/30/20, 70/30, WTA)
│   ├── three_max_unequal*/        # Soluções com vetores de stacks assimétricos (BTN, SB, BB)
│   └── *.json                     # Resumos de convergência, sweeps e validações individuais
│
├── live_bridge/                   # Ficheiros de comunicação IPC em direto com agentes de IA externos
│   ├── .gitkeep                   # Garante que a pasta é mantida no repositório
│   ├── bridge_history.jsonl       # Histórico acumulado de interações e pedidos
│   ├── latest_capture.png         # Última captura de ecrã enviada para análise
│   ├── request.json               # Pedido atual do Assistente com o estado da mesa
│   └── response.json              # Resposta e recomendação estratégica do agente externo
│
├── models/                        # Modelos supervisionados de frequências de ação pré-flop
│   ├── three_max_50_30_20_linear.json             # Modelo baseline linear padrão para 50/30/20 (stacks iguais)
│   ├── three_max_70_30_0_linear.json              # Modelo para prémios 70/30/0
│   ├── three_max_100_0_0_linear.json              # Modelo para Winner-Take-All
│   ├── three_max_unequal_*_additive_*.json        # Modelos aditivos com stacks desiguais
│   └── three_max_unequal_*_interactions_*.json    # Modelos com termos de interação BTN×SB, BTN×BB e SB×BB
│
├── training/                      # Ficheiros JSONL de treino rotulados produzidos pelos solvers
│   ├── three_max_push_fold_*_stratified*.jsonl    # Datasets estratificados com stacks iguais
│   └── three_max_unequal_*.jsonl                  # Datasets estratificados com stacks desiguais
│
├── preflop_build_progress.json    # Registo de progresso do cálculo de equidades pré-flop
├── preflop_equity_169.json        # Matriz pré-calculada de equidades all-in pré-flop (169×169 classes)
└── profiles.sqlite                # Base de dados persistente SQLite de históricos e perfis de jogadores
```

---

## Notas de Utilização

1. **Separação de Treino e Validação:**
   Ficheiros com o prefixo `three_max_unequal_train` nunca devem partilhar vetores ou sementes com conjuntos de validação (`three_max_unequal_holdout`).
2. **Idempotência do SQLite:**
   A base `profiles.sqlite` suporta importações repetidas de ficheiros JSON de histórico sem criar duplicados de `hand_id`.
3. **Privacidade de Capturas:**
   As imagens na pasta `captures/` contêm diagnósticos gerados localmente e são excluídas ou anonimizadas conforme a política de privacidade.
