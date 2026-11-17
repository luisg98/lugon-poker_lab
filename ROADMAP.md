# Roadmap — Player Intelligence, ICM/Nash e Modelo de Poker

Plano dos três blocos seguintes. Escrito antes de código, para fixar decisões técnicas e a ordem de execução.

## Âmbito e limite

A [política do PokerStars.PT](https://www.pokerstars.pt/poker/room/prohibited/) separa três grupos. Este roadmap
assume essa separação como regra de arquitetura, não como sugestão:

| Onde corre | O que é permitido | O que não é |
|---|---|---|
| **Durante o jogo** | HUD com estatísticas acumuladas do teu próprio jogo; pot odds; força absoluta da mão; charts pré-flop básicos | Conselho sobre a ação a tomar; qualquer modelo de IA sobre a mão atual; ICM/Nash/GTO; alimentar automaticamente a mesa para uma base de dados |
| **Cliente fechado** | Tudo: ICM, Nash, solvers, RAG, modelo local, revisão de sessões | — |

O **Bloco 2** (ICM/Nash) e o **Bloco 3** (modelo) são ferramentas de revisão: correm com o cliente fechado.
O **Bloco 1** produz dados que servem os dois, e a sua única saída em direto é o HUD.

Decisão estrutural que daqui decorre: a origem dos dados deixa de ser o screen capture e passa a ser o
**hand history do próprio cliente**. Isto não é uma concessão — é um upgrade. Ver Bloco 1.

---

## Bloco 1 — Recolha de dados de adversários

### Porquê mudar de fonte

O que o screen capture vê hoje: ~8 campos por frame, com erros de OCR, só as mãos em que estás a olhar para o
ecrã, sem showdowns, sem as mãos que deste fold, sem as ações que aconteceram entre capturas.

O que um hand history traz: **todas** as mãos, **todas** as ações por ordem, todos os showdowns, valores exatos,
posições, stacks iniciais, resultado líquido. Zero ambiguidade. É a razão por que nenhum tracker sério faz
screen scraping.

Efeito colateral: o trabalho de visão (`mercury_vision.py`, `live_vision.py`) deixa de ser o caminho crítico.
Continua útil para saber *em que mão estás agora* e desenhar o HUD, mas não para recolher estatística.

### Passo 0 — obter um ficheiro real (bloqueia tudo o resto)

A gravação está **desligada**: não existe `%LOCALAPPDATA%\PokerStars.PT\HandHistory\`. No cliente:
Definições → Histórico de Mãos → ativar gravação e escolher pasta.

Depois preciso de **um ficheiro .txt de um Spin & Go já terminado** para fixar o formato. Duas incógnitas reais:

1. **Idioma.** O cliente .PT pode escrever o histórico em português (`desiste`, `iguala`, `aumenta ... para`,
   `passa`, `aposta`) em vez de inglês. Há uma definição para forçar inglês — se existir, usá-la simplifica o
   parser e torna-o compatível com o formato documentado.
2. **Formato do torneio.** Spin & Go escreve cabeçalho de torneio (`PokerStars Tournament #...`), nível de
   blinds, e um *tournament summary* separado com prémio e posição final.

Sem esse ficheiro, o parser seria escrito contra suposições. É o primeiro pedido concreto.

### Passo 1 — `ps_history.py`: parser

Converte o .txt do PokerStars para o `HandHistory` que já existe em [history.py](lugon_poker/history.py), que já
valida ordem de streets, consistência de board, stacks e regras de ação.

Estrutura do ficheiro a tratar: cabeçalho → `*** HOLE CARDS ***` → `*** FLOP ***` / `TURN` / `RIVER` →
`*** SHOW DOWN ***` → `*** SUMMARY ***`. Ações a reconhecer: `posts small/big blind`, `folds`, `checks`,
`calls X`, `bets X`, `raises X to Y`, `is all-in`, `uncalled bet returned`, `collected X from pot`.

Regras de projeto:
- Mão que não parseia inteira é **rejeitada com motivo**, nunca parcialmente aceite. Mesma disciplina do resto
  do projeto: campo ambíguo é campo vazio, não é campo adivinhado.
- Idempotente por `hand_id` — reimportar a mesma pasta não duplica (já garantido em `persistence.py`).
- Testes contra ficheiros reais anonimizados, guardados em `examples/histories/pokerstars/`.

### Passo 2 — esquema de base de dados

O `persistence.py` atual guarda a mão serializada. Para estatística a sério é preciso normalizar:

```
tournaments  (tournament_id, buy_in, multiplier, prize_pool, finish_position, payout, played_at)
hands        (hand_id, tournament_id, level, sb, bb, ante, table_size, button_seat, played_at)
hand_players (hand_id, nickname, seat, position, starting_stack_bb, hole_cards|NULL, net_result)
actions      (hand_id, street, order, nickname, action, amount, pot_before, facing_bet, stack_before_bb)
```

`stack_before_bb` é o campo que faz a diferença toda — ver a seguir.

### Passo 3 — estatísticas que interessam num 3-max hyper-turbo

As métricas clássicas (VPIP/PFR/3-bet/AFq) já estão em [profiles.py](lugon_poker/profiles.py) com denominadores
por oportunidade e intervalos de Wilson — a base está certa. O que falta é o que distingue este formato:

- **Corte por profundidade de stack efetivo**: `>25bb`, `15–25bb`, `10–15bb`, `<10bb`. Num hyper-turbo o mesmo
  jogador é uma pessoa diferente a 25bb e a 8bb. Agregar tudo numa média esconde exatamente o que importa.
- Open / limp / fold por posição, em 3-max (BTN, SB, BB).
- **Frequência de open-shove** por posição e profundidade.
- Fold to steal na BB; call-off vs shove.
- **Frequência de limp** — indicador forte de jogador fraco neste formato.

### Passo 4 — o problema estatístico real, e a solução

Num Spin & Go vês cada adversário durante 1 a 3 torneios: **n por adversário quase nunca passa de 30 mãos**.
Com n=30, o intervalo de Wilson do VPIP é tão largo que a estatística individual não distingue um nit de um
maníaco. O `sample_adequacy` atual diz isto honestamente — e é por isso que, sozinho, o HUD individual vale pouco.

Duas respostas, ambas a construir:

1. **Estatísticas de população.** Agregar o pool inteiro por posição × profundidade. "O pool a 10bb na SB
   faz open-shove 48% das vezes" é uma afirmação com n grande e valor real, ao contrário de qualquer leitura
   individual com n=22.
2. **Encolhimento hierárquico (empirical Bayes).** Em vez do prior uniforme `Beta(1,1)` atual, derivar o prior
   da distribuição do pool. Com poucas observações a estimativa aproxima-se do pool; à medida que n cresce,
   aproxima-se do empírico. É a resposta estatisticamente correta para exatamente este problema, e substitui um
   prior que hoje é uma escolha arbitrária.

O `RateEstimate.from_counts` já aceita `prior_alpha`/`prior_beta` — a mudança encaixa sem reescrever nada.

### Passo 5 — HUD (a única saída em direto)

Janela por lugar, com: n de mãos, VPIP/PFR com indicação de amostra, tendência por profundidade atual, e
desvio face ao pool. Estático, sem mudar conforme a ação da mão, só dados do teu próprio jogo — dentro do
permitido. Mais pot odds e força da mão, que também são permitidos.

---

## Bloco 2 — ICM e push/fold Nash para 3-max

Ferramenta de revisão (cliente fechado). É o maior salto de qualidade estratégica disponível e não depende de
modelos. O ICM é exato; as equities pré-flop da matriz atual são estimativas Monte Carlo reproduzíveis.

Implementado: ICM Malmuth-Harville exato (`lugon_poker/icm.py`), solver heads-up chip-EV (`push_fold.py`) e protótipo 3-max ICM (`three_max.py`). O heads-up usa a matriz pré-flop de equities; o 3-max sorteia deals físicos para preservar blockers e resultados multiway.

Abaixo de ~15bb o Spin & Go colapsa para um jogo de push/fold, que é um problema resolvido:

1. **Modelo ICM** (Malmuth-Harville). Com 3 jogadores são 6 permutações de ordem de saída — cálculo exato e
   instantâneo, sem aproximações.
2. **Tabela de prémios por multiplicador.** A maioria dos multiplicadores é winner-take-all; os altos pagam 2.º
   e 3.º. A estrutura muda o ICM por completo, por isso entra como dado, não como constante.
3. **Equilíbrio push/fold.** O protótipo 3-max já modela stacks por lugar, árvore BTN → SB → BB, sete resultados terminais, side pots, cartas físicas e ICM nos pagamentos. Emite frequências para seis decisões e NashConv amostrado. Como CFR em multiplayer não garante equilíbrio de Nash, o resultado indica explicitamente amostragem, seed e tolerância; não é rotulado como solver exato.
4. **Gerador de charts** + comparação das tuas decisões reais (do Bloco 1) contra o equilíbrio: onde é que
   realmente perdes EV, medido, em vez de por intuição.

O ponto 4 é o que transforma isto numa ferramenta de estudo em vez de uma tabela: liga as tuas mãos jogadas ao
que era correto, e ordena os erros por EV perdido.

---

## Bloco 3 — Modelo de poker local

### O que não fazer, e porquê

**Fine-tuning com livros de poker não produz boas decisões.** O [PokerBench](https://arxiv.org/abs/2501.08328),
já citado no README, mede exatamente isso. A razão é estrutural: os livros ensinam conceitos em prosa, enquanto
jogar bem é frequências e EV. Um modelo treinado em prosa produz conselhos confiantes, plausíveis e medíocres —
o pior resultado possível numa ferramenta de estudo, porque não consegues detetar quando está errado.

### Arquitetura

1. **Os números vêm sempre do motor determinístico.** Já é a disciplina de [assistant.py](lugon_poker/assistant.py),
   cujo prompt proíbe inventar probabilidades. Mantém-se, agora com ICM e Nash do Bloco 2 a alimentá-lo.

2. **RAG sobre a biblioteca**, com citação de livro e página. Melhor que fine-tuning para conhecimento:
   verificável, atualizável, e vês de onde veio a afirmação. Encaixa na restrição de zero dependências do
   projeto — embeddings via `/api/embeddings` do Ollama (`nomic-embed-text` ou `bge-m3`) por HTTP com `urllib`,
   vetores guardados como blobs em SQLite, similaridade de cosseno em Python puro. Para alguns milhares de
   chunks o desempenho chega bem.

3. **Fine-tune só para a forma**, se chegar a valer a pena: português, JSON fiável, vocabulário consistente.
   LoRA sobre um modelo 7–8B. Sai do mundo stdlib (torch/peft) e por isso fica isolado num projeto à parte,
   fora do pacote `lugon_poker`.

4. **Dados de treino: as tuas mãos, não os livros.** Pares (situação → resposta correta do motor + explicação),
   gerados a partir do Bloco 1 e do Bloco 2. Ensina formato e estilo de raciocínio ancorados em verdade
   calculada, em vez de prosa.

5. **Avaliação antes de treinar.** Conjunto de spots reservado com resposta conhecida (do ICM/Nash), e medição
   de acerto antes e depois. Sem isto, fine-tuning é um ato de fé — e é a etapa que quase toda a gente salta.

Nota sobre os livros: uso pessoal num índice local é uma coisa; distribuir o modelo ou o índice treinado sobre
eles é outra. Fica local.

---

## Ordem de execução

| | Bloco | Desbloqueia | Depende de |
|---|---|---|---|
| 1.º | Parser + importador de hand histories | Tudo o resto | Um ficheiro .txt real |
| 2.º | Esquema, stats por profundidade, pool + encolhimento | HUD e dados de treino | Bloco 1 |
| 3.º | ICM + push/fold 3-max (protótipo amostrado) | Charts por stack/prémio; comparação com mãos reais depende do Bloco 1 | Nada |
| 4.º | HUD | — | Bloco 1 |
| 5.º | RAG sobre livros | Coach de revisão | Biblioteca em PDF/EPUB |
| 6.º | Fine-tune (opcional) | — | Blocos 2 e 3, e uma avaliação a funcionar |

O núcleo ICM/3-max está implementado, incluindo amostragem estratificada e validação cross-fit independente. Há grelhas com 845 labels para 50/30/20, 70/30/0 e winner-take-all, todas com stacks iguais de 6–15 BB e sem ante. O modelo 50/30/20 atual foi treinado com quatro seeds e 1 000 deals por classe; o arquivo v1 usa duas seeds e 500 deals por classe. A v2 reduziu o MAE leave-one-stack-out de 0,0392 para 0,0256 e o MAE externo médio de 0,0907 para 0,0681 em dois holdouts independentes.

O Assistente agora pode invocar diretamente o solver ICM 3-max a partir do painel manual **Solver ICM**. O resultado estruturado entra no contexto do LLM; a ação final fica limitada à estratégia do solver e ao conjunto de ações legais, enquanto o LLM explica as frequências. A primeira versão cobre seis nós pré-flop, 50/30/20, blinds 0,5/1 BB e sem ante; stacks iniciais são introduzidas manualmente em BB. O cálculo corre em segundo plano e usa cache por configuração. A integração não altera o solver nem liga o modelo de regressão v3 ao Assistente.

Nos dois holdouts, a v2 teve NashConv plug-in médio de 0,00096 contra 0,00112 do solver médio; o ganho cross-fit foi −0,00013 contra +0,00007. Estes ganhos são pequenos e próximos do ruído. O erro externo de frequência continua abaixo do desacordo entre seeds, que ronda 0,12. Os modelos continuam experimentais e não estão ligados ao Assistente nem substituem o solver.

Também há protótipos separados para stacks desiguais. A v2 foi treinada em 23 vetores e comparada em dois holdouts independentes: seis permutações de 8/10/12 BB e seis de 9/10/11 BB. O MAE ficou entre 0,0811 e 0,0848, abaixo do desacordo entre seeds (0,1317–0,1327); o ganho cross-fit ficou próximo do solver médio. As interações entre stacks não venceram consistentemente nos dois holdouts.

A candidata v3 acrescentou seis vetores centrais, totalizando 29 vetores e 4 901 labels. O MAE leave-one-vector-out melhorou para 0,0435 com interações, contra 0,0534 do baseline constante. No holdout externo de 8/10/12 BB, o MAE ficou em 0,0815 com interações, ligeiramente acima dos 0,0811 da v2; o ganho cross-fit médio foi 0,000414 contra 0,000517 do solver em M=3 000. No segundo holdout externo, as seis permutações de 7/10/13 BB deram MAE 0,09131 no aditivo e 0,09210 com interações, abaixo do desacordo entre seeds (0,13398), mas o ganho cross-fit ficou acima do solver: 0,000694 e 0,000674 contra 0,000506. A média dos dois holdouts também fica acima do solver nas duas versões. As interações não melhoraram o MAE nesse segundo holdout. v2/v3 continuam experimentais e desligadas do Assistente.

## Próximo passo concreto

Validar a integração Solver→LLM nos seis nós, confirmar que a estratégia exibida corresponde ao estado manual e medir latência com o LLM local. Depois, alargar a entrada automática de stacks/ação a capturas reais com validação estrita, parametrizar payouts e ante, e avaliar decisões com dados independentes. Manter o preditor de regressão v3 experimental enquanto o ganho estratégico médio continuar pior que o solver.
