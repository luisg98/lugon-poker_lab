# Lugon Poker Lab — Exemplos e Fixtures (`examples/`)

Esta diretoria reúne fixtures de teste e demonstrações que permitem experimentar e testar todas as funcionalidades do projeto sem necessidade de mesas reais ou sessões ativas.

---

## 1. Decisão Pré e Pós-Flop (`flop_call.json`)
Exemplo de estado da mesa (`HandState`) com cartas próprias do herói (`Ks`, `Qh`), board (`Kh`, `8c`, `2d`), pot e aposta do adversário.
Para avaliar a decisão matemática:
```powershell
py -m lugon_poker.cli examples/flop_call.json --iterations 20000 --seed 7
```

---

## 2. Históricos de Mãos (`histories/`)
- `sample_hands.json`: Conjunto de mãos completas sequenciais para testes do agregador de estatísticas de jogadores (`profiles.py`).
Para importar e visualizar no HUD local:
```powershell
py -m lugon_poker.profiles_cli import examples/histories/sample_hands.json --db data/profiles.sqlite
py -m lugon_poker.profiles_cli show --db data/profiles.sqlite --player Vitor
```

---

## 3. Visão e Mesas Sintéticas (`vision/`)
- `flop_table.json`: Descrição paramétrica dos elementos da mesa (posições de botões, cartas, stacks e apostas).
- `flop_960.png` e `flop_1280.png`: Renderizações sintéticas bitmap limpas geradas por `synthetic.py`.
Para testar a extração visual:
```powershell
py -m lugon_poker.vision_cli scan examples/vision/flop_960.png --analyze --iterations 10000 --seed 7
```
