# Lugon Poker Lab — Recursos Visuais (`assets/`)

Esta diretoria contém a identidade gráfica e ícones da aplicação.

---

## Ficheiros

- `lugon_poker_lab.png`: Imagem principal em alta resolução (1024×1024 px).
- `lugon_poker_lab_64.png`: Versão redimensionada (64×64 px) otimizada para o cabeçalho da interface gráfica (`companion_gui.py`).
- `lugon_poker_lab.ico`: Ícone do Windows multi-resolução (16×16, 32×32, 48×48, 64×64, 128×128, 256×256) incorporado no executável `LugonPokerLab.exe`.

Para regenerar o ficheiro `.ico` a partir do PNG original, executa o script PowerShell disponível em `tools/`:
```powershell
powershell -ExecutionPolicy Bypass -File tools/build_icon.ps1
```
