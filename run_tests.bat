@echo off
cd /d "%~dp0"
echo ========================================================
echo Running Lugon Poker Lab Test Suite...
echo ========================================================
py -m unittest discover -s tests -v
pause
