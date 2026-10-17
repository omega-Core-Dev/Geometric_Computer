@echo off
setlocal
set "LLVM_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20260908-ucrt-x86_64\bin"
set "PATH=%LLVM_BIN%;%PATH%"

if not exist "%~dp0bin" mkdir "%~dp0bin"
echo [GNC-X] Compilando Simulador Virtual (Incubadora C++20) com -O3 -march=native...
clang++.exe -std=c++20 -O3 -march=native -static -Wall -Wextra -I"%~dp0include" "%~dp0src\main.cpp" -o "%~dp0bin\gncx_simulator.exe"
if %ERRORLEVEL% neq 0 (
    echo [ERRO] Falha na compilacao.
    exit /b %ERRORLEVEL%
)

echo [GNC-X] Executando Simulador Virtual...
pushd "%~dp0bin"
"%~dp0bin\gncx_simulator.exe"
popd

