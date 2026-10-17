@echo off
setlocal
echo ============================================================================
echo   GNC-X Buildtool: Compilando gncx_core.dll (C++20) e Rodando Suite Python
echo ============================================================================

set "LLVM_BIN=C:\Users\orieb\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20260908-ucrt-x86_64\bin"
if exist "%LLVM_BIN%\clang++.exe" (
    set "PATH=%LLVM_BIN%;%PATH%"
)

if not exist "%~dp0bin" mkdir "%~dp0bin"

echo [1/2] Compilando C++20 Shared Library (bin\gncx_core.dll)...
clang++ -shared -std=c++20 -O3 -march=native -static ^
    -I"%~dp0..\02_gncx_vm_simulator\include" ^
    "%~dp0csrc\gncx_torch_c_api.cpp" ^
    -o "%~dp0bin\gncx_core.dll"

if %ERRORLEVEL% neq 0 (
    echo [ERRO] Falha na compilacao de gncx_core.dll
    exit /b %ERRORLEVEL%
)

echo [2/2] Executando Suite de Validacao Python/PyTorch (tests\benchmark_torch_gncx.py)...
python "%~dp0tests\benchmark_torch_gncx.py"
endlocal
