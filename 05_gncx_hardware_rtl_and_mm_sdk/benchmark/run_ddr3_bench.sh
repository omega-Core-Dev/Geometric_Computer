#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo ">>> [1/2] Compilando Biblioteca de Desenvolvimento GNC-X SDK (../lib/libgncx_mm.so)..."
mkdir -p ../lib
g++ -O3 -std=c++20 -march=native -shared -fPIC -I../include ../lib/gncx_sdk_lib.cpp -o ../lib/libgncx_mm.so -lpthread

echo ">>> [2/2] Compilando e Executando Benchmark Fisico de Silicio DDR3 + L1d (__rdtscp)..."
g++ -O3 -std=c++20 -march=native -I../include bench_ddr3_silicon.cpp -o bench_ddr3_silicon
./bench_ddr3_silicon
