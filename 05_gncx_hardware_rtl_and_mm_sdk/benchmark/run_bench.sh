#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo ">>> Compilando Suite de Benchmark C++20 (Passos 1 & 2 do Roadmap GNC-X)..."
g++ -O3 -std=c++20 -march=native -I../include bench_buddy_vs_gncx.cpp -o bench_buddy_vs_gncx
./bench_buddy_vs_gncx
