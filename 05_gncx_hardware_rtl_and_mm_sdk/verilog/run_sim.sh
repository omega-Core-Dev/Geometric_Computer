#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if command -v iverilog >/dev/null 2>&1 && command -v vvp >/dev/null 2>&1; then
    echo ">>> [Icarus Verilog] Compilando modulos IEEE 1364-2001/2005..."
    iverilog -g2005 -o gncx_pim_sim.vvp \
        gncx_phi_unit.v \
        gncx_matrix_ram_8x8.v \
        gncx_gmmu_rocc.v \
        tb_gncx_pim.v
    vvp gncx_pim_sim.vvp
    echo ""
fi

echo ">>> [GNC-X RTL C++20] Compilando e executando verificador ciclo-exato (1M ops + VCD)..."
g++ -O3 -std=c++20 -march=native sim_gncx_rtl.cpp -o sim_gncx_rtl
./sim_gncx_rtl
