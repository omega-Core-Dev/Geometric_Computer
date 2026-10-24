// ============================================================================
// Modulo: gncx_phi_unit.v
// Projeto: GNC-X (Geometric Navigational Core)
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao: Traducao Bijetiva O(1) de GeoAddr(k, d, R) <-> Coordenadas Fisicas (i, j)
// Latencia: 0 ciclos (Logica Combinacional Pura)
// Padrao: IEEE 1364-2001 / SystemVerilog compativel
// ============================================================================
`timescale 1ns / 1ps

module gncx_phi_unit (
    input  wire [1:0] k,          // Nivel de escala L0..L3 (0..3)
    input  wire [2:0] d,          // Indice diagonal (0..7)
    input  wire [1:0] rot_sel,    // Selecao de R: 00=1, 01=3, 10=5, 11=7 (coprimos com 8)
    output reg  [2:0] phys_i,     // Coordenada fisica da linha (0..7)
    output reg  [2:0] phys_j,     // Coordenada fisica da coluna (0..7)
    output wire [2:0] sieve_D,    // Crivo Diagonal Invariante D(k, d) = (d * R) mod 8
    output wire [5:0] frame_idx   // Indice linear fisico direto (0..63)
);

    // Mapeamento dos fatores de rotacao impares coprimos R_{k+1} em (Z/8Z)*
    reg [2:0] R;
    always @(*) begin
        case (rot_sel)
            2'b00: R = 3'd1;
            2'b01: R = 3'd3;
            2'b10: R = 3'd5;
            2'b11: R = 3'd7;
            default: R = 3'd1;
        endcase
    end

    // Calculo do Crivo Diagonal Invariante: D(k, d) = (d * R) mod 8
    wire [5:0] d_mult_R = d * R;
    wire [2:0] D        = d_mult_R[2:0]; // mod 8 equivale aos 3 LSBs
    assign sieve_D      = D;

    // Calculo de i = (d + k * R) mod 8
    wire [5:0] k_mult_R = k * R;
    wire [5:0] i_sum    = {3'b000, d} + k_mult_R;

    // Calculo de j = (D + k) mod 8
    wire [3:0] j_sum    = {1'b0, D} + {2'b00, k};

    always @(*) begin
        phys_i = i_sum[2:0]; // Truncamento mod 8
        phys_j = j_sum[2:0]; // Truncamento mod 8
    end

    // Frame linear = (i * 8) + j
    assign frame_idx = {phys_i, phys_j};

endmodule

// ============================================================================
// Modulo: gncx_phi_inverse_unit.v
// Funcao: Traducao Inversa O(1) Phi^-1(i, j, k, R) -> d e validacao de crivo
// Propriedade Algebrica: Em (Z/8Z)*, R^-1 mod 8 == R para todo R in {1,3,5,7}
// ============================================================================
module gncx_phi_inverse_unit (
    input  wire [2:0] phys_i,       // Linha fisica (0..7)
    input  wire [2:0] phys_j,       // Coluna fisica (0..7)
    input  wire [1:0] k,            // Nivel de escala L0..L3 (0..3)
    input  wire [1:0] rot_sel,      // Selecao de R: 00=1, 01=3, 10=5, 11=7
    output wire [2:0] d_from_i,     // Indice d recuperado via coordenada i
    output wire [2:0] d_from_j,     // Indice d recuperado via coordenada j (inverso modular)
    output wire       sieve_match   // 1 se (phys_i, phys_j) pertence ao crivo D(k, d, R)
);

    reg [2:0] R;
    always @(*) begin
        case (rot_sel)
            2'b00: R = 3'd1;
            2'b01: R = 3'd3;
            2'b10: R = 3'd5;
            2'b11: R = 3'd7;
            default: R = 3'd1;
        endcase
    end

    // Recuperacao via linha: d = (i - k * R) mod 8
    wire [5:0] k_mult_R = k * R;
    assign d_from_i     = phys_i - k_mult_R[2:0];

    // Recuperacao via coluna: D = (j - k) mod 8 => d = (D * R^-1) mod 8
    // Como R^2 == 1 (mod 8) para R impar, R^-1 == R (mod 8)
    wire [2:0] D_rec    = phys_j - {1'b0, k};
    wire [5:0] d_j_mult = D_rec * R;
    assign d_from_j     = d_j_mult[2:0];

    // Ponto fisico pertence a orbita diagonal se d_from_i == d_from_j
    assign sieve_match  = (d_from_i == d_from_j);

endmodule
