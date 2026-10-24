// ============================================================================
// Modulo: gncx_matrix_ram_8x8.v
// Projeto: GNC-X (Geometric Navigational Core)
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao: Matriz Fisica de SRAM 8x8 com Barramento Tri-Modal (Linha, Coluna,
//         Diagonal D(k,d)), Rotacao geo.rot em 0 Ciclos, Domain Keys de 4 bits
//         e Reducao PIM X-Folding em Silicio.
// Padrao: IEEE 1364-2001 / SystemVerilog compativel
// ============================================================================
`timescale 1ns / 1ps

module gncx_matrix_ram_8x8 #(
    parameter DATA_WIDTH = 32
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // Porta de Escrita com Protecao de Dominio (Domain Protection Keys - 4 bits)
    input  wire                      we,
    input  wire [2:0]                wr_row,
    input  wire [2:0]                wr_col,
    input  wire [3:0]                domain_id,       // Tag do processo atual (SAS Isolation)
    input  wire                      force_alloc,     // 1 = vincula celula ao domain_id atual
    input  wire [DATA_WIDTH-1:0]     wr_data,
    output reg                       domain_fault,    // 1 = violacao de intersecao algebrica

    // Porta de Leitura Tri-Modal + Rotacao Geometrica (geo.rot)
    input  wire [1:0]                read_mode,       // 00=Linha, 01=Coluna, 10=Diagonal D(k,d)
    input  wire [1:0]                k_level,         // Nivel de escala L0..L3 (0..3)
    input  wire [2:0]                target_idx,      // Indice base (linha, coluna ou d_offset)
    input  wire [1:0]                rot_sel,         // Fator R: 00=1, 01=3, 10=5, 11=7

    // Saidas Paralelas do Vetor de 8 Palavras (Compativel Verilog-2001 e SystemVerilog)
    output wire [8*DATA_WIDTH-1:0]   rd_vector_flat,  // Barramento paralelo de 256 bits (8x32)
    output reg  [DATA_WIDTH-1:0]     rd_word0,
    output reg  [DATA_WIDTH-1:0]     rd_word1,
    output reg  [DATA_WIDTH-1:0]     rd_word2,
    output reg  [DATA_WIDTH-1:0]     rd_word3,
    output reg  [DATA_WIDTH-1:0]     rd_word4,
    output reg  [DATA_WIDTH-1:0]     rd_word5,
    output reg  [DATA_WIDTH-1:0]     rd_word6,
    output reg  [DATA_WIDTH-1:0]     rd_word7,

    // Saidas PIM (Processing-In-Memory) X-Folding Nativo no Caminho de Saida
    output wire [3:0]                pim_zero_fold_mask, // Bit p=1 se par simetrico (p, 7-p) se anula
    output wire [DATA_WIDTH-1:0]     pim_folded_sum      // Reducao em arvore somadora pos-X-Folding
);

    // Grid fisico de 64 celulas SRAM de 32 bits + Tags de Dominio de 4 bits
    reg [DATA_WIDTH-1:0] mem_grid    [0:7][0:7];
    reg [3:0]            domain_grid [0:7][0:7];
    reg                  owned_grid  [0:7][0:7];

    integer r, c;

    // Escrita sincrona com verificacao de Domain Key (Anti-Intersecao em SAS)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            domain_fault <= 1'b0;
            for (r = 0; r < 8; r = r + 1) begin
                for (c = 0; c < 8; c = c + 1) begin
                    mem_grid[r][c]    <= {DATA_WIDTH{1'b0}};
                    domain_grid[r][c] <= 4'd0;
                    owned_grid[r][c]  <= 1'b0;
                end
            end
        end else if (we) begin
            if (force_alloc || !owned_grid[wr_row][wr_col] || (domain_grid[wr_row][wr_col] == domain_id)) begin
                mem_grid[wr_row][wr_col]    <= wr_data;
                domain_grid[wr_row][wr_col] <= domain_id;
                owned_grid[wr_row][wr_col]  <= 1'b1;
                domain_fault                <= 1'b0;
            end else begin
                // Colisao de intersecao modular entre processos com R distintos bloqueada!
                domain_fault <= 1'b1;
            end
        end else begin
            domain_fault <= 1'b0;
        end
    end

    // Instanciacao de 8 Unidades Phi em paralelo para varredura diagonal em 1 ciclo
    genvar idx;
    wire [2:0] diag_i [0:7];
    wire [2:0] diag_j [0:7];

    generate
        for (idx = 0; idx < 8; idx = idx + 1) begin : GEN_PHI
            wire [2:0] d_scan = target_idx + idx[2:0];
            gncx_phi_unit phi_inst (
                .k(k_level),
                .d(d_scan),
                .rot_sel(rot_sel),
                .phys_i(diag_i[idx]),
                .phys_j(diag_j[idx]),
                .sieve_D(),
                .frame_idx()
            );
        end
    endgenerate

    // Multiplexador Tri-Modal Combinacional (0 ciclos para geo.rot)
    reg [DATA_WIDTH-1:0] rd_vec [0:7];

    always @(*) begin
        case (read_mode)
            2'b00: begin // Modo Linha Horizontal: mem_grid[target_idx][0..7]
                for (c = 0; c < 8; c = c + 1)
                    rd_vec[c] = mem_grid[target_idx][c];
            end
            2'b01: begin // Modo Coluna Vertical (GLA Slicing): mem_grid[0..7][target_idx]
                for (r = 0; r < 8; r = r + 1)
                    rd_vec[r] = mem_grid[r][target_idx];
            end
            2'b10: begin // Modo Diagonal Geometrica D(k,d): celulas computadas via Phi
                for (r = 0; r < 8; r = r + 1)
                    rd_vec[r] = mem_grid[diag_i[r]][diag_j[r]];
            end
            default: begin
                for (c = 0; c < 8; c = c + 1)
                    rd_vec[c] = {DATA_WIDTH{1'b0}};
            end
        endcase

        rd_word0 = rd_vec[0];
        rd_word1 = rd_vec[1];
        rd_word2 = rd_vec[2];
        rd_word3 = rd_vec[3];
        rd_word4 = rd_vec[4];
        rd_word5 = rd_vec[5];
        rd_word6 = rd_vec[6];
        rd_word7 = rd_vec[7];
    end

    assign rd_vector_flat = {
        rd_word7, rd_word6, rd_word5, rd_word4,
        rd_word3, rd_word2, rd_word1, rd_word0
    };

    // ========================================================================
    // Unidade PIM (Processing-In-Memory): X-Folding de 4 Fases no Caminho de Saida
    // Dobramento simetrico: pares (0,7), (1,6), (2,5), (3,4)
    // Se A + B == 0 (complemento de 2) ou XOR simetrico, cancela na malha!
    // ========================================================================
    wire [DATA_WIDTH-1:0] pair_sum0 = rd_word0 + rd_word7;
    wire [DATA_WIDTH-1:0] pair_sum1 = rd_word1 + rd_word6;
    wire [DATA_WIDTH-1:0] pair_sum2 = rd_word2 + rd_word5;
    wire [DATA_WIDTH-1:0] pair_sum3 = rd_word3 + rd_word4;

    assign pim_zero_fold_mask[0] = (pair_sum0 == {DATA_WIDTH{1'b0}});
    assign pim_zero_fold_mask[1] = (pair_sum1 == {DATA_WIDTH{1'b0}});
    assign pim_zero_fold_mask[2] = (pair_sum2 == {DATA_WIDTH{1'b0}});
    assign pim_zero_fold_mask[3] = (pair_sum3 == {DATA_WIDTH{1'b0}});

    assign pim_folded_sum = (pair_sum0 + pair_sum1) + (pair_sum2 + pair_sum3);

endmodule
