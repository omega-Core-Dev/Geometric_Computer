// ============================================================================
// Modulo: gncx_gmmu_rocc.v
// Projeto: GNC-X (Geometric Navigational Core)
// Autor Original: Zaqueu Ribeiro da Costa (omega-Core-Dev)
// Funcao: Coprocessador RoCC (Rocket Custom Coprocessor) RISC-V 64-bit
//         Decodifica instrucoes ISA geometricas:
//           - custom0 (2'b00): geo.addr rd, (k, d, R) -> retorna frame_idx O(1)
//           - custom1 (2'b01): geo.ld   rd, (mode, idx) -> leitura vetorial + PIM
//           - custom2 (2'b10): geo.rot  R_new -> rotaciona visao em 0 ciclos de copia
//           - custom3 (2'b11): geo.st   (k, d), data -> escrita via crivo com Domain Key
// Padrao: IEEE 1364-2001 / SystemVerilog compativel
// ============================================================================
`timescale 1ns / 1ps

module gncx_gmmu_rocc (
    input  wire         clk,
    input  wire         rst_n,

    // Interface de Comando RoCC Simplificada
    input  wire         cmd_valid,
    input  wire [1:0]   cmd_opcode,     // 00=geo.addr, 01=geo.ld, 10=geo.rot, 11=geo.st
    input  wire [1:0]   cmd_k,          // Escala L0..L3
    input  wire [2:0]   cmd_d,          // Indice diagonal / alvo
    input  wire [1:0]   cmd_read_mode,  // 00=Linha, 01=Coluna, 10=Diagonal
    input  wire [1:0]   cmd_rot_imm,    // Novo R para geo.rot (00=1, 01=3, 10=5, 11=7)
    input  wire [3:0]   cmd_domain_id,  // Domain Tag do processo ativo
    input  wire         cmd_force_alloc,
    input  wire [31:0]  cmd_wr_data,

    // Respostas Imediatas do Coprocessador
    output reg  [1:0]   active_rot_reg, // Registrador arquitetural R ativo
    output wire [5:0]   resp_frame_idx, // Endereco fisico traduzido O(1)
    output wire [2:0]   resp_phys_i,
    output wire [2:0]   resp_phys_j,
    output wire [255:0] resp_vector,    // Linha/Coluna/Diagonal lida em 1 ciclo
    output wire [31:0]  resp_pim_sum,   // Reducao PIM X-Folding
    output wire [3:0]   resp_zero_mask, // Pares cancelados no X-Folding
    output wire         resp_domain_fault
);

    // Registrador de Rotacao Arquitetural (atualizado por geo.rot em 1 borda / 0 memcpy)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_rot_reg <= 2'b00; // R = 1 por padrao
        end else if (cmd_valid && (cmd_opcode == 2'b10)) begin
            active_rot_reg <= cmd_rot_imm;
        end
    end

    // Selecao combinacional de R: se a instrucao atual for geo.addr, usa cmd_rot_imm;
    // caso contrario, usa o registrador arquitetural active_rot_reg
    wire [1:0] eff_rot = (cmd_valid && cmd_opcode == 2'b00) ? cmd_rot_imm : active_rot_reg;

    // Unidade Phi acoplada ao estagio de decodificacao
    gncx_phi_unit u_phi_coupler (
        .k(cmd_k),
        .d(cmd_d),
        .rot_sel(eff_rot),
        .phys_i(resp_phys_i),
        .phys_j(resp_phys_j),
        .sieve_D(),
        .frame_idx(resp_frame_idx)
    );

    // Habilitacao de escrita via geo.st (escreve diretamente na coordenada traduzida por Phi)
    wire ram_we = cmd_valid && (cmd_opcode == 2'b11);

    // Matriz Fisica 8x8 SRAM Tri-Modal com PIM
    gncx_matrix_ram_8x8 #(
        .DATA_WIDTH(32)
    ) u_sram_grid (
        .clk(clk),
        .rst_n(rst_n),
        .we(ram_we),
        .wr_row(resp_phys_i),
        .wr_col(resp_phys_j),
        .domain_id(cmd_domain_id),
        .force_alloc(cmd_force_alloc),
        .wr_data(cmd_wr_data),
        .domain_fault(resp_domain_fault),
        .read_mode(cmd_read_mode),
        .k_level(cmd_k),
        .target_idx(cmd_d),
        .rot_sel(active_rot_reg),
        .rd_vector_flat(resp_vector),
        .rd_word0(),
        .rd_word1(),
        .rd_word2(),
        .rd_word3(),
        .rd_word4(),
        .rd_word5(),
        .rd_word6(),
        .rd_word7(),
        .pim_zero_fold_mask(resp_zero_mask),
        .pim_folded_sum(resp_pim_sum)
    );

endmodule
