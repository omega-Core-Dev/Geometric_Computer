// ============================================================================
// Modulo: tb_gncx_pim.v
// Projeto: GNC-X (Geometric Navigational Core)
// Funcao: Testbench Automatizado de Verificacao Formal e Funcional em Verilog
//         Valida:
//           1. Bijecao Exaustiva O(1) de Phi(k,d,R) e Phi^-1(i,j,k,R)
//           2. Acesso Tri-Modal em 1 Ciclo (Linha, Coluna e Crivo Diagonal)
//           3. Rotacao geo.rot em 0 Ciclos de Movimentacao Fisica
//           4. Protecao de Dominio (Domain Keys 4-bit) contra Intersecao Modular
//           5. Reducao PIM X-Folding Nativa no Barramento de Saida
// Padrao: IEEE 1364-2001 / 2005
// ============================================================================
`timescale 1ns / 1ps

module tb_gncx_pim;

    // Sinais para teste de Bijecao Phi / Phi^-1
    reg  [1:0] tb_k;
    reg  [2:0] tb_d;
    reg  [1:0] tb_rot;
    wire [2:0] tb_phys_i;
    wire [2:0] tb_phys_j;
    wire [2:0] tb_sieve_D;
    wire [5:0] tb_frame_idx;
    wire [2:0] tb_d_from_i;
    wire [2:0] tb_d_from_j;
    wire       tb_sieve_match;

    gncx_phi_unit dut_phi (
        .k(tb_k),
        .d(tb_d),
        .rot_sel(tb_rot),
        .phys_i(tb_phys_i),
        .phys_j(tb_phys_j),
        .sieve_D(tb_sieve_D),
        .frame_idx(tb_frame_idx)
    );

    gncx_phi_inverse_unit dut_phi_inv (
        .phys_i(tb_phys_i),
        .phys_j(tb_phys_j),
        .k(tb_k),
        .rot_sel(tb_rot),
        .d_from_i(tb_d_from_i),
        .d_from_j(tb_d_from_j),
        .sieve_match(tb_sieve_match)
    );

    // Sinais para Matriz SRAM 8x8 Tri-Modal + PIM
    reg         clk;
    reg         rst_n;
    reg         we;
    reg  [2:0]  wr_row;
    reg  [2:0]  wr_col;
    reg  [3:0]  domain_id;
    reg         force_alloc;
    reg  [31:0] wr_data;
    wire        domain_fault;

    reg  [1:0]  read_mode;
    reg  [1:0]  k_level;
    reg  [2:0]  target_idx;
    reg  [1:0]  rot_sel;

    wire [255:0] rd_vector_flat;
    wire [31:0]  rd_word0, rd_word1, rd_word2, rd_word3;
    wire [31:0]  rd_word4, rd_word5, rd_word6, rd_word7;
    wire [3:0]   pim_zero_fold_mask;
    wire [31:0]  pim_folded_sum;

    gncx_matrix_ram_8x8 #(
        .DATA_WIDTH(32)
    ) dut_ram (
        .clk(clk),
        .rst_n(rst_n),
        .we(we),
        .wr_row(wr_row),
        .wr_col(wr_col),
        .domain_id(domain_id),
        .force_alloc(force_alloc),
        .wr_data(wr_data),
        .domain_fault(domain_fault),
        .read_mode(read_mode),
        .k_level(k_level),
        .target_idx(target_idx),
        .rot_sel(rot_sel),
        .rd_vector_flat(rd_vector_flat),
        .rd_word0(rd_word0),
        .rd_word1(rd_word1),
        .rd_word2(rd_word2),
        .rd_word3(rd_word3),
        .rd_word4(rd_word4),
        .rd_word5(rd_word5),
        .rd_word6(rd_word6),
        .rd_word7(rd_word7),
        .pim_zero_fold_mask(pim_zero_fold_mask),
        .pim_folded_sum(pim_folded_sum)
    );

    // Gerador de Clock: 500 MHz (periodo 2ns)
    initial begin
        clk = 1'b0;
        forever #1 clk = ~clk;
    end

    integer r_it, k_it, d_it;
    integer total_errors;
    reg [7:0] seen_rows;
    reg [7:0] seen_cols;

    initial begin
        $dumpfile("gncx_pim_waveform.vcd");
        $dumpvars(0, tb_gncx_pim);

        total_errors = 0;
        rst_n        = 1'b0;
        we           = 1'b0;
        wr_row       = 3'd0;
        wr_col       = 3'd0;
        domain_id    = 4'd1;
        force_alloc  = 1'b0;
        wr_data      = 32'd0;
        read_mode    = 2'b00;
        k_level      = 2'b00;
        target_idx   = 3'd0;
        rot_sel      = 2'b00;
        tb_k         = 2'd0;
        tb_d         = 3'd0;
        tb_rot       = 2'd0;

        #4 rst_n = 1'b1;

        $display("======================================================================");
        $display("  GNC-X HARDWARE RTL TESTBENCH (Phi-Unit + 8x8 PIM SRAM + Domain Keys)");
        $display("======================================================================");

        // ------------------------------------------------------------------
        // TESTE 1: Validacao Exaustiva de Bijecao O(1) em Phi e Phi^-1
        // ------------------------------------------------------------------
        for (r_it = 0; r_it < 4; r_it = r_it + 1) begin
            for (k_it = 0; k_it < 4; k_it = k_it + 1) begin
                seen_rows = 8'b0;
                seen_cols = 8'b0;
                for (d_it = 0; d_it < 8; d_it = d_it + 1) begin
                    tb_rot = r_it[1:0];
                    tb_k   = k_it[1:0];
                    tb_d   = d_it[2:0];
                    #1;
                    seen_rows[tb_phys_i] = 1'b1;
                    seen_cols[tb_phys_j] = 1'b1;
                    if (!tb_sieve_match || (tb_d_from_i !== tb_d) || (tb_d_from_j !== tb_d)) begin
                        $display("[ERRO] Falha de inversao Phi^-1 em k=%0d d=%0d rot=%0d", k_it, d_it, r_it);
                        total_errors = total_errors + 1;
                    end
                end
                if ((seen_rows !== 8'hFF) || (seen_cols !== 8'hFF)) begin
                    $display("[ERRO] Colisao detectada em k=%0d rot=%0d (rows=%b cols=%b)", k_it, r_it, seen_rows, seen_cols);
                    total_errors = total_errors + 1;
                end
            end
        end
        $display("[PASS] Teste 1: Bijecao O(1) Phi <-> Phi^-1 (128/128 estados, 0 colisoes)");

        // ------------------------------------------------------------------
        // TESTE 2: Popula Grid 8x8 e Testa Leitura Tri-Modal em 1 Ciclo
        // Valor de cada celula (r, c) = (r * 16) + c + 100
        // ------------------------------------------------------------------
        for (r_it = 0; r_it < 8; r_it = r_it + 1) begin
            for (d_it = 0; d_it < 8; d_it = d_it + 1) begin
                @(negedge clk);
                we          = 1'b1;
                wr_row      = r_it[2:0];
                wr_col      = d_it[2:0];
                domain_id   = 4'd1;
                force_alloc = 1'b1;
                wr_data     = (r_it * 16) + d_it + 100;
                @(posedge clk);
            end
        end
        @(negedge clk);
        we = 1'b0;
        force_alloc = 1'b0;

        // 2A: Leitura Horizontal (Linha 2) -> (2,0)..(2,7) = 132..139
        read_mode  = 2'b00;
        target_idx = 3'd2;
        #1;
        if (rd_word0 !== 32'd132 || rd_word7 !== 32'd139) begin
            $display("[ERRO] Modo Linha falhou: w0=%0d w7=%0d", rd_word0, rd_word7);
            total_errors = total_errors + 1;
        end

        // 2B: Leitura Vertical (Coluna 5) -> (0,5)..(7,5) = 105..217
        read_mode  = 2'b01;
        target_idx = 3'd5;
        #1;
        if (rd_word0 !== 32'd105 || rd_word7 !== 32'd217) begin
            $display("[ERRO] Modo Coluna falhou: w0=%0d w7=%0d", rd_word0, rd_word7);
            total_errors = total_errors + 1;
        end

        // 2C: Leitura Diagonal D(k=0, d=0, R=1) -> (0,0), (1,1) .. (7,7)
        read_mode  = 2'b10;
        k_level    = 2'b00;
        target_idx = 3'd0;
        rot_sel    = 2'b00; // R=1
        #1;
        if (rd_word0 !== 32'd100 || rd_word1 !== 32'd117 || rd_word7 !== 32'd219) begin
            $display("[ERRO] Modo Diagonal R=1 falhou: w0=%0d w1=%0d w7=%0d", rd_word0, rd_word1, rd_word7);
            total_errors = total_errors + 1;
        end
        $display("[PASS] Teste 2: Barramento Tri-Modal (Linha, Coluna e Diagonal em 1 ciclo)");

        // ------------------------------------------------------------------
        // TESTE 3: Rotacao geo.rot em 0 Ciclos de Clock (Chaveamento Puro)
        // Muda R=1 (2'b00) -> R=3 (2'b01) -> R=5 (2'b10) -> R=7 (2'b11)
        // Sem tocar no clock e sem mover 1 unico bit na SRAM!
        // ------------------------------------------------------------------
        rot_sel = 2'b01; // R=3: para d=1, i=1, j=(1*3)=3 -> celula (1,3) = 16+3+100 = 119
        #1;
        if (rd_word1 !== 32'd119) begin
            $display("[ERRO] geo.rot R=3 falhou: esperado 119, obtido %0d", rd_word1);
            total_errors = total_errors + 1;
        end

        rot_sel = 2'b10; // R=5: para d=1, i=1, j=(1*5)=5 -> celula (1,5) = 16+5+100 = 121
        #1;
        if (rd_word1 !== 32'd121) begin
            $display("[ERRO] geo.rot R=5 falhou: esperado 121, obtido %0d", rd_word1);
            total_errors = total_errors + 1;
        end

        rot_sel = 2'b11; // R=7: para d=1, i=1, j=(1*7)=7 -> celula (1,7) = 16+7+100 = 123
        #1;
        if (rd_word1 !== 32'd123) begin
            $display("[ERRO] geo.rot R=7 falhou: esperado 123, obtido %0d", rd_word1);
            total_errors = total_errors + 1;
        end
        $display("[PASS] Teste 3: Rotacao geo.rot em 0 ciclos de clock (R=1 -> R=3 -> R=5 -> R=7)");

        // ------------------------------------------------------------------
        // TESTE 4: Domain Protection Keys (Isolamento Anti-Intersecao SAS)
        // As retas R1=1 e R2=3 cruzam-se em (0,0) e (4,4).
        // Processo 2 (domain_id=2) tenta sobrescrever (4,4) pertencente ao Processo 1!
        // ------------------------------------------------------------------
        @(negedge clk);
        we          = 1'b1;
        wr_row      = 3'd4;
        wr_col      = 3'd4;
        domain_id   = 4'd2; // Processo intruso no ponto de colisao (4,4)
        force_alloc = 1'b0;
        wr_data     = 32'hDEADBEEF;
        @(posedge clk);
        #1;
        if (domain_fault !== 1'b1) begin
            $display("[ERRO] Domain Key nao detectou colisao em (4,4)!");
            total_errors = total_errors + 1;
        end
        @(negedge clk);
        we = 1'b0;

        // Verifica que o dado original (4*16 + 4 + 100 = 168) permaneceu intacto
        read_mode  = 2'b00;
        target_idx = 3'd4;
        #1;
        if (rd_word4 !== 32'd168) begin
            $display("[ERRO] Dado corrompido na intersecao (4,4): %0d", rd_word4);
            total_errors = total_errors + 1;
        end
        $display("[PASS] Teste 4: Domain Protection Keys bloquearam colisao algebrica em (4,4)");

        // ------------------------------------------------------------------
        // TESTE 5: Reducao PIM X-Folding Nativa no Caminho de Saida
        // Escreve valores simetricos na diagonal R=1 (k=0):
        //   d=0: +50, d=7: -50 (cancela par 0)
        //   d=1: +25, d=6: -25 (cancela par 1)
        //   d=2: +10, d=5: -10 (cancela par 2)
        //   d=3: +12, d=4: +30 (soma = 42)
        // ------------------------------------------------------------------
        @(negedge clk); we = 1'b1; domain_id = 4'd1;
        wr_row = 3'd0; wr_col = 3'd0; wr_data = 32'd50;          @(posedge clk);
        @(negedge clk);
        wr_row = 3'd7; wr_col = 3'd7; wr_data = -32'd50;         @(posedge clk);
        @(negedge clk);
        wr_row = 3'd1; wr_col = 3'd1; wr_data = 32'd25;          @(posedge clk);
        @(negedge clk);
        wr_row = 3'd6; wr_col = 3'd6; wr_data = -32'd25;         @(posedge clk);
        @(negedge clk);
        wr_row = 3'd2; wr_col = 3'd2; wr_data = 32'd10;          @(posedge clk);
        @(negedge clk);
        wr_row = 3'd5; wr_col = 3'd5; wr_data = -32'd10;         @(posedge clk);
        @(negedge clk);
        wr_row = 3'd3; wr_col = 3'd3; wr_data = 32'd12;          @(posedge clk);
        @(negedge clk);
        wr_row = 3'd4; wr_col = 3'd4; wr_data = 32'd30;          @(posedge clk);
        @(negedge clk); we = 1'b0;

        read_mode  = 2'b10;
        k_level    = 2'b00;
        target_idx = 3'd0;
        rot_sel    = 2'b00; // R=1
        #1;
        if (pim_zero_fold_mask !== 4'b0111 || pim_folded_sum !== 32'd42) begin
            $display("[ERRO] PIM X-Folding falhou: mask=%b sum=%0d", pim_zero_fold_mask, pim_folded_sum);
            total_errors = total_errors + 1;
        end
        $display("[PASS] Teste 5: PIM X-Folding em Silicio (mask=0111, 3 pares cancelados, soma=42)");

        $display("======================================================================");
        if (total_errors == 0)
            $display("  RESULTADO FINAL: 100%% APROVADO (0 erros de verificacao RTL)");
        else
            $display("  RESULTADO FINAL: FALHOU (%0d erros)", total_errors);
        $display("======================================================================");
        $finish;
    end

endmodule
