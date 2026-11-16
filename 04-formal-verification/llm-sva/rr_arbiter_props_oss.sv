// rr_arbiter assertions lowered to the subset that open-source Yosys accepts:
// immediate assertions, $past, and small monitor registers instead of SVA sequences.
module rr_arbiter_props (
    input wire       clk,
    input wire       rst,
    input wire [3:0] req,
    input wire [3:0] gnt
);
    reg past_valid = 1'b0;
    always @(posedge clk) past_valid <= 1'b1;

    // S1-S3: combinational checks, disabled during reset
    always @* begin
        if (!rst) begin
            p_onehot:          assert ($onehot0(gnt));
            p_grant_valid:     assert ((gnt & ~req) == 4'b0000);
            p_work_conserving: assert (!(|req) || (|gnt));
        end
    end

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : g_req
            // S4: gnt[i] ##1 (req[i] && other request) |-> !gnt[i]
            always @(posedge clk) begin
                if (past_valid && !rst && !$past(rst) && $past(gnt[i])
                    && req[i] && |(req & ~(4'b0001 << i)))
                    assert (!gnt[i]);
            end

            // S5: not ((req[i] && !gnt[i]) [*4]); wait = consecutive earlier waiting cycles
            reg [1:0] wait_cnt = 2'd0;
            always @(posedge clk) begin
                if (rst || !(req[i] && !gnt[i])) wait_cnt <= 2'd0;
                else if (wait_cnt != 2'd3)       wait_cnt <= wait_cnt + 2'd1;
            end
            always @* begin
                if (!rst)
                    assert (!(req[i] && !gnt[i] && wait_cnt == 2'd3));
            end

            // Coverage
            always @* begin
                if (!rst) begin
                    cover (gnt[i]);
                    cover (req == 4'b1111 && gnt[i]);
                    cover (req[i] && !gnt[i] && wait_cnt == 2'd2);   // a 3-cycle wait happens
                end
            end
            always @(posedge clk) begin                              // S4's trigger happens
                if (past_valid && !rst && !$past(rst))
                    cover ($past(gnt[i]) && req[i] && |(req & ~(4'b0001 << i)));
            end
        end
    endgenerate
endmodule
