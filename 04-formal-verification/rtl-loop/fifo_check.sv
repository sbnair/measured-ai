// fifo_check.sv - black-box checker for spec.md, written by the verification engineer.
// It only looks at the FIFO's ports and keeps its own reference model.
module fifo_check (
    input wire       clk,
    input wire       rst,
    input wire       in_valid,
    input wire       in_ready,
    input wire [7:0] in_data,
    input wire       out_valid,
    input wire       out_ready,
    input wire [7:0] out_data
);
    wire do_write = in_valid && in_ready;
    wire do_read  = out_valid && out_ready;

    // Reference model: how many entries the FIFO should hold.
    reg [2:0] model_count = 3'd0;
    always @(posedge clk)
        if (rst) model_count <= 3'd0;
        else     model_count <= model_count + do_write - do_read;

    always @* if (!rst) begin
        a_no_overflow:  assert (model_count <= 3'd4);                       // F1, F4
        a_out_valid:    assert (out_valid == (model_count != 3'd0));        // F3
        a_ready_space:  assert (model_count == 3'd4 || in_ready);           // F4, not full
        a_ready_full:   assert (model_count != 3'd4 || in_ready == out_ready); // F4, full
    end

    // F5: follow one arbitrarily chosen write through the FIFO.
    (* anyseq *) wire pick;                 // the solver chooses which write to track
    reg       tracking = 1'b0;
    reg [7:0] t_data;
    reg [2:0] t_ahead;                      // entries still in front of the tracked one
    always @(posedge clk) begin
        if (rst) tracking <= 1'b0;
        else if (!tracking) begin
            if (do_write && pick) begin
                tracking <= 1'b1;
                t_data   <= in_data;
                t_ahead  <= model_count - do_read;
            end
        end else if (do_read) begin
            if (t_ahead == 3'd0) tracking <= 1'b0;   // the tracked entry left
            else                 t_ahead  <= t_ahead - 3'd1;
        end
    end
    always @* if (!rst && tracking && t_ahead == 3'd0) begin
        a_in_order: assert (out_valid && out_data == t_data);               // F3, F5
    end

    // Covers: the interesting corners are reachable
    always @* if (!rst) begin
        c_full:           cover (model_count == 3'd4);
        c_full_pass:      cover (model_count == 3'd4 && do_write && do_read);
        c_tracked_out:    cover (tracking && t_ahead == 3'd0 && do_read);
    end
endmodule
