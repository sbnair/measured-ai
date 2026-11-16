// 4-requester round-robin arbiter: combinational grant, registered priority pointer.
module rr_arbiter (
    input  wire       clk,
    input  wire       rst,      // synchronous, active high
    input  wire [3:0] req,
    output reg  [3:0] gnt
);
    reg [1:0] ptr;              // requester with the highest priority this cycle
    reg [1:0] gidx;             // index of the granted requester
    reg [1:0] idx;
    reg       found;
    integer   k;

    always @* begin
        gnt   = 4'b0000;
        gidx  = 2'd0;
        found = 1'b0;
        for (k = 0; k < 4; k = k + 1) begin
            idx = ptr + k[1:0];
            if (!found && req[idx]) begin
                gnt[idx] = 1'b1;
                gidx     = idx;
                found    = 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (rst)      ptr <= 2'd0;
        else if (found) ptr <= gidx + 2'd1;  // the winner gets the lowest priority next
    end
endmodule
