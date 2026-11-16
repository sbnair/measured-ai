module top (input wire clk, input wire rst, input wire [3:0] req);
    wire [3:0] gnt;
    // start every trace in reset
    reg init = 1'b1;
    always @(posedge clk) init <= 1'b0;
    always @* if (init) assume (rst);
    rr_arbiter       dut   (.clk(clk), .rst(rst), .req(req), .gnt(gnt));
    rr_arbiter_props props (.clk(clk), .rst(rst), .req(req), .gnt(gnt));
endmodule
