module top (input wire clk, input wire rst, input wire in_valid, input wire [7:0] in_data,
            input wire out_ready);
    wire in_ready, out_valid;
    wire [7:0] out_data;
    reg init = 1'b1;
    always @(posedge clk) init <= 1'b0;
    always @* if (init) assume (rst);
    fifo       dut (.clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
                    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data));
    fifo_check chk (.clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
                    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data));
endmodule
