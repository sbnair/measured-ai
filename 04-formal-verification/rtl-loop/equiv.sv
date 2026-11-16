// equiv.sv - sequential equivalence: the optimised FIFO must match the accepted one at every port.
module equiv (input wire clk, input wire rst, input wire in_valid, input wire [7:0] in_data,
              input wire out_ready);
    wire       a_in_ready, a_out_valid, b_in_ready, b_out_valid;
    wire [7:0] a_out_data, b_out_data;
    reg init = 1'b1;
    always @(posedge clk) init <= 1'b0;
    always @* if (init) assume (rst);

    fifo     ref_ (.clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(a_in_ready), .in_data(in_data),
                   .out_valid(a_out_valid), .out_ready(out_ready), .out_data(a_out_data));
    fifo_opt opt  (.clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(b_in_ready), .in_data(in_data),
                   .out_valid(b_out_valid), .out_ready(out_ready), .out_data(b_out_data));

    always @* if (!rst) begin
        e_in_ready:  assert (a_in_ready  == b_in_ready);
        e_out_valid: assert (a_out_valid == b_out_valid);
        e_out_data:  assert (!a_out_valid || a_out_data == b_out_data);  // data is a don't-care when empty
    end
endmodule
