// fifo_v2.v - same interface; count register removed, full/empty from 3-bit pointers.
module fifo_opt (
    input  wire       clk,
    input  wire       rst,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output wire       out_valid,
    input  wire       out_ready,
    output wire [7:0] out_data
);
    reg [7:0] mem [0:3];
    reg [2:0] wr_ptr, rd_ptr;               // bit 2 is the wrap bit

    wire empty = (wr_ptr == rd_ptr);
    wire full  = (wr_ptr[1:0] == rd_ptr[1:0]) && (wr_ptr[2] != rd_ptr[2]);

    assign out_valid = !empty;
    assign out_data  = mem[rd_ptr[1:0]];
    assign in_ready  = !full || out_ready;

    wire do_write = in_valid && in_ready;
    wire do_read  = out_valid && out_ready;

    always @(posedge clk) begin
        if (rst) begin
            wr_ptr <= 3'd0;
            rd_ptr <= 3'd0;
        end else begin
            if (do_write) begin
                mem[wr_ptr[1:0]] <= in_data;
                wr_ptr           <= wr_ptr + 3'd1;
            end
            if (do_read)
                rd_ptr <= rd_ptr + 3'd1;
        end
    end
endmodule
