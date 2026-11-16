// fifo.v - 4-entry, 8-bit valid/ready FIFO generated from spec.md (F1-F6).
module fifo (
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
    reg [1:0] wr_ptr, rd_ptr;
    reg [2:0] count;

    wire full  = (count == 3'd4);
    wire empty = (count == 3'd0);

    assign out_valid = !empty;
    assign out_data  = mem[rd_ptr];
    assign in_ready  = !full || out_ready;          // F4: full but reading this cycle

    wire do_write = in_valid && in_ready;
    wire do_read  = out_valid && out_ready;

    always @(posedge clk) begin
        if (rst) begin
            wr_ptr <= 2'd0;
            rd_ptr <= 2'd0;
            count  <= 3'd0;
        end else begin
            if (do_write) begin
                mem[wr_ptr] <= in_data;
                wr_ptr      <= wr_ptr + 2'd1;
            end
            if (do_read)
                rd_ptr <= rd_ptr + 2'd1;
            case ({do_write, do_read})
                2'b10:   count <= count + 3'd1;
                2'b01:   count <= count - 3'd1;
                default: count <= count;
            endcase
        end
    end
endmodule
