FIFO spec (8-bit data, 4 entries, synchronous active-high reset rst)
F1. Write side: in_valid, in_ready, in_data[7:0]. A write happens in a cycle where in_valid && in_ready.
F2. Read side: out_valid, out_ready, out_data[7:0]. A read happens in a cycle where out_valid && out_ready.
F3. out_valid is high exactly when the FIFO holds at least one entry; out_data is the oldest entry.
F4. in_ready is high when the FIFO is not full. When the FIFO is full, in_ready is high only if a read happens in the same cycle.
F5. Data leaves in the order it arrived, with no loss and no duplication.
F6. Reset empties the FIFO.
