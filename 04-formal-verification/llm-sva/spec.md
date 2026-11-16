Arbiter spec (4 requesters, req[3:0] in, gnt[3:0] out, synchronous active-high reset rst)
S1. At most one grant is high in any cycle.
S2. A requester is granted only in a cycle in which it is requesting.
S3. If any requester is requesting, some requester is granted in that cycle (work-conserving).
S4. Round-robin: a requester granted in one cycle has the lowest priority in the next cycle; if it requests again together with any other requester, the other requester wins.
S5. Fairness: a requester that keeps its request asserted is granted within 4 cycles.
