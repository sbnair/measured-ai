"""Mutation testing for the arbiter assertions.

Each mutant injects one realistic RTL bug. Each property is then proved on its own
against each mutant, so we see exactly which properties catch which bugs.
"""
import pathlib
import re
import shutil
import subprocess

RTL = pathlib.Path("rr_arbiter.v").read_text()
PROPS = pathlib.Path("rr_arbiter_props_oss.sv").read_text().splitlines()
TOP = pathlib.Path("top.sv").read_text()

# property name -> line number of its assert in rr_arbiter_props_oss.sv
PROPERTIES = {"S1 one-hot": 15, "S2 valid": 16, "S3 work-conserving": 17,
              "S4 round-robin": 28, "S5 fairness": 39}

MUTANTS = {
    "M1 pointer never moves":      ("else if (found) ptr <= gidx + 2'd1;", "else if (found) ptr <= ptr;"),
    "M2 winner keeps priority":    ("ptr <= gidx + 2'd1;", "ptr <= gidx;"),
    "M3 pointer skips one":        ("ptr <= gidx + 2'd1;", "ptr <= gidx + 2'd2;"),
    "M4 search starts one late":   ("idx = ptr + k[1:0];", "idx = ptr + k[1:0] + 2'd1;"),
    "M5 multiple grants":          ("if (!found && req[idx]) begin", "if (req[idx]) begin"),
    "M6 grant without request":    ("if (!found && req[idx]) begin", "if (!found && (req[idx] || k == 3)) begin"),
    "M7 last candidate ignored":   ("for (k = 0; k < 4; k = k + 1)", "for (k = 0; k < 3; k = k + 1)"),
    "M8 pointer moves when idle":  ("else if (found) ptr <= gidx + 2'd1;", "else ptr <= gidx + 2'd1;"),
    "M9 reset pointer to 3":       ("if (rst)      ptr <= 2'd0;", "if (rst)      ptr <= 2'd3;"),
}

SBY = """[options]
mode prove
depth 12
[engines]
smtbmc yices
[script]
read_verilog -formal rr_arbiter.v
read_verilog -formal -sv props.sv
read_verilog -formal -sv top.sv
prep -top top
[files]
rr_arbiter.v
props.sv
top.sv
"""


def props_only(line_no):
    """Keep one assert; disable the others by guarding them with a false condition."""
    out = []
    for n, text in enumerate(PROPS, start=1):
        if n in PROPERTIES.values() and n != line_no:
            text = re.sub(r"(\w+:\s*)?assert \(", "if (1'b0) assert (", text)
        out.append(text)
    return "\n".join(out)


def proved(rtl, props):
    work = pathlib.Path("mut_work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    (work / "rr_arbiter.v").write_text(rtl)
    (work / "props.sv").write_text(props)
    (work / "top.sv").write_text(TOP)
    (work / "job.sby").write_text(SBY)
    res = subprocess.run(["sby", "-f", "job.sby"], cwd=work, capture_output=True, text=True)
    if "DONE (PASS" in res.stdout:
        return True
    if "DONE (FAIL" in res.stdout:
        return False
    raise RuntimeError(res.stdout[-2000:])


rows = []
for mname, (old, new) in MUTANTS.items():
    assert RTL.count(old) == 1, mname
    rtl = RTL.replace(old, new)
    caught = [p for p, line in PROPERTIES.items() if not proved(rtl, props_only(line))]
    rows.append((mname, caught))
    print(f"{mname:<28} caught by: {', '.join(caught) if caught else 'NONE (survived)'}")

killed = sum(1 for _, c in rows if c)
print(f"\nmutation score: {killed}/{len(rows)} mutants killed")
