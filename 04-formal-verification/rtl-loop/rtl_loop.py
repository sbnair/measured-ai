"""Neurosymbolic RTL loop: an LLM writes the RTL, a formal tool decides whether to accept it.

The LLM never sees or edits the checker. It only receives the spec, its own RTL,
and the tool's verdict with a counterexample trace.
"""
import pathlib
import re
import shutil
import subprocess
import sys

SPEC = pathlib.Path("spec.md").read_text()
SBY = """[tasks]
prove
cover
[options]
prove: mode prove
cover: mode cover
cover: depth 12
[engines]
prove: abc pdr
cover: smtbmc yices
[script]
read_verilog -formal fifo.v
read_verilog -formal -sv fifo_check.sv
read_verilog -formal -sv top.sv
prep -top top
[files]
fifo.v
fifo_check.sv
top.sv
"""
TRACE_SIGNALS = ["rst", "in_valid", "in_ready", "in_data", "out_valid", "out_ready", "out_data"]


def run_formal(rtl_text, workdir="loop_work"):
    """Prove and cover the candidate RTL. Returns (ok, report)."""
    work = pathlib.Path(workdir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    (work / "fifo.v").write_text(rtl_text)
    for f in ("fifo_check.sv", "top.sv"):
        shutil.copy(f, work / f)
    (work / "job.sby").write_text(SBY)
    out = subprocess.run(["sby", "-f", "job.sby"], cwd=work, capture_output=True, text=True).stdout

    if "ERROR" in out and "DONE (PASS" not in out:
        errors = [l.split("base: ", 1)[-1] for l in out.splitlines() if "ERROR" in l][:5]
        return False, "The RTL does not compile:\n" + "\n".join(errors)
    m = re.search(r"failed assertion top\.chk\.(\w+) at .*? step (\d+)", out)
    if m:
        name, step = m.group(1), int(m.group(2))
        table = trace_table(work / "job_prove/engine_0/trace.vcd", step)
        return False, (f"Formal proof FAILED: assertion {name} is violated at cycle {step}.\n"
                       f"Counterexample (values at each cycle, top-level ports):\n{table}")
    if "Unreached cover statement" in out:
        return False, "Some cover statements are unreachable; the design cannot reach a required state."
    if out.count("DONE (PASS") == 2:
        return True, "All assertions proved for every reachable state; all covers reached."
    return False, "Unexpected tool output:\n" + out[-1500:]


def trace_table(vcd_path, last_step):
    """Turn a SymbiYosys VCD into a small cycle-by-cycle table of the FIFO's ports."""
    ids, values, rows, t = {}, {}, {}, None
    scope = []
    for line in pathlib.Path(vcd_path).read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "$scope":
            scope.append(tok[2])
        elif tok[0] == "$upscope":
            scope.pop()
        elif tok[0] == "$var" and scope == ["top", "dut"] and tok[4] in TRACE_SIGNALS:
            ids[tok[3]] = tok[4]
        elif line.startswith("#"):
            if t is not None and t % 10 == 0:
                rows[t // 10] = dict(values)
            t = int(line[1:])
        elif line[0] in "01" and line[1:] in ids:
            values[ids[line[1:]]] = int(line[0])
        elif line[0] == "b" and tok[1] in ids:
            values[ids[tok[1]]] = int(tok[0][1:], 2) if set(tok[0][1:]) <= {"0", "1"} else 0
    if t is not None and t % 10 == 0:
        rows[t // 10] = dict(values)
    head = "cycle | " + " | ".join(TRACE_SIGNALS)
    lines = [head, "-" * len(head)]
    for step in range(last_step + 1):
        v = rows.get(step, {})
        lines.append(f"{step:5} | " + " | ".join(str(v.get(s, "?")) for s in TRACE_SIGNALS))
    return "\n".join(lines)


def repair_prompt(rtl_text, report):
    return (f"You wrote this Verilog for the spec below. A formal tool rejected it.\n\n"
            f"SPEC:\n{SPEC}\nYOUR RTL:\n{rtl_text}\nTOOL REPORT:\n{report}\n\n"
            "Explain which spec item the trace violates, then return the complete corrected "
            "module in one ```verilog block. Do not change the interface.")


def ask_llm(prompt):
    """Call an LLM. Shown for completeness; this step was not run for the article."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=4000,
                                 messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    return re.search(r"```verilog\n(.*?)```", text, re.S).group(1)


def loop(rtl_text, max_rounds=5):
    for round_no in range(1, max_rounds + 1):
        ok, report = run_formal(rtl_text)
        print(f"round {round_no}: {'ACCEPTED' if ok else 'REJECTED'}")
        if ok:
            return rtl_text
        rtl_text = ask_llm(repair_prompt(rtl_text, report))
    raise RuntimeError("no verified RTL after max_rounds")


if __name__ == "__main__":
    # Verification-only mode: check one candidate and print what the LLM would receive.
    candidate = pathlib.Path(sys.argv[1]).read_text()
    ok, report = run_formal(candidate)
    print("ACCEPTED" if ok else "REJECTED")
    print(report)
