# Contributing

Thank you for your interest. This repo has one rule above all others: **a number is only accepted with the code and seed that produce it.**

## Ways to contribute

1. **Submit a result.** A new compression method, a better RL recipe, or the same experiment on new hardware (an M-series Mac, an iPhone, a Jetson, an FPGA). Open a *New result* issue first, then a pull request that adds:
   - the script (or a flag on an existing one) and the exact command,
   - the seed(s) and hardware,
   - the raw output file in `results/`,
   - a row in [LEADERBOARD.md](LEADERBOARD.md).
2. **Take an open challenge.** See [docs/first-issues.md](docs/first-issues.md). Comment on the issue before starting, so two people do not duplicate work.
3. **Find a bug in a claim.** If a number in an article or README does not reproduce, that is the most valuable issue you can open. Use the *Bug or failed reproduction* template.

## Standards for results

- Report at least 3 seeds for anything trained, or say clearly that it is one seed.
- Use the held-out evaluation the folder already uses, so rows are comparable.
- Say what was **not** measured (for example "logic checked in emulation, not timed on GPU").
- Timings: give the hardware, the software versions, the median of at least 100 runs, and whether other work was running.

## Code style

- Plain Python, PyTorch and NumPy. Keep each experiment runnable on a CPU in minutes where possible.
- Add or extend a test in `tests/` for any correctness property your change relies on. CI runs `python -m pytest -q tests` on every push.

## Credit

Every merged result is credited in the leaderboard and, when it is written up, in the article or post that uses it.
