# AGENTS.md

## Independent Verification

**After completing any non-trivial code change, always launch a separate verification subagent with a fresh context.**

The verification subagent must be independent of the implementation process:
- Do **not** reuse the current conversation or reasoning.
- Give the verifier only the repository state and a short description of the intended change.
- Treat the verifier as an independent reviewer.

The verification subagent should:

1. Review the implementation for correctness.
2. Look for logic bugs, edge cases, and regressions.
3. Verify that the implementation actually satisfies the requested behavior.
4. Check for style, maintainability, and consistency with the surrounding codebase.
5. Run or suggest appropriate tests, and verify that existing tests still pass.
6. Report any issues it finds, even if they contradict the original implementation.

The implementing agent should **not** consider its work complete until the verification subagent has finished and any significant issues have been addressed.

> Always prefer an independent verifier with a fresh context over self-review, since the implementation agent is prone to confirmation bias.

## Sanity check

When implementing a fix of a problem, make sure that the fix does resolve the problem. Run the test using the same configuration that encountered the problem.
