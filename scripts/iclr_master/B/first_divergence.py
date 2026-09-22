"""Where do two runs of the same episode first differ? For each agent trace, report the first differing
character and whether it falls in model output (after 'Thought:' / an action line) or in environment
feedback ('Result:' / 'Objects:' lines), with 200 chars of shared context.
Usage: first_divergence.py <results dir A> <results dir B> <ids>"""
import sys, pathlib
A, B, ids = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3].split(",")
for i in ids:
    for ag in (0, 1):
        fa = A / "traces" / str(ag) / f"trace-episode_{i}_0-{ag}.txt"
        fb = B / "traces" / str(ag) / f"trace-episode_{i}_0-{ag}.txt"
        if not (fa.exists() and fb.exists()):
            print(i, ag, "missing", fa.exists(), fb.exists()); continue
        a, b = fa.read_text(), fb.read_text()
        k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        if a == b:
            print(i, ag, "IDENTICAL", len(a)); continue
        line = a[: k].rsplit("\n", 1)[-1]
        kind = "env feedback" if line.startswith(("Result", "Objects")) else "model output"
        print(f"{i} agent{ag} first diff at char {k} of {len(a)}/{len(b)} in {kind}; line so far: {line[-120:]!r}")
        print("   A:", repr(a[k:k + 80])); print("   B:", repr(b[k:k + 80]))
