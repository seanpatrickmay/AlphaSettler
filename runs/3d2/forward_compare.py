"""Plan 10 Task 4: two forward_dump.py outputs (A: the encoder v3 build and net, B: v4 and the
upgrade) by the parity rule, |a - b| <= 1e-5 x max(1, |a|) on every output. Usage: forward_compare.py A B"""
import math
import sys
from array import array

a, b = sys.argv[1], sys.argv[2]
worst = 0.0
for mode in ("batch", "single"):
    for key in ("policy", "value", "aux"):
        x, y = array("f"), array("f")
        with open(f"{a}-{key}-{mode}.bin", "rb") as f:
            x.frombytes(f.read())
        with open(f"{b}-{key}-{mode}.bin", "rb") as f:
            y.frombytes(f.read())
        assert len(x) == len(y) > 0 and all(map(math.isfinite, x)) and all(map(math.isfinite, y)), (mode, key)
        err = max(abs(p - q) / max(1.0, abs(p)) for p, q in zip(x, y))
        same = sum(p == q for p, q in zip(x, y))
        worst = max(worst, err)
        print(f"{mode} {key}: {len(x)} outputs, max relative difference {err:.3g}, bit-equal {same} of {len(x)}")
print(f"worst {worst:.3g}: {'pass' if worst <= 1e-5 else 'FAIL'} (parity rule 1e-5)")
