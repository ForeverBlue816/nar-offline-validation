"""Probe the Llama-3.1-70B layer-3 down_proj Hessian that Cholesky rejects.

The dump is the damped fp64 matrix exactly as handed to torch.linalg.cholesky:
no NaN, no inf, symmetric, diagonal in [0.039, 645].  A matrix of the form
sum(x x^T) + damp I cannot fail in fp64, so either the matrix is not of that
form or the factorization backend is wrong.  This tries the same call on the
CPU (LAPACK), on the GPU with cuSOLVER and with MAGMA, and reads the smallest
eigenvalue of the leading block the failure names.
"""
import sys, time, json
from pathlib import Path
import torch

path = Path(sys.argv[1]); out = Path(sys.argv[2])
d = torch.load(path, map_location="cpu")
h = d["hessian"]; n = h.shape[0]; order = 6928
res = {"stats": d["stats"]}
print("loaded", h.shape, h.dtype, flush=True)

t = time.time(); ev = torch.linalg.eigvalsh(h[:order, :order])
res["leading_block"] = {"order": order, "min_eig": float(ev[0]), "max_eig": float(ev[-1]),
                        "negative_count": int((ev < 0).sum()), "seconds": time.time() - t}
print(res["leading_block"], flush=True)

def attempt(label, fn):
    t = time.time()
    try:
        fn(); res[label] = {"ok": True, "seconds": time.time() - t}
    except RuntimeError as e:
        res[label] = {"ok": False, "seconds": time.time() - t, "error": str(e)[:160]}
    print(label, res[label], flush=True)

attempt("cpu_fp64", lambda: torch.linalg.cholesky(h))
if torch.cuda.is_available():
    g = h.cuda()
    for lib in ("cusolver", "magma"):
        try:
            torch.backends.cuda.preferred_linalg_library(lib)
        except Exception as e:  # noqa: BLE001
            res[f"gpu_{lib}"] = {"ok": None, "error": f"backend unavailable: {e}"[:160]}; print(res[f"gpu_{lib}"]); continue
        attempt(f"gpu_fp64_{lib}", lambda: torch.linalg.cholesky(g))
        attempt(f"gpu_fp32_{lib}", lambda: torch.linalg.cholesky(g.float()))
    # cholesky_ex reports the failing pivot without raising
    L, info = torch.linalg.cholesky_ex(g)
    res["gpu_cholesky_ex_info"] = int(info)
    print("cholesky_ex info", int(info), flush=True)
out.write_text(json.dumps(res, indent=2) + "\n"); print("written", out)
