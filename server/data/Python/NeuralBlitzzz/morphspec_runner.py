
import json
from datetime import datetime

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def nbhs512_stub(data_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha512(data_bytes).hexdigest()

def simulate_field(params: dict, size=64, steps=128, seed="osmtriph"):
    import numpy as np
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    F = rng.standard_normal((size, size)) * 0.05
    activ = float(params.get("activator", 1.0))
    inhib  = float(params.get("inhibitor", 0.5))
    alpha  = 0.15 * activ
    beta   = 0.08 * inhib
    K = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    # Use simple manual laplacian to avoid scipy dependency
    for t in range(steps):
        # compute laplacian with wrap boundary
        lap = (
            np.roll(F, 1, 0) + np.roll(F, -1, 0) + np.roll(F, 1, 1) + np.roll(F, -1, 1) - 4*F
        )
        F += alpha * lap - beta * (F**3 - F)
    F = (F - F.mean()) / (F.std() + 1e-9)
    return F

def field_signature(F):
    import numpy as np
    s = 16
    H, W = F.shape
    Hs, Ws = (H//s)*s, (W//s)*s
    Fcrop = F[:Hs, :Ws]
    Fds = Fcrop.reshape(s, Hs//s, s, Ws//s).mean(-1).mean(1)
    data = Fds.round(5).astype("float32").tobytes()
    return nbhs512_stub(data)

def run_morph(spec: dict, seed="osmtriph"):
    RD = spec.get("RD", {})
    steps = int(RD.get("iterations", 256))
    params = dict(activator=RD.get("activator",1.0), inhibitor=RD.get("inhibitor",0.5))
    F = simulate_field(params=params, size=64, steps=steps, seed=seed)
    sig = field_signature(F)
    return {
        "timestamp": now_iso(),
        "params": {**params, "iterations": steps},
        "signature_nbhs512_stub": sig,
        "metrics": {
            "mean": float(F.mean()),
            "std": float(F.std()),
            "min": float(F.min()),
            "max": float(F.max())
        }
    }

if __name__ == "__main__":
    import argparse, sys, os, json
    ap = argparse.ArgumentParser(description="MorphSpec runner (stub)")
    ap.add_argument("--morph", required=True, help="Path to .morph.json")
    ap.add_argument("--seed", default="osmtriph", help="Deterministic seed")
    ap.add_argument("--out", default="morph_result.json")
    args = ap.parse_args()
    with open(args.morph, "r") as f:
        spec = json.load(f)
    res = run_morph(spec, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("Wrote", args.out)
