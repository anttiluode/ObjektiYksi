from __future__ import annotations

import argparse
import json
from pathlib import Path

from gate5_multisource_basin import summarize, train


def run(out_dir: Path, seeds: int = 3, attempts: int = 1600):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [train("worst_case_multi_source", seed=s, attempts=attempts) for s in range(seeds)]
    receipt = {
        "gate": "5-long",
        "purpose": "Attack the possibility that Gate 5's sub-rank-1 worst case is merely an insufficient proposal budget.",
        "seeds": seeds,
        "attempts_per_seed": attempts,
        "summary": summarize(runs),
        "runs": [x["result"] for x in runs],
    }
    (out_dir / "gate5_long_confirm.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--attempts", type=int, default=1600)
    args = p.parse_args()
    receipt = run(Path(args.out), seeds=args.seeds, attempts=args.attempts)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
