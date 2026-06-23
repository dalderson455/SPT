"""Generate a trace-level photobleaching dataset and save it as .npz.
Examples:
  python simulate_traces.py --n-traces 20000 --name val --seed 1
  python simulate_traces.py --config config.yaml --n-traces 50000 --name train
"""

import argparse
import os
import numpy as np
import yaml

from trace_simulator import DEFAULT_CFG, simulate_traces, save_dataset


def load_cfg(path):
    cfg = dict(DEFAULT_CFG)
    if path and os.path.exists(path):
        with open(path) as f:
            full = yaml.safe_load(f) or {}
        cfg.update(full.get("trace_simulator", {}))  # only this block; ignore the rest
    return cfg


def main():
    p = argparse.ArgumentParser(description="Simulate a photobleaching step-counting dataset.")
    p.add_argument("--config", type=str, default=None, help="YAML with a trace_simulator block")
    p.add_argument("--n-traces", type=int, default=20000)
    p.add_argument("--out-dir", type=str, default="TraceData")
    p.add_argument("--name", type=str, default="traces")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Simulating {args.n_traces} traces over {cfg['n_frames']} frames "
          f"(N in [{cfg['n_min']}, {cfg['n_max']}], seed={args.seed})...")
    data = simulate_traces(args.n_traces, cfg, rng)

    out_path = os.path.join(args.out_dir, f"{args.name}.npz")
    save_dataset(out_path, data)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB)")
    print(f"  traces {data['traces'].shape}  "
          f"N_total in [{data['n_total'].min()}, {data['n_total'].max()}]  "
          f"mean detected-able N_mature {data['n_mature'].mean():.1f}")


if __name__ == "__main__":
    main()
