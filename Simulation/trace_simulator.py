"""
Trace-level forward model for single-molecule photobleaching

Dependencies: numpy, pyyaml (yaml only in the CLI loader).
"""

from __future__ import annotations
import json
import numpy as np

ACTIVE, OFF, BLEACHED = 0, 1, 2

DEFAULT_CFG = {
    # assembly size prior (the label)
    "n_min": 1,
    "n_max": 64,                 # covers the real population (median ~32, tail ~95)
    "n_prior": "uniform",        # "uniform" or "loguniform"

    # acquisition (frames in the tracked window)
    "n_frames": 200,

    # photophysics (per-frame transition probabilities), bleach rate from data
    "p_bleach": 0.0022,          # ACTIVE -> BLEACHED (irreversible); lifetime ~455 frames
    "p_blink_off": 0.005,        # ACTIVE -> OFF (reversible)
    "p_blink_on": 0.30,          # OFF -> ACTIVE
    "p_bleach_from_off": 0.0,    # OFF -> BLEACHED (dark molecules ~don't bleach)

    # brightness: per-fluorophore intensity in col5 units (the single-molecule step)
    "i_single": 700.0,           # mean single-fluorophore step
    "brightness_cv": 0.30,       # 0 = identical; >0 = Gamma-distributed step sizes

    # maturation / labelling efficiency
    # Default 1.0: pipeline is calibrated to Nup96-mEGFP, which recovers ~32
    # reproducibly, so the effective visible fraction is 1 within this system.
    # A fixed sub-1.0 factor is an unmeasurable per-experiment constant. Knob
    # retained for robustness sweeps only (train at 1.0, test at <1.0).
    "p_mature": 1.0,

    # noise model: shot-like (variance proportional to signal) + small floor.
    # var(observed) = noise_gain * max(signal, 0) + sigma_add^2
    "noise_gain": 5400.0,        # variance per unit signal (col5 units); per-bin estimate std/sqrt(S)~73
    "sigma_add": 600.0,          # additive floor (noise on the bleached baseline)
    "allow_negative": True,      # background-subtracted intensity can dip below 0

    "store_latent": True,
}


def _sample_counts(n_traces, cfg, rng):
    lo, hi = int(cfg["n_min"]), int(cfg["n_max"])
    if cfg["n_prior"] == "loguniform":
        u = rng.uniform(np.log(lo), np.log(hi + 1), size=n_traces)
        n = np.clip(np.floor(np.exp(u)).astype(int), lo, hi)
    else:
        n = rng.integers(lo, hi + 1, size=n_traces)
    return n.astype(np.int16)


def _sample_brightness(shape, cfg, rng):
    mean, cv = float(cfg["i_single"]), float(cfg["brightness_cv"])
    if cv <= 0:
        return np.full(shape, mean, dtype=np.float32)
    k = 1.0 / (cv ** 2)
    theta = mean * (cv ** 2)
    return rng.gamma(k, theta, size=shape).astype(np.float32)


def apply_noise(signal, cfg, rng):
    """Shot-like noise: variance proportional to signal, plus an additive floor.

    var = noise_gain * max(signal, 0) + sigma_add^2. This reproduces the measured
    scaling (std/sqrt(signal) ~ const), so per-step detectability degrades with N
    as it does in the real data, rather than being a hand-set fraction.
    """
    s = np.clip(signal, 0, None)
    var = float(cfg["noise_gain"]) * s + float(cfg["sigma_add"]) ** 2
    obs = signal + rng.normal(0.0, np.sqrt(var)).astype(np.float32)
    if not cfg.get("allow_negative", True):
        obs = np.clip(obs, 0, None)
    return obs.astype(np.float32)


def simulate_traces(n_traces, cfg=None, rng=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rng = np.random.default_rng() if rng is None else rng

    n_frames = int(cfg["n_frames"])
    n_total = _sample_counts(n_traces, cfg, rng)
    N_max = int(cfg["n_max"])

    col = np.arange(N_max)[None, :]
    valid = col < n_total[:, None]
    mature = valid & (rng.random((n_traces, N_max)) < float(cfg["p_mature"]))
    n_mature = mature.sum(1).astype(np.int16)

    brightness = _sample_brightness((n_traces, N_max), cfg, rng)
    brightness[~mature] = 0.0

    state = np.full((n_traces, N_max), ACTIVE, dtype=np.int8)
    movable = mature

    signal = np.zeros((n_traces, n_frames), dtype=np.float32)   # clean intensity
    staircase = np.zeros((n_traces, n_frames), dtype=np.int16)  # true emitting count

    pb, poff = float(cfg["p_bleach"]), float(cfg["p_blink_off"])
    pon, pboff = float(cfg["p_blink_on"]), float(cfg["p_bleach_from_off"])

    for t in range(n_frames):
        emitting = movable & (state == ACTIVE)
        signal[:, t] = (brightness * emitting).sum(1)
        staircase[:, t] = emitting.sum(1)

        prev = state.copy()
        is_active = movable & (prev == ACTIVE)
        is_off = movable & (prev == OFF)

        u = rng.random((n_traces, N_max))
        state[is_active & (u < pb)] = BLEACHED
        state[is_active & (u >= pb) & (u < pb + poff)] = OFF

        v = rng.random((n_traces, N_max))
        state[is_off & (v < pon)] = ACTIVE
        if pboff > 0:
            state[is_off & (v >= pon) & (v < pon + pboff)] = BLEACHED

    traces = apply_noise(signal, cfg, rng)

    out = {"traces": traces, "n_total": n_total, "n_mature": n_mature}
    if cfg.get("store_latent", True):
        out.update({"clean_signal": signal, "staircase": staircase,
                    "brightness": brightness, "mature_mask": mature})
    out["params"] = cfg
    return out


def save_dataset(path, data):
    arrays = {k: v for k, v in data.items() if k != "params"}
    arrays["params_json"] = np.array(json.dumps(data.get("params", {})))
    np.savez_compressed(path, **arrays)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files if k != "params_json"}
        if "params_json" in f.files:
            data["params"] = json.loads(str(f["params_json"]))
    return data


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = simulate_traces(2000, None, rng)
    assert d["traces"].shape == (2000, 200)
    assert not np.isnan(d["traces"]).any()
    assert (d["staircase"][:, 0] == d["n_mature"]).all()
    assert (d["n_mature"] <= d["n_total"]).all()
    d2 = simulate_traces(500, {"p_blink_off": 0.0, "p_blink_on": 0.0}, rng)
    assert (np.diff(d2["staircase"], axis=1) <= 0).all()
    print("smoke test passed:", {k: v.shape for k, v in d.items() if hasattr(v, "shape")})
