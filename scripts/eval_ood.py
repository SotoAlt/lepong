"""Evaluate state head under in-distribution vs out-of-distribution state trajectories.

Generates two datasets on the same renderer (PongWorld.render(128)):
  - In-distribution: AI-tracked paddles (matches training data)
  - Out-of-distribution: random-policy paddles (ball visits novel states)

For each frame, runs encoder -> predictor -> state_head and measures
absolute prediction error on ball_x, ball_y, ball_vx, ball_vy, pad_l, pad_r.

Reports per-dim median / p95 / max error, in-dist vs OOD, and the OOD drop.

Usage:
    python scripts/eval_ood.py \
        --checkpoint checkpoints/lepong_statehead_frozen.pt \
        --episodes 20 --steps 100 --output results/eval_ood.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE
from model.pong_world import PongWorld


COURT_H = 0.6
BALL_SPEED_MAX = 0.025
STATE_NAMES = [
    "ball_x", "ball_y", "ball_vx", "ball_vy",
    "pad_l", "pad_r", "score_l", "score_r", "speed", "rally",
]
# Which state dims we actually care about for reporting (skip score/speed/rally)
REPORT_DIMS = [0, 1, 2, 3, 4, 5]


def collect_dataset(
    policy: str,
    n_episodes: int,
    steps_per_ep: int,
    frameskip: int,
    base_seed: int,
    image_size: int = 128,
) -> dict:
    """Collect a dataset of (frames, states) using either AI-tracked or random paddles."""
    env = PongWorld()
    rng = np.random.default_rng(base_seed)

    frames = []
    states = []
    ep_len = []

    for ep in range(n_episodes):
        env.reset(seed=base_seed + ep)
        noise = rng.uniform(0.0, 0.15)
        ep_frame_count = 0

        for step in range(steps_per_ep):
            if policy == "ai":
                action = env.ai_action(noise=noise)
            elif policy == "random":
                action = [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))]
            elif policy == "frozen":
                action = [0.0, 0.0]
            else:
                raise ValueError(f"unknown policy: {policy}")

            for _ in range(frameskip):
                env.step(action)

            frames.append(env.render(image_size))
            states.append(env.get_state())
            ep_frame_count += 1
        ep_len.append(ep_frame_count)

    return {
        "frames": np.stack(frames, axis=0),
        "states": np.stack(states, axis=0).astype(np.float32),
        "ep_len": ep_len,
        "policy": policy,
    }


def load_frozen_model(checkpoint_path: str, device: torch.device) -> tuple[JEPAPool, torch.Tensor, torch.Tensor]:
    """Load checkpoint and defensively freeze everything except state head."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", 10)
    model = JEPAPool(embed_dim=ckpt.get("embed_dim", EMBED_DIM), state_dim=state_dim)
    model.load_state_dict(ckpt["model"])

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False
    for p in model.action_encoder.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = False
    for p in model.pred_projector.parameters():
        p.requires_grad = False
    for p in model.sigreg.parameters():
        p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable <= 5000, f"Unexpected trainable param count: {n_trainable}"

    model.eval()
    model = model.to(device)
    state_mean = ckpt.get("state_mean", torch.zeros(state_dim)).to(device)
    state_std = ckpt.get("state_std", torch.ones(state_dim)).to(device)
    return model, state_mean, state_std


def evaluate_dataset(
    dataset: dict,
    model: JEPAPool,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run the state head on every frame and compute per-dim absolute errors."""
    frames = dataset["frames"]
    states = dataset["states"]
    N = len(frames)

    errs = {name: [] for name in STATE_NAMES}
    states_norm = (torch.from_numpy(states).to(device) - state_mean) / state_std

    with torch.no_grad():
        BATCH = 32
        i_start = HISTORY_SIZE
        i_end = N - 1
        indices = list(range(i_start, i_end))
        for batch_start in range(0, len(indices), BATCH):
            batch_idx = indices[batch_start : batch_start + BATCH]
            if not batch_idx:
                continue

            ctx_frames_list = []
            target_states_list = []
            for i in batch_idx:
                ctx = frames[i - HISTORY_SIZE + 1 : i + 1]
                tgt = states[i + 1]
                ctx_frames_list.append(ctx)
                target_states_list.append(tgt)

            ctx_np = np.stack(ctx_frames_list, axis=0)
            ctx_tensor = (
                torch.from_numpy(ctx_np)
                .float()
                .permute(0, 1, 4, 2, 3)
                / 255.0
            ).to(device)

            emb = model.encode(ctx_tensor)
            B = emb.shape[0]
            action = torch.zeros(B, HISTORY_SIZE, 2, device=device)
            action_emb = model.action_encoder(
                action.reshape(B * HISTORY_SIZE, 2)
            ).reshape(B, HISTORY_SIZE, -1)
            pred = model.predict_next(emb, action_emb)
            s_norm = model.state_head(pred)
            s_denorm = s_norm * state_std + state_mean

            tgt_norm = torch.from_numpy(np.stack(target_states_list)).to(device)
            tgt_denorm = tgt_norm

            abs_err = torch.abs(s_denorm - tgt_denorm).cpu().numpy()
            for dim_idx, name in enumerate(STATE_NAMES):
                errs[name].extend(abs_err[:, dim_idx].tolist())

    return {name: np.array(errs[name]) for name in STATE_NAMES}


def summary_stats(arr: np.ndarray) -> dict:
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        "p99": float(np.percentile(arr, 99)) if len(arr) else float("nan"),
        "max": float(arr.max()) if len(arr) else float("nan"),
    }


def report(label: str, errs: dict) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print(f"{'dim':10s}  {'n':>6s}  {'median':>8s}  {'p95':>8s}  {'p99':>8s}  {'max':>8s}", flush=True)
    out = {}
    for i in REPORT_DIMS:
        name = STATE_NAMES[i]
        s = summary_stats(errs[name])
        out[name] = s
        print(
            f"{name:10s}  {s['n']:6d}  "
            f"{s['median']:8.4f}  {s['p95']:8.4f}  {s['p99']:8.4f}  {s['max']:8.4f}",
            flush=True,
        )
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default="checkpoints/lepong_statehead_frozen.pt")
    pa.add_argument("--episodes", type=int, default=20)
    pa.add_argument("--steps", type=int, default=100)
    pa.add_argument("--frameskip", type=int, default=5)
    pa.add_argument("--seed", type=int, default=2349867)
    pa.add_argument("--output", default="results/eval_ood.json")
    pa.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = pa.parse_args()

    device = torch.device(args.device)
    print(f"=== State distribution OOD evaluation ===", flush=True)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    print(f"  episodes:   {args.episodes}", flush=True)
    print(f"  steps/ep:   {args.steps}", flush=True)
    print(f"  frameskip:  {args.frameskip}", flush=True)
    print(f"  device:     {device}", flush=True)
    print(f"  seed base:  {args.seed}", flush=True)

    # 1. Generate two datasets
    t0 = time.time()
    print("\nGenerating in-distribution dataset (AI-tracked paddles)...", flush=True)
    indist = collect_dataset(
        policy="ai",
        n_episodes=args.episodes,
        steps_per_ep=args.steps,
        frameskip=args.frameskip,
        base_seed=args.seed,
    )
    print(f"  {len(indist['frames'])} frames in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print("\nGenerating out-of-distribution dataset (random paddles)...", flush=True)
    ood = collect_dataset(
        policy="random",
        n_episodes=args.episodes,
        steps_per_ep=args.steps,
        frameskip=args.frameskip,
        base_seed=args.seed + 10000,
    )
    print(f"  {len(ood['frames'])} frames in {time.time() - t1:.1f}s", flush=True)

    # 2. Load the frozen state-head checkpoint
    print("\nLoading frozen state-head checkpoint...", flush=True)
    model, state_mean, state_std = load_frozen_model(args.checkpoint, device)
    print("  loaded, everything frozen except Linear(192, 10) state head", flush=True)

    # 3. Evaluate state head on both datasets
    t2 = time.time()
    print("\nEvaluating on in-distribution dataset...", flush=True)
    indist_errs = evaluate_dataset(indist, model, state_mean, state_std, device)
    print(f"  {len(indist_errs['ball_y'])} predictions in {time.time() - t2:.1f}s", flush=True)

    t3 = time.time()
    print("\nEvaluating on OOD dataset...", flush=True)
    ood_errs = evaluate_dataset(ood, model, state_mean, state_std, device)
    print(f"  {len(ood_errs['ball_y'])} predictions in {time.time() - t3:.1f}s", flush=True)

    # 4. Report side-by-side
    indist_report = report("IN-DISTRIBUTION (AI-tracked paddles)", indist_errs)
    ood_report = report("OUT-OF-DISTRIBUTION (random paddles)", ood_errs)

    # 5. Compute drops
    print("\n=== OOD DROP (OOD / IN-DIST) ===", flush=True)
    print(f"{'dim':10s}  {'in-dist median':>15s}  {'ood median':>12s}  {'drop':>8s}", flush=True)
    drops = {}
    for i in REPORT_DIMS:
        name = STATE_NAMES[i]
        ind_m = indist_report[name]["median"]
        ood_m = ood_report[name]["median"]
        drop = (ood_m / ind_m - 1) * 100 if ind_m > 1e-6 else float("inf")
        drops[name] = drop
        print(
            f"{name:10s}  {ind_m:15.4f}  {ood_m:12.4f}  {drop:+7.1f}%",
            flush=True,
        )

    # 6. Save results
    results = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "steps_per_ep": args.steps,
        "frameskip": args.frameskip,
        "seed": args.seed,
        "indist": indist_report,
        "ood": ood_report,
        "ood_drops_pct": drops,
        "methodology": (
            "In-distribution = AI-tracked paddles + uniform(0, 0.15) noise "
            "(matches training distribution). OOD = uniform random paddle "
            "actions. Same renderer (PongWorld.render(128)). State head: "
            "frozen backbone + Linear(192, 10) trained on AI-tracked data only."
        ),
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved results -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
