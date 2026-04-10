"""CNN JEPA world model for Pong — pixel encoder for 128x128 RGB frames.

Architecture:
  - PixelEncoder: CNN for 128x128 RGB images -> 192-dim embedding
  - ActionEncoder: 2-float [left_dy, right_dy] -> 192-dim embedding
  - ARPredictor: 6-layer causal transformer with AdaLN-zero conditioning
  - StateProbe: MLP that reads structured state from embeddings
  - Optional state_head: Linear(192, 10) auxiliary head for direct state readout

13M parameters total. Runs at ~20fps on CPU.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 192
HISTORY_SIZE = 3
PROJ_HIDDEN = 2048
ACTION_DIM = 2    # [left_paddle_dy, right_paddle_dy]
STATE_DIM = 28    # default probe output dim (overridden for Pong's 10-dim state)


# ---------------------------------------------------------------------------
# SIGReg — spectral implicit Gaussian regularization
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Projector MLP with BatchNorm
# ---------------------------------------------------------------------------

class ProjectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=PROJ_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Custom Attention (heads=16, dim_head=64, inner_dim=1024)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5

    def forward(self, x, causal=False):
        B, T, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, T, 3, self.heads, self.dim_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                              dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, ff_dim, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, causal=False):
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + sh1
        h = self.attn(h, causal=causal)
        x = x + g1 * h
        h = self.norm2(x) * (1 + s2) + sh2
        h = self.ff(h)
        x = x + g2 * h
        return x


# ---------------------------------------------------------------------------
# Pixel Encoder — CNN for 128x128 RGB
# ---------------------------------------------------------------------------

class PixelEncoder(nn.Module):
    """Encode 128x128 RGB image -> hidden_dim vector."""

    def __init__(self, hidden_dim=EMBED_DIM):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),    # 64x64
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, hidden_dim, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(hidden_dim), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, img):
        """img: (B, 3, 128, 128) float [0,1] -> (B, hidden_dim)"""
        x = self.convs(img)
        return self.pool(x).flatten(1)


# ---------------------------------------------------------------------------
# Action Encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    def __init__(self, action_dim=ACTION_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 4 * embed_dim), nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, action):
        return self.net(action)


# ---------------------------------------------------------------------------
# ARPredictor — causal transformer with AdaLN-zero
# ---------------------------------------------------------------------------

class ARPredictor(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, heads=16, dim_head=64, n_layers=6,
                 ff_dim=2048, dropout=0.1, max_len=HISTORY_SIZE):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, heads=heads, dim_head=dim_head,
                             ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cond):
        T = x.size(1)
        x = x + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, cond, causal=True)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full JEPA World Model
# ---------------------------------------------------------------------------

class JEPAPool(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, heads=16, dim_head=64,
                 n_layers=6, ff_dim=2048, state_dim=0):
        """
        state_dim: if > 0, adds an auxiliary state head that maps the predictor's
                   output embedding directly to a structured state vector. Trained
                   jointly with the predictor via L_total = L_embed + lambda * L_state.
                   When state_dim = 0 the model has no state head.
        """
        super().__init__()
        self.encoder = PixelEncoder(hidden_dim=embed_dim)
        self.projector = ProjectorMLP(embed_dim, embed_dim, hidden_dim=PROJ_HIDDEN)
        self.action_encoder = ActionEncoder(embed_dim=embed_dim)
        self.predictor = ARPredictor(embed_dim, heads=heads, dim_head=dim_head,
                                      n_layers=n_layers, ff_dim=ff_dim)
        self.pred_projector = ProjectorMLP(embed_dim, embed_dim, hidden_dim=PROJ_HIDDEN)
        self.sigreg = SIGReg()

        # Auxiliary state head
        self.state_dim = state_dim
        if state_dim > 0:
            self.state_head = nn.Linear(embed_dim, state_dim)
        else:
            self.state_head = None

    def encode(self, frames):
        """frames: (B, T, 3, H, W) float [0,1] -> (B, T, embed_dim)"""
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, *frames.shape[2:])
        h = self.encoder(flat)
        e = self.projector(h)
        return e.reshape(B, T, -1)

    def forward(self, frames, actions, states=None):
        """
        frames: (B, T, 3, H, W) -- T = history_size + 1 = 4
        actions: (B, T, 2) -- per frame
        states: (B, T, state_dim) -- ground-truth state per frame (optional)
        """
        B, T = frames.shape[:2]
        emb = self.encode(frames)

        action_flat = actions.reshape(B * T, -1)
        action_emb = self.action_encoder(action_flat).reshape(B, T, -1)

        ctx_emb = emb[:, :HISTORY_SIZE]
        ctx_action = action_emb[:, :HISTORY_SIZE]

        pred_raw = self.predictor(ctx_emb, ctx_action)
        pred_proj = self.pred_projector(pred_raw.reshape(B * HISTORY_SIZE, -1))
        pred_emb = pred_proj.reshape(B, HISTORY_SIZE, -1)

        tgt_emb = emb[:, 1:HISTORY_SIZE + 1]

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))

        # Auxiliary state head -- predicts NEXT-frame state from the predictor output
        if self.state_head is not None and states is not None:
            state_pred = self.state_head(pred_emb.reshape(B * HISTORY_SIZE, -1))
            state_pred = state_pred.reshape(B, HISTORY_SIZE, -1)
            tgt_states = states[:, 1:HISTORY_SIZE + 1]
            state_loss = (state_pred - tgt_states).pow(2).mean()
            return pred_loss, sigreg_loss, pred_emb, tgt_emb, state_loss, state_pred, tgt_states

        return pred_loss, sigreg_loss, pred_emb, tgt_emb

    def predict_next(self, ctx_emb, ctx_action):
        was_training = self.pred_projector.training
        self.pred_projector.eval()
        pred_raw = self.predictor(ctx_emb, ctx_action)
        pred_proj = self.pred_projector(pred_raw[:, -1])
        if was_training:
            self.pred_projector.train()
        return pred_proj

    def predict_state(self, ctx_emb, ctx_action):
        """Predict next embedding via predictor, then read state via state_head.

        ctx_emb: (B, HISTORY_SIZE, embed_dim)
        ctx_action: (B, HISTORY_SIZE, embed_dim)
        Returns: (B, state_dim) state predictions
        """
        if self.state_head is None:
            raise RuntimeError("predict_state requires state_head -- instantiate JEPAPool with state_dim > 0")
        pred = self.predict_next(ctx_emb, ctx_action)  # (B, embed_dim)
        return self.state_head(pred)                    # (B, state_dim)

    def rollout(self, seed_frames, seed_actions, future_actions, n_steps):
        emb = self.encode(seed_frames)
        B = emb.shape[0]

        action_flat = seed_actions.reshape(B * seed_actions.shape[1], -1)
        action_emb_list = list(self.action_encoder(action_flat).reshape(B, -1, EMBED_DIM).unbind(1))
        emb_list = list(emb.unbind(1))

        predictions = []
        for t in range(n_steps):
            ctx = torch.stack(emb_list[-HISTORY_SIZE:], dim=1)
            ctx_a = torch.stack(action_emb_list[-HISTORY_SIZE:], dim=1)
            pred = self.predict_next(ctx, ctx_a)
            predictions.append(pred)
            emb_list.append(pred)
            fa = self.action_encoder(future_actions[:, t])
            action_emb_list.append(fa)

        return torch.stack(predictions, dim=1)


# ---------------------------------------------------------------------------
# State Probe
# ---------------------------------------------------------------------------

class StateProbe(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, state_dim=STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train JEPA Pong World Model")
    pa.add_argument("--data", required=True)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch-size", type=int, default=64)
    pa.add_argument("--checkpoint", default="checkpoints/jepa_pong.pt")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--probe-epochs", type=int, default=50)
    args = pa.parse_args()

    # Load data
    d = np.load(args.data)
    frames_raw = d["frames"]   # (N, 128, 128, 3) uint8
    states = torch.from_numpy(d["states"]).float()
    actions = torch.from_numpy(d["actions"]).float()
    episodes = d["episodes"]

    # Convert frames to (N, 3, 128, 128) float [0, 1]
    frames = torch.from_numpy(frames_raw).float().permute(0, 3, 1, 2) / 255.0

    # Build 4-frame windows respecting episodes
    windows_f, windows_a, windows_s = [], [], []
    for i in range(HISTORY_SIZE, len(frames)):
        if episodes[i] != episodes[i - HISTORY_SIZE]:
            continue
        windows_f.append(frames[i - HISTORY_SIZE: i + 1])
        windows_a.append(actions[i - HISTORY_SIZE: i + 1])
        windows_s.append(states[i])

    windows_f = torch.stack(windows_f)
    windows_a = torch.stack(windows_a)
    windows_s = torch.stack(windows_s)
    print(f"Dataset: {len(windows_f)} windows from {len(frames)} frames")

    # Device
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # Model
    model = JEPAPool().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"JEPAPool: {n_params:,} params on {device}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup_epochs)
    SIGREG_LAMBDA = 0.09
    bs = args.batch_size

    # =====================================================
    # JEPA Training
    # =====================================================
    print(f"\n=== JEPA Training: {args.epochs} epochs ===")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(windows_f))
        e_pred, e_reg, n = 0.0, 0.0, 0
        model.train()

        for i in range(0, len(windows_f), bs):
            idx = perm[i:i + bs]
            xb = windows_f[idx].to(device)
            ab = windows_a[idx].to(device)

            pred_loss, sigreg_loss, _, _ = model(xb, ab)
            loss = pred_loss + SIGREG_LAMBDA * sigreg_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            e_pred += pred_loss.item() * len(idx)
            e_reg += sigreg_loss.item() * len(idx)
            n += len(idx)

        if epoch < warmup_epochs:
            for pg in opt.param_groups:
                pg['lr'] = 5e-5 * (epoch + 1) / warmup_epochs
        else:
            scheduler.step()

        lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"L_pred={e_pred/n:.4f}  SIGReg={e_reg/n:.4f}  lr={lr:.2e}")

        if (epoch + 1) % 10 == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                         "embed_dim": EMBED_DIM}, args.checkpoint)
            print(f"  -> saved: {args.checkpoint}")

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                 "embed_dim": EMBED_DIM}, args.checkpoint)

    # =====================================================
    # State Probe
    # =====================================================
    print(f"\n=== State Probe Training: {args.probe_epochs} epochs ===")
    actual_state_dim = windows_s.shape[1]  # auto-detect: 10 for Pong
    probe = StateProbe(state_dim=actual_state_dim).to(device)
    print(f"Probe output dim: {actual_state_dim}")
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    s_mean = windows_s.mean(0)
    s_std = windows_s.std(0).clamp(min=1e-6)

    model.eval()
    print("Encoding predicted latents for probe training...")
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(windows_f), bs):
            batch_f = windows_f[i:i+bs].to(device)
            batch_a = windows_a[i:i+bs].to(device)
            emb = model.encode(batch_f)
            ctx = emb[:, :HISTORY_SIZE]
            B_b = batch_a.shape[0]
            a_flat = batch_a[:, :HISTORY_SIZE].reshape(B_b * HISTORY_SIZE, -1)
            ctx_a = model.action_encoder(a_flat).reshape(B_b, HISTORY_SIZE, -1)
            pred = model.predict_next(ctx, ctx_a)
            all_emb.append(pred.cpu())
    all_emb = torch.cat(all_emb)
    states_n = (windows_s - s_mean) / s_std

    best_val = float('inf')
    n_tr = int(len(all_emb) * 0.9)

    for epoch in range(args.probe_epochs):
        probe.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, bs * 2):
            idx = perm[i:i + bs * 2]
            pred = probe(all_emb[idx].to(device))
            loss = F.mse_loss(pred, states_n[idx].to(device))
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()

        probe.eval()
        with torch.no_grad():
            val = F.mse_loss(probe(all_emb[n_tr:].to(device)),
                             states_n[n_tr:].to(device)).item()
        if val < best_val:
            best_val = val

        if (epoch + 1) % 10 == 0:
            print(f"  Probe epoch {epoch+1}/{args.probe_epochs}  "
                  f"val={val:.4f}  best={best_val:.4f}")

    probe.eval()
    with torch.no_grad():
        pred = probe(all_emb[n_tr:].to(device)).cpu()
        tgt = states_n[n_tr:]
        for i in range(actual_state_dim):
            c = np.corrcoef(pred[:, i].numpy(), tgt[:, i].numpy())[0, 1]
            print(f"  state[{i}]: corr={c:+.3f}")

    ckpt = torch.load(args.checkpoint, weights_only=False)
    ckpt["probe"] = probe.state_dict()
    ckpt["state_mean"] = s_mean
    ckpt["state_std"] = s_std
    torch.save(ckpt, args.checkpoint)
    print(f"Probe saved -> {args.checkpoint}")

    # Quick rollout test
    print("\n=== Rollout Test ===")
    model.eval()
    with torch.no_grad():
        seed = windows_f[:1].to(device)
        seed_a = windows_a[:1].to(device)
        future_a = torch.zeros(1, 10, ACTION_DIM, device=device)
        preds = model.rollout(seed, seed_a, future_a, n_steps=10)
        print(f"Rollout: {preds.shape}")
        print(f"  std={preds.std():.4f}  drift={((preds[:,0]-preds[:,-1]).norm()):.4f}")
