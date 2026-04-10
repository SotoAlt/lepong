"""FastAPI WebSocket server for lepong.

Loads a frozen JEPA checkpoint and serves real-time Pong predictions.

The server receives base64-encoded PNG frames from the client. It never
sees ball state as floats. The only model input is the client's canvas
pixels. The server reads state_head(predictor(encoder(client_frame)))
to extract ball_y and sends that value back as the paddle target.

A classical baseline (ball.y + 5*ball.vy with wall bouncing) is computed
from optional ground_truth telemetry for comparison purposes only.

Usage:
    python -m server.infer --checkpoint checkpoints/lepong_statehead_frozen.pt --port 8791
"""
import argparse
import asyncio
import base64
import io
import json
import logging
import os
import pathlib
import time as wallclock

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lepong")

COURT_H = 0.6
COURT_W = 1.0
BALL_SPEED_MAX = 0.025
BALL_R = 0.012
SERVER_TICK_STEPS = 5

_model = None
_device = None
_state_mean = None
_state_std = None
_checkpoint_path = None


def load_frozen_model():
    """Load the checkpoint and defensively freeze encoder + predictor."""
    global _model, _device, _state_mean, _state_std

    if _model is not None:
        return _model, _device

    device = torch.device("cpu")
    _device = device
    logger.info("Loading %s", _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    state_dim = ckpt.get("state_dim", 10)
    if state_dim <= 0:
        raise RuntimeError(
            f"lepong requires a state-head checkpoint but state_dim={state_dim}."
        )

    model = JEPAPool(
        embed_dim=ckpt.get("embed_dim", EMBED_DIM),
        state_dim=state_dim,
    )
    msg = model.load_state_dict(ckpt["model"])
    logger.info(
        "Loaded state dict: missing=%d unexpected=%d",
        len(msg.missing_keys), len(msg.unexpected_keys),
    )

    # Defensive freeze
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
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        "Trainable params: %d (state head only); frozen: %d (encoder + predictor)",
        n_trainable, n_frozen,
    )
    if n_trainable > 5000:
        raise RuntimeError(
            f"Expected ~2K trainable params for Linear(192, 10) but got {n_trainable}."
        )

    model.eval()
    model = model.to(device)
    _model = model
    _state_mean = ckpt.get("state_mean", torch.zeros(state_dim)).to(device)
    _state_std = ckpt.get("state_std", torch.ones(state_dim)).to(device)

    corrs = ckpt.get("val_correlations", {})
    logger.info(
        "Val correlations from training: ball_x=%.3f ball_y=%.3f pad_l=%.3f pad_r=%.3f",
        corrs.get("ball_x", float("nan")),
        corrs.get("ball_y", float("nan")),
        corrs.get("pad_l", float("nan")),
        corrs.get("pad_r", float("nan")),
    )

    return _model, _device


def decode_png_to_tensor(png_b64: str, target_size: int = 128) -> torch.Tensor:
    """Decode client-sent base64 PNG into a (1, 3, H, W) float tensor in [0, 1]."""
    if "," in png_b64:
        png_b64 = png_b64.split(",", 1)[-1]
    png_bytes = base64.b64decode(png_b64)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    arr_f = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr_f).permute(2, 0, 1).unsqueeze(0)
    return t


app = FastAPI(title="lepong", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "architecture": "CNN JEPA + Linear(192, 10) state head, encoder + predictor frozen",
        "input": "base64 PNG from client canvas",
        "paddle_target": "state_head(predictor(encoder(client_pixels)))[ball_y]",
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = pathlib.Path(__file__).parent.parent / "client" / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content=f"<h1>Client not found at {html_path}</h1>",
            status_code=404,
        )
    return HTMLResponse(html_path.read_text())


@app.websocket("/ws-pong")
async def pong_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    try:
        model, device = load_frozen_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    history_embs: list[torch.Tensor] = []
    history_action_embs: list[torch.Tensor] = []
    plan_count = 0
    plan_ms_total = 0.0

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            png_b64 = msg.get("frame_png")
            if not png_b64:
                await ws.send_json({"error": "missing frame_png (required)"})
                continue

            # Optional ground truth for verification and classical baseline only
            gt = msg.get("ground_truth") or {}
            gt_ball_x = gt.get("ball_x")
            gt_ball_y = gt.get("ball_y")
            gt_ball_vx = gt.get("ball_vx")
            gt_ball_vy = gt.get("ball_vy")
            occlusion_frac = float(msg.get("occlusion_frac", 0.0))

            # Classical baseline: integrate physics for SERVER_TICK_STEPS
            if (gt_ball_y is not None and gt_ball_vy is not None
                    and gt_ball_x is not None and gt_ball_vx is not None):
                bx = float(gt_ball_x)
                by = float(gt_ball_y)
                bvx = float(gt_ball_vx)
                bvy = float(gt_ball_vy)
                for _ in range(SERVER_TICK_STEPS):
                    bx += bvx
                    by += bvy
                    if by - BALL_R < 0:
                        by = BALL_R
                        bvy = abs(bvy)
                    if by + BALL_R > COURT_H:
                        by = COURT_H - BALL_R
                        bvy = -abs(bvy)
                baseline_ball_y = float(by)
                baseline_ball_x = float(bx)
            else:
                baseline_ball_y = None
                baseline_ball_x = None

            # Encode the client pixels
            frame = decode_png_to_tensor(png_b64, target_size=128)
            frame = frame.to(device)
            frame_seq = frame.unsqueeze(0)

            with torch.no_grad():
                emb = model.encode(frame_seq)[0, 0]
            history_embs.append(emb)

            with torch.no_grad():
                action = torch.zeros(1, 2, device=device)
                action_emb = model.action_encoder(action)[0]
            history_action_embs.append(action_emb)

            if len(history_embs) > HISTORY_SIZE:
                history_embs = history_embs[-HISTORY_SIZE:]
            if len(history_action_embs) > HISTORY_SIZE:
                history_action_embs = history_action_embs[-HISTORY_SIZE:]

            # Run JEPA prediction
            ai_paddle_y = None
            jepa_pred_state = None
            if len(history_embs) >= HISTORY_SIZE:
                t0 = wallclock.perf_counter()
                ctx = torch.stack(history_embs).unsqueeze(0)
                ctx_a = torch.stack(history_action_embs).unsqueeze(0)
                with torch.no_grad():
                    pred = model.predict_next(ctx, ctx_a)
                    s_norm = model.state_head(pred)[0]
                    state = (s_norm * _state_std + _state_mean).cpu().numpy()

                pred_bx = float(np.clip(state[0], 0, 1))
                pred_by = float(np.clip(state[1] * COURT_H, 0, COURT_H))
                pred_bvx = float(state[2] * BALL_SPEED_MAX)
                pred_bvy = float(state[3] * BALL_SPEED_MAX)

                jepa_pred_state = {
                    "ball_x": pred_bx,
                    "ball_y": pred_by,
                    "ball_vx": pred_bvx,
                    "ball_vy": pred_bvy,
                }
                ai_paddle_y = pred_by

                plan_ms = (wallclock.perf_counter() - t0) * 1000
                plan_count += 1
                plan_ms_total += plan_ms

            await ws.send_text(json.dumps({
                "ai_paddle_y": ai_paddle_y,
                "jepa_pred": jepa_pred_state,
                "classical_baseline_ball_y": baseline_ball_y,
                "classical_baseline_ball_x": baseline_ball_x,
                "plan_ms_avg": round(plan_ms_total / plan_count, 1) if plan_count else None,
                "history_ready": len(history_embs) >= HISTORY_SIZE,
            }))

    except WebSocketDisconnect:
        logger.info(
            "Client disconnected after %d plans (avg %.0f ms)",
            plan_count, plan_ms_total / max(plan_count, 1),
        )
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


def main():
    global _checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to state-head checkpoint")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    _checkpoint_path = args.checkpoint

    import uvicorn
    logger.info("Starting lepong server on %s:%d", args.host, args.port)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Endpoint: /ws-pong (accepts base64 PNG only)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
