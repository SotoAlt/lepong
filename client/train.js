// Federation training, in the same tab as gameplay.
//
// Lifted from /federated/?game=silent_v1 so silent.html can run a full
// SGD round on the JEPA predictor without users opening another tab.
// Compute is 100% browser-side — server only ships weights + a training
// batch and aggregates the resulting signSGD delta. No game-server CPU
// is spent on training; the predator-hunting loop on jepa-vps is
// unaffected.
//
// Phase 1: one-shot rounds triggered by a button click. Runs on the
//   main thread (post-match, so no contention with WS gameplay).
// Phase 2 (TODO): move SGD into a Web Worker (WASM backend) so it
//   can run while the game is still active without dropping frames.
// Phase 3 (TODO): auto-trigger after each match.

window.LepongFedTrain = (() => {
  'use strict';

  // Hard-coded hub origin — silent.html is served from sotoalt.dev (or
  // the silent docker container on jepa-vps), but the federation hub
  // always lives at jepa.waweapps.win. Cross-origin fetch needs CORS
  // headers from the hub; see CORSMiddleware in federated/ws_server.py.
  const HUB_BASE = 'https://jepa.waweapps.win/federated/';
  const GAME_ID = 'lepong_v1';
  const GAME_PREFIX = HUB_BASE + 'games/' + GAME_ID;

  // RELAY binary wire format constants — must match
  // federated/protocol.py and the parseRelayBlob in tfjs_forward.js.
  // We keep our own copy of MAGIC + DTYPE_* here so encodeDelta doesn't
  // depend on the per-game module beyond parseRelayBlob/makeForward.
  const DTYPE_F32 = 0;
  const DTYPE_I8 = 1;
  const MAGIC = [0x52, 0x45, 0x4c, 0x41, 0x59, 0, 0, 0]; // "RELAY\0\0\0"

  let tfjsLoaded = null;
  const ensureTfjs = () => {
    if (window.tf) return Promise.resolve(window.tf);
    if (tfjsLoaded) return tfjsLoaded;
    tfjsLoaded = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      // Pin to a known-good version (matches /federated/ deps).
      s.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
      s.onload = () => resolve(window.tf);
      s.onerror = () => reject(new Error('failed to load tfjs from CDN'));
      document.head.appendChild(s);
    });
    return tfjsLoaded;
  };

  const buildVariables = (tf, weightsArr) => {
    // BatchNorm running stats are NOT trainable — keep them as plain
    // tensors so the optimizer doesn't try to update them.
    const isBuffer = (n) => n.includes('running_mean') || n.includes('running_var');
    const variables = new Map();
    const buffers = new Map();
    for (const [name, info] of weightsArr) {
      const t = tf.tensor(info.data, info.shape, 'float32');
      if (isBuffer(name)) buffers.set(name, t);
      else variables.set(name, tf.variable(t, true, name));
    }
    return { variables, buffers };
  };

  // signSGD-compressed delta: one Int8Array(±1/0) + one fp32 scale per
  // layer. The two entries for each layer are emitted side-by-side so
  // pack_signSGD_delta on the server can pair them by name.
  const encodeDelta = (layers) => {
    const parts = [];
    const pushU32 = (v) => {
      const b = new Uint8Array(4);
      new DataView(b.buffer).setUint32(0, v, true);
      parts.push(b);
    };
    const pushU8 = (v) => parts.push(new Uint8Array([v]));
    parts.push(new Uint8Array(MAGIC));
    pushU32(layers.length * 2);
    const enc = new TextEncoder();
    for (const L of layers) {
      // signs entry
      const nameBytes = enc.encode(L.name);
      pushU32(nameBytes.length);
      parts.push(nameBytes);
      pushU8(DTYPE_I8);
      pushU8(L.shape.length);
      for (const d of L.shape) pushU32(d);
      pushU32(L.signs.byteLength);
      parts.push(new Uint8Array(L.signs.buffer, L.signs.byteOffset, L.signs.byteLength));
      // scalar fp32 scale entry, name-prefixed with "scale::"
      const scaleName = 'scale::' + L.name;
      const sBytes = enc.encode(scaleName);
      pushU32(sBytes.length);
      parts.push(sBytes);
      pushU8(DTYPE_F32);
      pushU8(0); // scalar
      pushU32(4);
      const sBuf = new Float32Array([L.scale]);
      parts.push(new Uint8Array(sBuf.buffer));
    }
    let total = 0;
    for (const p of parts) total += p.length;
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
  };

  let ctx = null;     // memoized across rounds
  let manifest = null;

  const fetchManifest = async () => {
    if (manifest) return manifest;
    const r = await fetch(GAME_PREFIX + '/manifest.json');
    if (!r.ok) throw new Error('manifest fetch failed: ' + r.status);
    manifest = await r.json();
    return manifest;
  };

  // ~30 MB of predictor-stack weights for silent_v1 (encoder stays on
  // the server — browser only sees pre-computed embeddings).
  const init = async (opts = {}) => {
    if (ctx) return ctx;
    const log = opts.onLog || (() => {});
    log('loading tf.js + tfjs_forward...');

    const tf = await ensureTfjs();
    const tfjsForwardUrl = GAME_PREFIX + '/tfjs_forward.js';
    const mod = await import(/* webpackIgnore: true */ tfjsForwardUrl);

    log('fetching predictor weights (one-time)...');
    const t0 = performance.now();
    const wResp = await fetch(GAME_PREFIX + '/weights/predictor.bin');
    if (!wResp.ok) throw new Error('weights fetch failed: ' + wResp.status);
    // Stream the body so the panel can report MB/MB progress live
    // instead of looking frozen for ~50s. Falls back to 63 MB estimate
    // if Content-Length is missing (HTTP/2 + gzip strips it on some
    // proxies). The status string format is parsed by fedProgressPct
    // in main.js to drive the progress bar in the 5%–20% slice.
    const totalBytes = parseInt(wResp.headers.get('content-length'), 10) || 63500000;
    const totalMB = Math.round(totalBytes / 1048576);
    const reader = wResp.body.getReader();
    const chunks = [];
    let received = 0;
    let lastUpdate = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      // Throttle status updates — every ~200 KB or 100 ms is enough.
      if (received - lastUpdate > 200_000 || received === totalBytes) {
        opts.onStatus?.('init: fetching predictor weights (' +
                        Math.round(received / 1048576) + '/' + totalMB + ' MB)');
        lastUpdate = received;
      }
    }
    const wBufArr = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { wBufArr.set(c, off); off += c.length; }
    const wBuf = wBufArr.buffer;
    const wArr = mod.parseRelayBlob(wBuf);
    log('weights: ' + wBuf.byteLength.toLocaleString() + ' B parsed (' +
        ((performance.now() - t0) | 0) + ' ms)');

    const { variables, buffers } = buildVariables(tf, wArr);
    const weights = new Map([...variables, ...buffers]);
    const { predictTrainable } = mod.makeForward(tf);
    const optimizer = tf.train.adam(5e-5);
    log('init done: ' + variables.size + ' trainable vars, ' +
        buffers.size + ' buffers');
    ctx = { tf, parseRelayBlob: mod.parseRelayBlob, weights, variables,
            buffers, optimizer, predictTrainable };
    return ctx;
  };

  // Resolves with { round_id, val_loss, delta_vs_baseline, accepted }
  // once the hub broadcasts round_done. Safe to call repeatedly — keeps
  // the tf context warm across calls so weights aren't re-downloaded.
  const runOneRound = async (opts = {}) => {
    const onLog = opts.onLog || (() => {});
    const onStatus = opts.onStatus || (() => {});
    const log = (msg, kind = 'info') => onLog(msg, kind);

    onStatus('init');
    await fetchManifest();
    await init({ onLog });

    onStatus('connecting');
    let cid = sessionStorage.getItem('silent_fed_cid');
    if (!cid) {
      cid = 'browser-' + Math.random().toString(36).slice(2, 8);
      sessionStorage.setItem('silent_fed_cid', cid);
    }

    const wsBase = HUB_BASE.replace(/^http/, 'ws');
    const wsUrl = wsBase + 'games/' + GAME_ID + '/ws';
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    return await new Promise((resolve, reject) => {
      let pendingEnv = null;
      let settled = false;
      const settle = (fn, val) => { if (!settled) { settled = true; fn(val); } };

      // 180s timeout — covers connect + round-wait + 63 MB weights
      // broadcast + 2 SGD steps + 16 MB delta upload + aggregate +
      // round_done. On a shared VPS where the federated container
      // contends with silent gameplay CPU, the broadcast + aggregate
      // phases each can run 30-60s. 90s was too tight; saw rounds
      // succeed hub-side after the client had given up.
      const tid = setTimeout(() => {
        try { ws.close(); } catch (_) {}
        settle(reject, new Error('round timeout (180s) — server slow or quorum unmet'));
      }, 180000);

      ws.onopen = () => {
        log('connected; hello cid=' + cid, 'send');
        ws.send(JSON.stringify({
          t: 'hello',
          client_id: cid,
          generation: manifest ? manifest.generation : undefined,
        }));
        onStatus('waiting for round announce');
      };

      ws.onmessage = async (ev) => {
        try {
          if (typeof ev.data === 'string') {
            const env = JSON.parse(ev.data);
            if (env.t === 'round') {
              pendingEnv = env;
              onStatus('round ' + env.round_id + ': syncing weights');
              log('round ' + env.round_id + ' announced', 'recv');
            } else if (env.t === 'round_done') {
              clearTimeout(tid);
              try { ws.close(1000); } catch (_) {}
              onStatus('round ' + env.round_id + ' done · val=' +
                       env.val_loss.toFixed(4));
              log('round ' + env.round_id + ' done · val=' +
                  env.val_loss.toFixed(4) + ' dvb=' +
                  env.delta_vs_baseline.toFixed(4), 'recv');
              settle(resolve, {
                round_id: env.round_id,
                val_loss: env.val_loss,
                delta_vs_baseline: env.delta_vs_baseline,
                accepted: env.accepted,
              });
            }
            return;
          }
          if (!pendingEnv) {
            log('binary frame before envelope', 'err');
            return;
          }
          const env = pendingEnv;
          pendingEnv = null;
          await trainAndPush(ctx, ev.data, env, ws, cid, log, onStatus);
          onStatus('uploaded delta · awaiting round_done');
        } catch (e) {
          clearTimeout(tid);
          try { ws.close(); } catch (_) {}
          settle(reject, e);
        }
      };

      ws.onclose = (ev) => {
        clearTimeout(tid);
        if (!settled && ev.code !== 1000) {
          settle(reject, new Error('ws closed: ' + ev.code +
                                   (ev.reason ? ' (' + ev.reason + ')' : '')));
        }
      };
    });
  };

  const trainAndPush = async (ctx, weightsBuf, env, ws, cid, log, onStatus) => {
    // Sync to server-broadcast weights — both trainable variables AND
    // BatchNorm running stats. Skipping buffers here would let the BN
    // stats drift from the aggregated model across rounds.
    const serverW = ctx.parseRelayBlob(weightsBuf);
    let syncedV = 0, syncedB = 0;
    for (const [name, v] of ctx.variables) {
      const a = serverW.get(name);
      if (!a) continue;
      const t = ctx.tf.tensor(a.data, a.shape, 'float32');
      v.assign(t);
      t.dispose();
      syncedV++;
    }
    for (const [name, oldT] of ctx.buffers) {
      const a = serverW.get(name);
      if (!a) continue;
      const t = ctx.tf.tensor(a.data, a.shape, 'float32');
      ctx.buffers.set(name, t);
      ctx.weights.set(name, t);
      oldT.dispose();
      syncedB++;
    }
    log('synced ' + syncedV + '/' + ctx.variables.size + ' vars, ' +
        syncedB + '/' + ctx.buffers.size + ' buffers', 'recv');

    // Snapshot pre-SGD weights so we can compute delta = post - pre.
    // Promise.all collapses the WebGL readback — sequential `await v.data()`
    // serializes GPU→CPU transfers and takes seconds for ~95 small tensors.
    const varNames = [...ctx.variables.keys()];
    const varList = [...ctx.variables.values()];
    const beforeArrs = await Promise.all(varList.map((v) => v.data()));
    const before = new Map();
    for (let i = 0; i < varNames.length; i++) {
      before.set(varNames[i], new Float32Array(beforeArrs[i]));
    }

    onStatus('round ' + env.round_id + ': fetching batch');
    const batchUrl = GAME_PREFIX + '/training_batch?client_id=' +
                     encodeURIComponent(cid);
    const batchResp = await fetch(batchUrl);
    if (!batchResp.ok) {
      throw new Error('training_batch fetch failed: ' + batchResp.status);
    }
    const batchBuf = await batchResp.arrayBuffer();
    const batch = ctx.parseRelayBlob(batchBuf);
    const embA = batch.get('emb');
    const actA = batch.get('actions');
    if (!embA || !actA) throw new Error('batch missing emb/actions');
    const emb = ctx.tf.tensor(embA.data, embA.shape, 'float32');
    const actions = ctx.tf.tensor(actA.data, actA.shape, 'float32');

    // K local SGD steps. tf.tidy inside the closure releases the
    // ~50 intermediate activations per forward pass.
    const steps = (manifest && manifest.local_steps_per_round) || 2;
    onStatus('round ' + env.round_id + ': training (0/' + steps + ')');
    for (let i = 0; i < steps; i++) {
      const t0 = performance.now();
      const lossT = ctx.optimizer.minimize(() => {
        return ctx.tf.tidy(() => {
          const out = ctx.predictTrainable(emb, actions, ctx.weights);
          return ctx.tf.mean(ctx.tf.square(ctx.tf.sub(out.pred_emb, out.tgt_emb)));
        });
      }, true, varList);
      const lossVal = (await lossT.data())[0];
      lossT.dispose();
      const dt = (performance.now() - t0) | 0;
      log('  step ' + (i + 1) + '/' + steps + '  loss=' +
          lossVal.toExponential(4) + '  (' + dt + ' ms)', 'recv');
      onStatus('round ' + env.round_id + ': training (' + (i + 1) +
               '/' + steps + ')');
      // Yield so the UI repaints between expensive SGD steps.
      await new Promise((r) => setTimeout(r, 0));
    }
    emb.dispose();
    actions.dispose();

    // signSGD delta = sign(post - pre) per parameter, with a per-layer
    // fp32 scale = ||delta||_2 / sqrt(N).
    onStatus('round ' + env.round_id + ': encoding delta');
    const curArrs = await Promise.all(varList.map((v) => v.data()));

    const layers = [];
    for (let li = 0; li < varNames.length; li++) {
      const name = varNames[li];
      const cur = curArrs[li];
      const prev = before.get(name);
      const signs = new Int8Array(cur.length);
      let sumSq = 0;
      for (let i = 0; i < cur.length; i++) {
        const d = cur[i] - prev[i];
        sumSq += d * d;
        signs[i] = d > 0 ? 1 : (d < 0 ? -1 : 0);
      }
      const norm = Math.sqrt(sumSq);
      const scale = norm / Math.max(Math.sqrt(cur.length), 1.0);
      layers.push({ name, signs, scale, shape: varList[li].shape });
    }
    const blob = encodeDelta(layers);
    ws.send(JSON.stringify({
      t: 'delta',
      round_id: env.round_id,
      n_local_steps: steps,
    }));
    ws.send(blob.buffer);
    log('uploaded delta (' + blob.length.toLocaleString() + ' B, ' +
        layers.length + ' layers)', 'send');
  };

  return { runOneRound };
})();
