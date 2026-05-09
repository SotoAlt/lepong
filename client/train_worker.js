// Phase 2: train-while-playing Web Worker. Runs the same SGD round
// protocol as train.js, but in a worker (TF.js WASM backend) so SGD
// doesn't block the gameplay main thread. Single-threaded WASM is
// 3-5x slower than WebGL but doesn't compete with canvas + WS receive.

importScripts(
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js',
);

const HUB_BASE = 'https://jepa.waweapps.win/federated/';
const GAME_ID = 'lepong_v1';
const GAME_PREFIX = HUB_BASE + 'games/' + GAME_ID;

const DTYPE_F32 = 0;
const DTYPE_I8 = 1;
const MAGIC = [0x52, 0x45, 0x4c, 0x41, 0x59, 0, 0, 0];

const post = (msg) => self.postMessage(msg);
const status = (text) => post({ type: 'status', text });
const log = (text) => post({ type: 'log', text });

let ctx = null;
let manifest = null;
let stopRequested = false;

// Sequence number persists across rounds in this worker session;
// each round's cid is the same so the hub treats them as one
// contributor. Random suffix avoids collisions across browser tabs.
const cid = 'worker-' + Math.random().toString(36).slice(2, 8);

const buildVariables = (tf, weightsArr) => {
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
    const nameBytes = enc.encode(L.name);
    pushU32(nameBytes.length);
    parts.push(nameBytes);
    pushU8(DTYPE_I8);
    pushU8(L.shape.length);
    for (const d of L.shape) pushU32(d);
    pushU32(L.signs.byteLength);
    parts.push(new Uint8Array(L.signs.buffer, L.signs.byteOffset, L.signs.byteLength));
    const scaleName = 'scale::' + L.name;
    const sBytes = enc.encode(scaleName);
    pushU32(sBytes.length);
    parts.push(sBytes);
    pushU8(DTYPE_F32);
    pushU8(0);
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

const init = async () => {
  if (ctx) return ctx;
  status('init: loading wasm backend');
  // Tell tf-backend-wasm where to fetch the .wasm files from. Same
  // jsdelivr origin as the JS, so no CORS gymnastics. Without this,
  // setBackend('wasm') 404s on tfjs-backend-wasm.wasm.
  tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
  await tf.setBackend('wasm');
  await tf.ready();

  status('init: importing tfjs_forward');
  // Dynamic import works in both classic + module workers since 2020.
  const mod = await import(GAME_PREFIX + '/tfjs_forward.js');

  status('init: fetching predictor weights (0/63 MB)');
  const t0 = performance.now();
  const wResp = await fetch(GAME_PREFIX + '/weights/predictor.bin');
  if (!wResp.ok) throw new Error('weights fetch failed: ' + wResp.status);
  // Stream so the panel reports live MB/MB progress instead of
  // looking frozen for ~50s. Status text format matches train.js so
  // fedProgressPct in main.js handles both code paths uniformly.
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
    if (received - lastUpdate > 200_000 || received === totalBytes) {
      status('init: fetching predictor weights (' +
             Math.round(received / 1048576) + '/' + totalMB + ' MB)');
      lastUpdate = received;
    }
  }
  const wBufArr = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { wBufArr.set(c, off); off += c.length; }
  const wBuf = wBufArr.buffer;
  const wArr = mod.parseRelayBlob(wBuf);
  log(`weights: ${wBuf.byteLength.toLocaleString()} B in ${((performance.now()-t0)|0)} ms`);

  const { variables, buffers } = buildVariables(tf, wArr);
  const weights = new Map([...variables, ...buffers]);
  const { predictTrainable } = mod.makeForward(tf);
  const optimizer = tf.train.adam(5e-5);
  ctx = { tf, parseRelayBlob: mod.parseRelayBlob, weights, variables,
          buffers, optimizer, predictTrainable };
  log(`init done: ${variables.size} vars, ${buffers.size} buffers`);
  return ctx;
};

const fetchManifest = async () => {
  if (manifest) return manifest;
  const r = await fetch(GAME_PREFIX + '/manifest.json');
  if (!r.ok) throw new Error('manifest fetch failed: ' + r.status);
  manifest = await r.json();
  return manifest;
};

const trainAndPush = async (ctx, weightsBuf, env, ws) => {
  // 1) Sync trainable variables + BN buffers from broadcast.
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
  log(`synced ${syncedV}/${ctx.variables.size} vars, ${syncedB}/${ctx.buffers.size} buffers`);

  // 2) Snapshot pre-SGD weights — Promise.all parallelizes the GPU/WASM readback.
  const varNames = [...ctx.variables.keys()];
  const varList = [...ctx.variables.values()];
  const beforeArrs = await Promise.all(varList.map((v) => v.data()));
  const before = new Map();
  for (let i = 0; i < varNames.length; i++) {
    before.set(varNames[i], new Float32Array(beforeArrs[i]));
  }

  // 3) Fetch training batch (server pre-encodes audio → embeddings).
  status('round ' + env.round_id + ': fetching batch');
  const batchUrl = GAME_PREFIX + '/training_batch?client_id=' + encodeURIComponent(cid);
  const batchResp = await fetch(batchUrl);
  if (!batchResp.ok) throw new Error('training_batch fetch failed: ' + batchResp.status);
  const batchBuf = await batchResp.arrayBuffer();
  const batch = ctx.parseRelayBlob(batchBuf);
  const embA = batch.get('emb');
  const actA = batch.get('actions');
  if (!embA || !actA) throw new Error('batch missing emb/actions');
  const emb = ctx.tf.tensor(embA.data, embA.shape, 'float32');
  const actions = ctx.tf.tensor(actA.data, actA.shape, 'float32');

  // try/finally so emb + actions release even on mid-round error.
  // Without this, an SGD step throwing leaks two big tensors per
  // failure across an indefinitely-running worker.
  try {
    // 4) K SGD steps. tf.tidy releases per-step intermediates.
    const steps = (manifest && manifest.local_steps_per_round) || 2;
    for (let i = 0; i < steps; i++) {
      if (stopRequested) throw new Error('stop requested mid-round');
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
      log(`  step ${i + 1}/${steps}  loss=${lossVal.toExponential(4)}  (${dt} ms)`);
      status('round ' + env.round_id + ': training (' + (i + 1) + '/' + steps + ')');
    }

    // 5) signSGD encode + upload.
    status('round ' + env.round_id + ': encoding delta');
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
    ws.send(JSON.stringify({ t: 'delta', round_id: env.round_id, n_local_steps: steps }));
    ws.send(blob.buffer);
    log(`uploaded delta (${blob.length.toLocaleString()} B, ${layers.length} layers)`);
    return { steps };
  } finally {
    emb.dispose();
    actions.dispose();
  }
};

const runOneRound = async () => {
  await fetchManifest();
  await init();

  status('connecting');
  const wsBase = HUB_BASE.replace(/^http/, 'ws');
  const wsUrl = wsBase + 'games/' + GAME_ID + '/ws';
  const ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  return await new Promise((resolve, reject) => {
    let pendingEnv = null;
    let settled = false;
    const settle = (fn, val) => { if (!settled) { settled = true; fn(val); } };

    const tid = setTimeout(() => {
      try { ws.close(); } catch (_) {}
      settle(reject, new Error('round timeout (180s)'));
    }, 180000);

    ws.onopen = () => {
      ws.send(JSON.stringify({
        t: 'hello',
        client_id: cid,
        generation: manifest ? manifest.generation : undefined,
      }));
      status('waiting for round announce');
    };

    ws.onmessage = async (ev) => {
      try {
        if (typeof ev.data === 'string') {
          const env = JSON.parse(ev.data);
          if (env.t === 'round') {
            pendingEnv = env;
            status('round ' + env.round_id + ': syncing weights');
          } else if (env.t === 'round_done') {
            clearTimeout(tid);
            try { ws.close(1000); } catch (_) {}
            settle(resolve, {
              round_id: env.round_id,
              val_loss: env.val_loss,
              delta_vs_baseline: env.delta_vs_baseline,
              accepted: env.accepted,
            });
          }
          return;
        }
        if (!pendingEnv) return;
        const env = pendingEnv;
        pendingEnv = null;
        await trainAndPush(ctx, ev.data, env, ws);
        status('uploaded delta · awaiting round_done');
      } catch (e) {
        clearTimeout(tid);
        try { ws.close(); } catch (_) {}
        settle(reject, e);
      }
    };

    ws.onclose = (ev) => {
      clearTimeout(tid);
      if (!settled && ev.code !== 1000) {
        settle(reject, new Error('ws closed: ' + ev.code));
      }
    };
  });
};

let loopActive = false;
const runLoop = async () => {
  if (loopActive) return;
  loopActive = true;
  stopRequested = false;
  log(`worker started cid=${cid}`);
  while (!stopRequested) {
    try {
      const result = await runOneRound();
      post({ type: 'roundDone', ...result });
      if (!stopRequested) await new Promise((r) => setTimeout(r, 1000));
    } catch (e) {
      // "stop requested mid-round" is expected when the user toggles
      // off — surface it as a stopped event, not an error.
      if (stopRequested) break;
      post({ type: 'error', message: e.message || String(e) });
      await new Promise((r) => setTimeout(r, 5000));
    }
  }
  loopActive = false;
  status('idle — gameplay feeds the pool');
  post({ type: 'stopped' });
  log(`worker stopped cid=${cid}`);
};

self.onmessage = (e) => {
  const { type } = e.data || {};
  if (type === 'start') runLoop();
  else if (type === 'stop') stopRequested = true;
};
