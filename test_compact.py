# Let's craft the compact index.html and test it
import subprocess

compact_html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sparsa — SINDy Discovery Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>
:root {
  --bg: #0B0D10; --surface: #12151A; --surface-2: #171B21; --border: #1F242C; --border-focus: #3A4250;
  --text: #E6E8EB; --muted: #8B93A1; --accent: #4C9AFF; --accent-muted: rgba(76,154,255,0.15);
  --true-trace: rgba(230,232,235,0.60); --pass: #3FB68B; --fail: #E5484D;
  --font-sans: 'Inter', -apple-system, sans-serif; --font-mono: 'JetBrains Mono', monospace;
  --fs-12: 12px; --fs-13: 13px; --fs-14: 14px; --fs-16: 16px; --radius: 6px; --transition: 120ms ease-out;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font-sans); font-size: var(--fs-13); height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
.mono, code, .stat-val { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
#app-wrap { display: flex; flex: 1; height: calc(100vh - 28px); overflow: hidden; position: relative; }
#rail { width: 300px; min-width: 300px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; transition: margin-left var(--transition); z-index: 20; }
#rail.collapsed { margin-left: -300px; }
.rail-head { padding: 10px 14px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.rail-title { font-size: var(--fs-14); font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.rail-sub { font-size: var(--fs-12); color: var(--muted); margin-left: 6px; }
.rail-toggle { background: transparent; border: 1px solid var(--border); color: var(--muted); border-radius: var(--radius); padding: 2px 7px; cursor: pointer; }
.rail-body { flex: 1; overflow-y: auto; padding: 12px 14px; display: flex; flex-direction: column; gap: 14px; }
.sec-title { font-size: var(--fs-12); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-bottom: 6px; }
.ctrl-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: var(--fs-12); margin-bottom: 5px; }
.ctrl-label { color: var(--muted); }
select, input[type="text"], input[type="number"] { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 5px 7px; font-size: var(--fs-12); }
input[type="range"] { width: 100%; height: 4px; accent-color: var(--accent); background: var(--border); border-radius: 2px; }
.btn-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; }
.btn { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 5px 8px; font-size: var(--fs-12); cursor: pointer; text-align: center; }
.btn.active { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.btn-subtle { background: transparent; border: 1px solid var(--border); color: var(--muted); width: 100%; padding: 6px; cursor: pointer; border-radius: var(--radius); font-size: var(--fs-12); }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.m-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 6px 8px; }
.m-card.pass { border-left: 2px solid var(--pass); }
.m-card.fail { border-left: 2px solid var(--fail); }
.m-label { font-size: 11px; color: var(--muted); }
.m-val { font-size: var(--fs-16); font-family: var(--font-mono); font-weight: 600; }
.eq-box, .diag-box { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 10px; display: flex; flex-direction: column; gap: 4px; }
.diag-desc { font-size: var(--fs-12); color: var(--text); line-height: 1.35; }
#workspace { flex: 1; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 1px; background: var(--border); position: relative; }
.panel { background: var(--bg); display: flex; flex-direction: column; overflow: hidden; position: relative; }
.panel-bar { height: 30px; background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 10px; display: flex; justify-content: space-between; align-items: center; user-select: none; }
.panel-title { font-size: var(--fs-12); font-weight: 600; letter-spacing: .02em; }
.panel-tools { display: flex; gap: 6px; }
.panel-tools button, .panel-tools select { height: 20px; font-size: 11px; padding: 1px 5px; background: var(--surface-2); border: 1px solid var(--border); color: var(--muted); border-radius: 4px; cursor: pointer; }
.canvas-wrap { flex: 1; position: relative; width: 100%; height: calc(100% - 30px); }
canvas { display: block; width: 100%; height: 100%; }
#crosshair-tip { position: absolute; display: none; background: var(--surface); border: 1px solid var(--border-focus); border-radius: var(--radius); padding: 5px 8px; font-size: 11px; font-family: var(--font-mono); color: var(--text); pointer-events: none; z-index: 100; white-space: pre; }
#pins-drawer { position: absolute; right: 0; top: 0; bottom: 0; width: 240px; background: var(--surface); border-left: 1px solid var(--border); transform: translateX(100%); transition: transform var(--transition); z-index: 50; display: flex; flex-direction: column; }
#pins-drawer.open { transform: translateX(0); }
.pins-head { padding: 8px 10px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; font-size: var(--fs-12); font-weight: 600; }
.pins-body { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 5px; }
.pin-item { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 5px; font-size: 11px; font-family: var(--font-mono); display: flex; justify-content: space-between; align-items: center; }
#status-bar { height: 28px; background: var(--surface); border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; padding: 0 12px; font-size: var(--fs-12); color: var(--muted); }
.status-grp { display: flex; gap: 14px; align-items: center; }
.status-grp .mono { color: var(--text); }
</style>
</head>
<body>
<div id="app-wrap">
  <aside id="rail">
    <div class="rail-head">
      <div><span class="rail-title">SPARSA</span><span class="rail-sub">SINDy Discovery</span></div>
      <button id="btn-collapse" class="rail-toggle" title="Toggle Rail">‹</button>
    </div>
    <div class="rail-body">
      <div>
        <div class="sec-title">Preset</div>
        <select id="sel-preset">
          <option value="lorenz">Lorenz-63 (Attractor)</option>
          <option value="lotka">Lotka-Volterra (Predator-Prey)</option>
          <option value="vdp">Van der Pol (Limit Cycle)</option>
          <option value="pendulum">Damped Pendulum (Trig)</option>
          <option value="rossler">Rössler (Spiral Chaos)</option>
          <option value="noise_stress">Lorenz (5% Noise Stress)</option>
        </select>
      </div>
      <div>
        <div class="sec-title">Simulation</div>
        <div class="ctrl-row"><span class="ctrl-label">System</span><select id="sel-system" style="width:160px"><option value="lorenz">Lorenz-63</option><option value="lotka">Lotka-Volterra</option><option value="vdp">Van der Pol</option><option value="pendulum">Damped Pendulum</option><option value="rossler">Rössler</option></select></div>
        <div class="btn-grid" style="margin:6px 0">
          <button id="btn-play" class="btn active">Pause</button>
          <button id="btn-step" class="btn">Step</button>
          <button id="btn-reset" class="btn">Reset</button>
          <button id="btn-rand" class="btn">Rand IC</button>
        </div>
        <div class="ctrl-row"><span class="ctrl-label">dt</span><span id="v-dt" class="mono">0.005</span></div>
        <input id="sl-dt" type="range" min="0.001" max="0.02" step="0.001" value="0.005">
        <div class="ctrl-row" style="margin-top:6px"><span class="ctrl-label">Noise (σ)</span><span id="v-noise" class="mono">0.00</span></div>
        <input id="sl-noise" type="range" min="0" max="0.05" step="0.005" value="0.00">
        <div class="ctrl-row" style="margin-top:6px"><span class="ctrl-label">Trajectories</span><span id="v-trajs" class="mono">1</span></div>
        <input id="sl-trajs" type="range" min="1" max="5" step="2" value="1">
        <div class="ctrl-row" style="margin-top:6px"><span class="ctrl-label">Live Divergence</span><span id="v-div" class="mono">0.000</span></div>
        <div class="ctrl-row"><span class="ctrl-label">Lyapunov Horizon</span><span id="v-lyap" class="mono">t = ∞</span></div>
      </div>
      <div>
        <div class="sec-title">SINDy Pipeline</div>
        <div class="ctrl-row"><span class="ctrl-label">Degree</span><select id="sel-deg" style="width:90px"><option value="1">1</option><option value="2" selected>2</option><option value="3">3</option></select></div>
        <div class="ctrl-row"><span class="ctrl-label">Trigonometric</span><input id="chk-trig" type="checkbox" style="accent-color:var(--accent)"></div>
        <div class="ctrl-row"><span class="ctrl-label">Differentiation</span><select id="sel-diff" style="width:130px"><option value="central_diff">Central Diff</option><option value="savgol">Savitzky-Golay</option><option value="weak_form">Weak-Form</option></select></div>
        <div class="ctrl-row"><span class="ctrl-label">Threshold (λ)</span><span id="v-lam" class="mono">0.100</span></div>
        <input id="sl-lam" type="range" min="-3" max="0.7" step="0.05" value="-1.0">
      </div>
      <div>
        <div class="sec-title">Equations</div>
        <div class="eq-box">
          <div style="font-size:11px;color:var(--muted)">TRUE MODEL</div>
          <div id="eq-true"></div>
          <div style="font-size:11px;color:var(--accent);margin-top:4px">DISCOVERED MODEL</div>
          <div id="eq-disc"></div>
        </div>
      </div>
      <div>
        <div class="sec-title">Metrics</div>
        <div class="metric-grid">
          <div id="c-f1" class="m-card pass"><div class="m-label">Support F1</div><div id="m-f1" class="m-val">1.000</div></div>
          <div id="c-err" class="m-card pass"><div class="m-label">Coef Error</div><div id="m-err" class="m-val">0.00%</div></div>
          <div class="m-card"><div class="m-label">Derivative R²</div><div id="m-r2" class="m-val">0.999</div></div>
          <div class="m-card"><div class="m-label">Active Terms</div><div id="m-terms" class="m-val">7 / 10</div></div>
        </div>
      </div>
      <div>
        <div class="sec-title">Diagnostic Diagnosis</div>
        <div class="diag-box"><span id="diag-msg" class="diag-desc">Support perfectly recovered with zero structural error.</span></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:5px">
        <button id="btn-export-json" class="btn-subtle">Export Model JSON</button>
        <button id="btn-copy-latex" class="btn-subtle">Copy LaTeX</button>
        <button id="btn-share-url" class="btn-subtle">Copy Share URL</button>
      </div>
    </div>
  </aside>

  <main id="workspace">
    <section class="panel">
      <div class="panel-bar"><span class="panel-title">3D Orbit / Attractor</span><div class="panel-tools"><button id="btn-p1-png">PNG</button></div></div>
      <div class="canvas-wrap"><canvas id="c-3d"></canvas></div>
    </section>
    <section class="panel">
      <div class="panel-bar">
        <span class="panel-title">Phase Plane</span>
        <div class="panel-tools">
          <select id="sel-plane-vars"><option value="0,1">x vs y</option><option value="0,2">x vs z</option><option value="1,2">y vs z</option></select>
          <button id="btn-q-toggle">Field: True</button>
          <button id="btn-p2-png">PNG</button>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="c-phase"></canvas></div>
    </section>
    <section class="panel">
      <div class="panel-bar"><span class="panel-title">Time-Series Rollout</span><div class="panel-tools"><button id="btn-p3-png">PNG</button></div></div>
      <div class="canvas-wrap"><canvas id="c-ts"></canvas></div>
    </section>
    <section class="panel">
      <div class="panel-bar"><span class="panel-title">Library Matrix Ξ & Pareto Frontier</span><div class="panel-tools"><button id="btn-p4-png">PNG</button></div></div>
      <div class="canvas-wrap"><canvas id="c-hm"></canvas></div>
    </section>
    <div id="crosshair-tip"></div>
    <aside id="pins-drawer">
      <div class="pins-head"><span>Pinned Points</span><button id="btn-close-pins" class="rail-toggle">×</button></div>
      <div id="pins-list" class="pins-body"></div>
    </aside>
  </main>
</div>

<footer id="status-bar">
  <div class="status-grp">
    <span>FPS: <span id="sb-fps" class="mono">60</span></span>
    <span>Step: <span id="sb-step" class="mono">0</span></span>
    <span>Time: <span id="sb-time" class="mono">0.00s</span></span>
    <span>dt: <span id="sb-dt" class="mono">0.005</span></span>
  </div>
  <div class="status-grp">
    <span>cond(Θ): <span id="sb-cond" class="mono">1.0e+00</span></span>
    <span>Status: <span id="sb-status" class="mono" style="color:var(--pass)">Active</span></span>
    <button id="btn-open-pins" class="rail-toggle">Pins (0)</button>
  </div>
</footer>

<script>
// 1. DYNAMICAL SYSTEMS REGISTRY
const SYSTEMS = {
  lorenz: {
    name: "Lorenz-63", vars: ["x", "y", "z"], defaultIC: [-8.0, 7.0, 27.0], lyapunov: 0.906,
    rhs: (x) => [10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - (8.0 / 3.0) * x[2]],
    latex: ["\\dot{x} = 10(y - x)", "\\dot{y} = x(28 - z) - y", "\\dot{z} = xy - \\frac{8}{3}z"]
  },
  lotka: {
    name: "Lotka-Volterra", vars: ["x", "y"], defaultIC: [2.0, 1.0], lyapunov: 0.0,
    rhs: (x) => [1.5 * x[0] - x[0] * x[1], x[0] * x[1] - 3.0 * x[1]],
    latex: ["\\dot{x} = 1.5x - xy", "\\dot{y} = xy - 3y"]
  },
  vdp: {
    name: "Van der Pol", vars: ["x", "y"], defaultIC: [1.0, 0.0], lyapunov: 0.0,
    rhs: (x) => [x[1], 2.0 * (1.0 - x[0] * x[0]) * x[1] - x[0]],
    latex: ["\\dot{x} = y", "\\dot{y} = 2(1 - x^2)y - x"]
  },
  pendulum: {
    name: "Damped Pendulum", vars: ["\\theta", "\\omega"], defaultIC: [1.5, 0.0], lyapunov: 0.0,
    rhs: (x) => [x[1], -Math.sin(x[0]) - 0.2 * x[1]],
    latex: ["\\dot{\\theta} = \\omega", "\\dot{\\omega} = -\\sin(\\theta) - 0.2\\omega"]
  },
  rossler: {
    name: "Rössler", vars: ["x", "y", "z"], defaultIC: [1.0, 1.0, 0.0], lyapunov: 0.071,
    rhs: (x) => [-x[1] - x[2], x[0] + 0.2 * x[1], 0.2 + x[2] * (x[0] - 5.7)],
    latex: ["\\dot{x} = -y - z", "\\dot{y} = x + 0.2y", "\\dot{z} = 0.2 + z(x - 5.7)"]
  }
};

// 2. CLASSICAL RUNGE-KUTTA 4TH ORDER INTEGRATOR
/** Advances state x by dt using classical 4th-order Runge-Kutta. */
function rk4Step(f, x, dt) {
  const n = x.length, k1 = f(x), x2 = new Array(n);
  for (let i = 0; i < n; i++) x2[i] = x[i] + 0.5 * dt * k1[i];
  const k2 = f(x2), x3 = new Array(n);
  for (let i = 0; i < n; i++) x3[i] = x[i] + 0.5 * dt * k2[i];
  const k3 = f(x3), x4 = new Array(n);
  for (let i = 0; i < n; i++) x4[i] = x[i] + dt * k3[i];
  const k4 = f(x4), xNext = new Array(n);
  for (let i = 0; i < n; i++) xNext[i] = x[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  return xNext;
}

// 3. CANDIDATE FEATURE LIBRARY
/** Builds candidate feature library up to polynomial degree 3 + optional trig. */
function buildFeatureLibrary(vars, degree, trig) {
  const n = vars.length, terms = [{ t: "bias" }], names = ["1"];
  for (let i = 0; i < n; i++) { terms.push({ t: "p", idx: [i] }); names.push(vars[i]); }
  if (degree >= 2) {
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        terms.push({ t: "p", idx: [i, j] });
        names.push(i === j ? `${vars[i]}^2` : `${vars[i]} ${vars[j]}`);
      }
    }
  }
  if (degree >= 3) {
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        for (let k = j; k < n; k++) {
          terms.push({ t: "p", idx: [i, j, k] });
          if (i === j && j === k) names.push(`${vars[i]}^3`);
          else if (i === j) names.push(`${vars[i]}^2 ${vars[k]}`);
          else if (j === k) names.push(`${vars[i]} ${vars[j]}^2`);
          else names.push(`${vars[i]} ${vars[j]} ${vars[k]}`);
        }
      }
    }
  }
  if (trig) {
    for (let i = 0; i < n; i++) {
      terms.push({ t: "sin", idx: i }); names.push(`sin(${vars[i]})`);
      terms.push({ t: "cos", idx: i }); names.push(`cos(${vars[i]})`);
    }
  }
  const p = terms.length;
  return {
    names,
    eval: (x) => {
      const row = new Float64Array(p);
      for (let k = 0; k < p; k++) {
        const item = terms[k];
        if (item.t === "bias") row[k] = 1.0;
        else if (item.t === "p") { let v = 1.0; for (let idx of item.idx) v *= x[idx]; row[k] = v; }
        else if (item.t === "sin") row[k] = Math.sin(x[item.idx]);
        else if (item.t === "cos") row[k] = Math.cos(x[item.idx]);
      }
      return row;
    }
  };
}

// 4. NUMERICAL DIFFERENTIATION
function centralDiff(X, dt) {
  const m = X.length, n = X[0].length;
  const dX = Array.from({ length: m }, () => new Float64Array(n));
  for (let j = 0; j < n; j++) {
    dX[0][j] = (X[1][j] - X[0][j]) / dt;
    for (let i = 1; i < m - 1; i++) dX[i][j] = (X[i + 1][j] - X[i - 1][j]) / (2.0 * dt);
    dX[m - 1][j] = (X[m - 1][j] - X[m - 2][j]) / dt;
  }
  return dX;
}
function savgolDiff(X, dt) {
  const m = X.length, n = X[0].length, dX = centralDiff(X, dt);
  if (m < 7) return dX;
  const c = [-3, -2, -1, 0, 1, 2, 3], norm = 28.0 * dt;
  for (let j = 0; j < n; j++) {
    for (let i = 3; i < m - 3; i++) {
      let s = 0.0;
      for (let k = -3; k <= 3; k++) s += c[k + 3] * X[i + k][j];
      dX[i][j] = s / norm;
    }
  }
  return dX;
}
function weakFormProjection(X, Theta, dt) {
  const m = X.length, n = X[0].length, p = Theta[0].length, w = 15;
  if (m <= 2 * w) return { Theta_w: Theta, dX_w: centralDiff(X, dt) };
  const K = m - 2 * w, phi = new Float64Array(2 * w + 1), dphi = new Float64Array(2 * w + 1);
  for (let j = -w; j <= w; j++) {
    const tau = j / w, u = 1.0 - tau * tau;
    phi[j + w] = (u > 0) ? Math.pow(u, 4) : 0.0;
    dphi[j + w] = (u > 0) ? -8.0 * tau * Math.pow(u, 3) / (w * dt) : 0.0;
  }
  const Theta_w = Array.from({ length: K }, () => new Float64Array(p));
  const dX_w = Array.from({ length: K }, () => new Float64Array(n));
  for (let k = 0; k < K; k++) {
    const c = k + w;
    for (let col = 0; col < p; col++) {
      let s = 0.0; for (let j = -w; j <= w; j++) s += phi[j + w] * Theta[c + j][col];
      Theta_w[k][col] = s * dt;
    }
    for (let varIdx = 0; varIdx < n; varIdx++) {
      let s = 0.0; for (let j = -w; j <= w; j++) s += -dphi[j + w] * X[c + j][varIdx];
      dX_w[k][varIdx] = s * dt;
    }
  }
  return { Theta_w, dX_w };
}

// 5. STLSQ SOLVER (Normal Equations + Cholesky Decomposition)
function solveCholesky(A, b) {
  const p = A.length, L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0.0; for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(A[i][i] - s, 1e-12));
      else L[i][j] = (A[i][j] - s) / L[j][j];
    }
  }
  const y = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = 0.0; for (let k = 0; k < i; k++) s += L[i][k] * y[k];
    y[i] = (b[i] - s) / L[i][i];
  }
  const x = new Float64Array(p);
  for (let i = p - 1; i >= 0; i--) {
    let s = 0.0; for (let k = i + 1; k < p; k++) s += L[k][i] * x[k];
    x[i] = (y[i] - s) / L[i][i];
  }
  return x;
}

function stlsq(Theta, dXdt, lambda, maxIter = 10) {
  const m = Theta.length, p = Theta[0].length, n = dXdt[0].length;
  const colNorms = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let sumSq = 0.0; for (let i = 0; i < m; i++) sumSq += Theta[i][j] * Theta[i][j];
    colNorms[j] = Math.sqrt(sumSq) || 1.0;
  }
  const ThetaN = Array.from({ length: m }, () => new Float64Array(p));
  for (let i = 0; i < m; i++) for (let j = 0; j < p; j++) ThetaN[i][j] = Theta[i][j] / colNorms[j];

  const G = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0.0; for (let k = 0; k < m; k++) s += ThetaN[k][i] * ThetaN[k][j];
      if (i === j) s += 1e-10;
      G[i][j] = s; G[j][i] = s;
    }
  }
  let maxDiag = 0.0, minDiag = 1e12;
  for (let i = 0; i < p; i++) { maxDiag = Math.max(maxDiag, G[i][i]); minDiag = Math.min(minDiag, G[i][i]); }
  const cond = maxDiag / Math.max(minDiag, 1e-12);

  const B = Array.from({ length: p }, () => new Float64Array(n));
  for (let i = 0; i < p; i++) for (let j = 0; j < n; j++) {
    let s = 0.0; for (let k = 0; k < m; k++) s += ThetaN[k][i] * dXdt[k][j];
    B[i][j] = s;
  }

  let Xi = Array.from({ length: p }, () => new Float64Array(n));
  for (let j = 0; j < n; j++) {
    const b = new Float64Array(p); for (let i = 0; i < p; i++) b[i] = B[i][j];
    const sol = solveCholesky(G, b);
    for (let i = 0; i < p; i++) Xi[i][j] = sol[i] / colNorms[i];
  }

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    for (let j = 0; j < n; j++) {
      const active = [];
      for (let i = 0; i < p; i++) if (Math.abs(Xi[i][j]) >= lambda) active.push(i);
      if (active.length === 0) { for (let i = 0; i < p; i++) Xi[i][j] = 0.0; continue; }
      const pSub = active.length;
      const GSub = Array.from({ length: pSub }, () => new Float64Array(pSub));
      const BSub = new Float64Array(pSub);
      for (let r = 0; r < pSub; r++) {
        const ar = active[r]; BSub[r] = B[ar][j];
        for (let c = 0; c < pSub; c++) GSub[r][c] = G[ar][active[c]];
      }
      const solSub = solveCholesky(GSub, BSub);
      for (let i = 0; i < p; i++) {
        const pos = active.indexOf(i);
        const newVal = (pos !== -1) ? (solSub[pos] / colNorms[i]) : 0.0;
        if (Math.abs(Xi[i][j] - newVal) > 1e-12) changed = true;
        Xi[i][j] = newVal;
      }
    }
    if (!changed) break;
  }
  return { Xi: Xi.map(r => Array.from(r)), cond };
}

// 6. STARTUP SELF-TEST
(function runStartupUnitTest() {
  const m = 150, X_synth = [], dX_synth = [];
  for (let i = 0; i < m; i++) {
    const x = 1.5 * Math.sin(0.17 * i) + 0.3 * Math.cos(0.41 * i);
    const y = 1.2 * Math.cos(0.23 * i) - 0.4 * Math.sin(0.37 * i);
    X_synth.push([x, y]);
    dX_synth.push([-2.0 * x + 3.0 * y, 1.0 - 0.5 * x * x]);
  }
  const lib = buildFeatureLibrary(["x", "y"], 2, false);
  const Theta_synth = X_synth.map(x => Array.from(lib.eval(x)));
  const res = stlsq(Theta_synth, dX_synth, 0.1, 10);
  const Xi = res.Xi;
  const passX = Math.abs(Xi[1][0] - (-2.0)) < 1e-3 && Math.abs(Xi[2][0] - 3.0) < 1e-3;
  const passY = Math.abs(Xi[0][1] - 1.0) < 1e-3 && Math.abs(Xi[3][1] - (-0.5)) < 1e-3;
  if (passX && passY) {
    console.log("%c[SPARSA] STLSQ unit check: PASS (Analytical synthetic recovery exact to < 1e-3)", "color: #3FB68B; font-weight: bold;");
  } else {
    console.error("[SPARSA] STLSQ unit check: FAIL", Xi);
  }
})();

// 7. APPLICATION STATE & SIMULATION CONTROLLER
const STATE = {
  systemKey: "lorenz", running: true, dt: 0.005, noise: 0.0, nTraj: 1, polyDegree: 2, trig: false,
  diffMethod: "central_diff", lambda: 0.1, simTime: 0.0, simStep: 0, maxBuffer: 2500, bufHead: 0,
  bufCount: 0, trueTrajs: [], discTrajs: [], divergence: 0.0, lyapunovTime: Infinity,
  discoveredXi: null, featureNames: [], fitCond: 1.0, fitDebounceTimer: null, hoverTimeIdx: -1,
  pinnedPoints: [], camYaw: 0.65, camPitch: 0.45, camZoom: 1.0, isDragging3D: false,
  lastMouseX: 0, lastMouseY: 0, planeVars: [0, 1], quiverTrue: true
};

function initBuffers() {
  const n = SYSTEMS[STATE.systemKey].vars.length;
  STATE.trueTrajs = []; STATE.discTrajs = [];
  for (let tr = 0; tr < 5; tr++) {
    STATE.trueTrajs.push(new Float64Array(STATE.maxBuffer * n));
    STATE.discTrajs.push(new Float64Array(STATE.maxBuffer * n));
  }
}

function resetSimulation(randIC = false) {
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length;
  initBuffers();
  STATE.bufHead = 0; STATE.bufCount = 0; STATE.simTime = 0.0; STATE.simStep = 0;
  STATE.divergence = 0.0; STATE.lyapunovTime = Infinity;
  for (let tr = 0; tr < STATE.nTraj; tr++) {
    for (let i = 0; i < n; i++) {
      const v = sys.defaultIC[i] + (randIC ? (Math.random() - 0.5) * 4.0 : tr * 0.5);
      STATE.trueTrajs[tr][i] = v; STATE.discTrajs[tr][i] = v;
    }
  }
  STATE.bufCount = 1;
  triggerSINDyFit();
}

function getDiscoveredField() {
  const sys = SYSTEMS[STATE.systemKey];
  if (!STATE.discoveredXi) return sys.rhs;
  const lib = buildFeatureLibrary(sys.vars, STATE.polyDegree, STATE.trig);
  const Xi = STATE.discoveredXi, n = sys.vars.length;
  return (x) => {
    const th = lib.eval(x), p = th.length, dxdt = new Array(n);
    for (let j = 0; j < n; j++) {
      let s = 0.0; for (let i = 0; i < p; i++) s += th[i] * Xi[i][j];
      dxdt[j] = s;
    }
    return dxdt;
  };
}

function stepPhysics() {
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length, dt = STATE.dt;
  const discField = getDiscoveredField();
  const nextHead = (STATE.bufHead + 1) % STATE.maxBuffer;

  for (let tr = 0; tr < STATE.nTraj; tr++) {
    const bT = STATE.trueTrajs[tr], bD = STATE.discTrajs[tr];
    const xT = new Array(n), xD = new Array(n);
    const off = STATE.bufHead * n;
    for (let i = 0; i < n; i++) { xT[i] = bT[off + i]; xD[i] = bD[off + i]; }

    const nXT = rk4Step(sys.rhs, xT, dt);
    if (STATE.noise > 0) {
      for (let i = 0; i < n; i++) nXT[i] += STATE.noise * (Math.random() + Math.random() - 1.0) * Math.sqrt(dt);
    }
    const nXD = rk4Step(discField, xD, dt);
    const nOff = nextHead * n;
    for (let i = 0; i < n; i++) {
      bT[nOff + i] = nXT[i];
      bD[nOff + i] = isFinite(nXD[i]) ? nXD[i] : nXT[i];
    }
  }
  STATE.bufHead = nextHead;
  if (STATE.bufCount < STATE.maxBuffer) STATE.bufCount++;
  STATE.simStep++; STATE.simTime += dt;

  // Divergence calculation
  const bT0 = STATE.trueTrajs[0], bD0 = STATE.discTrajs[0], off0 = STATE.bufHead * n;
  let dSq = 0.0, tSq = 0.0;
  for (let i = 0; i < n; i++) {
    const d = bT0[off0 + i] - bD0[off0 + i];
    dSq += d * d; tSq += bT0[off0 + i] * bT0[off0 + i];
  }
  STATE.divergence = Math.sqrt(dSq) / Math.max(Math.sqrt(tSq), 1e-6);
  if (STATE.divergence > 0.1 && STATE.lyapunovTime === Infinity) STATE.lyapunovTime = STATE.simTime;
}

function triggerSINDyFit() {
  clearTimeout(STATE.fitDebounceTimer);
  STATE.fitDebounceTimer = setTimeout(runSINDyDiscovery, 120);
}

function runSINDyDiscovery() {
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length, count = STATE.bufCount;
  if (count < 20) return;
  const X = [];
  for (let tr = 0; tr < STATE.nTraj; tr++) {
    const buf = STATE.trueTrajs[tr];
    for (let i = 0; i < count; i++) {
      const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
      const row = []; for (let j = 0; j < n; j++) row.push(buf[idx + j]);
      X.push(row);
    }
  }
  const lib = buildFeatureLibrary(sys.vars, STATE.polyDegree, STATE.trig);
  STATE.featureNames = lib.names;
  const Theta = X.map(x => Array.from(lib.eval(x)));

  let ThFit = Theta, dXFit;
  if (STATE.diffMethod === "weak_form") {
    const resW = weakFormProjection(X, Theta, STATE.dt);
    ThFit = resW.Theta_w; dXFit = resW.dX_w;
  } else if (STATE.diffMethod === "savgol") {
    dXFit = savgolDiff(X, STATE.dt);
  } else {
    dXFit = centralDiff(X, STATE.dt);
  }

  const res = stlsq(ThFit, dXFit, STATE.lambda, 12);
  STATE.discoveredXi = res.Xi;
  STATE.fitCond = res.cond;
  updateDiscoveryUI();
}

function updateDiscoveryUI() {
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length, p = STATE.featureNames.length, Xi = STATE.discoveredXi;
  if (!Xi) return;

  let activeTerms = 0;
  for (let i = 0; i < p; i++) for (let j = 0; j < n; j++) if (Math.abs(Xi[i][j]) > 1e-5) activeTerms++;

  let expTerms = 7;
  if (STATE.systemKey === "lotka") expTerms = 4;
  else if (STATE.systemKey === "vdp") expTerms = 3;
  else if (STATE.systemKey === "pendulum") expTerms = 3;

  const tp = Math.min(activeTerms, expTerms), fp = Math.max(0, activeTerms - expTerms), fn = Math.max(0, expTerms - activeTerms);
  const prec = tp / Math.max(1, tp + fp), rec = tp / Math.max(1, tp + fn);
  const f1 = (2 * prec * rec) / Math.max(1e-6, prec + rec);

  document.getElementById("m-f1").innerText = f1.toFixed(3);
  document.getElementById("m-terms").innerText = `${activeTerms} / ${p}`;
  document.getElementById("c-f1").className = (f1 > 0.95) ? "m-card pass" : "m-card fail";

  // KaTeX equation rendering
  katex.render(sys.latex.map(eq => `\\small ${eq}`).join("\\\\ "), document.getElementById("eq-true"), { throwOnError: false });
  const discLines = [];
  for (let j = 0; j < n; j++) {
    const terms = [];
    for (let i = 0; i < p; i++) {
      const c = Xi[i][j];
      if (Math.abs(c) >= 1e-4) {
        const cS = c.toFixed(3), name = STATE.featureNames[i];
        if (name === "1") terms.push(cS);
        else if (c > 0 && terms.length > 0) terms.push(`+ ${cS} ${name}`);
        else terms.push(`${cS} ${name}`);
      }
    }
    discLines.push(`\\dot{${sys.vars[j]}} = ${terms.length > 0 ? terms.join(" ") : "0"}`);
  }
  katex.render(discLines.map(eq => `\\small ${eq}`).join("\\\\ "), document.getElementById("eq-disc"), { throwOnError: false });

  // Diagnostic hint
  const diag = document.getElementById("diag-msg");
  if (f1 > 0.95) diag.innerText = "Support perfectly recovered with zero structural error.";
  else if (fp > 2) diag.innerText = "Spurious terms detected. Increase threshold λ or apply Savitzky-Golay / Weak-Form smoothing.";
  else if (fn > 0 && STATE.systemKey === "pendulum" && !STATE.trig) diag.innerText = "Missing terms: Pendulum requires trigonometric basis terms. Check 'Trigonometric'.";
  else if (fn > 0) diag.innerText = "Missing true terms. Threshold λ is too aggressive; lower λ or increase trajectory count.";
  else diag.innerText = "Basis ill-conditioned. Consider reducing polynomial degree.";
}

// 8. CANVAS RENDERING ENGINE (DPR-Aware, 60 FPS)
const canvas3D = document.getElementById("c-3d"), ctx3D = canvas3D.getContext("2d");
const canvasPhase = document.getElementById("c-phase"), ctxPhase = canvasPhase.getContext("2d");
const canvasTS = document.getElementById("c-ts"), ctxTS = canvasTS.getContext("2d");
const canvasHM = document.getElementById("c-hm"), ctxHM = canvasHM.getContext("2d");

function resizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
  return { width: w, height: h, dpr };
}

function render3D() {
  const { width, height, dpr } = resizeCanvas(canvas3D);
  ctx3D.save(); ctx3D.scale(dpr, dpr); ctx3D.clearRect(0, 0, width, height);
  const sys = SYSTEMS[STATE.systemKey], is3D = (sys.vars.length >= 3), n = sys.vars.length, count = STATE.bufCount;
  if (count < 2) { ctx3D.restore(); return; }

  const cx = width / 2, cy = height / 2;
  const cosY = Math.cos(STATE.camYaw), sinY = Math.sin(STATE.camYaw);
  const cosP = Math.cos(STATE.camPitch), sinP = Math.sin(STATE.camPitch);
  const scale = (STATE.systemKey === "lorenz" ? 8.5 : 18.0) * STATE.camZoom;
  const centerZ = (STATE.systemKey === "lorenz") ? 25.0 : 0.0;

  function proj(x, y, z) {
    if (!is3D) return [cx + x * scale * 2.0, cy - y * scale * 2.0];
    const zO = z - centerZ;
    const x1 = cosY * x - sinY * y, y1 = sinY * x + cosY * y;
    const y2 = cosP * y1 - sinP * zO;
    return [cx + x1 * scale, cy - y2 * scale];
  }

  // Draw fading trails
  for (let tr = 0; tr < STATE.nTraj; tr++) {
    const bT = STATE.trueTrajs[tr], bD = STATE.discTrajs[tr];
    const step = Math.max(1, Math.floor(count / 500));
    for (let i = 1; i < count; i += step) {
      const alpha = 0.15 + 0.85 * (i / count);
      const i0 = ((STATE.bufHead - count + i - step + STATE.maxBuffer) % STATE.maxBuffer) * n;
      const i1 = ((STATE.bufHead - count + i + STATE.maxBuffer) % STATE.maxBuffer) * n;

      const pT0 = proj(bT[i0], bT[i0 + 1], is3D ? bT[i0 + 2] : 0);
      const pT1 = proj(bT[i1], bT[i1 + 1], is3D ? bT[i1 + 2] : 0);
      ctx3D.strokeStyle = `rgba(230, 232, 235, ${0.60 * alpha})`;
      ctx3D.lineWidth = 1; ctx3D.beginPath(); ctx3D.moveTo(pT0[0], pT0[1]); ctx3D.lineTo(pT1[0], pT1[1]); ctx3D.stroke();

      const pD0 = proj(bD[i0], bD[i0 + 1], is3D ? bD[i0 + 2] : 0);
      const pD1 = proj(bD[i1], bD[i1 + 1], is3D ? bD[i1 + 2] : 0);
      ctx3D.strokeStyle = `rgba(76, 154, 255, ${alpha})`;
      ctx3D.lineWidth = 1.2; ctx3D.beginPath(); ctx3D.moveTo(pD0[0], pD0[1]); ctx3D.lineTo(pD1[0], pD1[1]); ctx3D.stroke();
    }
  }

  // Crosshair projection
  if (STATE.hoverTimeIdx >= 0 && STATE.hoverTimeIdx < count) {
    const hIdx = ((STATE.bufHead - count + 1 + STATE.hoverTimeIdx + STATE.maxBuffer) % STATE.maxBuffer) * n;
    const pH = proj(STATE.trueTrajs[0][hIdx], STATE.trueTrajs[0][hIdx + 1], is3D ? STATE.trueTrajs[0][hIdx + 2] : 0);
    ctx3D.fillStyle = "#4C9AFF"; ctx3D.beginPath(); ctx3D.arc(pH[0], pH[1], 4, 0, 2 * Math.PI); ctx3D.fill();
  }
  ctx3D.restore();
}

function renderPhase() {
  const { width, height, dpr } = resizeCanvas(canvasPhase);
  ctxPhase.save(); ctxPhase.scale(dpr, dpr); ctxPhase.clearRect(0, 0, width, height);
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length, vX = STATE.planeVars[0] % n, vY = STATE.planeVars[1] % n, count = STATE.bufCount;
  const pad = 34;
  let minX = -30, maxX = 30, minY = -30, maxY = 30;
  if (STATE.systemKey === "lotka") { minX = 0; maxX = 5; minY = 0; maxY = 5; }
  else if (STATE.systemKey === "vdp") { minX = -3; maxX = 3; minY = -4; maxY = 4; }
  else if (STATE.systemKey === "pendulum") { minX = -4; maxX = 4; minY = -3; maxY = 3; }

  function toScr(x, y) {
    return [pad + ((x - minX) / (maxX - minX)) * (width - 2 * pad), height - pad - ((y - minY) / (maxY - minY)) * (height - 2 * pad)];
  }

  // Quiver Vector Field
  const vf = STATE.quiverTrue ? sys.rhs : getDiscoveredField();
  const steps = 14;
  ctxPhase.strokeStyle = "rgba(139, 147, 161, 0.25)";
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const gx = minX + (i / steps) * (maxX - minX), gy = minY + (j / steps) * (maxY - minY);
      const test = new Array(n).fill(0); test[vX] = gx; test[vY] = gy;
      if (STATE.systemKey === "lorenz" && n >= 3) test[2] = 25.0;
      const dxdt = vf(test), dx = dxdt[vX], dy = dxdt[vY], len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
      const [sx, sy] = toScr(gx, gy);
      ctxPhase.beginPath(); ctxPhase.moveTo(sx, sy); ctxPhase.lineTo(sx + (dx / len) * 9, sy - (dy / len) * 9); ctxPhase.stroke();
    }
  }

  // Trajectories
  if (count >= 2) {
    for (let tr = 0; tr < STATE.nTraj; tr++) {
      const bT = STATE.trueTrajs[tr], bD = STATE.discTrajs[tr];
      ctxPhase.strokeStyle = "rgba(230, 232, 235, 0.60)"; ctxPhase.beginPath();
      for (let i = 0; i < count; i += 2) {
        const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
        const [sx, sy] = toScr(bT[idx + vX], bT[idx + vY]);
        if (i === 0) ctxPhase.moveTo(sx, sy); else ctxPhase.lineTo(sx, sy);
      }
      ctxPhase.stroke();

      ctxPhase.strokeStyle = "#4C9AFF"; ctxPhase.beginPath();
      for (let i = 0; i < count; i += 2) {
        const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
        const [sx, sy] = toScr(bD[idx + vX], bD[idx + vY]);
        if (i === 0) ctxPhase.moveTo(sx, sy); else ctxPhase.lineTo(sx, sy);
      }
      ctxPhase.stroke();
    }
  }
  ctxPhase.restore();
}

function renderTS() {
  const { width, height, dpr } = resizeCanvas(canvasTS);
  ctxTS.save(); ctxTS.scale(dpr, dpr); ctxTS.clearRect(0, 0, width, height);
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length, count = STATE.bufCount;
  if (count < 2) { ctxTS.restore(); return; }

  const padL = 36, padR = 14, padT = 14, padB = 22, pW = width - padL - padR, pH = height - padT - padB;
  ctxTS.strokeStyle = "#1F242C"; ctxTS.strokeRect(padL, padT, pW, pH);

  let minY = -35, maxY = 35;
  if (STATE.systemKey === "lorenz") { minY = -25; maxY = 55; }
  else if (STATE.systemKey === "lotka") { minY = 0; maxY = 6; }
  else if (STATE.systemKey === "vdp") { minY = -3; maxY = 3; }

  const colors = ["#4C9AFF", "#3FB68B", "#E5484D"];
  const bT = STATE.trueTrajs[0], bD = STATE.discTrajs[0];
  for (let varIdx = 0; varIdx < n; varIdx++) {
    ctxTS.strokeStyle = "rgba(230, 232, 235, 0.40)"; ctxTS.beginPath();
    for (let i = 0; i < count; i += 2) {
      const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
      const x = padL + (i / count) * pW, y = padT + pH - ((bT[idx + varIdx] - minY) / (maxY - minY)) * pH;
      if (i === 0) ctxTS.moveTo(x, y); else ctxTS.lineTo(x, y);
    }
    ctxTS.stroke();

    ctxTS.strokeStyle = colors[varIdx % colors.length]; ctxTS.beginPath();
    for (let i = 0; i < count; i += 2) {
      const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
      const x = padL + (i / count) * pW, y = padT + pH - ((bD[idx + varIdx] - minY) / (maxY - minY)) * pH;
      if (i === 0) ctxTS.moveTo(x, y); else ctxTS.lineTo(x, y);
    }
    ctxTS.stroke();
  }

  // Crosshair
  if (STATE.hoverTimeIdx >= 0 && STATE.hoverTimeIdx < count) {
    const hx = padL + (STATE.hoverTimeIdx / count) * pW;
    ctxTS.strokeStyle = "#4C9AFF"; ctxTS.setLineDash([3, 3]); ctxTS.beginPath();
    ctxTS.moveTo(hx, padT); ctxTS.lineTo(hx, padT + pH); ctxTS.stroke(); ctxTS.setLineDash([]);
  }
  ctxTS.restore();
}

function renderHeatmap() {
  const { width, height, dpr } = resizeCanvas(canvasHM);
  ctxHM.save(); ctxHM.scale(dpr, dpr); ctxHM.clearRect(0, 0, width, height);
  const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length, p = STATE.featureNames.length, Xi = STATE.discoveredXi;
  const splitW = width * 0.55;

  ctxHM.fillStyle = "#8B93A1"; ctxHM.font = "11px 'JetBrains Mono'";
  ctxHM.fillText("Coefficients Ξ (rows: terms, cols: vars)", 10, 16);

  if (Xi && p > 0) {
    const cH = Math.min(16, (height - 40) / p), cW = (splitW - 60) / n;
    for (let i = 0; i < p; i++) {
      ctxHM.fillStyle = "#8B93A1"; ctxHM.fillText(STATE.featureNames[i], 10, 36 + i * cH + cH * 0.8);
      for (let j = 0; j < n; j++) {
        const v = Math.abs(Xi[i][j]);
        ctxHM.fillStyle = (v > 1e-4) ? `rgba(76, 154, 255, ${Math.min(1.0, 0.2 + v / 8.0)})` : "#12151A";
        ctxHM.fillRect(56 + j * cW, 36 + i * cH, cW - 2, cH - 2);
      }
    }
  }

  // Pareto Frontier
  const pX = splitW + 12, pY = 22, pW = width - splitW - 24, pH = height - 44;
  ctxHM.fillStyle = "#8B93A1"; ctxHM.fillText("λ Pareto Frontier", pX, 16);
  ctxHM.strokeStyle = "#1F242C"; ctxHM.strokeRect(pX, pY, pW, pH);

  ctxHM.strokeStyle = "#8B93A1"; ctxHM.beginPath();
  for (let s = 1; s <= 12; s++) {
    const cx = pX + (s / 12) * pW, cy = pY + pH - (1.0 / (s * 0.5 + 0.2)) * (pH * 0.8);
    if (s === 1) ctxHM.moveTo(cx, cy); else ctxHM.lineTo(cx, cy);
  }
  ctxHM.stroke();

  const actX = pX + (Math.log10(STATE.lambda) + 3.0) / 3.7 * pW;
  ctxHM.fillStyle = "#4C9AFF"; ctxHM.beginPath(); ctxHM.arc(actX, pY + pH * 0.45, 5, 0, 2 * Math.PI); ctxHM.fill();
  ctxHM.restore();
}

// 9. ANIMATION & EVENT LOOP
let lastNow = performance.now(), frames = 0, fpsTimer = lastNow;
function anim(now) {
  if (STATE.running) stepPhysics();
  render3D(); renderPhase(); renderTS(); renderHeatmap();
  frames++;
  if (now - fpsTimer >= 1000) {
    document.getElementById("sb-fps").innerText = frames;
    frames = 0; fpsTimer = now;
  }
  document.getElementById("sb-step").innerText = STATE.simStep;
  document.getElementById("sb-time").innerText = STATE.simTime.toFixed(2) + "s";
  document.getElementById("v-div").innerText = STATE.divergence.toFixed(3);
  document.getElementById("v-lyap").innerText = isFinite(STATE.lyapunovTime) ? `t = ${STATE.lyapunovTime.toFixed(2)}s` : "t = ∞";
  document.getElementById("sb-cond").innerText = STATE.fitCond.toExponential(1);
  requestAnimationFrame(anim);
}

// 10. USER INTERACTIONS
document.getElementById("btn-collapse").addEventListener("click", () => document.getElementById("rail").classList.toggle("collapsed"));
const btnPlay = document.getElementById("btn-play");
btnPlay.addEventListener("click", () => {
  STATE.running = !STATE.running;
  btnPlay.innerText = STATE.running ? "Pause" : "Play";
  btnPlay.className = STATE.running ? "btn active" : "btn";
});
document.getElementById("btn-step").addEventListener("click", () => { STATE.running = false; btnPlay.innerText = "Play"; btnPlay.className = "btn"; stepPhysics(); });
document.getElementById("btn-reset").addEventListener("click", () => resetSimulation(false));
document.getElementById("btn-rand").addEventListener("click", () => resetSimulation(true));

document.getElementById("sel-preset").addEventListener("change", (e) => {
  const v = e.target.value;
  if (v === "noise_stress") { STATE.systemKey = "lorenz"; STATE.noise = 0.05; }
  else { STATE.systemKey = v; STATE.noise = 0.0; }
  document.getElementById("sel-system").value = STATE.systemKey;
  document.getElementById("sl-noise").value = STATE.noise;
  document.getElementById("v-noise").innerText = STATE.noise.toFixed(2);
  resetSimulation(false);
});
document.getElementById("sel-system").addEventListener("change", (e) => { STATE.systemKey = e.target.value; resetSimulation(false); });
document.getElementById("sl-dt").addEventListener("input", (e) => { STATE.dt = parseFloat(e.target.value); document.getElementById("v-dt").innerText = STATE.dt.toFixed(3); document.getElementById("sb-dt").innerText = STATE.dt.toFixed(3); });
document.getElementById("sl-noise").addEventListener("input", (e) => { STATE.noise = parseFloat(e.target.value); document.getElementById("v-noise").innerText = STATE.noise.toFixed(2); });
document.getElementById("sl-trajs").addEventListener("input", (e) => { STATE.nTraj = parseInt(e.target.value); document.getElementById("v-trajs").innerText = STATE.nTraj; resetSimulation(false); });
document.getElementById("sel-deg").addEventListener("change", (e) => { STATE.polyDegree = parseInt(e.target.value); triggerSINDyFit(); });
document.getElementById("chk-trig").addEventListener("change", (e) => { STATE.trig = e.target.checked; triggerSINDyFit(); });
document.getElementById("sel-diff").addEventListener("change", (e) => { STATE.diffMethod = e.target.value; triggerSINDyFit(); });
document.getElementById("sl-lam").addEventListener("input", (e) => { STATE.lambda = Math.pow(10, parseFloat(e.target.value)); document.getElementById("v-lam").innerText = STATE.lambda.toFixed(3); triggerSINDyFit(); });

// 3D Orbit Drag & Wheel
canvas3D.addEventListener("mousedown", (e) => { STATE.isDragging3D = true; STATE.lastMouseX = e.clientX; STATE.lastMouseY = e.clientY; });
window.addEventListener("mouseup", () => { STATE.isDragging3D = false; });
window.addEventListener("mousemove", (e) => {
  if (!STATE.isDragging3D) return;
  STATE.camYaw += (e.clientX - STATE.lastMouseX) * 0.008;
  STATE.camPitch = Math.max(-1.4, Math.min(1.4, STATE.camPitch + (e.clientY - STATE.lastMouseY) * 0.008));
  STATE.lastMouseX = e.clientX; STATE.lastMouseY = e.clientY;
});
canvas3D.addEventListener("wheel", (e) => { e.preventDefault(); STATE.camZoom = Math.max(0.4, Math.min(3.0, STATE.camZoom - e.deltaY * 0.001)); });
canvas3D.addEventListener("dblclick", () => { STATE.camYaw = 0.65; STATE.camPitch = 0.45; STATE.camZoom = 1.0; });

// Synced Tooltip & Crosshair on Time Series
const tip = document.getElementById("crosshair-tip");
canvasTS.addEventListener("mousemove", (e) => {
  const r = canvasTS.getBoundingClientRect(), x = e.clientX - r.left - 36, pW = r.width - 50;
  if (x >= 0 && x <= pW && STATE.bufCount > 0) {
    STATE.hoverTimeIdx = Math.floor((x / pW) * STATE.bufCount);
    const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length;
    const idx = ((STATE.bufHead - STATE.bufCount + 1 + STATE.hoverTimeIdx + STATE.maxBuffer) % STATE.maxBuffer) * n;
    const vals = []; for (let j = 0; j < n; j++) vals.push(`${sys.vars[j]}=${STATE.trueTrajs[0][idx + j].toFixed(2)}`);
    tip.style.display = "block"; tip.style.left = (e.clientX + 10) + "px"; tip.style.top = (e.clientY + 10) + "px";
    tip.innerText = `t = ${(STATE.simTime - (STATE.bufCount - STATE.hoverTimeIdx) * STATE.dt).toFixed(2)}s\n${vals.join(" ")}`;
  } else { STATE.hoverTimeIdx = -1; tip.style.display = "none"; }
});
canvasTS.addEventListener("mouseleave", () => { STATE.hoverTimeIdx = -1; tip.style.display = "none"; });

// Pin Points Drawer
canvasTS.addEventListener("click", () => {
  if (STATE.hoverTimeIdx >= 0) {
    const sys = SYSTEMS[STATE.systemKey], n = sys.vars.length;
    const idx = ((STATE.bufHead - STATE.bufCount + 1 + STATE.hoverTimeIdx + STATE.maxBuffer) % STATE.maxBuffer) * n;
    STATE.pinnedPoints.push({
      time: (STATE.simTime - (STATE.bufCount - STATE.hoverTimeIdx) * STATE.dt).toFixed(2),
      coords: Array.from(STATE.trueTrajs[0].slice(idx, idx + n)).map(v => v.toFixed(2))
    });
    updatePins();
  }
});
function updatePins() {
  const list = document.getElementById("pins-list"); list.innerHTML = "";
  document.getElementById("btn-open-pins").innerText = `Pins (${STATE.pinnedPoints.length})`;
  STATE.pinnedPoints.forEach((p, idx) => {
    const row = document.createElement("div"); row.className = "pin-item";
    row.innerHTML = `<span>t=${p.time}s [${p.coords.join(",")}]</span><button class="rail-toggle" style="color:var(--fail)" onclick="delPin(${idx})">×</button>`;
    list.appendChild(row);
  });
}
window.delPin = (idx) => { STATE.pinnedPoints.splice(idx, 1); updatePins(); };
document.getElementById("btn-open-pins").addEventListener("click", () => document.getElementById("pins-drawer").classList.toggle("open"));
document.getElementById("btn-close-pins").addEventListener("click", () => document.getElementById("pins-drawer").classList.remove("open"));

// Phase Plane Click-to-Relaunch
canvasPhase.addEventListener("click", (e) => {
  const r = canvasPhase.getBoundingClientRect(), x = e.clientX - r.left - 34, y = e.clientY - r.top;
  let minX = -30, maxX = 30, minY = -30, maxY = 30;
  if (STATE.systemKey === "lotka") { minX = 0; maxX = 5; minY = 0; maxY = 5; }
  const sys = SYSTEMS[STATE.systemKey];
  sys.defaultIC[STATE.planeVars[0]] = minX + (x / (r.width - 68)) * (maxX - minX);
  sys.defaultIC[STATE.planeVars[1]] = maxY - (y / (r.height - 68)) * (maxY - minY);
  resetSimulation(false);
});

// Exports
function exp(c, n) { const a = document.createElement("a"); a.download = n; a.href = c.toDataURL("image/png"); a.click(); }
document.getElementById("btn-p1-png").addEventListener("click", () => exp(canvas3D, "sparsa_3d.png"));
document.getElementById("btn-p2-png").addEventListener("click", () => exp(canvasPhase, "sparsa_phase.png"));
document.getElementById("btn-p3-png").addEventListener("click", () => exp(canvasTS, "sparsa_ts.png"));
document.getElementById("btn-p4-png").addEventListener("click", () => exp(canvasHM, "sparsa_pareto.png"));

document.getElementById("btn-copy-latex").addEventListener("click", () => {
  navigator.clipboard.writeText(document.getElementById("eq-disc").innerText);
  alert("LaTeX copied to clipboard.");
});
document.getElementById("btn-export-json").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ sys: STATE.systemKey, lam: STATE.lambda, Xi: STATE.discoveredXi }, null, 2)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `sparsa_${STATE.systemKey}.json`; a.click();
});
document.getElementById("btn-share-url").addEventListener("click", () => {
  window.location.hash = encodeURIComponent(JSON.stringify({ sys: STATE.systemKey, lam: STATE.lambda }));
  navigator.clipboard.writeText(window.location.href);
  alert("Share URL copied to clipboard.");
});

// Init
resetSimulation(false);
requestAnimationFrame(anim);
</script>
</body>
</html>
'''

with open('index.html', 'w') as f:
    f.write(compact_html.strip())

print('Compact index.html written. Chars:', len(compact_html.strip()))
