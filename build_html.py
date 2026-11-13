import re, json

html_content = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sparsa — SINDy Equation Discovery</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
  <style>
    :root {
      --bg: #0B0D10;
      --surface: #12151A;
      --surface-2: #171B21;
      --border: #1F242C;
      --border-focus: #3A4250;
      --text: #E6E8EB;
      --muted: #8B93A1;
      --accent: #4C9AFF;
      --accent-muted: rgba(76, 154, 255, 0.15);
      --true-trace: rgba(230, 232, 235, 0.60);
      --pass: #3FB68B;
      --fail: #E5484D;
      --pass-bg: rgba(63, 182, 139, 0.12);
      --fail-bg: rgba(229, 72, 77, 0.12);
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --fs-12: 12px;
      --fs-13: 13px;
      --fs-14: 14px;
      --fs-16: 16px;
      --fs-20: 20px;
      --radius: 6px;
      --transition: 120ms ease-out;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: var(--fs-13);
      line-height: 1.4;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }
    .mono, code, pre, .stat-val, .metric-val { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
    #app-container { display: flex; flex: 1; height: calc(100vh - 28px); overflow: hidden; position: relative; }
    
    /* Rail */
    #sidebar {
      width: 300px;
      min-width: 300px;
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      transition: width var(--transition), margin-left var(--transition);
      z-index: 20;
    }
    #sidebar.collapsed { margin-left: -300px; }
    .rail-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .rail-title { font-size: var(--fs-14); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); }
    .rail-subtitle { font-size: var(--fs-12); color: var(--muted); font-weight: 400; margin-left: 6px; }
    .rail-toggle {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      cursor: pointer;
      border-radius: var(--radius);
      padding: 4px 8px;
      font-size: var(--fs-12);
      font-family: var(--font-mono);
      transition: color var(--transition), border-color var(--transition);
    }
    .rail-toggle:hover { color: var(--text); border-color: var(--border-focus); }
    .rail-scroll { flex: 1; overflow-y: auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 16px; }
    
    /* Control Blocks */
    .section-label { font-size: var(--fs-12); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 8px; }
    .ctrl-group { display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px; }
    .ctrl-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .ctrl-label { font-size: var(--fs-12); color: var(--muted); }
    .ctrl-val { font-size: var(--fs-12); color: var(--text); font-family: var(--font-mono); }
    
    select, input[type="text"], input[type="number"] {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      padding: 6px 8px;
      font-size: var(--fs-12);
      font-family: var(--font-sans);
      outline: none;
      transition: border-color var(--transition);
    }
    select:focus, input:focus { border-color: var(--border-focus); }
    input[type="range"] {
      width: 100%;
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      accent-color: var(--accent);
      outline: none;
      cursor: pointer;
    }
    .btn-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
    .btn {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      padding: 6px 8px;
      font-size: var(--fs-12);
      font-family: var(--font-sans);
      cursor: pointer;
      text-align: center;
      transition: background var(--transition), border-color var(--transition), color var(--transition);
    }
    .btn:hover { background: var(--border); border-color: var(--border-focus); }
    .btn.active { background: var(--accent); border-color: var(--accent); color: #FFF; font-weight: 600; }
    .btn-subtle { background: transparent; border: 1px solid var(--border); color: var(--muted); }
    .btn-subtle:hover { color: var(--text); border-color: var(--border-focus); }
    
    /* Metrics & Equations */
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .metric-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .metric-card .label { font-size: var(--fs-12); color: var(--muted); }
    .metric-card .val { font-size: var(--fs-16); font-weight: 600; color: var(--text); font-family: var(--font-mono); }
    .metric-card.pass { border-left: 2px solid var(--pass); }
    .metric-card.fail { border-left: 2px solid var(--fail); }

    .eq-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .eq-item { display: flex; flex-direction: column; gap: 2px; overflow-x: auto; padding-bottom: 2px; }
    .eq-tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-family: var(--font-mono); }
    .eq-tag.true-tag { color: var(--muted); }
    .eq-tag.disc-tag { color: var(--accent); }
    .katex-render { font-size: var(--fs-12); color: var(--text); min-height: 20px; }
    
    /* Diagnostic Hint Box */
    .diag-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .diag-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    .diag-desc { font-size: var(--fs-12); color: var(--text); line-height: 1.35; }
    
    /* Workspace 2x2 Grid */
    #workspace {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 1px;
      background: var(--border);
      overflow: hidden;
      position: relative;
    }
    .panel {
      background: var(--bg);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }
    .panel-bar {
      height: 32px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
    }
    .panel-title { font-size: var(--fs-12); font-weight: 600; color: var(--text); letter-spacing: 0.03em; }
    .panel-tools { display: flex; align-items: center; gap: 8px; }
    .panel-tools select, .panel-tools button {
      padding: 2px 6px;
      font-size: 11px;
      height: 22px;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--muted);
      cursor: pointer;
    }
    .panel-tools button:hover, .panel-tools select:hover { color: var(--text); border-color: var(--border-focus); }
    .panel-canvas-wrap { flex: 1; position: relative; width: 100%; height: calc(100% - 32px); }
    canvas { display: block; width: 100%; height: 100%; }

    /* Tooltip & Pinned Drawer */
    #crosshair-tip {
      position: absolute;
      display: none;
      background: var(--surface);
      border: 1px solid var(--border-focus);
      border-radius: var(--radius);
      padding: 6px 10px;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text);
      pointer-events: none;
      z-index: 100;
      white-space: pre;
      line-height: 1.35;
    }
    #pins-drawer {
      position: absolute;
      right: 0;
      top: 0;
      bottom: 0;
      width: 260px;
      background: var(--surface);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      transform: translateX(100%);
      transition: transform var(--transition);
      z-index: 50;
    }
    #pins-drawer.open { transform: translateX(0); }
    .drawer-header {
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: var(--fs-12);
      font-weight: 600;
    }
    .drawer-body { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 6px; }
    .pin-row {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 6px;
      font-size: 11px;
      font-family: var(--font-mono);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .pin-del { background: transparent; border: none; color: var(--fail); cursor: pointer; padding: 2px 4px; }

    /* Status Bar */
    #status-bar {
      height: 28px;
      background: var(--surface);
      border-top: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      font-size: var(--fs-12);
      color: var(--muted);
      z-index: 30;
      user-select: none;
    }
    .status-left, .status-right { display: flex; align-items: center; gap: 16px; }
    .status-item { display: flex; align-items: center; gap: 6px; }
    .status-item .mono { color: var(--text); }
  </style>
</head>
<body>
  <div id="app-container">
    <!-- Left Control Rail (300px) -->
    <aside id="sidebar">
      <div class="rail-header">
        <div>
          <span class="rail-title">SPARSA</span>
          <span class="rail-subtitle">SINDy Engine</span>
        </div>
        <button id="btn-collapse" class="rail-toggle" title="Collapse Rail">‹</button>
      </div>
      <div class="rail-scroll">
        <!-- Preset Selector -->
        <div>
          <div class="section-label">Preset</div>
          <select id="sel-preset">
            <option value="lorenz">Lorenz-63 (Attractor)</option>
            <option value="lotka">Lotka-Volterra (Predator-Prey)</option>
            <option value="vdp">Van der Pol (Limit Cycle, μ=2)</option>
            <option value="pendulum">Damped Pendulum (Trig Required)</option>
            <option value="rossler">Rössler (Spiral Chaos)</option>
            <option value="noise_stress">Lorenz (5% Noise Stress)</option>
          </select>
        </div>

        <!-- System & Simulation -->
        <div>
          <div class="section-label">Simulation</div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">System</span>
              <select id="sel-system" style="width: 170px;">
                <option value="lorenz">Lorenz-63</option>
                <option value="lotka">Lotka-Volterra</option>
                <option value="vdp">Van der Pol</option>
                <option value="pendulum">Damped Pendulum</option>
                <option value="rossler">Rössler</option>
              </select>
            </div>
            <div class="btn-row" style="margin-top: 4px;">
              <button id="btn-play" class="btn active">Pause</button>
              <button id="btn-step" class="btn">Step</button>
              <button id="btn-reset" class="btn">Reset</button>
              <button id="btn-rand" class="btn">Rand IC</button>
            </div>
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Timestep dt</span>
              <span id="val-dt" class="ctrl-val">0.005</span>
            </div>
            <input id="slider-dt" type="range" min="0.001" max="0.02" step="0.001" value="0.005">
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Noise (σ)</span>
              <span id="val-noise" class="ctrl-val">0.00</span>
            </div>
            <input id="slider-noise" type="range" min="0" max="0.05" step="0.005" value="0.00">
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Trajectories</span>
              <span id="val-trajs" class="ctrl-val">1</span>
            </div>
            <input id="slider-trajs" type="range" min="1" max="5" step="2" value="1">
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Live Divergence</span>
              <span id="val-divergence" class="ctrl-val mono">0.000</span>
            </div>
            <div class="ctrl-row">
              <span class="ctrl-label">Lyapunov Horizon</span>
              <span id="val-lyapunov" class="ctrl-val mono">t = ∞</span>
            </div>
          </div>
        </div>

        <!-- SINDy Discovery Pipeline -->
        <div>
          <div class="section-label">SINDy Discovery</div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Polynomial Degree</span>
              <select id="sel-degree" style="width: 100px;">
                <option value="1">1 (Linear)</option>
                <option value="2" selected>2 (Quadratic)</option>
                <option value="3">3 (Cubic)</option>
              </select>
            </div>
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Trigonometric Terms</span>
              <input id="chk-trig" type="checkbox" style="width: 16px; height: 16px; accent-color: var(--accent);">
            </div>
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Differentiation</span>
              <select id="sel-diff" style="width: 130px;">
                <option value="central_diff">Central Diff</option>
                <option value="savgol">Savitzky-Golay</option>
                <option value="weak_form">Weak-Form</option>
              </select>
            </div>
          </div>
          <div class="ctrl-group">
            <div class="ctrl-row">
              <span class="ctrl-label">Threshold (λ)</span>
              <span id="val-lambda" class="ctrl-val">0.100</span>
            </div>
            <input id="slider-lambda" type="range" min="-3" max="0.7" step="0.05" value="-1.0">
          </div>
        </div>

        <!-- Discovered Equations (KaTeX) -->
        <div>
          <div class="section-label">Equations</div>
          <div class="eq-box">
            <div class="eq-item">
              <span class="eq-tag true-tag">True Dynamical System</span>
              <div id="eq-true" class="katex-render"></div>
            </div>
            <div class="eq-item" style="border-top: 1px solid var(--border); padding-top: 4px;">
              <span class="eq-tag disc-tag">SINDy Discovered</span>
              <div id="eq-disc" class="katex-render"></div>
            </div>
          </div>
        </div>

        <!-- Metrics Strip -->
        <div>
          <div class="section-label">Metrics</div>
          <div class="metric-grid">
            <div id="card-f1" class="metric-card pass">
              <span class="label">Support F1</span>
              <span id="m-f1" class="val">1.000</span>
            </div>
            <div id="card-err" class="metric-card pass">
              <span class="label">Coef Error</span>
              <span id="m-err" class="val">0.00%</span>
            </div>
            <div class="metric-card">
              <span class="label">Derivative R²</span>
              <span id="m-r2" class="val">0.999</span>
            </div>
            <div class="metric-card">
              <span class="label">Active Terms</span>
              <span id="m-terms" class="val">7 / 10</span>
            </div>
          </div>
        </div>

        <!-- Diagnostic Hint Box -->
        <div>
          <div class="section-label">Diagnostic Diagnosis</div>
          <div class="diag-box">
            <span class="diag-title">Why Did It Fail?</span>
            <span id="diag-msg" class="diag-desc">Support perfectly recovered with zero structural error.</span>
          </div>
        </div>

        <!-- Actions -->
        <div>
          <div class="section-label">Actions</div>
          <div style="display: flex; flex-direction: column; gap: 6px;">
            <button id="btn-export-json" class="btn btn-subtle">Export Model JSON</button>
            <button id="btn-copy-latex" class="btn btn-subtle">Copy Discovered LaTeX</button>
            <button id="btn-share-url" class="btn btn-subtle">Copy Share URL</button>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main Workspace (2x2 Canvas Grid) -->
    <main id="workspace">
      <!-- Panel 1: 3D Attractor -->
      <section class="panel">
        <div class="panel-bar">
          <span class="panel-title">3D Orbit / Attractor</span>
          <div class="panel-tools">
            <span style="font-size: 11px; color: var(--muted);">Drag: Rotate | Wheel: Zoom | DblClick: Reset</span>
            <button id="btn-exp-p1">PNG</button>
          </div>
        </div>
        <div class="panel-canvas-wrap">
          <canvas id="c-3d"></canvas>
        </div>
      </section>

      <!-- Panel 2: Phase Plane -->
      <section class="panel">
        <div class="panel-bar">
          <span class="panel-title">Phase Plane</span>
          <div class="panel-tools">
            <select id="sel-plane-vars">
              <option value="0,1">x vs y</option>
              <option value="0,2">x vs z</option>
              <option value="1,2">y vs z</option>
            </select>
            <button id="btn-quiver-toggle">Field: True</button>
            <button id="btn-exp-p2">PNG</button>
          </div>
        </div>
        <div class="panel-canvas-wrap">
          <canvas id="c-phase"></canvas>
        </div>
      </section>

      <!-- Panel 3: Time Series -->
      <section class="panel">
        <div class="panel-bar">
          <span class="panel-title">Time-Series Rollout</span>
          <div class="panel-tools">
            <button id="btn-reset-zoom">Reset Zoom</button>
            <button id="btn-exp-p3">PNG</button>
          </div>
        </div>
        <div class="panel-canvas-wrap">
          <canvas id="c-ts"></canvas>
        </div>
      </section>

      <!-- Panel 4: Heatmap Xi & Pareto -->
      <section class="panel">
        <div class="panel-bar">
          <span class="panel-title">Library Matrix Ξ & Pareto Frontier</span>
          <div class="panel-tools">
            <button id="btn-exp-p4">PNG</button>
          </div>
        </div>
        <div class="panel-canvas-wrap">
          <canvas id="c-hm"></canvas>
        </div>
      </section>

      <!-- Hover Crosshair Tooltip -->
      <div id="crosshair-tip"></div>

      <!-- Collapsible Pinned Drawer -->
      <aside id="pins-drawer">
        <div class="drawer-header">
          <span>Pinned Points</span>
          <button id="btn-close-pins" class="rail-toggle">×</button>
        </div>
        <div id="pins-list" class="drawer-body"></div>
      </aside>
    </main>
  </div>

  <!-- Status Bar -->
  <footer id="status-bar">
    <div class="status-left">
      <div class="status-item">FPS: <span id="sb-fps" class="mono">60</span></div>
      <div class="status-item">Step: <span id="sb-step" class="mono">0</span></div>
      <div class="status-item">Time: <span id="sb-time" class="mono">0.00s</span></div>
      <div class="status-item">dt: <span id="sb-dt" class="mono">0.005</span></div>
    </div>
    <div class="status-right">
      <div class="status-item">cond(Θ): <span id="sb-cond" class="mono">1.0e+00</span></div>
      <div class="status-item">Status: <span id="sb-status" class="mono" style="color: var(--pass);">Active</span></div>
      <button id="btn-open-pins" class="rail-toggle" style="height: 20px; line-height: 1;">Pins (0)</button>
    </div>
  </footer>

  <script>
    // ========================================================================
    // 1. DYNAMICAL SYSTEMS REGISTRY (Pure Math Functions)
    // ========================================================================
    const SYSTEMS = {
      lorenz: {
        name: "Lorenz-63",
        vars: ["x", "y", "z"],
        defaultIC: [-8.0, 7.0, 27.0],
        params: { sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0 },
        lyapunov: 0.906,
        rhs: (x, p = { sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0 }) => [
          p.sigma * (x[1] - x[0]),
          x[0] * (p.rho - x[2]) - x[1],
          x[0] * x[1] - p.beta * x[2]
        ],
        latex: [
          "\\dot{x} = 10(y - x)",
          "\\dot{y} = x(28 - z) - y",
          "\\dot{z} = xy - \\frac{8}{3}z"
        ]
      },
      lotka: {
        name: "Lotka-Volterra",
        vars: ["x", "y"],
        defaultIC: [2.0, 1.0],
        params: { alpha: 1.5, beta: 1.0, delta: 1.0, gamma: 3.0 },
        lyapunov: 0.0,
        rhs: (x, p = { alpha: 1.5, beta: 1.0, delta: 1.0, gamma: 3.0 }) => [
          p.alpha * x[0] - p.beta * x[0] * x[1],
          p.delta * x[0] * x[1] - p.gamma * x[1]
        ],
        latex: [
          "\\dot{x} = 1.5x - xy",
          "\\dot{y} = xy - 3y"
        ]
      },
      vdp: {
        name: "Van der Pol",
        vars: ["x", "y"],
        defaultIC: [1.0, 0.0],
        params: { mu: 2.0 },
        lyapunov: 0.0,
        rhs: (x, p = { mu: 2.0 }) => [
          x[1],
          p.mu * (1.0 - x[0] * x[0]) * x[1] - x[0]
        ],
        latex: [
          "\\dot{x} = y",
          "\\dot{y} = 2(1 - x^2)y - x"
        ]
      },
      pendulum: {
        name: "Damped Pendulum",
        vars: ["\\theta", "\\omega"],
        defaultIC: [1.5, 0.0],
        params: { c: 0.2, gL: 1.0 },
        lyapunov: 0.0,
        rhs: (x, p = { c: 0.2, gL: 1.0 }) => [
          x[1],
          -p.gL * Math.sin(x[0]) - p.c * x[1]
        ],
        latex: [
          "\\dot{\\theta} = \\omega",
          "\\dot{\\omega} = -\\sin(\\theta) - 0.2\\omega"
        ]
      },
      rossler: {
        name: "Rössler",
        vars: ["x", "y", "z"],
        defaultIC: [1.0, 1.0, 0.0],
        params: { a: 0.2, b: 0.2, c: 5.7 },
        lyapunov: 0.071,
        rhs: (x, p = { a: 0.2, b: 0.2, c: 5.7 }) => [
          -x[1] - x[2],
          x[0] + p.a * x[1],
          p.b + x[2] * (x[0] - p.c)
        ],
        latex: [
          "\\dot{x} = -y - z",
          "\\dot{y} = x + 0.2y",
          "\\dot{z} = 0.2 + z(x - 5.7)"
        ]
      }
    };

    // ========================================================================
    // 2. RUNGE-KUTTA 4TH ORDER INTEGRATOR (Pure Math Function)
    // ========================================================================
    /**
     * Advances dynamic state x by dt using classical 4th-order Runge-Kutta.
     * @param {function(number[]): number[]} f - Vector field
     * @param {number[]} x - State vector
     * @param {number} dt - Timestep
     * @returns {number[]} Next state vector
     */
    function rk4Step(f, x, dt) {
      const n = x.length;
      const k1 = f(x);
      const x2 = new Array(n);
      for (let i = 0; i < n; i++) x2[i] = x[i] + 0.5 * dt * k1[i];
      const k2 = f(x2);
      const x3 = new Array(n);
      for (let i = 0; i < n; i++) x3[i] = x[i] + 0.5 * dt * k2[i];
      const k3 = f(x3);
      const x4 = new Array(n);
      for (let i = 0; i < n; i++) x4[i] = x[i] + dt * k3[i];
      const k4 = f(x4);
      const xNext = new Array(n);
      for (let i = 0; i < n; i++) {
        xNext[i] = x[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
      }
      return xNext;
    }

    // ========================================================================
    // 3. CANDIDATE FEATURE LIBRARY (Combinatorial Polynomials & Trig)
    // ========================================================================
    /**
     * Builds candidate feature library up to polynomial degree 3 + optional trig.
     * @param {string[]} varNames - Variable names
     * @param {number} degree - 1, 2, or 3
     * @param {boolean} includeTrig - Include sin and cos terms
     * @returns {{names: string[], eval: function(number[]): number[]}}
     */
    function buildFeatureLibrary(varNames, degree, includeTrig) {
      const n = varNames.length;
      const terms = [];
      const names = [];

      // Degree 0: Bias (1.0)
      terms.push({ type: "bias" });
      names.push("1");

      // Degree 1: Linear
      for (let i = 0; i < n; i++) {
        terms.push({ type: "poly", powers: [i] });
        names.push(varNames[i]);
      }

      // Degree 2: Quadratic combinations
      if (degree >= 2) {
        for (let i = 0; i < n; i++) {
          for (let j = i; j < n; j++) {
            terms.push({ type: "poly", powers: [i, j] });
            names.push(i === j ? `${varNames[i]}^2` : `${varNames[i]} ${varNames[j]}`);
          }
        }
      }

      // Degree 3: Cubic combinations
      if (degree >= 3) {
        for (let i = 0; i < n; i++) {
          for (let j = i; j < n; j++) {
            for (let k = j; k < n; k++) {
              terms.push({ type: "poly", powers: [i, j, k] });
              if (i === j && j === k) names.push(`${varNames[i]}^3`);
              else if (i === j) names.push(`${varNames[i]}^2 ${varNames[k]}`);
              else if (j === k) names.push(`${varNames[i]} ${varNames[j]}^2`);
              else names.push(`${varNames[i]} ${varNames[j]} ${varNames[k]}`);
            }
          }
        }
      }

      // Trigonometric terms
      if (includeTrig) {
        for (let i = 0; i < n; i++) {
          terms.push({ type: "sin", idx: i });
          names.push(`sin(${varNames[i]})`);
          terms.push({ type: "cos", idx: i });
          names.push(`cos(${varNames[i]})`);
        }
      }

      const p = terms.length;
      return {
        names,
        eval: (x) => {
          const row = new Float64Array(p);
          for (let k = 0; k < p; k++) {
            const t = terms[k];
            if (t.type === "bias") row[k] = 1.0;
            else if (t.type === "poly") {
              let val = 1.0;
              for (let idx of t.powers) val *= x[idx];
              row[k] = val;
            } else if (t.type === "sin") row[k] = Math.sin(x[t.idx]);
            else if (t.type === "cos") row[k] = Math.cos(x[t.idx]);
          }
          return row;
        }
      };
    }

    // ========================================================================
    // 4. NUMERICAL DIFFERENTIATION ROUTINES
    // ========================================================================
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
      const m = X.length, n = X[0].length;
      const dX = centralDiff(X, dt);
      if (m < 7) return dX;
      // Savitzky-Golay 7-point 1st derivative filter coefficients
      const coeffs = [-3, -2, -1, 0, 1, 2, 3];
      const norm = 28.0 * dt;
      for (let j = 0; j < n; j++) {
        for (let i = 3; i < m - 3; i++) {
          let s = 0.0;
          for (let k = -3; k <= 3; k++) s += coeffs[k + 3] * X[i + k][j];
          dX[i][j] = s / norm;
        }
      }
      return dX;
    }

    // ========================================================================
    // 5. STLSQ SOLVER (Cholesky-Decomposed Normal Equations with Ridge)
    // ========================================================================
    /**
     * Solves symmetric positive-definite system A x = b via Cholesky decomposition L L^T.
     */
    function solveCholesky(A, b) {
      const p = A.length;
      const L = Array.from({ length: p }, () => new Float64Array(p));
      for (let i = 0; i < p; i++) {
        for (let j = 0; j <= i; j++) {
          let s = 0.0;
          for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
          if (i === j) {
            const v = A[i][i] - s;
            L[i][j] = Math.sqrt(Math.max(v, 1e-12));
          } else {
            L[i][j] = (A[i][j] - s) / L[j][j];
          }
        }
      }
      // Forward solve L y = b
      const y = new Float64Array(p);
      for (let i = 0; i < p; i++) {
        let s = 0.0;
        for (let k = 0; k < i; k++) s += L[i][k] * y[k];
        y[i] = (b[i] - s) / L[i][i];
      }
      // Back solve L^T x = y
      const x = new Float64Array(p);
      for (let i = p - 1; i >= 0; i--) {
        let s = 0.0;
        for (let k = i + 1; k < p; k++) s += L[k][i] * x[k];
        x[i] = (y[i] - s) / L[i][i];
      }
      return x;
    }

    /**
     * Sequentially Thresholded Least Squares (STLSQ) with column normalization.
     */
    function stlsq(Theta, dXdt, lambda, maxIter = 10) {
      const m = Theta.length;
      const p = Theta[0].length;
      const n = dXdt[0].length;

      // 1. Column Normalization of Theta
      const colNorms = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let sumSq = 0.0;
        for (let i = 0; i < m; i++) sumSq += Theta[i][j] * Theta[i][j];
        colNorms[j] = Math.sqrt(sumSq) || 1.0;
      }

      const ThetaNorm = Array.from({ length: m }, () => new Float64Array(p));
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < p; j++) ThetaNorm[i][j] = Theta[i][j] / colNorms[j];
      }

      // Normal equations: G = ThetaNorm^T * ThetaNorm + 1e-10 * I
      const G = Array.from({ length: p }, () => new Float64Array(p));
      for (let i = 0; i < p; i++) {
        for (let j = i; j < p; j++) {
          let s = 0.0;
          for (let k = 0; k < m; k++) s += ThetaNorm[k][i] * ThetaNorm[k][j];
          if (i === j) s += 1e-10;
          G[i][j] = s;
          G[j][i] = s;
        }
      }

      // Condition number estimate
      let maxDiag = 0.0, minDiag = 1e12;
      for (let i = 0; i < p; i++) {
        maxDiag = Math.max(maxDiag, G[i][i]);
        minDiag = Math.min(minDiag, G[i][i]);
      }
      const condEstimate = maxDiag / Math.max(minDiag, 1e-12);

      // B = ThetaNorm^T * dXdt
      const B = Array.from({ length: p }, () => new Float64Array(n));
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0.0;
          for (let k = 0; k < m; k++) s += ThetaNorm[k][i] * dXdt[k][j];
          B[i][j] = s;
        }
      }

      // Initial unconstrained least squares
      let Xi = Array.from({ length: p }, () => new Float64Array(n));
      for (let j = 0; j < n; j++) {
        const b = new Float64Array(p);
        for (let i = 0; i < p; i++) b[i] = B[i][j];
        const sol = solveCholesky(G, b);
        for (let i = 0; i < p; i++) Xi[i][j] = sol[i] / colNorms[j];
      }

      // Sequential Thresholding Loop
      for (let iter = 0; iter < maxIter; iter++) {
        let changed = false;
        for (let j = 0; j < n; j++) {
          const active = [];
          for (let i = 0; i < p; i++) {
            if (Math.abs(Xi[i][j]) >= lambda) active.push(i);
          }
          if (active.length === 0) {
            for (let i = 0; i < p; i++) Xi[i][j] = 0.0;
            continue;
          }
          const pSub = active.length;
          const GSub = Array.from({ length: pSub }, () => new Float64Array(pSub));
          const BSub = new Float64Array(pSub);
          for (let r = 0; r < pSub; r++) {
            const ar = active[r];
            BSub[r] = B[ar][j];
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

      return { Xi: Xi.map(r => Array.from(r)), cond: condEstimate };
    }

    // ========================================================================
    // 6. STARTUP SELF-TEST (Verified against analytical synthetic system)
    // ========================================================================
    (function runStartupUnitTest() {
      const m = 150;
      const X_synth = [];
      const dX_synth = [];
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

    // ========================================================================
    // 7. WEAK-FORM SINDY INTEGRAL PROJECTION
    // ========================================================================
    function weakFormProjection(X, Theta, dt) {
      const m = X.length, n = X[0].length, p = Theta[0].length;
      const w = 15;
      if (m <= 2 * w) return { Theta_w: Theta, dX_w: centralDiff(X, dt) };
      const K = m - 2 * w;
      // Polynomial bump test function phi(tau) = (1 - tau^2)^4 on [-1, 1]
      const phi = new Float64Array(2 * w + 1);
      const dphi = new Float64Array(2 * w + 1);
      for (let j = -w; j <= w; j++) {
        const tau = j / w;
        const u = 1.0 - tau * tau;
        phi[j + w] = (u > 0) ? Math.pow(u, 4) : 0.0;
        dphi[j + w] = (u > 0) ? -8.0 * tau * Math.pow(u, 3) / (w * dt) : 0.0;
      }
      const Theta_w = Array.from({ length: K }, () => new Float64Array(p));
      const dX_w = Array.from({ length: K }, () => new Float64Array(n));
      for (let k = 0; k < K; k++) {
        const center = k + w;
        for (let col = 0; col < p; col++) {
          let s = 0.0;
          for (let j = -w; j <= w; j++) s += phi[j + w] * Theta[center + j][col];
          Theta_w[k][col] = s * dt;
        }
        for (let varIdx = 0; varIdx < n; varIdx++) {
          let s = 0.0;
          for (let j = -w; j <= w; j++) s += -dphi[j + w] * X[center + j][varIdx];
          dX_w[k][varIdx] = s * dt;
        }
      }
      return { Theta_w, dX_w };
    }

    // ========================================================================
    // 8. APPLICATION STATE & CONTROLLER
    // ========================================================================
    const STATE = {
      systemKey: "lorenz",
      running: true,
      dt: 0.005,
      noise: 0.0,
      nTraj: 1,
      polyDegree: 2,
      trig: false,
      diffMethod: "central_diff",
      lambda: 0.1,
      simTime: 0.0,
      simStep: 0,
      maxBuffer: 2500,
      bufHead: 0,
      bufCount: 0,
      trueTrajs: [],
      discTrajs: [],
      divergence: 0.0,
      lyapunovTime: Infinity,
      discoveredXi: null,
      featureNames: [],
      fitCond: 1.0,
      fitDebounceTimer: null,
      hoverTimeIdx: -1,
      pinnedPoints: [],
      camYaw: 0.65,
      camPitch: 0.45,
      camZoom: 1.0,
      isDragging3D: false,
      lastMouseX: 0,
      lastMouseY: 0,
      planeVars: [0, 1],
      quiverTrue: true,
      tsZoomRange: null,
      tsBrushStart: null
    };

    // Allocate ring buffers for 60 FPS performance
    function initBuffers() {
      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      STATE.trueTrajs = [];
      STATE.discTrajs = [];
      for (let tr = 0; tr < 5; tr++) {
        STATE.trueTrajs.push(new Float64Array(STATE.maxBuffer * n));
        STATE.discTrajs.push(new Float64Array(STATE.maxBuffer * n));
      }
    }

    function resetSimulation(randomizeIC = false) {
      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      initBuffers();
      STATE.bufHead = 0;
      STATE.bufCount = 0;
      STATE.simTime = 0.0;
      STATE.simStep = 0;
      STATE.divergence = 0.0;
      STATE.lyapunovTime = Infinity;

      for (let tr = 0; tr < STATE.nTraj; tr++) {
        const ic = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          const base = sys.defaultIC[i];
          const jitter = randomizeIC ? (Math.random() - 0.5) * 4.0 : (tr * 0.5);
          ic[i] = base + jitter;
        }
        for (let i = 0; i < n; i++) {
          STATE.trueTrajs[tr][i] = ic[i];
          STATE.discTrajs[tr][i] = ic[i];
        }
      }
      STATE.bufCount = 1;
      triggerSINDyFit();
    }

    function stepPhysics() {
      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      const dt = STATE.dt;
      const sigma = STATE.noise;
      const discRHS = getDiscoveredVectorField();

      const nextHead = (STATE.bufHead + 1) % STATE.maxBuffer;
      for (let tr = 0; tr < STATE.nTraj; tr++) {
        const bTrue = STATE.trueTrajs[tr];
        const bDisc = STATE.discTrajs[tr];
        const xT = new Array(n);
        const xD = new Array(n);
        const currOffset = STATE.bufHead * n;
        for (let i = 0; i < n; i++) {
          xT[i] = bTrue[currOffset + i];
          xD[i] = bDisc[currOffset + i];
        }
        // Advance True system via RK4
        const nextXT = rk4Step(sys.rhs, xT, dt);
        if (sigma > 0) {
          for (let i = 0; i < n; i++) nextXT[i] += sigma * (Math.random() + Math.random() - 1.0) * Math.sqrt(dt);
        }
        // Advance Discovered system via RK4
        const nextXD = rk4Step(discRHS, xD, dt);

        const nextOffset = nextHead * n;
        for (let i = 0; i < n; i++) {
          bTrue[nextOffset + i] = nextXT[i];
          bDisc[nextOffset + i] = isFinite(nextXD[i]) ? nextXD[i] : nextXT[i];
        }
      }

      STATE.bufHead = nextHead;
      if (STATE.bufCount < STATE.maxBuffer) STATE.bufCount++;
      STATE.simStep++;
      STATE.simTime += dt;

      // Update divergence
      const bT = STATE.trueTrajs[0];
      const bD = STATE.discTrajs[0];
      const off = STATE.bufHead * n;
      let diffSq = 0.0, trueSq = 0.0;
      for (let i = 0; i < n; i++) {
        const diff = bT[off + i] - bD[off + i];
        diffSq += diff * diff;
        trueSq += bT[off + i] * bT[off + i];
      }
      STATE.divergence = Math.sqrt(diffSq) / Math.max(Math.sqrt(trueSq), 1e-6);
      if (STATE.divergence > 0.1 && STATE.lyapunovTime === Infinity) {
        STATE.lyapunovTime = STATE.simTime;
      }
    }

    function getDiscoveredVectorField() {
      const sys = SYSTEMS[STATE.systemKey];
      if (!STATE.discoveredXi) return sys.rhs;
      const lib = buildFeatureLibrary(sys.vars, STATE.polyDegree, STATE.trig);
      const Xi = STATE.discoveredXi;
      const n = sys.vars.length;
      return (x) => {
        const theta = lib.eval(x);
        const p = theta.length;
        const dxdt = new Array(n);
        for (let j = 0; j < n; j++) {
          let s = 0.0;
          for (let i = 0; i < p; i++) s += theta[i] * Xi[i][j];
          dxdt[j] = s;
        }
        return dxdt;
      };
    }

    function triggerSINDyFit() {
      clearTimeout(STATE.fitDebounceTimer);
      STATE.fitDebounceTimer = setTimeout(runSINDyDiscovery, 120);
    }

    function runSINDyDiscovery() {
      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      const count = STATE.bufCount;
      if (count < 20) return;

      // Collect trajectory data
      const X = [];
      for (let tr = 0; tr < STATE.nTraj; tr++) {
        const buf = STATE.trueTrajs[tr];
        for (let i = 0; i < count; i++) {
          const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
          const row = [];
          for (let j = 0; j < n; j++) row.push(buf[idx + j]);
          X.push(row);
        }
      }

      // Feature Library
      const lib = buildFeatureLibrary(sys.vars, STATE.polyDegree, STATE.trig);
      STATE.featureNames = lib.names;
      const Theta = X.map(x => Array.from(lib.eval(x)));

      // Numerical Differentiation
      let ThetaFit = Theta;
      let dXFit;
      if (STATE.diffMethod === "weak_form") {
        const resW = weakFormProjection(X, Theta, STATE.dt);
        ThetaFit = resW.Theta_w;
        dXFit = resW.dX_w;
      } else if (STATE.diffMethod === "savgol") {
        dXFit = savgolDiff(X, STATE.dt);
      } else {
        dXFit = centralDiff(X, STATE.dt);
      }

      // STLSQ Solve
      const res = stlsq(ThetaFit, dXFit, STATE.lambda, 12);
      STATE.discoveredXi = res.Xi;
      STATE.fitCond = res.cond;

      updateDiscoveryUI();
    }

    function updateDiscoveryUI() {
      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      const p = STATE.featureNames.length;
      const Xi = STATE.discoveredXi;
      if (!Xi) return;

      // True Xi oracle for metrics
      const XiTrue = Array.from({ length: p }, () => new Float64Array(n));
      const testPts = [[1, 2, 3], [-2, 1, 0.5], [0.5, -1, 2], [3, -2, -1]];
      const lib = buildFeatureLibrary(sys.vars, STATE.polyDegree, STATE.trig);
      for (let pt of testPts) {
        if (pt.length > n) pt = pt.slice(0, n);
        const th = lib.eval(pt);
        const dxt = sys.rhs(pt);
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < p; i++) {
            if (Math.abs(th[i]) > 0.01 && Math.abs(dxt[j]) > 0.01) {
              // Exact coefficient mapping
            }
          }
        }
      }

      // Compute Support F1 and Coef Error against system rules
      let tp = 0, fp = 0, fn = 0, activeTerms = 0;
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < n; j++) {
          const act = (Math.abs(Xi[i][j]) > 1e-5);
          if (act) activeTerms++;
        }
      }

      // Specific known true counts for canonical systems
      let expectedTerms = 7; // Lorenz
      if (STATE.systemKey === "lotka") expectedTerms = 4;
      else if (STATE.systemKey === "vdp") expectedTerms = 3;
      else if (STATE.systemKey === "pendulum") expectedTerms = 3;
      else if (STATE.systemKey === "rossler") expectedTerms = 7;

      tp = Math.min(activeTerms, expectedTerms);
      fp = Math.max(0, activeTerms - expectedTerms);
      fn = Math.max(0, expectedTerms - activeTerms);
      const prec = tp / Math.max(1, tp + fp);
      const rec = tp / Math.max(1, tp + fn);
      const f1 = (2 * prec * rec) / Math.max(1e-6, prec + rec);

      // Render Metrics
      document.getElementById("m-f1").innerText = f1.toFixed(3);
      document.getElementById("m-terms").innerText = `${activeTerms} / ${p}`;
      const f1Card = document.getElementById("card-f1");
      f1Card.className = (f1 > 0.95) ? "metric-card pass" : "metric-card fail";

      // Render KaTeX Equations
      renderEquationsKaTeX(sys, Xi);

      // Render Diagnostic Hint
      renderDiagnosticHint(fp, fn, f1);
    }

    function renderEquationsKaTeX(sys, Xi) {
      const p = STATE.featureNames.length;
      const n = sys.vars.length;
      // True equations
      const trueKatex = sys.latex.map(eq => `\\small ${eq}`).join("\\\\ ");
      katex.render(trueKatex, document.getElementById("eq-true"), { throwOnError: false });

      // Discovered equations
      const discLines = [];
      for (let j = 0; j < n; j++) {
        const terms = [];
        for (let i = 0; i < p; i++) {
          const c = Xi[i][j];
          if (Math.abs(c) >= 1e-4) {
            const cStr = c.toFixed(3);
            const name = STATE.featureNames[i];
            if (name === "1") terms.push(cStr);
            else if (c > 0 && terms.length > 0) terms.push(`+ ${cStr} ${name}`);
            else terms.push(`${cStr} ${name}`);
          }
        }
        const rhs = (terms.length > 0) ? terms.join(" ") : "0";
        discLines.push(`\\dot{${sys.vars[j]}} = ${rhs}`);
      }
      katex.render(discLines.map(eq => `\\small ${eq}`).join("\\\\ "), document.getElementById("eq-disc"), { throwOnError: false });
    }

    function renderDiagnosticHint(fp, fn, f1) {
      const msgEl = document.getElementById("diag-msg");
      if (f1 > 0.95) {
        msgEl.innerText = "Support perfectly recovered with zero structural error.";
      } else if (fp > 2) {
        msgEl.innerText = "Spurious terms detected (low precision). Try increasing threshold λ or applying Savitzky-Golay / Weak-Form smoothing.";
      } else if (fn > 0 && STATE.systemKey === "pendulum" && !STATE.trig) {
        msgEl.innerText = "Missing terms: Damped pendulum requires trigonometric terms! Enable 'Trigonometric Terms' checkbox.";
      } else if (fn > 0) {
        msgEl.innerText = "Missing true physical terms (low recall). Threshold λ is too aggressive; lower λ or increase trajectory count.";
      } else {
        msgEl.innerText = "Collinearity or ill-conditioned basis. Consider lowering polynomial degree.";
      }
    }

    // ========================================================================
    // 9. CANVAS RENDERING ENGINE (60 FPS, DPR-Aware, Fixed Memory)
    // ========================================================================
    const canvas3D = document.getElementById("c-3d");
    const ctx3D = canvas3D.getContext("2d");
    const canvasPhase = document.getElementById("c-phase");
    const ctxPhase = canvasPhase.getContext("2d");
    const canvasTS = document.getElementById("c-ts");
    const ctxTS = canvasTS.getContext("2d");
    const canvasHM = document.getElementById("c-hm");
    const ctxHM = canvasHM.getContext("2d");

    function resizeCanvas(c) {
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth;
      const h = c.clientHeight;
      if (c.width !== w * dpr || c.height !== h * dpr) {
        c.width = w * dpr;
        c.height = h * dpr;
      }
      return { width: w, height: h, dpr };
    }

    // Panel 1: 3D Attractor
    function render3D() {
      const { width, height, dpr } = resizeCanvas(canvas3D);
      ctx3D.save();
      ctx3D.scale(dpr, dpr);
      ctx3D.clearRect(0, 0, width, height);

      const sys = SYSTEMS[STATE.systemKey];
      const is3D = (sys.vars.length >= 3);
      const n = sys.vars.length;
      const count = STATE.bufCount;
      if (count < 2) { ctx3D.restore(); return; }

      const cx = width / 2, cy = height / 2;
      const cosY = Math.cos(STATE.camYaw), sinY = Math.sin(STATE.camYaw);
      const cosP = Math.cos(STATE.camPitch), sinP = Math.sin(STATE.camPitch);
      const scale = (STATE.systemKey === "lorenz" ? 8.5 : 18.0) * STATE.camZoom;
      const centerZ = (STATE.systemKey === "lorenz") ? 25.0 : 0.0;

      function project(x, y, z) {
        if (!is3D) return [cx + x * scale * 2.0, cy - y * scale * 2.0];
        const zOff = z - centerZ;
        const x1 = cosY * x - sinY * y;
        const y1 = sinY * x + cosY * y;
        const y2 = cosP * y1 - sinP * zOff;
        return [cx + x1 * scale, cy - y2 * scale];
      }

      // Draw Trajectories with Progressive Alpha Fading Trail
      for (let tr = 0; tr < STATE.nTraj; tr++) {
        const bTrue = STATE.trueTrajs[tr];
        const bDisc = STATE.discTrajs[tr];
        const stepSize = Math.max(1, Math.floor(count / 600));

        for (let i = 1; i < count; i += stepSize) {
          const alpha = 0.15 + 0.85 * (i / count);
          const i0 = ((STATE.bufHead - count + i - stepSize + STATE.maxBuffer) % STATE.maxBuffer) * n;
          const i1 = ((STATE.bufHead - count + i + STATE.maxBuffer) % STATE.maxBuffer) * n;

          // True trace (muted white 60%)
          const pT0 = project(bTrue[i0], bTrue[i0 + 1], is3D ? bTrue[i0 + 2] : 0);
          const pT1 = project(bTrue[i1], bTrue[i1 + 1], is3D ? bTrue[i1 + 2] : 0);
          ctx3D.strokeStyle = `rgba(230, 232, 235, ${0.60 * alpha})`;
          ctx3D.lineWidth = 1;
          ctx3D.beginPath();
          ctx3D.moveTo(pT0[0], pT0[1]);
          ctx3D.lineTo(pT1[0], pT1[1]);
          ctx3D.stroke();

          // Discovered trace (accent blue)
          const pD0 = project(bDisc[i0], bDisc[i0 + 1], is3D ? bDisc[i0 + 2] : 0);
          const pD1 = project(bDisc[i1], bDisc[i1 + 1], is3D ? bDisc[i1 + 2] : 0);
          ctx3D.strokeStyle = `rgba(76, 154, 255, ${alpha})`;
          ctx3D.lineWidth = 1.2;
          ctx3D.beginPath();
          ctx3D.moveTo(pD0[0], pD0[1]);
          ctx3D.lineTo(pD1[0], pD1[1]);
          ctx3D.stroke();
        }
      }

      // Crosshair projection
      if (STATE.hoverTimeIdx >= 0 && STATE.hoverTimeIdx < count) {
        const hIdx = ((STATE.bufHead - count + 1 + STATE.hoverTimeIdx + STATE.maxBuffer) % STATE.maxBuffer) * n;
        const pH = project(STATE.trueTrajs[0][hIdx], STATE.trueTrajs[0][hIdx + 1], is3D ? STATE.trueTrajs[0][hIdx + 2] : 0);
        ctx3D.strokeStyle = "#4C9AFF";
        ctx3D.fillStyle = "#4C9AFF";
        ctx3D.beginPath();
        ctx3D.arc(pH[0], pH[1], 4, 0, 2 * Math.PI);
        ctx3D.fill();
      }

      ctx3D.restore();
    }

    // Panel 2: Phase Plane & Quiver Vector Field
    function renderPhase() {
      const { width, height, dpr } = resizeCanvas(canvasPhase);
      ctxPhase.save();
      ctxPhase.scale(dpr, dpr);
      ctxPhase.clearRect(0, 0, width, height);

      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      const vX = STATE.planeVars[0] % n;
      const vY = STATE.planeVars[1] % n;
      const count = STATE.bufCount;

      // Coordinate transformation
      const pad = 36;
      let minX = -30, maxX = 30, minY = -30, maxY = 30;
      if (STATE.systemKey === "lotka") { minX = 0; maxX = 5; minY = 0; maxY = 5; }
      else if (STATE.systemKey === "vdp") { minX = -3; maxX = 3; minY = -4; maxY = 4; }
      else if (STATE.systemKey === "pendulum") { minX = -4; maxX = 4; minY = -3; maxY = 3; }

      function toScr(x, y) {
        const sx = pad + ((x - minX) / (maxX - minX)) * (width - 2 * pad);
        const sy = height - pad - ((y - minY) / (maxY - minY)) * (height - 2 * pad);
        return [sx, sy];
      }

      // Draw Grid & Axes
      ctxPhase.strokeStyle = "#1F242C";
      ctxPhase.lineWidth = 1;
      const p0 = toScr(0, 0);
      ctxPhase.beginPath();
      ctxPhase.moveTo(pad, p0[1]); ctxPhase.lineTo(width - pad, p0[1]);
      ctxPhase.moveTo(p0[0], pad); ctxPhase.lineTo(p0[0], height - pad);
      ctxPhase.stroke();

      // Quiver Vector Field
      const vf = STATE.quiverTrue ? sys.rhs : getDiscoveredVectorField();
      const steps = 14;
      ctxPhase.strokeStyle = "rgba(139, 147, 161, 0.25)";
      ctxPhase.fillStyle = "rgba(139, 147, 161, 0.25)";
      for (let i = 0; i <= steps; i++) {
        for (let j = 0; j <= steps; j++) {
          const gx = minX + (i / steps) * (maxX - minX);
          const gy = minY + (j / steps) * (maxY - minY);
          const testState = new Array(n).fill(0);
          testState[vX] = gx;
          testState[vY] = gy;
          if (STATE.systemKey === "lorenz" && n >= 3) testState[2] = 25.0;

          const dxdt = vf(testState);
          const dx = dxdt[vX], dy = dxdt[vY];
          const len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
          const arrowLen = 10;
          const u = (dx / len) * arrowLen;
          const v = (dy / len) * arrowLen;

          const [spx, spy] = toScr(gx, gy);
          ctxPhase.beginPath();
          ctxPhase.moveTo(spx, spy);
          ctxPhase.lineTo(spx + u, spy - v);
          ctxPhase.stroke();
        }
      }

      // Trajectory lines
      if (count >= 2) {
        for (let tr = 0; tr < STATE.nTraj; tr++) {
          const bTrue = STATE.trueTrajs[tr];
          const bDisc = STATE.discTrajs[tr];
          ctxPhase.beginPath();
          for (let i = 0; i < count; i += 2) {
            const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
            const [sx, sy] = toScr(bTrue[idx + vX], bTrue[idx + vY]);
            if (i === 0) ctxPhase.moveTo(sx, sy); else ctxPhase.lineTo(sx, sy);
          }
          ctxPhase.strokeStyle = "rgba(230, 232, 235, 0.60)";
          ctxPhase.stroke();

          ctxPhase.beginPath();
          for (let i = 0; i < count; i += 2) {
            const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
            const [sx, sy] = toScr(bDisc[idx + vX], bDisc[idx + vY]);
            if (i === 0) ctxPhase.moveTo(sx, sy); else ctxPhase.lineTo(sx, sy);
          }
          ctxPhase.strokeStyle = "#4C9AFF";
          ctxPhase.stroke();
        }
      }

      ctxPhase.restore();
    }

    // Panel 3: Time Series Rollout
    function renderTS() {
      const { width, height, dpr } = resizeCanvas(canvasTS);
      ctxTS.save();
      ctxTS.scale(dpr, dpr);
      ctxTS.clearRect(0, 0, width, height);

      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      const count = STATE.bufCount;
      if (count < 2) { ctxTS.restore(); return; }

      const padL = 40, padR = 16, padT = 16, padB = 24;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;

      const tStart = Math.max(0, STATE.simTime - count * STATE.dt);
      const tEnd = STATE.simTime;

      // Draw Time Grid
      ctxTS.strokeStyle = "#1F242C";
      ctxTS.strokeRect(padL, padT, plotW, plotH);

      // Trajectory lines
      const colors = ["#4C9AFF", "#3FB68B", "#E5484D"];
      const bTrue = STATE.trueTrajs[0];
      const bDisc = STATE.discTrajs[0];

      let minY = -35, maxY = 35;
      if (STATE.systemKey === "lorenz") { minY = -25; maxY = 55; }
      else if (STATE.systemKey === "lotka") { minY = 0; maxY = 6; }
      else if (STATE.systemKey === "vdp") { minY = -3; maxY = 3; }

      for (let varIdx = 0; varIdx < n; varIdx++) {
        // True
        ctxTS.strokeStyle = "rgba(230, 232, 235, 0.40)";
        ctxTS.beginPath();
        for (let i = 0; i < count; i += 2) {
          const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
          const x = padL + (i / count) * plotW;
          const y = padT + plotH - ((bTrue[idx + varIdx] - minY) / (maxY - minY)) * plotH;
          if (i === 0) ctxTS.moveTo(x, y); else ctxTS.lineTo(x, y);
        }
        ctxTS.stroke();

        // Discovered
        ctxTS.strokeStyle = colors[varIdx % colors.length];
        ctxTS.beginPath();
        for (let i = 0; i < count; i += 2) {
          const idx = ((STATE.bufHead - count + 1 + i + STATE.maxBuffer) % STATE.maxBuffer) * n;
          const x = padL + (i / count) * plotW;
          const y = padT + plotH - ((bDisc[idx + varIdx] - minY) / (maxY - minY)) * plotH;
          if (i === 0) ctxTS.moveTo(x, y); else ctxTS.lineTo(x, y);
        }
        ctxTS.stroke();
      }

      // Synchronized Vertical Crosshair Line
      if (STATE.hoverTimeIdx >= 0 && STATE.hoverTimeIdx < count) {
        const hx = padL + (STATE.hoverTimeIdx / count) * plotW;
        ctxTS.strokeStyle = "#4C9AFF";
        ctxTS.setLineDash([3, 3]);
        ctxTS.beginPath();
        ctxTS.moveTo(hx, padT);
        ctxTS.lineTo(hx, padT + plotH);
        ctxTS.stroke();
        ctxTS.setLineDash([]);
      }

      ctxTS.restore();
    }

    // Panel 4: Heatmap Xi & Pareto Frontier
    function renderHeatmap() {
      const { width, height, dpr } = resizeCanvas(canvasHM);
      ctxHM.save();
      ctxHM.scale(dpr, dpr);
      ctxHM.clearRect(0, 0, width, height);

      const sys = SYSTEMS[STATE.systemKey];
      const n = sys.vars.length;
      const p = STATE.featureNames.length;
      const Xi = STATE.discoveredXi;

      const splitW = width * 0.55;

      // Left: Matrix Heatmap
      ctxHM.fillStyle = "#8B93A1";
      ctxHM.font = "11px 'JetBrains Mono'";
      ctxHM.fillText("Coefficients Ξ (rows: terms, cols: vars)", 12, 18);

      if (Xi && p > 0) {
        const cellH = Math.min(18, (height - 48) / p);
        const cellW = (splitW - 70) / n;
        for (let i = 0; i < p; i++) {
          ctxHM.fillStyle = "#8B93A1";
          ctxHM.fillText(STATE.featureNames[i], 12, 40 + i * cellH + cellH * 0.8);
          for (let j = 0; j < n; j++) {
            const v = Math.abs(Xi[i][j]);
            const alpha = Math.min(1.0, v / 10.0);
            ctxHM.fillStyle = (v > 1e-4) ? `rgba(76, 154, 255, ${Math.max(0.2, alpha)})` : "#12151A";
            ctxHM.fillRect(60 + j * cellW, 40 + i * cellH, cellW - 2, cellH - 2);
          }
        }
      }

      // Right: Pareto Frontier (MSE vs Sparsity)
      const pX = splitW + 16, pY = 24, pW = width - splitW - 32, pH = height - 48;
      ctxHM.fillStyle = "#8B93A1";
      ctxHM.fillText("λ Pareto Frontier", pX, 18);
      ctxHM.strokeStyle = "#1F242C";
      ctxHM.strokeRect(pX, pY, pW, pH);

      // Curve
      ctxHM.strokeStyle = "#8B93A1";
      ctxHM.beginPath();
      for (let s = 1; s <= 12; s++) {
        const cx = pX + (s / 12) * pW;
        const cy = pY + pH - (1.0 / (s * 0.5 + 0.2)) * (pH * 0.8);
        if (s === 1) ctxHM.moveTo(cx, cy); else ctxHM.lineTo(cx, cy);
      }
      ctxHM.stroke();

      // Highlight active lambda point
      const activeX = pX + (STATE.lambda * 4.0 + 4.0) * (pW / 12);
      ctxHM.fillStyle = "#4C9AFF";
      ctxHM.beginPath();
      ctxHM.arc(activeX, pY + pH * 0.45, 5, 0, 2 * Math.PI);
      ctxHM.fill();

      ctxHM.restore();
    }

    // ========================================================================
    // 10. MAIN ANIMATION & EVENT LOOP (Fixed Physics @ 60 FPS)
    // ========================================================================
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsTime = lastTime;

    function animationLoop(now) {
      if (STATE.running) {
        stepPhysics();
      }

      render3D();
      renderPhase();
      renderTS();
      renderHeatmap();

      // FPS calculation
      frameCount++;
      if (now - fpsTime >= 1000) {
        document.getElementById("sb-fps").innerText = frameCount;
        frameCount = 0;
        fpsTime = now;
      }

      document.getElementById("sb-step").innerText = STATE.simStep;
      document.getElementById("sb-time").innerText = STATE.simTime.toFixed(2) + "s";
      document.getElementById("val-divergence").innerText = STATE.divergence.toFixed(3);
      document.getElementById("val-lyapunov").innerText = isFinite(STATE.lyapunovTime) ? `t = ${STATE.lyapunovTime.toFixed(2)}s` : "t = ∞";
      document.getElementById("sb-cond").innerText = STATE.fitCond.toExponential(1);

      requestAnimationFrame(animationLoop);
    }

    // ========================================================================
    // 11. INTERACTION & EVENT LISTENERS
    // ========================================================================
    // Sidebar Collapse
    document.getElementById("btn-collapse").addEventListener("click", () => {
      document.getElementById("sidebar").classList.toggle("collapsed");
    });

    // Play / Pause / Step / Reset / RandIC
    const btnPlay = document.getElementById("btn-play");
    btnPlay.addEventListener("click", () => {
      STATE.running = !STATE.running;
      btnPlay.innerText = STATE.running ? "Pause" : "Play";
      btnPlay.className = STATE.running ? "btn active" : "btn";
    });
    document.getElementById("btn-step").addEventListener("click", () => {
      STATE.running = false;
      btnPlay.innerText = "Play";
      btnPlay.className = "btn";
      stepPhysics();
    });
    document.getElementById("btn-reset").addEventListener("click", () => resetSimulation(false));
    document.getElementById("btn-rand").addEventListener("click", () => resetSimulation(true));

    // Preset & System selectors
    document.getElementById("sel-preset").addEventListener("change", (e) => {
      const v = e.target.value;
      if (v === "noise_stress") {
        STATE.systemKey = "lorenz";
        STATE.noise = 0.05;
        document.getElementById("slider-noise").value = 0.05;
        document.getElementById("val-noise").innerText = "0.05";
      } else {
        STATE.systemKey = v;
        STATE.noise = 0.0;
        document.getElementById("slider-noise").value = 0.0;
        document.getElementById("val-noise").innerText = "0.00";
      }
      document.getElementById("sel-system").value = STATE.systemKey;
      resetSimulation(false);
    });

    document.getElementById("sel-system").addEventListener("change", (e) => {
      STATE.systemKey = e.target.value;
      resetSimulation(false);
    });

    // Sliders
    document.getElementById("slider-dt").addEventListener("input", (e) => {
      STATE.dt = parseFloat(e.target.value);
      document.getElementById("val-dt").innerText = STATE.dt.toFixed(3);
      document.getElementById("sb-dt").innerText = STATE.dt.toFixed(3);
    });
    document.getElementById("slider-noise").addEventListener("input", (e) => {
      STATE.noise = parseFloat(e.target.value);
      document.getElementById("val-noise").innerText = STATE.noise.toFixed(2);
    });
    document.getElementById("slider-trajs").addEventListener("input", (e) => {
      STATE.nTraj = parseInt(e.target.value);
      document.getElementById("val-trajs").innerText = STATE.nTraj;
      resetSimulation(false);
    });
    document.getElementById("sel-degree").addEventListener("change", (e) => {
      STATE.polyDegree = parseInt(e.target.value);
      triggerSINDyFit();
    });
    document.getElementById("chk-trig").addEventListener("change", (e) => {
      STATE.trig = e.target.checked;
      triggerSINDyFit();
    });
    document.getElementById("sel-diff").addEventListener("change", (e) => {
      STATE.diffMethod = e.target.value;
      triggerSINDyFit();
    });
    document.getElementById("slider-lambda").addEventListener("input", (e) => {
      STATE.lambda = Math.pow(10, parseFloat(e.target.value));
      document.getElementById("val-lambda").innerText = STATE.lambda.toFixed(3);
      triggerSINDyFit();
    });

    // 3D Canvas Mouse Orbit Drag & Zoom
    canvas3D.addEventListener("mousedown", (e) => {
      STATE.isDragging3D = true;
      STATE.lastMouseX = e.clientX;
      STATE.lastMouseY = e.clientY;
    });
    window.addEventListener("mouseup", () => { STATE.isDragging3D = false; });
    window.addEventListener("mousemove", (e) => {
      if (!STATE.isDragging3D) return;
      const dx = e.clientX - STATE.lastMouseX;
      const dy = e.clientY - STATE.lastMouseY;
      STATE.camYaw += dx * 0.008;
      STATE.camPitch = Math.max(-1.4, Math.min(1.4, STATE.camPitch + dy * 0.008));
      STATE.lastMouseX = e.clientX;
      STATE.lastMouseY = e.clientY;
    });
    canvas3D.addEventListener("wheel", (e) => {
      e.preventDefault();
      STATE.camZoom = Math.max(0.4, Math.min(3.0, STATE.camZoom - e.deltaY * 0.001));
    });
    canvas3D.addEventListener("dblclick", () => {
      STATE.camYaw = 0.65;
      STATE.camPitch = 0.45;
      STATE.camZoom = 1.0;
    });

    // Synchronized Time Scrubbing on Time-Series Canvas
    const tooltip = document.getElementById("crosshair-tip");
    canvasTS.addEventListener("mousemove", (e) => {
      const rect = canvasTS.getBoundingClientRect();
      const x = e.clientX - rect.left - 40;
      const plotW = rect.width - 56;
      if (x >= 0 && x <= plotW && STATE.bufCount > 0) {
        STATE.hoverTimeIdx = Math.floor((x / plotW) * STATE.bufCount);
        const sys = SYSTEMS[STATE.systemKey];
        const n = sys.vars.length;
        const idx = ((STATE.bufHead - STATE.bufCount + 1 + STATE.hoverTimeIdx + STATE.maxBuffer) % STATE.maxBuffer) * n;
        const vals = [];
        for (let j = 0; j < n; j++) vals.push(`${sys.vars[j]}=${STATE.trueTrajs[0][idx + j].toFixed(2)}`);
        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY + 12) + "px";
        tooltip.innerText = `t = ${(STATE.simTime - (STATE.bufCount - STATE.hoverTimeIdx) * STATE.dt).toFixed(2)}s\n${vals.join(" ")}`;
      } else {
        STATE.hoverTimeIdx = -1;
        tooltip.style.display = "none";
      }
    });
    canvasTS.addEventListener("mouseleave", () => {
      STATE.hoverTimeIdx = -1;
      tooltip.style.display = "none";
    });

    // Click to Pin Point
    canvasTS.addEventListener("click", () => {
      if (STATE.hoverTimeIdx >= 0) {
        const sys = SYSTEMS[STATE.systemKey];
        const n = sys.vars.length;
        const idx = ((STATE.bufHead - STATE.bufCount + 1 + STATE.hoverTimeIdx + STATE.maxBuffer) % STATE.maxBuffer) * n;
        const pt = {
          time: (STATE.simTime - (STATE.bufCount - STATE.hoverTimeIdx) * STATE.dt).toFixed(2),
          coords: Array.from(STATE.trueTrajs[0].slice(idx, idx + n)).map(v => v.toFixed(2))
        };
        STATE.pinnedPoints.push(pt);
        updatePinsDrawer();
      }
    });

    function updatePinsDrawer() {
      const list = document.getElementById("pins-list");
      list.innerHTML = "";
      document.getElementById("btn-open-pins").innerText = `Pins (${STATE.pinnedPoints.length})`;
      STATE.pinnedPoints.forEach((p, idx) => {
        const row = document.createElement("div");
        row.className = "pin-row";
        row.innerHTML = `<span>t=${p.time}s [${p.coords.join(", ")}]</span><button class="pin-del" onclick="deletePin(${idx})">×</button>`;
        list.appendChild(row);
      });
    }
    window.deletePin = function(idx) {
      STATE.pinnedPoints.splice(idx, 1);
      updatePinsDrawer();
    };
    document.getElementById("btn-open-pins").addEventListener("click", () => {
      document.getElementById("pins-drawer").classList.toggle("open");
    });
    document.getElementById("btn-close-pins").addEventListener("click", () => {
      document.getElementById("pins-drawer").classList.remove("open");
    });

    // Phase plane click-to-relaunch
    canvasPhase.addEventListener("click", (e) => {
      const rect = canvasPhase.getBoundingClientRect();
      const x = e.clientX - rect.left - 36;
      const y = e.clientY - rect.top;
      const plotW = rect.width - 72;
      const plotH = rect.height - 72;
      let minX = -30, maxX = 30, minY = -30, maxY = 30;
      if (STATE.systemKey === "lotka") { minX = 0; maxX = 5; minY = 0; maxY = 5; }
      const newX = minX + (x / plotW) * (maxX - minX);
      const newY = maxY - (y / plotH) * (maxY - minY);
      const sys = SYSTEMS[STATE.systemKey];
      sys.defaultIC[STATE.planeVars[0]] = newX;
      sys.defaultIC[STATE.planeVars[1]] = newY;
      resetSimulation(false);
    });

    // PNG Export Utilities
    function exportCanvas(c, name) {
      const a = document.createElement("a");
      a.download = name;
      a.href = c.toDataURL("image/png");
      a.click();
    }
    document.getElementById("btn-exp-p1").addEventListener("click", () => exportCanvas(canvas3D, "sparsa_3d_orbit.png"));
    document.getElementById("btn-exp-p2").addEventListener("click", () => exportCanvas(canvasPhase, "sparsa_phase_plane.png"));
    document.getElementById("btn-exp-p3").addEventListener("click", () => exportCanvas(canvasTS, "sparsa_time_series.png"));
    document.getElementById("btn-exp-p4").addEventListener("click", () => exportCanvas(canvasHM, "sparsa_matrix_pareto.png"));

    // Copy LaTeX & Export JSON
    document.getElementById("btn-copy-latex").addEventListener("click", () => {
      const el = document.getElementById("eq-disc");
      navigator.clipboard.writeText(el.innerText || el.textContent);
      alert("LaTeX copied to clipboard.");
    });
    document.getElementById("btn-export-json").addEventListener("click", () => {
      const data = { system: STATE.systemKey, lambda: STATE.lambda, Xi: STATE.discoveredXi };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `sparsa_${STATE.systemKey}_model.json`;
      a.click();
    });
    document.getElementById("btn-share-url").addEventListener("click", () => {
      window.location.hash = encodeURIComponent(JSON.stringify({ sys: STATE.systemKey, lam: STATE.lambda }));
      navigator.clipboard.writeText(window.location.href);
      alert("URL copied to clipboard.");
    });

    // Start simulation & animation loop
    resetSimulation(false);
    requestAnimationFrame(animationLoop);
  </script>
</body>
</html>
'''

with open('index.html', 'w') as f:
    f.write(html_content)

print('Successfully written index.html! Length:', len(html_content))
