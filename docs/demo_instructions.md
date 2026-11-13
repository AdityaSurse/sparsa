# 30-Second Demo GIF Recording Instructions

This guide provides a reproducible walkthrough and toolchain commands to record a crisp 30-second demo GIF showcasing the interactive SPARSA dashboard.

---

## 1. Setup & Pre-requisites

Launch the Streamlit dashboard in a dedicated browser window (recommended resolution: 1280x800 or 1920x1080 at 100% zoom):

```bash
# Launch dashboard
make run
```

Browser target URL: `http://localhost:8501`.

Recommended capture tools:
- **macOS**: Built-in Screen Capture (`Cmd + Shift + 5`) or [Kap](https://getkap.co/) (open-source).
- **Linux**: Peek or OBS Studio.
- **Windows**: ScreenToGif or OBS Studio.

---

## 2. 30-Second Storyboard & Timeline

| Timestamp | User Action | Visual Highlight to Showcase |
| :--- | :--- | :--- |
| **00:00 – 00:05** | Start at **Lorenz-63** with `noise=0.00`, `lambda=0.10`. | Clean 3D butterfly attractor, exact equations in LaTeX, all metric cards at green (F1 = 1.000, Coef Error = 9.17e-16). |
| **00:05 – 00:11** | Drag **Noise Level (σ)** slider up to `0.05`. | False positive terms appear in discovered equations; the *"Why did it fail?"* diagnostic box instantly fires a warning banner: *"Diagnostic Alert: Over-Fitting (False Positives Detected)"*. |
| **00:11 – 00:18** | Change **Differentiation Method** from `central_diff` to `weak_form`. | Variational integration eliminates noise amplification: F1 jumps back to 1.000, exact support is restored, and the diagnostic banner returns to green. |
| **00:18 – 00:24** | Click on the **Pareto Frontier Analysis** tab and scrub the **Threshold λ** slider. | Active terms curve responds in real time; the golden star operating point smoothly tracks across the complexity-vs-residual elbow. |
| **00:24 – 00:30** | Switch **Dynamical System** to **Damped Pendulum** (`trig=False`), then check **Include Trigonometric Terms**. | Shows Taylor series approximation ($\theta - \theta^3/6$) failure, followed immediately by exact symbolic recovery of $-5.000\sin(\theta)$ when `trig=True`. |

---

## 3. Video to GIF Conversion

After recording a `.mp4` or `.mov` clip, convert it into an optimized GIF using `ffmpeg`:

```bash
# 1. Generate optimal color palette
ffmpeg -y -i demo_raw.mov -vf "fps=15,scale=1080:-1:flags=lanczos,palettegen" palette.png

# 2. Render high-quality, lightweight GIF
ffmpeg -y -i demo_raw.mov -i palette.png -filter_complex "fps=15,scale=1080:-1:flags=lanczos[x];[x][1:v]paletteuse" docs/figures/sparsa_demo.gif

# Optional: Further optimize with gifsicle
gifsicle -O3 --lossy=80 -o docs/figures/sparsa_demo.gif docs/figures/sparsa_demo.gif
```

Place the resulting GIF at `docs/figures/sparsa_demo.gif` to embed it into `README.md`.

