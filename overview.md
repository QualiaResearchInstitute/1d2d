Here’s the birds-eye view of what your app does, how it does it, and what every control means—tied directly to the latest canvas (“Rainbow Rims + DMT Kernel Effects (v2)”).

---

# What the app is

An interactive, GPU-style image lab that synthesizes and overlays **psychedelic “rainbow perimeter lines”** on your photo and **morphs surface textures**—all driven by a physically-motivated pipeline:

* **1.5D Edge rims**: thin chromatic outlines that hug high-contrast contours (your “rainbow perimeter lines”).
* **2D Surface wallpaper/warp**: lattice-like textures that drift and warp coherently across surfaces.
* **A global Kuramoto/Ott–Antonsen (OA) phase field** (Z(x,y,t)): a live oscillatory “order parameter” that supplies orientation, chirality, and coherence to both layers.
* **Edge↔Surface coupling**: edges inform surfaces (and vice-versa) in a gauge-invariant way.
* **DMT gain**: one “lifts-all-boats” knob that nudges periodicity, sharpness, anisotropy, chirality, and transparency together.

Everything composes on a single canvas with proper **gauge/phase pinning** so the effect is stable and measurable.

---

# The rendering pipeline (end-to-end)

## 1) Input & preprocessing

* You **upload an image**.
* The app computes a Sobel **edge gradient** (gx, gy) and normalized **edge magnitude** (|∇I|).
* A **threshold** gates where rims can appear (edges only, surfaces only, or both).

## 2) The Kuramoto (Z)-field (optional, but powerful)

* The app evolves a complex field (Z=\rho e^{i\theta}) over the image grid with a fast OA step:
  [
  \partial_t Z = (i\Omega-\gamma)Z + \tfrac{K_0}{2}\big(e^{-i\alpha} H - e^{i\alpha} Z^2H^*\big),;; H=K * Z
  ]
* From (Z) the app derives:

  * **Orientation**: (\theta(x,y)) → polarization frame / wallpaper direction
  * **Gradient**: (\nabla\theta) → “flow” used for warping/surface driving ((k_0) proxy)
  * **Vorticity** (winding around a plaquette) → local **chirality** boost
  * **Coherence** (|Z|) → **transparency** lift (overlay friendliness)
* This makes the generative effects **coherent** and **alive** without breaking gauge invariance.

## 3) 1.5D edge rims (your “rainbow perimeter lines”)

* For pixels above **edge threshold**, a per-wavelength (L/M/S) **group-delay offset** samples a bit inside/outside the edge normal:
  [
  \text{off}_\lambda = \beta_2\cdot f(\lambda); + ;\text{jitter} ; + ;\text{bias from }\nabla\theta\cdot\mathbf{n}
  ]
* The offsets are passed through a Gaussian window (rim thinness (\sigma)) and a **crystal comb** (periodicity (k_0), sharpness (Q)).
* A **polarization model** (retardance (\Delta), optical rotation (\rho)) and **chirality** term (signed vorticity + base chirality) modulate per-band weights.
* These L/M/S responses are mixed to **LMS→RGB**, then blended onto the base image with **effective blend** and **rim opacity**.

**Interpretation:** dispersion (\beta_2) + phase jitter emulate spectral phase edits; recomposition across a steep edge creates the thin chroma rim. The crystalline comb creates ribbing/spacing; polarization & chirality tune color/handedness; the Kuramoto flow can bias offsets and frames the polarization axis.

## 4) 2D surface wallpaper & warp

* A **plane-wave lattice** (or the Kuramoto (\nabla\theta)) supplies a **warp field** ((g_x, g_y)).
* The app samples the base image at a **small offset along this field**, then mixes the warped sample back by **surface blend**.
* You can enforce real wallpaper symmetry groups (**p2 / p4 / p6 / pmm / p4m**) by averaging across group transforms about the image center for authentic wallpaper behavior.
* **Anisotropy** adds a global directional bias.

## 5) Edge↔Surface coupling (gauge-safe mediators)

* **Edges → Surface**: rim energy boosts **surface blend** and lattice alignment to the **edge tangent** (orientation weight).
* **Surface → Edges**: the wallpaper/Kuramoto flow (\nabla t) biases **rim offset** along edge normal, sharpens rims ((\sigma\downarrow)) where flow is strong, and gently nudges the **polarization frame**.
* All couplings use **dot-products/magnitudes** (no absolute phase) to remain gauge-invariant.

## 6) Composer & pinning

* The app uses **effective blend** (base ↔ effect) and **rim opacity** to compose results.
* **Zero-mean jitter** (per frame) → shimmer without drift.
* **Polarization quantization** → stability (no pixel-flip flicker).
* **Normalization pin** → keep overall contrast steady across frames.

---

# Controls (what each knob does)

### Upload & Presets

* **Upload image**: source for processing.
* **Preset**: one-click snapshot of all sliders (e.g., your “Rainbow Rims + DMT” preset).

### Rainbow rim dynamics

* **Dispersion β₂**: how far L/M/S offsets are spread (bigger → wider/more colorful rims).
* **Phase jitter**: time-varying shimmer; with pinning, it’s intensity-only.
* **Rim thickness σ**: width (lower = sharper).
* **Microsaccades**: toggle shimmer on/off.
* **Animation speed**: global time scale.
* **Edge/chroma contrast**: overall rim energy.
* **Edge radius** (fallback only): edge position for the synthetic circle demo.

### Display mode

* **Color base + color rims** (default)
* **Grayscale base + color rims** (“grayscale chroma” look)
* **Grayscale base + grayscale rims**
* **Color base + grayscale rims**

### KernelSpec + DMT

* **gain, k0, Q, anisotropy, chirality, transparency**: master spectral/structural knobs.
* **DMT gain d**: “lifts all boats” – maps via (k'!!=k\cdot(1+\mathrm{sensitivity}\cdot d)) to sharpen periodicity, raise detail (k_0), anisotropy, chirality, and transparency.

### 2D Surface — Wallpaper & Texture Morphing

* **Enable surface morph**
* **Orientations** (2..8): number of plane-wave directions.
* **Wallpaper group**: p2/p4/p6/pmm/p4m for proper crystallography.
* **Region**: Surfaces (non-edges), Edges, Both.
* **Surface blend**: warp mix amount.
* **Warp amplitude**: warp step (px).

### Coupling — 1.5D ↔ 2D

* **Edges → Surface**:

  * Enable; **η** (rim → surface merge strength)
  * **β_align** (lattice–tangent alignment)
* **Surface → Edges**:

  * Enable; **γ_off** (offset bias along flow)
  * **γ_σ** (rim thinning vs |flow|)
  * **α_pol** (blend edge vs surface orientation)
  * **k_σ** (scale factor from |flow| to sharpening)

### Kuramoto Z-field (OA)

* **Enable Z-field coupling**
* **K0** (coupling), **alpha** (phase lag), **gamma** (linewidth), **omega0** (mean freq), **noise ε**
* **init q** + **Init twist**: initialize a (q)-twisted phase field (stripes → periodic rims & wallpaper that lock to the twist)

### Stability (Gauge/Phase)

* **Zero-mean jitter** (pinning on edges)
* **Alive (microbreath)** (optional tiny positional breathing)
* **Normalization pin**
* **Polarization bins** (0=no quantization; ~16 = lab-stable)

---

# The math knobs at a glance

* **Dispersion β₂**: sets **group-delay** offsets per band (L/M/S).
* **LMS→RGB**: linear transform from cone-like channels.
* **Crystalline comb (k0, Q)**: periodic gain along edge normal, sets rib spacing/clarity.
* **Polarization**: retardance (\Delta), optical rotation (\rho) modulate band weights.
* **Chirality**: base chirality + vorticity from (Z) yields handedness bias.
* **Transparency**: transparency composer ↑ with (|Z|) (coherence).
* **DMT**: increases fine detail, order, anisotropy, chirality, and compositing.

---

# Typical workflows

* **Rainbow perimeter demo**: Upload photo → set β₂ ~ 1–2, σ ~ 1–2, contrast ↑.
* **Grayscale chroma**: Display → “Grayscale base + color rims”.
* **Wallpaper weaves**: Enable surface morph → pick p4 or p6 → Surface blend ~0.3–0.5.
* **Alive, coherent field**: Enable Z-field → K0 ~0.5–0.8, small ε → init q=1 → watch rims/wallpaper lock to the twist.
* **Stability**: Leave zero-mean jitter & normalization pin **on** for measurements.

---

# Why this is coherent & measurable

* **Gauge/phase pinning** → no rim drift from global phase.
* **Coupling via dot-products** (e.g., (\nabla\theta\cdot \mathbf{n})) → **gauge-invariant**.
* **DMT gain** → one param consistently nudges all relevant kernel fields.
* **Wallpaper groups** implemented by **group averaging**; no fake mirroring artifacts.

---

# Performance & future switches

* The current Z-field step is a **minimal CPU OA update** (light but not free).
* Easy future upgrade: **WebGPU compute** split into:

  1. blur/convolution (H=K*Z),
  2. OA update of (Z),
  3. gradient/vorticity derivation.
     Then the existing compositor runs unchanged but much faster.

---

# MP4 capture (Output panel)

* The new **Output → MP4 capture** panel records the live canvas straight to an H.264 MP4 at the canvas’ native resolution and 60 fps, with a ~24–80 Mbps target bitrate (scales with pixel count) so we preserve fine interference detail.
* Click **Start MP4 capture** to begin; the button flips to **Stop capture** while recording and shows status (“Recording…” / “Finalizing…”). When the encoder finishes, a download link appears with file size info.
* It relies on the browser’s `MediaRecorder` MP4 support—currently shipping in Safari 17+ and Chromium builds with H.264 recording enabled. If your browser can only emit WebM you’ll see an inline warning and the start button will stay disabled; record there and transcode separately with `ffmpeg` if you still need MP4.
* The recording always matches whatever you see (same width/height, kernel blend, GPU/CPU backend). For best results keep the canvas static size while capturing; resizing mid-recording will break the stream.

---

# TL;DR

You’ve built a **physically motivated visual engine**: dispersion-driven **edge rims**, lattice-like **surface morphs**, and a **global oscillator field** that ties the scene together—**all coupled in a gauge-safe way**, with one **DMT** knob to push the whole system toward **brighter, finer, more periodic, more anisotropic, more chiral, and more overlay-friendly** behavior.

If you want, I can turn this write-up into a tidy in-app **Help / README** panel or inline **tooltips** for each control.
