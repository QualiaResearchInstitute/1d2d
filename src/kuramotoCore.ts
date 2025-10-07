const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));

const clamp01 = (v: number) => clamp(v, 0, 1);

const wrapAngle = (a: number) => {
  let ang = a;
  while (ang > Math.PI) ang -= 2 * Math.PI;
  while (ang <= -Math.PI) ang += 2 * Math.PI;
  return ang;
};

export type KuramotoParams = {
  alphaKur: number;
  gammaKur: number;
  omega0: number;
  K0: number;
  epsKur: number;
};

export type KuramotoState = {
  width: number;
  height: number;
  Zr: Float32Array;
  Zi: Float32Array;
};

export type KuramotoDerived = {
  gradX: Float32Array;
  gradY: Float32Array;
  vort: Float32Array;
  coh: Float32Array;
};

const wrapIndex = (x: number, y: number, width: number, height: number) => {
  const xx = ((x % width) + width) % width;
  const yy = ((y % height) + height) % height;
  return yy * width + xx;
};

export const createKuramotoState = (width: number, height: number): KuramotoState => {
  const total = width * height;
  return {
    width,
    height,
    Zr: new Float32Array(total),
    Zi: new Float32Array(total)
  };
};

export const derivedFieldCount = 4;

export const derivedBufferSize = (width: number, height: number) =>
  width * height * derivedFieldCount * Float32Array.BYTES_PER_ELEMENT;

export const createDerivedViews = (
  buffer: ArrayBuffer,
  width: number,
  height: number
): KuramotoDerived => {
  const total = width * height;
  const view = new Float32Array(buffer);
  const gradX = view.subarray(0, total);
  const gradY = view.subarray(total, total * 2);
  const vort = view.subarray(total * 2, total * 3);
  const coh = view.subarray(total * 3, total * 4);
  return { gradX, gradY, vort, coh };
};

export const initKuramotoState = (
  state: KuramotoState,
  q: number,
  derived?: KuramotoDerived
) => {
  const { width, height, Zr, Zi } = state;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const theta = (2 * Math.PI * q * x) / width;
      const idx = y * width + x;
      Zr[idx] = Math.cos(theta);
      Zi[idx] = Math.sin(theta);
    }
  }
  if (derived) {
    derived.gradX.fill(0);
    derived.gradY.fill(0);
    derived.vort.fill(0);
    derived.coh.fill(0.5);
  }
};

export const stepKuramotoState = (
  state: KuramotoState,
  params: KuramotoParams,
  dt: number,
  randn: () => number
) => {
  const { width, height, Zr, Zi } = state;
  const { alphaKur, gammaKur, omega0, K0, epsKur } = params;
  const ca = Math.cos(alphaKur);
  const sa = Math.sin(alphaKur);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const left = wrapIndex(x - 1, y, width, height);
      const right = wrapIndex(x + 1, y, width, height);
      const up = wrapIndex(x, y - 1, width, height);
      const down = wrapIndex(x, y + 1, width, height);
      const Hr =
        0.2 * (Zr[idx] + Zr[left] + Zr[right] + Zr[up] + Zr[down]);
      const Hi =
        0.2 * (Zi[idx] + Zi[left] + Zi[right] + Zi[up] + Zi[down]);
      const Zre = Zr[idx];
      const Zim = Zi[idx];
      const Z2r = Zre * Zre - Zim * Zim;
      const Z2i = 2 * Zre * Zim;
      const H1r = ca * Hr + sa * Hi;
      const H1i = -sa * Hr + ca * Hi;
      const HrConj = Hr;
      const HiConj = -Hi;
      const Tr = Z2r * HrConj - Z2i * HiConj;
      const Ti = Z2r * HiConj + Z2i * HrConj;
      const H2r = ca * Tr - sa * Ti;
      const H2i = sa * Tr + ca * Ti;
      const dZr =
        -gammaKur * Zre - omega0 * Zim + 0.5 * K0 * (H1r - H2r);
      const dZi =
        -gammaKur * Zim + omega0 * Zre + 0.5 * K0 * (H1i - H2i);
      const noise = Math.sqrt(Math.max(dt * epsKur, 0));
      Zr[idx] = Zre + dt * dZr + noise * randn();
      Zi[idx] = Zim + dt * dZi + noise * randn();
    }
  }
};

export const deriveKuramotoFields = (
  state: KuramotoState,
  derived: KuramotoDerived
) => {
  const { width, height, Zr, Zi } = state;
  const { gradX, gradY, vort, coh } = derived;
  const theta = (idx: number) => Math.atan2(Zi[idx], Zr[idx]);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const left = wrapIndex(x - 1, y, width, height);
      const right = wrapIndex(x + 1, y, width, height);
      const up = wrapIndex(x, y - 1, width, height);
      const down = wrapIndex(x, y + 1, width, height);
      gradX[idx] = 0.5 * wrapAngle(theta(right) - theta(left));
      gradY[idx] = 0.5 * wrapAngle(theta(down) - theta(up));
      coh[idx] = clamp01(Math.hypot(Zr[idx], Zi[idx]));
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i00 = y * width + x;
      const i10 = wrapIndex(x + 1, y, width, height);
      const i11 = wrapIndex(x + 1, y + 1, width, height);
      const i01 = wrapIndex(x, y + 1, width, height);
      const a = wrapAngle(theta(i10) - theta(i00));
      const b = wrapAngle(theta(i11) - theta(i10));
      const c = wrapAngle(theta(i01) - theta(i11));
      const d = wrapAngle(theta(i00) - theta(i01));
      vort[i00] = (a + b + c + d) / (2 * Math.PI);
    }
  }
};

const mulberry32 = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

export const createNormalGenerator = (seed?: number) => {
  const rng = seed == null ? Math.random : mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare != null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    spare = z1;
    return z0;
  };
};

