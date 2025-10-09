export type SyntheticCaseId = "circles" | "checkerboard" | "texturedPlane";

export type SyntheticCase = {
  id: SyntheticCaseId;
  label: string;
  description: string;
  generate: (width: number, height: number) => ImageData;
};

const createImage = (width: number, height: number) =>
  new ImageData(new Uint8ClampedArray(width * height * 4), width, height);

const generateCircles = (width: number, height: number): ImageData => {
  const image = createImage(width, height);
  const data = image.data;
  const cx = width * 0.5;
  const cy = height * 0.5;
  const maxR = Math.hypot(cx, cy);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.hypot(dx, dy);
      const ring = Math.abs(Math.sin(r * 0.12));
      const atten = Math.exp(-Math.pow(r / (maxR * 0.9), 2));
      const hue = (Math.atan2(dy, dx) / (2 * Math.PI) + 1) % 1;
      const base = 0.15 + 0.75 * atten;
      const chroma = 0.5 * ring + 0.2 * atten;
      const angle = hue * 6;
      const xComp = chroma * (1 - Math.abs((angle % 2) - 1));
      let r1 = 0;
      let g1 = 0;
      let b1 = 0;
      if (angle >= 0 && angle < 1) {
        r1 = chroma;
        g1 = xComp;
      } else if (angle < 2) {
        r1 = xComp;
        g1 = chroma;
      } else if (angle < 3) {
        g1 = chroma;
        b1 = xComp;
      } else if (angle < 4) {
        g1 = xComp;
        b1 = chroma;
      } else if (angle < 5) {
        r1 = xComp;
        b1 = chroma;
      } else {
        r1 = chroma;
        b1 = xComp;
      }
      const m = base - chroma / 2;
      data[idx + 0] = Math.round((r1 + m) * 255);
      data[idx + 1] = Math.round((g1 + m) * 255);
      data[idx + 2] = Math.round((b1 + m) * 255);
      data[idx + 3] = 255;
    }
  }
  return image;
};

const generateCheckerboard = (width: number, height: number): ImageData => {
  const image = createImage(width, height);
  const data = image.data;
  const tile = Math.max(8, Math.floor(Math.min(width, height) / 12));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const isDark = ((Math.floor(x / tile) + Math.floor(y / tile)) & 1) === 0;
      const luminance = isDark ? 40 : 220;
      data[idx + 0] = luminance;
      data[idx + 1] = luminance;
      data[idx + 2] = luminance;
      data[idx + 3] = 255;
    }
  }
  return image;
};

const generateTexturedPlane = (width: number, height: number): ImageData => {
  const image = createImage(width, height);
  const data = image.data;
  const freqX = 2 * Math.PI / Math.max(1, width);
  const freqY = 2 * Math.PI / Math.max(1, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const nx = x / width;
      const ny = y / height;
      const wave =
        0.5 +
        0.5 *
          (Math.sin(nx * 5 * Math.PI + Math.cos(ny * 2 * Math.PI)) +
            0.5 * Math.sin((nx + ny) * 3 * Math.PI + Math.sin(nx * freqX * 40)));
      const ridge = Math.abs(Math.sin((nx + ny) * Math.PI * 6));
      const r = Math.round(255 * Math.min(1, wave * 0.8 + ridge * 0.2));
      const g = Math.round(255 * Math.min(1, wave * 0.6 + 0.3));
      const b = Math.round(255 * Math.min(1, 0.4 + ridge * 0.5));
      data[idx + 0] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  return image;
};

export const DEFAULT_SYNTHETIC_SIZE = { width: 512, height: 512 } as const;

export const SYNTHETIC_CASES: SyntheticCase[] = [
  {
    id: "circles",
    label: "Concentric Circles",
    description: "Radial rings with chromatic gradients for rim energy checks.",
    generate: generateCircles
  },
  {
    id: "checkerboard",
    label: "High-Contrast Checkerboard",
    description: "Binary checkerboard to stress edge detection and coupling deltas.",
    generate: generateCheckerboard
  },
  {
    id: "texturedPlane",
    label: "Textured Plane",
    description: "Sinusoidal warp plane for surface morph and |Z| baselines.",
    generate: generateTexturedPlane
  }
];
