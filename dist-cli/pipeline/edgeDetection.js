import { makeResolution } from '../fields/contracts.js';
export const computeEdgeField = (image) => {
  const { data, width, height } = image;
  const gray = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      gray[y * width + x] =
        (0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2]) / 255;
    }
  }
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  const idx = (ix, iy) => iy * width + ix;
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sx = 0;
      let sy = 0;
      let k = 0;
      for (let j = -1; j <= 1; j++) {
        for (let i = -1; i <= 1; i++) {
          const v = gray[idx(x + i, y + j)];
          sx += v * kx[k];
          sy += v * ky[k];
          k++;
        }
      }
      gx[idx(x, y)] = sx;
      gy[idx(x, y)] = sy;
    }
  }
  const mag = new Float32Array(width * height);
  let maxMag = 1e-6;
  for (let i = 0; i < mag.length; i++) {
    const m = Math.hypot(gx[i], gy[i]);
    mag[i] = m;
    if (m > maxMag) maxMag = m;
  }
  const inv = 1 / maxMag;
  for (let i = 0; i < mag.length; i++) {
    mag[i] *= inv;
  }
  return {
    kind: 'rim',
    resolution: makeResolution(width, height),
    gx,
    gy,
    mag,
  };
};
