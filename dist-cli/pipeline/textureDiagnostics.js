const EPS = 1e-6;
const ORIENTATION_OFFSET = 0.9;
const ORIENTATION_INV_RANGE = 1 / (1 - ORIENTATION_OFFSET);
const ORIENTATION_POWER = 2;
const DEFAULT_RESULT = {
  wallpapericityMean: 0,
  wallpapericityStd: 0,
  beatEnergyMean: 0,
  beatEnergyStd: 0,
  resonanceRate: 0,
  sampleCount: 0,
  dogMean: 0,
  dogStd: 0,
  orientationMean: 0,
  orientationStd: 0,
  divisiveMean: 0,
  divisiveStd: 0,
};
const toGray = (surface) => {
  const { width, height } = surface.resolution;
  const total = width * height;
  const gray = new Float32Array(total);
  const data = surface.rgba;
  for (let i = 0; i < total; i++) {
    const idx = i * 4;
    const r = data[idx] / 255;
    const g = data[idx + 1] / 255;
    const b = data[idx + 2] / 255;
    gray[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return gray;
};
const buildGaussianKernel = (sigma) => {
  const radius = Math.max(1, Math.floor(sigma * 3));
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  let sum = 0;
  const inv = 1 / (2 * sigma * sigma + EPS);
  for (let i = -radius; i <= radius; i++) {
    const value = Math.exp(-(i * i) * inv);
    kernel[i + radius] = value;
    sum += value;
  }
  if (sum <= 0) {
    kernel.fill(1 / size);
  } else {
    for (let i = 0; i < size; i++) {
      kernel[i] /= sum;
    }
  }
  return kernel;
};
const applySeparableBlur = (input, width, height, kernel) => {
  const radius = (kernel.length - 1) >> 1;
  const temp = new Float32Array(width * height);
  const output = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let accum = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = Math.min(width - 1, Math.max(0, x + k));
        accum += input[y * width + xx] * kernel[k + radius];
      }
      temp[y * width + x] = accum;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let accum = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = Math.min(height - 1, Math.max(0, y + k));
        accum += temp[yy * width + x] * kernel[k + radius];
      }
      output[y * width + x] = accum;
    }
  }
  return output;
};
const computeDoG = (gray, width, height, sigma, factor = 1.6) => {
  const kernelA = buildGaussianKernel(Math.max(sigma, EPS));
  const kernelB = buildGaussianKernel(Math.max(sigma * factor, EPS));
  const blurA = applySeparableBlur(gray, width, height, kernelA);
  const blurB = applySeparableBlur(gray, width, height, kernelB);
  const total = width * height;
  const result = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    result[i] = blurA[i] - blurB[i];
  }
  return result;
};
const computeGradients = (gray, width, height) => {
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
          const value = gray[idx(x + i, y + j)];
          sx += value * kx[k];
          sy += value * ky[k];
          k++;
        }
      }
      const id = idx(x, y);
      gx[id] = sx;
      gy[id] = sy;
    }
  }
  return { gx, gy };
};
const stats = (sum, sumSq, count) => {
  if (count <= 0) {
    return { mean: 0, std: 0 };
  }
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return { mean, std: Math.sqrt(variance) };
};
export const computeTextureDiagnostics = (surface, config) => {
  if (!surface) {
    return { ...DEFAULT_RESULT };
  }
  const { width, height } = surface.resolution;
  const total = width * height;
  if (total === 0) {
    return { ...DEFAULT_RESULT };
  }
  const gray = toGray(surface);
  const { gx, gy } = computeGradients(gray, width, height);
  const sigmas = [0.7, 1.2, 2.4];
  const scaleCount = sigmas.length;
  const dogAbs = new Float32Array(total);
  for (let s = 0; s < sigmas.length; s++) {
    const dog = computeDoG(gray, width, height, sigmas[s]);
    for (let i = 0; i < total; i++) {
      const v = Math.abs(dog[i]);
      dogAbs[i] += v;
    }
  }
  const orientations = config.orientations.length > 0 ? config.orientations : [0];
  const orientationCount = orientations.length;
  const cos = orientations.map((angle) => Math.cos(angle));
  const sin = orientations.map((angle) => Math.sin(angle));
  const includeMaps = Boolean(config.includeMaps);
  const wallpapericityMap = includeMaps ? new Float32Array(total) : undefined;
  const beatEnvelopeMap = includeMaps && orientationCount > 1 ? new Float32Array(total) : undefined;
  let wallpaperSum = 0;
  let wallpaperSumSq = 0;
  let beatSum = 0;
  let beatSumSq = 0;
  let orientEnergySum = 0;
  let orientEnergySumSq = 0;
  let divisiveSum = 0;
  let divisiveSumSq = 0;
  let dogSum = 0;
  let dogSumSq = 0;
  let resonanceHits = 0;
  for (let i = 0; i < total; i++) {
    const localDog = dogAbs[i] / scaleCount;
    dogSum += localDog;
    dogSumSq += localDog * localDog;
    const gxVal = gx[i];
    const gyVal = gy[i];
    const gMagSq = gxVal * gxVal + gyVal * gyVal;
    const gMag = Math.sqrt(gMagSq);
    const invMag = gMag > EPS ? 1 / gMag : 0;
    let sumOrient = 0;
    let maxEnergy = 0;
    let secondEnergy = 0;
    for (let k = 0; k < orientationCount; k++) {
      let energy = 0;
      if (invMag > 0) {
        const alignment = Math.abs(gxVal * cos[k] + gyVal * sin[k]) * invMag;
        if (alignment >= ORIENTATION_OFFSET) {
          const norm = (alignment - ORIENTATION_OFFSET) * ORIENTATION_INV_RANGE;
          const tuned = ORIENTATION_POWER === 2 ? norm * norm : Math.pow(norm, ORIENTATION_POWER);
          const alignmentSq = alignment * alignment;
          energy = gMagSq * alignmentSq * tuned;
        }
      }
      sumOrient += energy;
      if (energy > maxEnergy) {
        secondEnergy = maxEnergy;
        maxEnergy = energy;
      } else if (energy > secondEnergy) {
        secondEnergy = energy;
      }
    }
    if (sumOrient <= EPS && gMagSq > EPS) {
      sumOrient = gMagSq;
      if (gMagSq > maxEnergy) {
        secondEnergy = maxEnergy;
        maxEnergy = gMagSq;
      } else if (gMagSq > secondEnergy) {
        secondEnergy = gMagSq;
      }
    }
    orientEnergySum += sumOrient;
    orientEnergySumSq += sumOrient * sumOrient;
    const divisive = sumOrient / (sumOrient + localDog + EPS);
    divisiveSum += divisive;
    divisiveSumSq += divisive * divisive;
    wallpaperSum += divisive;
    wallpaperSumSq += divisive * divisive;
    if (wallpapericityMap) {
      wallpapericityMap[i] = divisive;
    }
    let beatLocal = 0;
    if (orientationCount > 1 && maxEnergy > EPS) {
      const ratio = secondEnergy / (maxEnergy + EPS);
      beatLocal = ratio * divisive;
      if (ratio > 0.6 && divisive > 0.1) {
        resonanceHits++;
      }
    }
    beatSum += beatLocal;
    beatSumSq += beatLocal * beatLocal;
    if (beatEnvelopeMap) {
      beatEnvelopeMap[i] = beatLocal;
    }
  }
  const wallpaperStats = stats(wallpaperSum, wallpaperSumSq, total);
  const beatStats = stats(beatSum, beatSumSq, total);
  const dogStats = stats(dogSum, dogSumSq, total);
  const orientStats = stats(orientEnergySum, orientEnergySumSq, total);
  const divisiveStats = stats(divisiveSum, divisiveSumSq, total);
  const resonanceRate = total > 0 ? resonanceHits / total : 0;
  return {
    wallpapericityMean: wallpaperStats.mean,
    wallpapericityStd: wallpaperStats.std,
    beatEnergyMean: beatStats.mean,
    beatEnergyStd: beatStats.std,
    resonanceRate,
    sampleCount: total,
    dogMean: dogStats.mean,
    dogStd: dogStats.std,
    orientationMean: orientStats.mean,
    orientationStd: orientStats.std,
    divisiveMean: divisiveStats.mean,
    divisiveStd: divisiveStats.std,
    wallpapericityMap,
    beatEnvelopeMap,
  };
};
