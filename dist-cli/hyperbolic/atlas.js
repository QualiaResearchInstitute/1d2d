import { makeResolution } from '../fields/contracts.js';
const EPS = 1e-6;
const MAX_LIMIT = 0.999999;
const clamp01 = (value) => {
  if (Number.isNaN(value) || value === Infinity || value === -Infinity) return 0;
  return Math.min(1, Math.max(0, value));
};
const clampAbs = (value, limit) => {
  const bounded = Math.min(Math.abs(value), limit);
  return value >= 0 ? bounded : -bounded;
};
const safeAtanh = (x) => {
  const clamped = Math.min(MAX_LIMIT, Math.max(-MAX_LIMIT, x));
  return 0.5 * Math.log((1 + clamped) / (1 - clamped));
};
const safeTanh = (x) => {
  if (x > 8) return 0.9999997749;
  if (x < -8) return -0.9999997749;
  const e2 = Math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
};
export const createHyperbolicAtlas = (params) => {
  const width = Math.max(1, Math.floor(params.width));
  const height = Math.max(1, Math.floor(params.height));
  const curvatureStrength = clamp01(Math.abs(params.curvatureStrength));
  const diskLimit = Math.min(MAX_LIMIT, Math.max(0.1, params.diskLimit ?? 0.999));
  const defaultCenter = (size) => Math.floor(size / 2);
  const centerX = params.centerX ?? defaultCenter(width);
  const centerY = params.centerY ?? defaultCenter(height);
  const extentX = Math.max(centerX, width - 1 - centerX);
  const extentY = Math.max(centerY, height - 1 - centerY);
  const maxRadius = Math.max(1, Math.hypot(extentX, extentY));
  const curvatureScale = 1 + curvatureStrength * 3;
  const resolution = makeResolution(width, height);
  const texels = resolution.texels;
  const coords = new Float32Array(texels * 2);
  const polar = new Float32Array(texels * 2);
  const jacobians = new Float32Array(texels * 4);
  const areaWeights = new Float32Array(texels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      const coordIndex = p * 2;
      const jacIndex = p * 4;
      const nx = (x - centerX) / maxRadius;
      const ny = (y - centerY) / maxRadius;
      const r = Math.hypot(nx, ny);
      const directionX = r > EPS ? nx / r : 1;
      const directionY = r > EPS ? ny / r : 0;
      const clipped = Math.min(r, diskLimit);
      const rho = safeAtanh(clipped);
      const hyperRadius = 2 * curvatureScale * rho;
      const targetRadius = safeTanh(rho * curvatureScale);
      const localRadius = clipped > EPS ? clipped : 1;
      const baseFactor = clipped > EPS ? targetRadius / localRadius : curvatureScale;
      let diskX = nx * baseFactor;
      let diskY = ny * baseFactor;
      let dDiskX_dNx;
      let dDiskX_dNy;
      let dDiskY_dNx;
      let dDiskY_dNy;
      if (clipped <= EPS) {
        diskX = nx * curvatureScale;
        diskY = ny * curvatureScale;
        dDiskX_dNx = curvatureScale;
        dDiskX_dNy = 0;
        dDiskY_dNx = 0;
        dDiskY_dNy = curvatureScale;
      } else if (clipped >= diskLimit - 1e-6) {
        const factor = targetRadius / localRadius;
        diskX = directionX * targetRadius;
        diskY = directionY * targetRadius;
        dDiskX_dNx = factor * (directionX * directionX);
        dDiskX_dNy = factor * (directionX * directionY);
        dDiskY_dNx = factor * (directionY * directionX);
        dDiskY_dNy = factor * (directionY * directionY);
      } else {
        const drhoDr = 1 / (1 - clipped * clipped);
        const targetPrime = curvatureScale * (1 - targetRadius * targetRadius) * drhoDr;
        const g = targetRadius / clipped;
        const gPrime = (targetPrime * clipped - targetRadius) / (clipped * clipped);
        const drdNx = nx / r;
        const drdNy = ny / r;
        dDiskX_dNx = g + nx * gPrime * drdNx;
        dDiskX_dNy = nx * gPrime * drdNy;
        dDiskY_dNx = ny * gPrime * drdNx;
        dDiskY_dNy = g + ny * gPrime * drdNy;
        diskX = nx * g;
        diskY = ny * g;
      }
      if (params.mode === 'klein') {
        const sq = diskX * diskX + diskY * diskY;
        const kFactor = 2 / (1 + sq);
        const factorSq = -4 / (1 + sq) ** 2;
        const dFactor_dPx = factorSq * diskX;
        const dFactor_dPy = factorSq * diskY;
        const dk_dNx = dFactor_dPx * dDiskX_dNx + dFactor_dPy * dDiskY_dNx;
        const dk_dNy = dFactor_dPx * dDiskX_dNy + dFactor_dPy * dDiskY_dNy;
        const px = diskX;
        const py = diskY;
        diskX = px * kFactor;
        diskY = py * kFactor;
        dDiskX_dNx = kFactor * dDiskX_dNx + px * dk_dNx;
        dDiskX_dNy = kFactor * dDiskX_dNy + px * dk_dNy;
        dDiskY_dNx = kFactor * dDiskY_dNx + py * dk_dNx;
        dDiskY_dNy = kFactor * dDiskY_dNy + py * dk_dNy;
      }
      const sampleX = clampAbs(diskX, MAX_LIMIT) * maxRadius + centerX;
      const sampleY = clampAbs(diskY, MAX_LIMIT) * maxRadius + centerY;
      coords[coordIndex] = Math.min(width - 1, Math.max(0, sampleX));
      coords[coordIndex + 1] = Math.min(height - 1, Math.max(0, sampleY));
      polar[coordIndex] = hyperRadius;
      polar[coordIndex + 1] = Math.atan2(ny, nx);
      const j11 = dDiskX_dNx / maxRadius;
      const j12 = dDiskX_dNy / maxRadius;
      const j21 = dDiskY_dNx / maxRadius;
      const j22 = dDiskY_dNy / maxRadius;
      jacobians[jacIndex] = j11;
      jacobians[jacIndex + 1] = j12;
      jacobians[jacIndex + 2] = j21;
      jacobians[jacIndex + 3] = j22;
      const det = j11 * j22 - j12 * j21;
      const denom = Math.max(EPS, 1 - (diskX * diskX + diskY * diskY));
      const metricScale = 4 / (denom * denom);
      areaWeights[p] = Math.abs(det) * metricScale;
    }
  }
  return {
    resolution,
    coords,
    polar,
    jacobians,
    areaWeights,
    metadata: {
      curvatureStrength,
      curvatureScale,
      diskLimit,
      centerX,
      centerY,
      maxRadius,
      mode: params.mode,
    },
  };
};
export const packageHyperbolicAtlasForGpu = (atlas) => {
  const { resolution, coords, polar, jacobians, areaWeights, metadata } = atlas;
  const texels = resolution.texels;
  const stride = 9;
  const buffer = new Float32Array(texels * stride);
  for (let i = 0; i < texels; i++) {
    const outOffset = i * stride;
    const coordOffset = i * 2;
    const jacOffset = i * 4;
    buffer[outOffset + 0] = coords[coordOffset + 0];
    buffer[outOffset + 1] = coords[coordOffset + 1];
    buffer[outOffset + 2] = polar[coordOffset + 0];
    buffer[outOffset + 3] = polar[coordOffset + 1];
    buffer[outOffset + 4] = jacobians[jacOffset + 0];
    buffer[outOffset + 5] = jacobians[jacOffset + 1];
    buffer[outOffset + 6] = jacobians[jacOffset + 2];
    buffer[outOffset + 7] = jacobians[jacOffset + 3];
    buffer[outOffset + 8] = areaWeights[i];
  }
  return {
    width: resolution.width,
    height: resolution.height,
    buffer,
    layout: {
      stride,
      components: stride,
      meaning: [
        'sampleX',
        'sampleY',
        'radius',
        'theta',
        'jacobianXX',
        'jacobianXY',
        'jacobianYX',
        'jacobianYY',
        'hyperbolicAreaWeight',
      ],
    },
    metadata,
  };
};
export const mapHyperbolicPolarToPixel = (atlas, radius, theta) => {
  const { curvatureScale, centerX, centerY, maxRadius, diskLimit, mode } = atlas.metadata;
  const rho = radius / (2 * curvatureScale);
  const diskRadius = safeTanh(rho);
  const clamped = Math.min(diskLimit, Math.max(0, diskRadius));
  let px = Math.cos(theta) * clamped;
  let py = Math.sin(theta) * clamped;
  if (mode === 'klein') {
    const denom = 1 + px * px + py * py;
    if (denom > EPS) {
      const factor = 2 / denom;
      px *= factor;
      py *= factor;
    }
  }
  const x = px * maxRadius + centerX;
  const y = py * maxRadius + centerY;
  return [x, y];
};
