const EPSILON = 1e-6;
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const gaussian = (distance, sigma) => {
  const scale = 2 * sigma * sigma + EPSILON;
  return Math.exp(-(distance * distance) / scale);
};
const sanitizeFluxSource = (source, width, height) => {
  const safeX = Number.isFinite(source.x) ? clamp(source.x, 0, width - 1) : width * 0.5;
  const safeY = Number.isFinite(source.y) ? clamp(source.y, 0, height - 1) : height * 0.5;
  const rawCharge = Number.isFinite(source.charge) ? source.charge : 1;
  const charge = rawCharge === 0 ? 1 : rawCharge;
  const strength = Number.isFinite(source.strength)
    ? Math.max(Math.abs(source.strength), EPSILON)
    : 1;
  return {
    x: safeX,
    y: safeY,
    charge,
    strength,
  };
};
const toPairs = (sources) => {
  const positives = [];
  const negatives = [];
  sources.forEach((source) => {
    if (source.charge >= 0) {
      positives.push(source);
    } else {
      negatives.push(source);
    }
  });
  if (positives.length === 0 || negatives.length === 0) {
    return [];
  }
  const pairs = [];
  const negTaken = new Set();
  positives.forEach((pos) => {
    let bestIdx = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    negatives.forEach((neg, idx) => {
      if (negTaken.has(idx)) return;
      const dx = pos.x - neg.x;
      const dy = pos.y - neg.y;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    });
    if (bestIdx >= 0) {
      const neg = negatives[bestIdx];
      negTaken.add(bestIdx);
      pairs.push({
        ax: pos.x + 0.5,
        ay: pos.y + 0.5,
        bx: neg.x + 0.5,
        by: neg.y + 0.5,
        strength: 0.5 * (pos.strength ?? 1) + 0.5 * (neg.strength ?? 1),
      });
    }
  });
  negatives.forEach((neg, idx) => {
    if (negTaken.has(idx)) {
      return;
    }
    let bestIdx = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    positives.forEach((pos, pIdx) => {
      const dx = pos.x - neg.x;
      const dy = pos.y - neg.y;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = pIdx;
      }
    });
    if (bestIdx >= 0) {
      const pos = positives[bestIdx];
      pairs.push({
        ax: pos.x + 0.5,
        ay: pos.y + 0.5,
        bx: neg.x + 0.5,
        by: neg.y + 0.5,
        strength: 0.5 * (pos.strength ?? 1) + 0.5 * (neg.strength ?? 1),
      });
    }
  });
  return pairs;
};
const accumulatePairEnergy = (pair, energy, direction, width, height, tubeRadius) => {
  const ax = pair.ax;
  const ay = pair.ay;
  const bx = pair.bx;
  const by = pair.by;
  const dx = bx - ax;
  const dy = by - ay;
  const length = Math.hypot(dx, dy);
  if (length <= EPSILON) {
    return;
  }
  const dirX = dx / length;
  const dirY = dy / length;
  const radius = tubeRadius;
  const radiusPad = radius * 3;
  const minX = clamp(Math.floor(Math.min(ax, bx) - radiusPad), 0, width - 1);
  const maxX = clamp(Math.ceil(Math.max(ax, bx) + radiusPad), 0, width - 1);
  const minY = clamp(Math.floor(Math.min(ay, by) - radiusPad), 0, height - 1);
  const maxY = clamp(Math.ceil(Math.max(ay, by) + radiusPad), 0, height - 1);
  const lengthSq = length * length + EPSILON;
  for (let y = minY; y <= maxY; y++) {
    const py = y + 0.5;
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const apx = px - ax;
      const apy = py - ay;
      let t = (apx * dx + apy * dy) / lengthSq;
      t = clamp(t, 0, 1);
      const closestX = ax + dx * t;
      const closestY = ay + dy * t;
      const distX = px - closestX;
      const distY = py - closestY;
      const distance = Math.hypot(distX, distY);
      if (distance > radiusPad) continue;
      const tubeProfile = gaussian(distance, radius);
      if (tubeProfile <= EPSILON) continue;
      const chord = Math.sin(Math.PI * clamp(t, 0, 1));
      const weight = pair.strength * tubeProfile * clamp(chord, 0.1, 1);
      const idx = y * width + x;
      energy[idx] += weight;
      const dirIndex = idx * 2;
      direction[dirIndex + 0] += dirX * weight;
      direction[dirIndex + 1] += dirY * weight;
    }
  }
};
const accumulateSourceEnergy = (source, energy, width, height, sourceRadius) => {
  const centerX = source.x + 0.5;
  const centerY = source.y + 0.5;
  const radius = sourceRadius;
  const minX = clamp(Math.floor(centerX - radius * 2), 0, width - 1);
  const maxX = clamp(Math.ceil(centerX + radius * 2), 0, width - 1);
  const minY = clamp(Math.floor(centerY - radius * 2), 0, height - 1);
  const maxY = clamp(Math.ceil(centerY + radius * 2), 0, height - 1);
  for (let y = minY; y <= maxY; y++) {
    const py = y + 0.5;
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const distance = Math.hypot(px - centerX, py - centerY);
      const profile = gaussian(distance, radius);
      if (profile <= EPSILON) continue;
      const idx = y * width + x;
      energy[idx] += profile * source.strength * 0.6;
    }
  }
};
export const computeFluxOverlayState = (params) => {
  const { width, height } = params;
  if (width <= 0 || height <= 0) {
    return null;
  }
  const sanitizedSources = params.sources.map((source) =>
    sanitizeFluxSource(source, width, height),
  );
  if (sanitizedSources.length < 2) {
    return null;
  }
  const pairs = toPairs(sanitizedSources);
  if (pairs.length === 0) {
    return null;
  }
  const total = width * height;
  const energy = new Float32Array(total);
  const direction = new Float32Array(total * 2);
  const baseRadius = params.tubeRadius ?? Math.max(3, Math.min(width, height) * 0.03);
  const sourceRadius = params.sourceRadius ?? Math.max(2, baseRadius * 0.65);
  pairs.forEach((pair) => accumulatePairEnergy(pair, energy, direction, width, height, baseRadius));
  sanitizedSources.forEach((source) =>
    accumulateSourceEnergy(source, energy, width, height, sourceRadius),
  );
  let maxEnergy = 0;
  for (let idx = 0; idx < total; idx++) {
    const value = energy[idx];
    if (value > maxEnergy) {
      maxEnergy = value;
    }
  }
  if (maxEnergy <= EPSILON) {
    return null;
  }
  const invMax = 1 / maxEnergy;
  for (let idx = 0; idx < total; idx++) {
    const weight = energy[idx];
    const dirIndex = idx * 2;
    if (weight <= EPSILON) {
      direction[dirIndex + 0] = 0;
      direction[dirIndex + 1] = 0;
      continue;
    }
    const vx = direction[dirIndex + 0] / weight;
    const vy = direction[dirIndex + 1] / weight;
    const norm = Math.hypot(vx, vy);
    if (norm <= EPSILON) {
      direction[dirIndex + 0] = 0;
      direction[dirIndex + 1] = 0;
    } else {
      direction[dirIndex + 0] = vx / norm;
      direction[dirIndex + 1] = vy / norm;
    }
  }
  return {
    width,
    height,
    energy,
    direction,
    energyScale: invMax,
    maxEnergy,
  };
};
export const buildFluxSu3Block = (normalizedEnergy, angle) => {
  const theta = clamp(normalizedEnergy, 0, 1) * clamp(angle, -Math.PI, Math.PI);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  return [
    [
      { re: cos, im: 0 },
      { re: -sin, im: 0 },
      { re: 0, im: 0 },
    ],
    [
      { re: sin, im: 0 },
      { re: cos, im: 0 },
      { re: 0, im: 0 },
    ],
    [
      { re: 0, im: 0 },
      { re: 0, im: 0 },
      { re: 1, im: 0 },
    ],
  ];
};
const hsvToRgb = (h, s, v) => {
  const hue = ((h % 1) + 1) % 1;
  const i = Math.floor(hue * 6);
  const f = hue * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0:
      return [v, t, p];
    case 1:
      return [q, v, p];
    case 2:
      return [p, v, t];
    case 3:
      return [p, q, v];
    case 4:
      return [t, p, v];
    default:
      return [v, p, q];
  }
};
export const fluxEnergyToOverlayColor = (normalizedEnergy, angle) => {
  const energy = clamp(normalizedEnergy, 0, 1);
  const hue = (((angle / (Math.PI * 2)) % 1) + 1) % 1;
  const saturation = clamp(0.45 + 0.35 * energy, 0, 1);
  const value = clamp(0.35 + 0.6 * energy, 0, 1);
  return hsvToRgb(hue, saturation, value);
};
