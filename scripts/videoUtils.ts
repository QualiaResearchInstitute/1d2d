export const parseRationalFps = (value: string): number => {
  if (!value) {
    throw new Error('Missing FPS value from ffprobe.');
  }
  if (value.includes('/')) {
    const [num, den] = value.split('/');
    const numerator = Number(num);
    const denominator = Number(den);
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
      throw new Error(`Invalid FPS rational: ${value}`);
    }
    return numerator / denominator;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    throw new Error(`Invalid FPS value: ${value}`);
  }
  return numeric;
};
