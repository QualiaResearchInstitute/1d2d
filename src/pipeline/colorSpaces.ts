const clamp01 = (value: number): number => (value < 0 ? 0 : value > 1 ? 1 : value);

export const srgbToLinear = (value: number): number => {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return Math.pow((value + 0.055) / 1.055, 2.4);
};

export const linearToSrgb = (value: number): number => {
  const clamped = clamp01(value);
  if (clamped <= 0.0031308) {
    return clamped * 12.92;
  }
  return 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
};

export const rgbToLms = (r: number, g: number, b: number): [number, number, number] => [
  0.31399022 * r + 0.63951294 * g + 0.04649755 * b,
  0.15537241 * r + 0.75789446 * g + 0.08670142 * b,
  0.01775239 * r + 0.10944209 * g + 0.87256922 * b,
];

export const lmsToRgb = (L: number, M: number, S: number): [number, number, number] => [
  5.47221206 * L - 4.6419601 * M + 0.16963708 * S,
  -1.1252419 * L + 2.29317094 * M - 0.1678952 * S,
  0.02980165 * L - 0.19318073 * M + 1.16364789 * S,
];

export const linearLuma = (r: number, g: number, b: number): number =>
  0.2126 * r + 0.7152 * g + 0.0722 * b;
