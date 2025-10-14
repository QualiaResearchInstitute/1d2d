const isObject = (value) => typeof value === 'object' && value !== null && !Array.isArray(value);
const toPath = (parts) =>
  parts
    .map((part, index) => {
      if (typeof part === 'number') {
        return `[${part}]`;
      }
      return index === 0 ? part : `.${part}`;
    })
    .join('');
const comparePrimitives = (left, right, acc, path) => {
  if (left !== right) {
    acc.push({ kind: 'changed', path: toPath(path), left, right });
  }
};
export const diffValues = (left, right, path = [], acc = []) => {
  if (left === right) {
    return acc;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    const max = Math.max(left.length, right.length);
    for (let i = 0; i < max; i++) {
      if (i >= left.length) {
        acc.push({ kind: 'added', path: toPath([...path, i]), value: right[i] });
      } else if (i >= right.length) {
        acc.push({ kind: 'removed', path: toPath([...path, i]), value: left[i] });
      } else {
        diffValues(left[i], right[i], [...path, i], acc);
      }
    }
    return acc;
  }
  if (isObject(left) && isObject(right)) {
    const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
    for (const key of keys) {
      if (!(key in right)) {
        acc.push({ kind: 'removed', path: toPath([...path, key]), value: left[key] });
      } else if (!(key in left)) {
        acc.push({ kind: 'added', path: toPath([...path, key]), value: right[key] });
      } else {
        diffValues(left[key], right[key], [...path, key], acc);
      }
    }
    return acc;
  }
  comparePrimitives(left, right, acc, path);
  return acc;
};
