type Primitive = string | number | boolean | null;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export type DiffEntry =
  | { kind: 'added'; path: string; value: unknown }
  | { kind: 'removed'; path: string; value: unknown }
  | { kind: 'changed'; path: string; left: unknown; right: unknown };

const toPath = (parts: Array<string | number>): string =>
  parts
    .map((part, index) => {
      if (typeof part === 'number') {
        return `[${part}]`;
      }
      return index === 0 ? part : `.${part}`;
    })
    .join('');

const comparePrimitives = (
  left: Primitive,
  right: Primitive,
  acc: DiffEntry[],
  path: Array<string | number>,
) => {
  if (left !== right) {
    acc.push({ kind: 'changed', path: toPath(path), left, right });
  }
};

export const diffValues = (
  left: unknown,
  right: unknown,
  path: Array<string | number> = [],
  acc: DiffEntry[] = [],
): DiffEntry[] => {
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
  comparePrimitives(left as Primitive, right as Primitive, acc, path);
  return acc;
};
