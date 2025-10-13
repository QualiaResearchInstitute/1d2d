import { createHash } from 'blake3';

const textEncoder = new TextEncoder();

const bytesToHex = (bytes: Uint8Array): string => {
  let result = '';
  for (let i = 0; i < bytes.length; i++) {
    result += bytes[i].toString(16).padStart(2, '0');
  }
  return result;
};

type CanonicalScalar = null | boolean | number | string;

type CanonicalValue = CanonicalScalar | CanonicalValue[] | { [key: string]: CanonicalValue };

const formatCanonicalNumber = (value: number): string => {
  if (!Number.isFinite(value)) {
    throw new TypeError(`Canonical JSON cannot encode non-finite numbers (received ${value})`);
  }
  if (Object.is(value, -0)) {
    return '0';
  }
  if (value === 0) {
    return '0';
  }
  let text = Number(value).toString();
  if (!text.includes('e')) {
    if (!text.includes('.')) {
      return text;
    }
    while (text.endsWith('0')) {
      text = text.slice(0, -1);
    }
    if (text.endsWith('.')) {
      text = text.slice(0, -1);
    }
    return text;
  }
  const [mantissa, exponentRaw] = text.split('e');
  let mantissaText = mantissa;
  if (mantissaText.includes('.')) {
    while (mantissaText.endsWith('0')) {
      mantissaText = mantissaText.slice(0, -1);
    }
    if (mantissaText.endsWith('.')) {
      mantissaText = mantissaText.slice(0, -1);
    }
  }
  const exponent = exponentRaw.startsWith('+') ? exponentRaw.slice(1) : exponentRaw;
  return `${mantissaText}e${exponent}`;
};

type StringifyConfig = {
  indentUnit?: string;
};

const stringifyCanonicalValue = (
  value: CanonicalValue,
  config: StringifyConfig,
  depth: number,
): string => {
  if (value === null) {
    return 'null';
  }
  const valueType = typeof value;
  if (valueType === 'number') {
    return formatCanonicalNumber(value as number);
  }
  if (valueType === 'string') {
    return JSON.stringify(value as string);
  }
  if (valueType === 'boolean') {
    return value ? 'true' : 'false';
  }

  const indentUnit = config.indentUnit;
  const hasIndent = typeof indentUnit === 'string';

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '[]';
    }
    if (!hasIndent) {
      const parts = value.map((entry) => stringifyCanonicalValue(entry, config, depth));
      return `[${parts.join(',')}]`;
    }
    const nextIndent = indentUnit.repeat(depth + 1);
    const baseIndent = indentUnit.repeat(depth);
    const parts = value.map(
      (entry) => `${nextIndent}${stringifyCanonicalValue(entry, config, depth + 1)}`,
    );
    return `[\n${parts.join(',\n')}\n${baseIndent}]`;
  }

  if (value && typeof value === 'object') {
    const entries = Object.keys(value);
    if (entries.length === 0) {
      return '{}';
    }
    if (!hasIndent) {
      const parts = entries.map(
        (key) =>
          `${JSON.stringify(key)}:${stringifyCanonicalValue(
            (value as Record<string, CanonicalValue>)[key],
            config,
            depth,
          )}`,
      );
      return `{${parts.join(',')}}`;
    }
    const nextIndent = indentUnit.repeat(depth + 1);
    const baseIndent = indentUnit.repeat(depth);
    const parts = entries.map((key) => {
      const fieldValue = (value as Record<string, CanonicalValue>)[key];
      return `${nextIndent}${JSON.stringify(key)}: ${stringifyCanonicalValue(fieldValue, config, depth + 1)}`;
    });
    return `{\n${parts.join(',\n')}\n${baseIndent}}`;
  }

  throw new TypeError('Unsupported canonical JSON value encountered during serialization');
};

type NormalizeContext = {
  inArray: boolean;
};

const normalizeNumber = (value: number): number => {
  if (!Number.isFinite(value)) {
    throw new TypeError(`Canonical JSON cannot encode non-finite numbers (received ${value})`);
  }
  return Object.is(value, -0) ? 0 : value;
};

const normalizeArray = (source: ArrayLike<unknown>): CanonicalValue[] => {
  const result: CanonicalValue[] = [];
  for (let i = 0; i < source.length; i++) {
    const normalized = normalizeValue((source as any)[i], { inArray: true });
    result.push(normalized ?? null);
  }
  return result;
};

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
};

const normalizeObject = (value: Record<string, unknown>): CanonicalValue => {
  const entries = Object.keys(value)
    .filter(
      (key) =>
        value[key] !== undefined &&
        typeof value[key] !== 'function' &&
        typeof value[key] !== 'symbol',
    )
    .sort();
  const result: Record<string, CanonicalValue> = {};
  for (const key of entries) {
    const normalized = normalizeValue(value[key], { inArray: false });
    if (normalized !== undefined) {
      result[key] = normalized;
    }
  }
  return result;
};

const normalizeValue = (value: unknown, context: NormalizeContext): CanonicalValue | undefined => {
  if (value === null) {
    return null;
  }
  if (value === undefined) {
    return context.inArray ? null : undefined;
  }
  if (typeof value === 'number') {
    return normalizeNumber(value);
  }
  if (typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'bigint') {
    throw new TypeError('Canonical JSON does not support bigint values');
  }
  if (typeof value === 'function' || typeof value === 'symbol') {
    return context.inArray ? null : undefined;
  }
  if (Array.isArray(value)) {
    return normalizeArray(value);
  }
  if (ArrayBuffer.isView(value)) {
    return normalizeArray(Array.from(value as unknown as number[]));
  }
  if (value instanceof ArrayBuffer) {
    return normalizeArray(Array.from(new Uint8Array(value)));
  }

  if (value && typeof (value as { toJSON?: () => unknown }).toJSON === 'function') {
    const transformed = (value as { toJSON: () => unknown }).toJSON();
    return normalizeValue(transformed, context);
  }

  if (value instanceof Map) {
    const pairs: [string, unknown][] = [];
    for (const [key, entry] of value.entries()) {
      pairs.push([String(key), entry]);
    }
    pairs.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    const result: Record<string, CanonicalValue> = {};
    for (const [key, entry] of pairs) {
      const normalized = normalizeValue(entry, { inArray: false });
      if (normalized !== undefined) {
        result[key] = normalized;
      }
    }
    return result;
  }

  if (value instanceof Set) {
    const elements = Array.from(value).sort();
    return normalizeArray(elements);
  }

  if (isPlainObject(value)) {
    return normalizeObject(value);
  }

  return normalizeObject(Object(value as Record<string, unknown>));
};

const cleanParsed = (value: unknown): unknown => {
  if (value === null) {
    return null;
  }
  if (typeof value === 'number') {
    return Object.is(value, -0) ? 0 : value;
  }
  if (typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => cleanParsed(entry));
  }
  if (value && typeof value === 'object') {
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(value)) {
      result[key] = cleanParsed((value as Record<string, unknown>)[key]);
    }
    return result;
  }
  return value;
};

export type CanonicalJsonWriteOptions = {
  indent?: number;
};

export const writeCanonicalJson = (
  value: unknown,
  options: CanonicalJsonWriteOptions = {},
): string => {
  const normalized = normalizeValue(value, { inArray: false });
  const indentSize =
    typeof options.indent === 'number' && options.indent > 0
      ? Math.min(options.indent, 10)
      : undefined;
  const config: StringifyConfig =
    indentSize !== undefined ? { indentUnit: ' '.repeat(indentSize) } : {};
  return stringifyCanonicalValue(normalized ?? null, config, 0);
};

export const readCanonicalJson = <T>(text: string): T => {
  const parsed = JSON.parse(text) as unknown;
  return cleanParsed(parsed) as T;
};

export const hashCanonicalJsonString = (json: string): string => {
  const bytes = textEncoder.encode(json);
  const digest = createHash().update(bytes).digest();
  return bytesToHex(digest);
};

export const hashCanonicalJson = (
  value: unknown,
  options: CanonicalJsonWriteOptions = {},
): { json: string; hash: string } => {
  const json = writeCanonicalJson(value, options);
  const hash = hashCanonicalJsonString(json);
  return { json, hash };
};
