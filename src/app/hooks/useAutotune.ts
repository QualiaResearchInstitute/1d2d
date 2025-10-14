import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useControlPanels } from '../../state/AppState';
import type { ControlPanelState } from '../../state/types';

const TRIAL_COUNT = 32;
const YIELD_INTERVAL = 4;
const MIN_CHANGE_EPSILON = 1e-3;

type JsonSchema = Record<string, unknown>;

interface AutotuneField {
  readonly panelId: string;
  readonly panelLabel: string;
  readonly pointer: string;
  readonly key: string;
  readonly label: string;
  readonly currentValue: number;
  readonly minimum?: number;
  readonly maximum?: number;
  readonly step?: number;
  readonly weight: number;
}

interface Candidate {
  readonly values: Map<string, number>;
}

export interface AutotuneSuggestion {
  readonly panelId: string;
  readonly panelLabel: string;
  readonly pointer: string;
  readonly label: string;
  readonly from: number;
  readonly to: number;
  readonly delta: number;
}

export interface AutotuneResult {
  readonly candidate: Candidate;
  readonly suggestions: AutotuneSuggestion[];
  readonly score: number;
  readonly baselineScore: number;
  readonly trials: number;
  readonly durationMs: number;
}

export interface AutotuneState {
  readonly status: 'idle' | 'running' | 'completed' | 'error';
  readonly progress: number;
  readonly currentTrial: number;
  readonly totalTrials: number;
  readonly result?: AutotuneResult;
  readonly error?: string;
  readonly errorMessage?: string;
}

const decodePointerSegment = (segment: string): string =>
  segment.replace(/~1/g, '/').replace(/~0/g, '~');

const getValueByPointer = (formData: unknown, pointer: string): unknown => {
  if (!pointer || pointer === '/') {
    return formData;
  }
  const segments = pointer.split('/').slice(1).map(decodePointerSegment);
  let current: unknown = formData;
  for (const segment of segments) {
    if (current === null || current === undefined) {
      return undefined;
    }
    if (Array.isArray(current)) {
      const index = Number.parseInt(segment, 10);
      if (!Number.isFinite(index) || index < 0 || index >= current.length) {
        return undefined;
      }
      current = current[index];
    } else if (typeof current === 'object') {
      current = (current as Record<string, unknown>)[segment];
    } else {
      return undefined;
    }
  }
  return current;
};

const cloneDeep = <T>(value: T): T => {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
};

const setValueByPointer = (
  formData: Record<string, unknown>,
  pointer: string,
  value: unknown,
): Record<string, unknown> => {
  const segments = pointer.split('/').slice(1).map(decodePointerSegment);
  if (segments.length === 0) {
    return formData;
  }
  const next = cloneDeep(formData);
  let cursor: any = next;
  for (let index = 0; index < segments.length - 1; index++) {
    const segment = segments[index];
    if (Array.isArray(cursor)) {
      const arrayIndex = Number.parseInt(segment, 10);
      if (!Number.isFinite(arrayIndex) || arrayIndex < 0) {
        return next;
      }
      if (!cursor[arrayIndex]) {
        cursor[arrayIndex] = {};
      }
      cursor = cursor[arrayIndex];
    } else if (cursor && typeof cursor === 'object') {
      if (!(segment in cursor)) {
        cursor[segment] = {};
      }
      cursor = cursor[segment];
    } else {
      return next;
    }
  }
  const finalSegment = segments[segments.length - 1];
  if (Array.isArray(cursor)) {
    const arrayIndex = Number.parseInt(finalSegment, 10);
    if (Number.isFinite(arrayIndex) && arrayIndex >= 0) {
      cursor[arrayIndex] = value;
    }
  } else if (cursor && typeof cursor === 'object') {
    cursor[finalSegment] = value;
  }
  return next;
};

const resolveSchemaForPointer = (schema: JsonSchema, pointer: string): JsonSchema | undefined => {
  const segments = pointer.split('/').slice(1).map(decodePointerSegment);
  let current: any = schema;
  for (const segment of segments) {
    if (!current) {
      return undefined;
    }
    if (
      current.type === 'object' &&
      current.properties &&
      segment in (current.properties as Record<string, unknown>)
    ) {
      current = (current.properties as Record<string, unknown>)[segment];
      continue;
    }
    if (Array.isArray(current.oneOf) || Array.isArray(current.anyOf)) {
      const candidates = (current.oneOf ?? current.anyOf) as JsonSchema[] | undefined;
      if (candidates && candidates.length > 0) {
        current = candidates[0];
      }
    }
    if (current.items) {
      current = current.items;
      continue;
    }
    if (typeof current === 'object' && current !== null && segment in current) {
      current = current[segment as keyof typeof current];
      continue;
    }
    return undefined;
  }
  return current as JsonSchema;
};

const toNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

const deriveWeight = (label: string, pointer: string): number => {
  const token = `${label} ${pointer}`.toLowerCase();
  let weight = 1;
  if (token.includes('gain') || token.includes('blend') || token.includes('weight')) {
    weight += 0.35;
  }
  if (token.includes('symmetry') || token.includes('balance') || token.includes('curvature')) {
    weight += 0.4;
  }
  if (token.includes('flux') || token.includes('phase')) {
    weight += 0.2;
  }
  return weight;
};

const collectFields = (panels: readonly ControlPanelState[]): AutotuneField[] => {
  const fields: AutotuneField[] = [];
  panels.forEach((panel) => {
    panel.bindings.forEach((binding) => {
      const pointer = binding.pointer;
      const value = getValueByPointer(panel.formData, pointer);
      const numericValue = toNumber(value);
      if (numericValue === undefined) {
        return;
      }
      const schemaNode = resolveSchemaForPointer(panel.schema as JsonSchema, pointer);
      const labelFromSchema =
        (schemaNode?.title as string | undefined) ??
        (schemaNode?.description as string | undefined) ??
        decodePointerSegment(pointer.split('/').slice(-1)[0] ?? '');
      const minimum = toNumber(schemaNode?.minimum) ?? toNumber(schemaNode?.min);
      const maximum = toNumber(schemaNode?.maximum) ?? toNumber(schemaNode?.max);
      const step = toNumber(schemaNode?.multipleOf) ?? toNumber(schemaNode?.step);
      const key = `${panel.id}::${pointer}`;
      fields.push({
        panelId: panel.id,
        panelLabel: panel.label,
        pointer,
        key,
        label: labelFromSchema,
        currentValue: numericValue,
        minimum,
        maximum,
        step,
        weight: deriveWeight(labelFromSchema, pointer),
      });
    });
  });
  return fields;
};

const createCandidateFromBaseline = (fields: AutotuneField[]): Candidate => {
  const values = new Map<string, number>();
  fields.forEach((field) => {
    values.set(field.key, field.currentValue);
  });
  return { values };
};

const clampValue = (value: number, field: AutotuneField): number => {
  let next = value;
  if (typeof field.minimum === 'number') {
    next = Math.max(field.minimum, next);
  }
  if (typeof field.maximum === 'number') {
    next = Math.min(field.maximum, next);
  }
  if (typeof field.step === 'number' && field.step > 0) {
    next = Math.round(next / field.step) * field.step;
  }
  return Number(next.toFixed(6));
};

const sampleCandidate = (baseline: Candidate, fields: AutotuneField[]): Candidate => {
  const values = new Map(baseline.values);
  fields.forEach((field) => {
    const current = values.get(field.key) ?? field.currentValue;
    const span =
      typeof field.minimum === 'number' && typeof field.maximum === 'number'
        ? Math.max(field.maximum - field.minimum, field.step ?? 0.1)
        : Math.max(Math.abs(current) || 1, field.step ? field.step * 10 : 1);
    const jitterScale = span * 0.2;
    const random = (Math.random() * 2 - 1) * jitterScale;
    const mutate = Math.random() > 0.3;
    const candidateValue = mutate ? clampValue(current + random, field) : current;
    values.set(field.key, candidateValue);
  });
  return { values };
};

const normalise = (value: number, min?: number, max?: number): number => {
  if (typeof min === 'number' && typeof max === 'number' && max > min) {
    return (value - min) / (max - min);
  }
  const range = Math.max(Math.abs(value) * 2, 1);
  return 0.5 + value / (2 * range);
};

const evaluateCandidate = (candidate: Candidate, fields: AutotuneField[]): number => {
  let scoreSum = 0;
  let weightSum = 0;
  fields.forEach((field) => {
    const value = candidate.values.get(field.key) ?? field.currentValue;
    const normalised = Math.min(1, Math.max(0, normalise(value, field.minimum, field.maximum)));
    const centerPreference = 1 - Math.abs(normalised - 0.5) * 1.8;
    const delta = Math.abs(value - field.currentValue);
    const span =
      typeof field.minimum === 'number' && typeof field.maximum === 'number'
        ? Math.max(field.maximum - field.minimum, field.step ?? 0.1)
        : Math.max(Math.abs(field.currentValue) * 2, field.step ? field.step * 20 : 2);
    const smoothness = 1 - Math.min(1, delta / (span === 0 ? 1 : span));
    const component = centerPreference * 0.65 + smoothness * 0.35;
    scoreSum += component * field.weight;
    weightSum += field.weight;
  });
  if (weightSum === 0) {
    return 0;
  }
  return scoreSum / weightSum;
};

const deriveSuggestions = (
  fields: AutotuneField[],
  baseline: Candidate,
  candidate: Candidate,
): AutotuneSuggestion[] => {
  const suggestions: AutotuneSuggestion[] = [];
  fields.forEach((field) => {
    const from = baseline.values.get(field.key) ?? field.currentValue;
    const to = candidate.values.get(field.key) ?? from;
    const delta = to - from;
    if (Math.abs(delta) < (field.step ?? MIN_CHANGE_EPSILON)) {
      return;
    }
    suggestions.push({
      panelId: field.panelId,
      panelLabel: field.panelLabel,
      pointer: field.pointer,
      label: field.label,
      from,
      to,
      delta,
    });
  });
  suggestions.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return suggestions;
};

export function useAutotune() {
  const { controlsState, updateForm } = useControlPanels();
  const [state, setState] = useState<AutotuneState>({
    status: 'idle',
    progress: 0,
    currentTrial: 0,
    totalTrials: TRIAL_COUNT,
  });
  const fieldsRef = useRef<AutotuneField[]>([]);
  const panelsRef = useRef<Map<string, ControlPanelState>>(new Map());
  const baselineRef = useRef<Candidate | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const setSafeState = useCallback((updater: (prev: AutotuneState) => AutotuneState) => {
    setState((prev) => {
      if (!mountedRef.current) {
        return prev;
      }
      return updater(prev);
    });
  }, []);

  const run = useCallback(async () => {
    if (!controlsState?.panels || controlsState.panels.length === 0) {
      setSafeState(() => ({
        status: 'error',
        error: 'no-controls',
        errorMessage: undefined,
        progress: 0,
        currentTrial: 0,
        totalTrials: TRIAL_COUNT,
      }));
      return;
    }
    setSafeState(() => ({
      status: 'running',
      progress: 0,
      currentTrial: 0,
      totalTrials: TRIAL_COUNT,
      error: undefined,
      errorMessage: undefined,
    }));

    try {
      const panels = controlsState.panels;
      const fields = collectFields(panels);
      fieldsRef.current = fields;
      panelsRef.current = new Map(panels.map((panel) => [panel.id, panel]));

      if (fields.length === 0) {
        setSafeState(() => ({
          status: 'error',
          error: 'no-numeric-fields',
          errorMessage: undefined,
          progress: 0,
          currentTrial: 0,
          totalTrials: TRIAL_COUNT,
        }));
        return;
      }

      const baseline = createCandidateFromBaseline(fields);
      baselineRef.current = baseline;
      let bestCandidate = baseline;
      let bestScore = evaluateCandidate(baseline, fields);
      const start =
        typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();

      for (let trial = 0; trial < TRIAL_COUNT; trial++) {
        const candidate = sampleCandidate(baseline, fields);
        const score = evaluateCandidate(candidate, fields);
        if (score > bestScore + 1e-4) {
          bestScore = score;
          bestCandidate = candidate;
        }
        const progress = (trial + 1) / TRIAL_COUNT;
        const currentTrial = trial + 1;
        setSafeState((prev) =>
          prev.status === 'running'
            ? {
                status: 'running',
                progress,
                currentTrial,
                totalTrials: TRIAL_COUNT,
                error: undefined,
                errorMessage: undefined,
                result: undefined,
              }
            : prev,
        );
        if ((trial + 1) % YIELD_INTERVAL === 0) {
          await new Promise<void>((resolve) => {
            if (typeof requestAnimationFrame === 'function') {
              requestAnimationFrame(() => resolve());
            } else {
              setTimeout(() => resolve(), 0);
            }
          });
        }
      }

      const end =
        typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
      const baselineScore = evaluateCandidate(baseline, fields);
      const suggestions = deriveSuggestions(fields, baseline, bestCandidate);
      const result: AutotuneResult = {
        candidate: bestCandidate,
        suggestions,
        score: bestScore,
        baselineScore,
        trials: TRIAL_COUNT,
        durationMs: end - start,
      };

      setSafeState((prev) => ({
        status: 'completed',
        progress: 1,
        currentTrial: TRIAL_COUNT,
        totalTrials: TRIAL_COUNT,
        result,
        error: undefined,
        errorMessage: undefined,
      }));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSafeState(() => ({
        status: 'error',
        error: 'failed',
        errorMessage: message,
        progress: 0,
        currentTrial: 0,
        totalTrials: TRIAL_COUNT,
      }));
    }
  }, [controlsState?.panels, setSafeState]);

  const reset = useCallback(() => {
    setSafeState(() => ({
      status: 'idle',
      progress: 0,
      currentTrial: 0,
      totalTrials: TRIAL_COUNT,
      error: undefined,
      errorMessage: undefined,
      result: undefined,
    }));
  }, [setSafeState]);

  const apply = useCallback(() => {
    if (!state.result || !controlsState?.panels) {
      return;
    }
    const fields = fieldsRef.current;
    const candidate = state.result.candidate;
    const panelMap = panelsRef.current;
    const updates = new Map<string, Record<string, unknown>>();
    fields.forEach((field) => {
      const value = candidate.values.get(field.key);
      if (value === undefined) {
        return;
      }
      const panel = panelMap.get(field.panelId);
      if (!panel) {
        return;
      }
      const baseForm = updates.get(field.panelId) ?? cloneDeep(panel.formData);
      const updated = setValueByPointer(baseForm, field.pointer, value);
      updates.set(field.panelId, updated);
    });
    updates.forEach((formData, panelId) => {
      updateForm(panelId, formData);
    });
  }, [controlsState?.panels, state.result, updateForm]);

  return useMemo(
    () => ({
      state,
      run,
      apply,
      reset,
    }),
    [apply, reset, run, state],
  );
}
