import { FIELD_CONTRACTS, FIELD_KINDS, type FieldKind, type FieldResolution } from "./contracts.js";

export type FieldStatus = {
  kind: FieldKind;
  available: boolean;
  resolution: FieldResolution | null;
  updatedAt: number | null;
  stalenessMs: number;
  stale: boolean;
  lastSource: string | null;
};

export type FieldStatusMap = Record<FieldKind, FieldStatus>;

const initialStatus = (kind: FieldKind): FieldStatus => ({
  kind,
  available: false,
  resolution: null,
  updatedAt: null,
  stalenessMs: Number.POSITIVE_INFINITY,
  stale: false,
  lastSource: null
});

export const createInitialStatuses = (): FieldStatusMap =>
  FIELD_KINDS.reduce<FieldStatusMap>((acc, kind) => {
    acc[kind] = initialStatus(kind);
    return acc;
  }, {} as FieldStatusMap);

export const markFieldUpdate = (
  prev: FieldStatusMap,
  kind: FieldKind,
  resolution: FieldResolution,
  source: string,
  now: number
): FieldStatusMap => {
  const nextStatus: FieldStatus = {
    ...prev[kind],
    available: true,
    resolution,
    updatedAt: now,
    stalenessMs: 0,
    stale: false,
    lastSource: source
  };
  if (prev[kind] === nextStatus) {
    return prev;
  }
  return {
    ...prev,
    [kind]: nextStatus
  };
};

export const markFieldUnavailable = (
  prev: FieldStatusMap,
  kind: FieldKind,
  source: string,
  now: number
): FieldStatusMap => {
  const status = prev[kind];
  if (!status.available && status.lastSource === source) {
    return prev;
  }
  const nextStatus: FieldStatus = {
    ...status,
    available: false,
    resolution: null,
    updatedAt: now,
    stalenessMs: Number.POSITIVE_INFINITY,
    stale: false,
    lastSource: source
  };
  return {
    ...prev,
    [kind]: nextStatus
  };
};

export type FieldStalenessUpdate = {
  kind: FieldKind;
  becameStale: boolean;
  recovered: boolean;
  stalenessMs: number;
};

export const refreshFieldStaleness = (
  prev: FieldStatusMap,
  now: number
): { next: FieldStatusMap; changes: FieldStalenessUpdate[] } => {
  let changed = false;
  const updates: FieldStalenessUpdate[] = [];
  const next: FieldStatusMap = { ...prev };
  for (const kind of FIELD_KINDS) {
    const status = prev[kind];
    const contract = FIELD_CONTRACTS[kind];
    const updatedAt = status.updatedAt;
    const stalenessMs = status.available && updatedAt != null ? Math.max(0, now - updatedAt) : Number.POSITIVE_INFINITY;
    const staleThreshold = contract.lifetime.staleMs;
    const stale = status.available && updatedAt != null && stalenessMs > staleThreshold;
    const stableStaleness = status.available ? stalenessMs : Number.POSITIVE_INFINITY;
    if (
      Math.abs(status.stalenessMs - stableStaleness) > 0.5 ||
      status.stale !== stale
    ) {
      changed = true;
      next[kind] = {
        ...status,
        stalenessMs: stableStaleness,
        stale
      };
      updates.push({
        kind,
        becameStale: !status.stale && stale,
        recovered: status.stale && !stale,
        stalenessMs: stableStaleness
      });
    }
  }
  return {
    next: changed ? next : prev,
    changes: updates
  };
};
