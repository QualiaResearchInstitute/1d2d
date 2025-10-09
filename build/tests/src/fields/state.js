import { FIELD_CONTRACTS, FIELD_KINDS } from "./contracts.js";
const initialStatus = (kind) => ({
    kind,
    available: false,
    resolution: null,
    updatedAt: null,
    stalenessMs: Number.POSITIVE_INFINITY,
    stale: false,
    lastSource: null
});
export const createInitialStatuses = () => FIELD_KINDS.reduce((acc, kind) => {
    acc[kind] = initialStatus(kind);
    return acc;
}, {});
export const markFieldUpdate = (prev, kind, resolution, source, now) => {
    const nextStatus = {
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
export const markFieldUnavailable = (prev, kind, source, now) => {
    const status = prev[kind];
    if (!status.available && status.lastSource === source) {
        return prev;
    }
    const nextStatus = {
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
export const refreshFieldStaleness = (prev, now) => {
    let changed = false;
    const updates = [];
    const next = { ...prev };
    for (const kind of FIELD_KINDS) {
        const status = prev[kind];
        const contract = FIELD_CONTRACTS[kind];
        const updatedAt = status.updatedAt;
        const stalenessMs = status.available && updatedAt != null ? Math.max(0, now - updatedAt) : Number.POSITIVE_INFINITY;
        const staleThreshold = contract.lifetime.staleMs;
        const stale = status.available && updatedAt != null && stalenessMs > staleThreshold;
        const stableStaleness = status.available ? stalenessMs : Number.POSITIVE_INFINITY;
        if (Math.abs(status.stalenessMs - stableStaleness) > 0.5 ||
            status.stale !== stale) {
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
