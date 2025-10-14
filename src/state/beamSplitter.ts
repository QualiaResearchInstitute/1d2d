import type { SceneGraphState, SceneNode } from './types.js';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const createId = (prefix: string): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
};

export type BeamSplitterBranchSource = 'source' | 'edge' | 'phase' | 'oscillator' | 'surface';

export type BeamSplitterTransformStep =
  | {
      kind: 'rotate';
      degrees: number;
    }
  | {
      kind: 'mirror';
      axis: 'x' | 'y';
    }
  | {
      kind: 'scale';
      factor: number;
    };

export type BeamSplitterBranch = {
  id: string;
  label: string;
  symmetry?: string;
  weight?: number;
  priority?: number;
  source?: BeamSplitterBranchSource;
  transformStack: BeamSplitterTransformStep[];
};

const clampDegrees = (degrees: unknown) => {
  if (typeof degrees !== 'number' || !Number.isFinite(degrees)) {
    return 0;
  }
  let norm = degrees % 360;
  if (norm < -180) norm += 360;
  if (norm > 180) norm -= 360;
  return norm;
};

const clampFactor = (factor: unknown) => {
  if (typeof factor !== 'number' || !Number.isFinite(factor) || factor === 0) {
    return 1;
  }
  const clamped = Math.max(0.05, Math.min(20, factor));
  return clamped;
};

const parseTransformStep = (value: unknown): BeamSplitterTransformStep | null => {
  if (!isRecord(value) || typeof value.kind !== 'string') {
    return null;
  }
  switch (value.kind) {
    case 'rotate':
      return {
        kind: 'rotate',
        degrees: clampDegrees(value.degrees),
      };
    case 'mirror':
      return {
        kind: 'mirror',
        axis: value.axis === 'y' ? 'y' : 'x',
      };
    case 'scale':
      return {
        kind: 'scale',
        factor: clampFactor(value.factor),
      };
    default:
      return null;
  }
};

const parseTransformStack = (value: unknown): BeamSplitterTransformStep[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const stack: BeamSplitterTransformStep[] = [];
  for (const entry of value) {
    const step = parseTransformStep(entry);
    if (step) {
      stack.push(step);
    }
  }
  return stack.length > 0 ? stack : [];
};

export const parseBranches = (metadata: unknown): BeamSplitterBranch[] => {
  if (!isRecord(metadata)) {
    return [];
  }
  const rawBranches = Array.isArray(metadata.branches) ? metadata.branches : [];
  const seenIds = new Set<string>();
  return rawBranches.map((entry, index) => {
    const record = isRecord(entry) ? entry : {};
    let id = typeof record.id === 'string' && record.id.trim() ? record.id : createId('branch');
    while (seenIds.has(id)) {
      id = createId('branch');
    }
    seenIds.add(id);
    return {
      id,
      label:
        typeof record.label === 'string' && record.label.trim()
          ? record.label
          : `Branch ${index + 1}`,
      symmetry: typeof record.symmetry === 'string' ? record.symmetry : 'identity',
      weight: typeof record.weight === 'number' ? record.weight : 1,
      priority: typeof record.priority === 'number' ? record.priority : index,
      source:
        record.source === 'edge' ||
        record.source === 'phase' ||
        record.source === 'oscillator' ||
        record.source === 'surface'
          ? (record.source as BeamSplitterBranchSource)
          : 'source',
      transformStack: (() => {
        const parsed = parseTransformStack(record.transformStack);
        if (parsed.length > 0) {
          return parsed;
        }
        const legacy = parseTransformStep(record.transform);
        if (legacy) {
          return [legacy];
        }
        return [
          {
            kind: 'rotate',
            degrees: 0,
          },
        ];
      })(),
    };
  });
};

export const createBranch = (index: number): BeamSplitterBranch => ({
  id: createId('branch'),
  label: `Branch ${index + 1}`,
  symmetry: 'identity',
  weight: 1,
  priority: index,
  source: 'source',
  transformStack: [
    {
      kind: 'rotate',
      degrees: 0,
    },
  ],
});

export const serialiseBranches = (branches: BeamSplitterBranch[]): Record<string, unknown>[] =>
  branches.map((branch, index) => ({
    id: branch.id,
    label: branch.label || `Branch ${index + 1}`,
    symmetry: branch.symmetry ?? 'identity',
    weight: typeof branch.weight === 'number' ? branch.weight : 1,
    priority: typeof branch.priority === 'number' ? branch.priority : index,
    source: branch.source ?? 'source',
    transformStack:
      branch.transformStack && branch.transformStack.length > 0
        ? branch.transformStack.map((step) => {
            if (step.kind === 'rotate') {
              return { kind: 'rotate', degrees: clampDegrees(step.degrees) };
            }
            if (step.kind === 'mirror') {
              return { kind: 'mirror', axis: step.axis === 'y' ? 'y' : 'x' };
            }
            if (step.kind === 'scale') {
              return { kind: 'scale', factor: clampFactor(step.factor) };
            }
            return step;
          })
        : [
            {
              kind: 'rotate',
              degrees: 0,
            },
          ],
  }));

export const adjustBeamSplitterBranches = (
  scene: SceneGraphState,
  targetCount: number,
): SceneGraphState => {
  if (!Number.isFinite(targetCount) || targetCount <= 0) {
    return scene;
  }
  let changed = false;
  const nodes = scene.nodes.map((node) => {
    if (node.type !== 'BeamSplitter') {
      return node;
    }
    const metadata = isRecord(node.metadata) ? { ...node.metadata } : {};
    const branches = parseBranches(metadata);
    const nextBranches = branches.slice(0, targetCount);
    while (nextBranches.length < targetCount) {
      nextBranches.push(createBranch(nextBranches.length));
    }
    if (
      branches.length !== nextBranches.length ||
      branches.some(
        (branch, index) => JSON.stringify(branch) !== JSON.stringify(nextBranches[index]),
      )
    ) {
      changed = true;
    }
    const nextParameters = node.parameters.map((parameter) =>
      parameter.id === 'branchCount' ? { ...parameter, value: targetCount } : parameter,
    );
    if (
      !changed &&
      node.parameters.some(
        (parameter) => parameter.id === 'branchCount' && parameter.value !== targetCount,
      )
    ) {
      changed = true;
    }
    if (!changed) {
      return node;
    }
    return {
      ...node,
      parameters: nextParameters,
      metadata: {
        ...metadata,
        branches: serialiseBranches(nextBranches),
      },
    } as SceneNode;
  });
  if (!changed) {
    return scene;
  }
  return {
    ...scene,
    nodes,
  };
};
