import type {
  ManifestControlBinding,
  ManifestControlPanel,
  ManifestControls,
  ManifestControlTransform,
  ManifestParameterValue,
} from '../manifest/types.js';
import type {
  ControlBindingState,
  ControlPanelState,
  ControlsState,
  SceneGraphState,
  SceneNode,
  SceneNodeParameter,
} from './types.js';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const deepClone = <T>(value: T): T => {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
};

const decodePointerSegment = (segment: string): string =>
  segment.replace(/~1/g, '/').replace(/~0/g, '~');

const getPointerSegments = (pointer: string): string[] => {
  if (!pointer || pointer === '/') {
    return [];
  }
  if (!pointer.startsWith('/')) {
    return [decodePointerSegment(pointer)];
  }
  return pointer.split('/').slice(1).map(decodePointerSegment);
};

const readPointerValue = (source: unknown, pointer: string): unknown => {
  const segments = getPointerSegments(pointer);
  if (segments.length === 0) {
    return source;
  }
  let current: unknown = source;
  for (const segment of segments) {
    if (Array.isArray(current)) {
      const index = Number.parseInt(segment, 10);
      if (!Number.isFinite(index) || index < 0 || index >= current.length) {
        return undefined;
      }
      current = current[index];
      continue;
    }
    if (!isRecord(current)) {
      return undefined;
    }
    current = current[segment];
    if (current === undefined) {
      return undefined;
    }
  }
  return current;
};

const assignPointerValue = (source: unknown, pointer: string, value: unknown): unknown => {
  const segments = getPointerSegments(pointer);
  if (segments.length === 0) {
    return value;
  }
  const clone = Array.isArray(source) ? [...source] : isRecord(source) ? { ...source } : {};
  let current: any = clone;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const segment = segments[i]!;
    const isArrayIndex = Array.isArray(current) || /^\d+$/.test(segment);
    const nextExisting = Array.isArray(current)
      ? current[Number.parseInt(segment, 10)]
      : current[segment];
    let nextValue: any;
    if (nextExisting === undefined) {
      nextValue = isArrayIndex ? [] : {};
    } else {
      nextValue = Array.isArray(nextExisting)
        ? [...nextExisting]
        : isRecord(nextExisting)
          ? { ...nextExisting }
          : nextExisting;
    }
    if (Array.isArray(current)) {
      const idx = Number.parseInt(segment, 10);
      current[idx] = nextValue;
      current = nextValue;
    } else {
      current[segment] = nextValue;
      current = nextValue;
    }
  }
  const lastSegment = segments[segments.length - 1]!;
  if (Array.isArray(current)) {
    const idx = Number.parseInt(lastSegment, 10);
    current[idx] = value;
  } else {
    current[lastSegment] = value;
  }
  return clone;
};

const areValuesEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) {
    return true;
  }
  if (typeof a !== typeof b) {
    return false;
  }
  if (isRecord(a) && isRecord(b)) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const key of keys) {
      if (!areValuesEqual(a[key], b[key])) {
        return false;
      }
    }
    return true;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i += 1) {
      if (!areValuesEqual(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }
  return false;
};

const applyTransform = (
  value: unknown,
  transform: ManifestControlTransform | undefined,
  direction: 'toParameter' | 'fromParameter',
): unknown => {
  if (!transform) {
    return value;
  }
  const transformKey =
    direction === 'toParameter' ? transform.toParameter : transform.fromParameter;
  let numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return value;
  }
  switch (transformKey) {
    case 'degreesToRadians':
      numeric = (numeric * Math.PI) / 180;
      break;
    case 'radiansToDegrees':
      numeric = (numeric * 180) / Math.PI;
      break;
    case 'percentToUnit':
      numeric = numeric / 100;
      break;
    case 'unitToPercent':
      numeric = numeric * 100;
      break;
    default:
      break;
  }
  if (typeof transform.scale === 'number') {
    numeric *= transform.scale;
  }
  if (typeof transform.offset === 'number') {
    numeric += transform.offset;
  }
  return numeric;
};

const findSceneNode = (scene: SceneGraphState, nodeId: string): SceneNode | undefined =>
  scene.nodes.find((node) => node.id === nodeId);

const findSceneParameter = (
  node: SceneNode | undefined,
  parameterId: string,
): SceneNodeParameter | undefined =>
  node?.parameters.find((parameter) => parameter.id === parameterId);

const toPlainValue = (value: ManifestParameterValue): unknown => {
  if (isRecord(value) && value.kind === 'vector' && Array.isArray(value.values)) {
    return {
      kind: 'vector',
      size: value.size,
      values: [...value.values],
    };
  }
  return value;
};

const fromPlainValue = (
  original: ManifestParameterValue,
  nextValue: unknown,
): ManifestParameterValue => {
  if (isRecord(original) && original.kind === 'vector') {
    const next = isRecord(nextValue) ? nextValue : {};
    const values = Array.isArray(next.values) ? next.values.map(Number) : [];
    const size =
      typeof next.size === 'number' && Number.isFinite(next.size) ? next.size : values.length;
    return {
      kind: 'vector',
      size,
      values: values.slice(0, size),
    };
  }
  if (typeof original === 'number') {
    const numeric = typeof nextValue === 'number' ? nextValue : Number(nextValue);
    return Number.isFinite(numeric) ? numeric : original;
  }
  if (typeof original === 'boolean') {
    if (typeof nextValue === 'boolean') {
      return nextValue;
    }
    if (nextValue === 'true' || nextValue === 'false') {
      return nextValue === 'true';
    }
    return original;
  }
  if (typeof original === 'string') {
    return typeof nextValue === 'string' ? nextValue : String(nextValue ?? original);
  }
  return original;
};

const extractBindingValue = (scene: SceneGraphState, binding: ManifestControlBinding): unknown => {
  if (binding.kind === 'nodeParameter') {
    const node = findSceneNode(scene, binding.nodeId);
    const parameter = findSceneParameter(node, binding.parameterId);
    if (!parameter) {
      return undefined;
    }
    const plain = toPlainValue(parameter.value);
    const value = binding.valuePath ? readPointerValue(plain, binding.valuePath) : plain;
    return applyTransform(value, binding.transform, 'fromParameter');
  }
  return undefined;
};

const updateParameterValue = (
  scene: SceneGraphState,
  binding: ManifestControlBinding,
  incomingValue: unknown,
): SceneGraphState => {
  if (binding.kind !== 'nodeParameter') {
    return scene;
  }
  const nodeIndex = scene.nodes.findIndex((node) => node.id === binding.nodeId);
  if (nodeIndex < 0) {
    return scene;
  }
  const node = scene.nodes[nodeIndex]!;
  const parameterIndex = node.parameters.findIndex(
    (parameter) => parameter.id === binding.parameterId,
  );
  if (parameterIndex < 0) {
    return scene;
  }
  const parameter = node.parameters[parameterIndex]!;
  const targetValue = applyTransform(incomingValue, binding.transform, 'toParameter');
  const currentPlain = toPlainValue(parameter.value);
  const nextPlain = binding.valuePath
    ? assignPointerValue(currentPlain, binding.valuePath, targetValue)
    : targetValue;
  const nextValue = fromPlainValue(parameter.value, nextPlain);
  if (areValuesEqual(parameter.value, nextValue)) {
    return scene;
  }
  const nextParameters = node.parameters.slice();
  nextParameters[parameterIndex] = { ...parameter, value: nextValue };
  const nextNode = { ...node, parameters: nextParameters };
  const nextNodes = scene.nodes.slice();
  nextNodes[nodeIndex] = nextNode;
  return {
    ...scene,
    nodes: nextNodes,
  };
};

const buildBindingStates = (panel: ManifestControlPanel): ControlBindingState[] =>
  Object.entries(panel.bindings).map(([pointer, binding]) => ({
    pointer,
    binding,
  }));

const createPanelState = (
  panel: ManifestControlPanel,
  scene: SceneGraphState,
  collapsedOverride?: boolean,
): ControlPanelState => {
  const bindingStates = buildBindingStates(panel);
  let formData: Record<string, unknown> = {};
  bindingStates.forEach(({ pointer, binding }) => {
    const value = extractBindingValue(scene, binding);
    if (value !== undefined) {
      formData = assignPointerValue(formData, pointer, value) as Record<string, unknown>;
    }
  });
  return {
    id: panel.id,
    label: panel.label,
    description: panel.description,
    category: panel.category,
    icon: panel.icon,
    collapsed: collapsedOverride ?? panel.collapsed,
    schema: deepClone(panel.schema),
    uiSchema: panel.uiSchema ? deepClone(panel.uiSchema) : undefined,
    bindings: bindingStates,
    formData,
  };
};

export const createControlsState = (
  controls: ManifestControls | undefined,
  scene: SceneGraphState,
): ControlsState | undefined => {
  if (!controls) {
    return undefined;
  }
  const panels = controls.panels.map((panel) => createPanelState(panel, scene));
  return {
    panels,
    source: controls,
  };
};

export const syncControlPanels = (
  current: ControlsState | undefined,
  scene: SceneGraphState,
): ControlsState | undefined => {
  if (!current?.source) {
    return current;
  }
  const collapsedMap = new Map<string, boolean | undefined>(
    current.panels.map((panel) => [panel.id, panel.collapsed]),
  );
  const panels = current.source.panels.map((panel) =>
    createPanelState(panel, scene, collapsedMap.get(panel.id)),
  );
  return {
    panels,
    source: current.source,
  };
};

export const applyControlPanelFormData = (
  controls: ControlsState | undefined,
  panelId: string,
  formData: Record<string, unknown>,
  scene: SceneGraphState,
): { scene: SceneGraphState; panels: ControlPanelState[] } => {
  if (!controls?.source) {
    return { scene, panels: controls?.panels ?? [] };
  }
  const sourcePanel = controls.source.panels.find((panel) => panel.id === panelId);
  const currentPanel = controls.panels.find((panel) => panel.id === panelId);
  if (!sourcePanel || !currentPanel) {
    return { scene, panels: controls.panels };
  }

  let nextScene = scene;
  currentPanel.bindings.forEach(({ pointer, binding }) => {
    const newValue = readPointerValue(formData, pointer);
    nextScene = updateParameterValue(nextScene, binding, newValue);
  });

  const updatedPanels = controls.panels.map((panel) =>
    panel.id === panelId ? { ...panel, formData } : panel,
  );
  return { scene: nextScene, panels: updatedPanels };
};

export const updatePanelCollapsedState = (
  panels: ControlPanelState[],
  panelId: string,
  collapsed: boolean,
): ControlPanelState[] =>
  panels.map((panel) => (panel.id === panelId ? { ...panel, collapsed } : panel));
