import {
  type ManifestLink,
  type ManifestMetadata,
  type ManifestNode,
  type ManifestNodeParameter,
  type ManifestTimeline,
  type ManifestTimelineClip,
  type ManifestTimelineKeyframe,
  type ManifestParameterValue,
  type ManifestParameterControl,
  type ManifestValidationIssue,
  type ManifestValidationResult,
  type ManifestPrimitive,
  type ManifestParameterVector,
  type ManifestControls,
  type ManifestControlPanel,
  type ManifestControlBinding,
  type ManifestPreset,
  type ManifestControlTransform,
  type SceneManifest,
} from './types.js';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown): string | null => (typeof value === 'string' ? value : null);
const asNumber = (value: unknown): number | null => (typeof value === 'number' ? value : null);
const asBoolean = (value: unknown): boolean | null => (typeof value === 'boolean' ? value : null);

const toPath = (...parts: (string | number)[]): readonly (string | number)[] => parts;

const pushIssue = (
  issues: ManifestValidationIssue[],
  code: string,
  message: string,
  path: readonly (string | number)[],
  severity: ManifestValidationIssue['severity'] = 'error',
) => {
  issues.push({ code, message, path, severity });
};

const normalisePrimitive = (value: unknown): ManifestPrimitive | null => {
  const numberValue = asNumber(value);
  if (numberValue !== null) {
    return numberValue;
  }
  const booleanValue = asBoolean(value);
  if (booleanValue !== null) {
    return booleanValue;
  }
  const stringValue = asString(value);
  if (stringValue !== null) {
    return stringValue;
  }
  return null;
};

const normaliseVector = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestParameterVector | null => {
  if (!isRecord(value)) {
    pushIssue(issues, 'parameter/vector/type', 'Vector parameter must be an object', path);
    return null;
  }
  const kind = asString(value.kind);
  if (kind !== 'vector') {
    pushIssue(issues, 'parameter/vector/kind', 'Vector parameter must set kind to "vector"', path);
    return null;
  }

  const values = Array.isArray(value.values)
    ? value.values.filter((entry) => typeof entry === 'number')
    : null;
  if (!values) {
    pushIssue(issues, 'parameter/vector/values', 'Vector parameter must specify numeric values[]', [
      ...path,
      'values',
    ]);
    return null;
  }

  const size = value.size;
  const declaredSize =
    typeof size === 'number' && Number.isInteger(size) && size > 0 ? size : values.length;
  if (typeof size !== 'number' || size !== values.length) {
    pushIssue(
      issues,
      'parameter/vector/size-mismatch',
      'Vector parameter size does not match the number of components; using actual length.',
      [...path, 'size'],
      'warning',
    );
  }

  return {
    kind: 'vector',
    size: declaredSize,
    values: values.slice(0, declaredSize),
  };
};

const normaliseParameterControl = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestParameterControl | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    pushIssue(issues, 'parameter/control/type', 'Parameter control must be an object', path);
    return undefined;
  }

  const kind = asString(value.kind);
  if (!kind) {
    pushIssue(issues, 'parameter/control/kind', 'Parameter control must define a kind', [
      ...path,
      'kind',
    ]);
    return undefined;
  }

  switch (kind) {
    case 'slider':
    case 'range': {
      const min = asNumber(value.min) ?? undefined;
      const max = asNumber(value.max) ?? undefined;
      const step = asNumber(value.step) ?? undefined;
      const unit = asString(value.unit) ?? undefined;
      return {
        kind,
        min,
        max,
        step,
        unit,
      };
    }
    case 'toggle': {
      const trueLabel = asString(value.trueLabel) ?? undefined;
      const falseLabel = asString(value.falseLabel) ?? undefined;
      return {
        kind,
        trueLabel,
        falseLabel,
      };
    }
    case 'select': {
      if (!Array.isArray(value.options)) {
        pushIssue(
          issues,
          'parameter/control/options',
          'Select controls must provide an options array',
          [...path, 'options'],
        );
        return undefined;
      }
      const options: Array<{ value: string | number | boolean; label?: string }> = [];
      value.options.forEach((option, index) => {
        if (!isRecord(option)) {
          pushIssue(
            issues,
            'parameter/control/options/type',
            'Select control options must be objects',
            [...path, 'options', index],
          );
          return;
        }
        const optionValue =
          asString(option.value) ??
          asNumber(option.value) ??
          (typeof option.value === 'boolean' ? option.value : null);
        if (optionValue === null) {
          pushIssue(
            issues,
            'parameter/control/options/value',
            'Select control option must provide a string, number, or boolean value',
            [...path, 'options', index, 'value'],
          );
          return;
        }
        const optionLabel = asString(option.label) ?? undefined;
        options.push({ value: optionValue, label: optionLabel });
      });
      return {
        kind,
        options,
      };
    }
    case 'color': {
      const formatValue = asString(value.format);
      const format = formatValue === 'hex' || formatValue === 'rgb' ? formatValue : undefined;
      return {
        kind,
        format,
      };
    }
    case 'vector': {
      const componentLabels = Array.isArray(value.componentLabels)
        ? value.componentLabels.filter((entry): entry is string => typeof entry === 'string')
        : undefined;
      const min = asNumber(value.min) ?? undefined;
      const max = asNumber(value.max) ?? undefined;
      const step = asNumber(value.step) ?? undefined;
      return {
        kind,
        componentLabels,
        min,
        max,
        step,
      };
    }
    case 'curve': {
      const description = asString(value.description) ?? undefined;
      return {
        kind,
        description,
      };
    }
    case 'custom': {
      const widget = asString(value.widget) ?? undefined;
      const options = isRecord(value.options) ? value.options : undefined;
      return {
        kind,
        widget,
        options,
      };
    }
    default: {
      pushIssue(
        issues,
        'parameter/control/kind',
        `Unsupported parameter control kind "${kind}"`,
        [...path, 'kind'],
        'warning',
      );
      return undefined;
    }
  }
};

const normaliseParameterValue = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestParameterValue | null => {
  const primitive = normalisePrimitive(value);
  if (primitive !== null) {
    return primitive;
  }
  const vector = normaliseVector(value, issues, path);
  if (vector) {
    return vector;
  }
  pushIssue(issues, 'parameter/value/unsupported', 'Unsupported parameter value', path);
  return null;
};

const normaliseNodeParameters = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestNodeParameter[] | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    pushIssue(issues, 'node/parameters/type', 'Node parameters must be an array', path);
    return undefined;
  }

  const result: ManifestNodeParameter[] = [];
  const seen = new Set<string>();

  value.forEach((parameterValue, index) => {
    if (!isRecord(parameterValue)) {
      pushIssue(issues, 'node/parameters/entry/type', 'Parameter entries must be objects', [
        ...path,
        index,
      ]);
      return;
    }

    const id = asString(parameterValue.id) ?? '';
    if (!id) {
      pushIssue(issues, 'node/parameters/entry/id', 'Parameter must define an id', [
        ...path,
        index,
        'id',
      ]);
      return;
    }
    if (seen.has(id)) {
      pushIssue(issues, 'node/parameters/entry/duplicate', `Duplicate parameter id "${id}"`, [
        ...path,
        index,
        'id',
      ]);
      return;
    }
    seen.add(id);

    const label = asString(parameterValue.label) ?? id;
    const valuePath = [...path, index, 'value'] as const;
    const valuePayload = normaliseParameterValue(parameterValue.value, issues, valuePath);
    if (!valuePayload) {
      return;
    }
    result.push({
      id,
      label,
      value: valuePayload,
      description: asString(parameterValue.description) ?? undefined,
      control: normaliseParameterControl(parameterValue.control, issues, [
        ...path,
        index,
        'control',
      ]),
      panel: asString(parameterValue.panel) ?? undefined,
    });
  });

  return result;
};

const normaliseNodes = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestNode[] => {
  if (!Array.isArray(value)) {
    pushIssue(issues, 'nodes/type', 'Manifest nodes must be an array', path);
    return [];
  }

  const seen = new Set<string>();
  const nodes: ManifestNode[] = [];

  value.forEach((nodeValue, index) => {
    if (!isRecord(nodeValue)) {
      pushIssue(issues, 'node/type', 'Node entries must be objects', [...path, index]);
      return;
    }

    const id = asString(nodeValue.id);
    if (!id) {
      pushIssue(issues, 'node/id', 'Node must have an id', [...path, index, 'id']);
      return;
    }
    if (seen.has(id)) {
      pushIssue(issues, 'node/duplicate-id', `Duplicate node id "${id}"`, [...path, index, 'id']);
      return;
    }
    seen.add(id);

    const type = asString(nodeValue.type) ?? 'UnknownNode';
    const label = asString(nodeValue.label) ?? id;

    let position: ManifestNode['position'];
    if (isRecord(nodeValue.position)) {
      const x = asNumber(nodeValue.position.x);
      const y = asNumber(nodeValue.position.y);
      if (x !== null && y !== null) {
        position = { x, y };
      }
    }

    const parameters = normaliseNodeParameters(nodeValue.parameters, issues, [
      ...path,
      index,
      'parameters',
    ]);

    nodes.push({
      id,
      type,
      label,
      parameters,
      position,
      metadata: isRecord(nodeValue.metadata) ? nodeValue.metadata : undefined,
    });
  });

  return nodes;
};

const ALLOWED_TRANSFORMS = new Set([
  'degreesToRadians',
  'radiansToDegrees',
  'percentToUnit',
  'unitToPercent',
]);

const normaliseControlTransform = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestControlTransform | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    pushIssue(issues, 'controls/transform/type', 'Control transform must be an object', path);
    return undefined;
  }

  const toParameterRaw = asString(value.toParameter) ?? undefined;
  const fromParameterRaw = asString(value.fromParameter) ?? undefined;
  if (toParameterRaw && !ALLOWED_TRANSFORMS.has(toParameterRaw)) {
    pushIssue(
      issues,
      'controls/transform/unsupported',
      `Unsupported transform "${toParameterRaw}"`,
      [...path, 'toParameter'],
      'warning',
    );
  }
  if (fromParameterRaw && !ALLOWED_TRANSFORMS.has(fromParameterRaw)) {
    pushIssue(
      issues,
      'controls/transform/unsupported',
      `Unsupported transform "${fromParameterRaw}"`,
      [...path, 'fromParameter'],
      'warning',
    );
  }

  const scale = asNumber(value.scale) ?? undefined;
  const offset = asNumber(value.offset) ?? undefined;

  if (!toParameterRaw && !fromParameterRaw && scale === undefined && offset === undefined) {
    return undefined;
  }

  return {
    toParameter:
      toParameterRaw && ALLOWED_TRANSFORMS.has(toParameterRaw)
        ? (toParameterRaw as ManifestControlTransform['toParameter'])
        : undefined,
    fromParameter:
      fromParameterRaw && ALLOWED_TRANSFORMS.has(fromParameterRaw)
        ? (fromParameterRaw as ManifestControlTransform['fromParameter'])
        : undefined,
    scale,
    offset,
  };
};

const normaliseControlBinding = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
  nodesById: Map<string, ManifestNode>,
): ManifestControlBinding | null => {
  if (!isRecord(value)) {
    pushIssue(issues, 'controls/binding/type', 'Control bindings must be objects', path);
    return null;
  }

  const kind = asString(value.kind) ?? 'nodeParameter';

  if (kind === 'nodeParameter') {
    const nodeId = asString(value.nodeId);
    const parameterId = asString(value.parameterId);
    if (!nodeId || !parameterId) {
      pushIssue(
        issues,
        'controls/binding/nodeParameter/fields',
        'Node parameter bindings must specify nodeId and parameterId',
        path,
      );
      return null;
    }
    const node = nodesById.get(nodeId);
    if (!node) {
      pushIssue(
        issues,
        'controls/binding/nodeParameter/missing-node',
        `Control binding references missing node "${nodeId}"`,
        [...path, 'nodeId'],
      );
      return null;
    }
    const hasParameter =
      node.parameters?.some((parameter) => parameter.id === parameterId) ?? false;
    if (!hasParameter) {
      pushIssue(
        issues,
        'controls/binding/nodeParameter/missing-parameter',
        `Control binding references missing parameter "${parameterId}" on node "${nodeId}"`,
        [...path, 'parameterId'],
        'warning',
      );
    }
    const valuePath = asString(value.valuePath) ?? undefined;
    const transform = normaliseControlTransform(value.transform, issues, [...path, 'transform']);
    return {
      kind: 'nodeParameter',
      nodeId,
      parameterId,
      valuePath,
      transform,
    };
  }

  if (kind === 'environment') {
    const envPath = asString((value as Record<string, unknown>).path);
    if (!envPath) {
      pushIssue(
        issues,
        'controls/binding/environment/path',
        'Environment bindings must supply a path string',
        [...path, 'path'],
      );
      return null;
    }
    return {
      kind: 'environment',
      path: envPath,
    };
  }

  if (kind === 'timeline') {
    const clipId = asString(value.clipId);
    const parameterId = asString(value.parameterId);
    if (!clipId || !parameterId) {
      pushIssue(
        issues,
        'controls/binding/timeline/fields',
        'Timeline bindings must include clipId and parameterId',
        path,
      );
      return null;
    }
    const valuePath = asString(value.valuePath) ?? undefined;
    return {
      kind: 'timeline',
      clipId,
      parameterId,
      valuePath,
    };
  }

  pushIssue(
    issues,
    'controls/binding/kind',
    `Unsupported binding kind "${kind}"`,
    [...path, 'kind'],
    'warning',
  );
  return null;
};

const normaliseControlPanel = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
  nodesById: Map<string, ManifestNode>,
): ManifestControlPanel | null => {
  if (!isRecord(value)) {
    pushIssue(issues, 'controls/panel/type', 'Control panels must be objects', path);
    return null;
  }

  const id = asString(value.id);
  if (!id) {
    pushIssue(issues, 'controls/panel/id', 'Control panel must define an id', [...path, 'id']);
    return null;
  }

  const label = asString(value.label) ?? id;
  const description = asString(value.description) ?? undefined;
  const category = asString(value.category) ?? undefined;
  const icon = asString(value.icon) ?? undefined;
  const collapsed = typeof value.collapsed === 'boolean' ? value.collapsed : undefined;

  if (!isRecord(value.schema)) {
    pushIssue(issues, 'controls/panel/schema', 'Control panels must provide a JSON schema object', [
      ...path,
      'schema',
    ]);
    return null;
  }
  const schema = value.schema;

  const uiSchema = isRecord(value.uiSchema)
    ? (value.uiSchema as Record<string, unknown>)
    : undefined;

  if (!isRecord(value.bindings)) {
    pushIssue(issues, 'controls/panel/bindings', 'Control panels must provide a bindings object', [
      ...path,
      'bindings',
    ]);
    return null;
  }

  const bindings: Record<string, ManifestControlBinding> = {};
  Object.entries(value.bindings).forEach(([pointer, bindingValue]) => {
    const binding = normaliseControlBinding(
      bindingValue,
      issues,
      [...path, 'bindings', pointer],
      nodesById,
    );
    if (binding) {
      bindings[pointer] = binding;
    }
  });

  return {
    id,
    label,
    description,
    category,
    icon,
    collapsed,
    schema,
    uiSchema,
    bindings,
  };
};

const normalisePresets = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestPreset[] | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    pushIssue(issues, 'controls/presets/type', 'Presets must be an array', path);
    return undefined;
  }

  const presets: ManifestPreset[] = [];
  const seen = new Set<string>();

  value.forEach((entry, index) => {
    if (!isRecord(entry)) {
      pushIssue(issues, 'controls/presets/entry/type', 'Preset entries must be objects', [
        ...path,
        index,
      ]);
      return;
    }

    const id = asString(entry.id);
    if (!id) {
      pushIssue(issues, 'controls/presets/entry/id', 'Preset must define an id', [
        ...path,
        index,
        'id',
      ]);
      return;
    }
    if (seen.has(id)) {
      pushIssue(issues, 'controls/presets/entry/duplicate', `Duplicate preset id "${id}"`, [
        ...path,
        index,
        'id',
      ]);
      return;
    }
    seen.add(id);

    const label = asString(entry.label) ?? id;
    const description = asString(entry.description) ?? undefined;
    const thumbnail = asString(entry.thumbnail) ?? undefined;

    const panelsRaw = entry.panels;
    let panels: Record<string, Record<string, unknown>> | undefined;
    if (panelsRaw !== undefined) {
      if (!isRecord(panelsRaw)) {
        pushIssue(
          issues,
          'controls/presets/entry/panels',
          'Preset panels must be an object mapping panel ids to form data',
          [...path, index, 'panels'],
        );
      } else {
        const mapped: Record<string, Record<string, unknown>> = {};
        Object.entries(panelsRaw).forEach(([panelId, formData]) => {
          if (!isRecord(formData)) {
            pushIssue(
              issues,
              'controls/presets/entry/panels/type',
              'Preset panel data must be objects',
              [...path, index, 'panels', panelId],
            );
            return;
          }
          mapped[panelId] = formData;
        });
        panels = mapped;
      }
    }

    presets.push({
      id,
      label,
      description,
      thumbnail,
      panels,
    });
  });

  return presets;
};

const normaliseControls = (
  value: unknown,
  issues: ManifestValidationIssue[],
  nodes: ManifestNode[],
  path: readonly (string | number)[],
): ManifestControls | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    pushIssue(issues, 'controls/type', 'Controls definition must be an object', path);
    return undefined;
  }

  const nodesById = new Map<string, ManifestNode>();
  nodes.forEach((node) => nodesById.set(node.id, node));

  const panelsRaw = value.panels;
  const panels: ManifestControlPanel[] = [];
  if (panelsRaw !== undefined) {
    if (!Array.isArray(panelsRaw)) {
      pushIssue(issues, 'controls/panels/type', 'Control panels must be an array', [
        ...path,
        'panels',
      ]);
    } else {
      const seen = new Set<string>();
      panelsRaw.forEach((panelValue, index) => {
        const panel = normaliseControlPanel(
          panelValue,
          issues,
          [...path, 'panels', index],
          nodesById,
        );
        if (!panel) {
          return;
        }
        if (seen.has(panel.id)) {
          pushIssue(
            issues,
            'controls/panels/duplicate',
            `Duplicate control panel id "${panel.id}"`,
            [...path, 'panels', index, 'id'],
          );
          return;
        }
        seen.add(panel.id);
        panels.push(panel);
      });
    }
  }

  const presets = normalisePresets(value.presets, issues, [...path, 'presets']);

  if (panels.length === 0 && (!presets || presets.length === 0)) {
    return undefined;
  }

  return {
    panels,
    presets,
  };
};

const normaliseLinkEndpoint = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestLink['from'] | null => {
  if (!isRecord(value)) {
    pushIssue(issues, 'link/endpoint/type', 'Link endpoints must be objects', path);
    return null;
  }
  const nodeId = asString(value.nodeId);
  const port = asString(value.port);
  if (!nodeId || !port) {
    pushIssue(issues, 'link/endpoint/fields', 'Link endpoint must include nodeId and port', path);
    return null;
  }
  return { nodeId, port };
};

const normaliseLinks = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestLink[] => {
  if (!Array.isArray(value)) {
    pushIssue(issues, 'links/type', 'Manifest links must be an array', path);
    return [];
  }

  const seen = new Set<string>();
  const links: ManifestLink[] = [];

  value.forEach((linkValue, index) => {
    if (!isRecord(linkValue)) {
      pushIssue(issues, 'link/type', 'Link entries must be objects', [...path, index]);
      return;
    }

    const id = asString(linkValue.id);
    if (!id) {
      pushIssue(issues, 'link/id', 'Link must include an id', [...path, index, 'id']);
      return;
    }
    if (seen.has(id)) {
      pushIssue(issues, 'link/duplicate-id', `Duplicate link id "${id}"`, [...path, index, 'id']);
      return;
    }
    seen.add(id);

    const from = normaliseLinkEndpoint(linkValue.from, issues, [...path, index, 'from']);
    const to = normaliseLinkEndpoint(linkValue.to, issues, [...path, index, 'to']);
    if (!from || !to) {
      return;
    }

    links.push({
      id,
      from,
      to,
      metadata: isRecord(linkValue.metadata) ? linkValue.metadata : undefined,
    });
  });

  return links;
};

const normaliseKeyframes = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestTimelineKeyframe[] => {
  if (!Array.isArray(value)) {
    pushIssue(issues, 'timeline/keyframes/type', 'Keyframes must be an array', path);
    return [];
  }
  const frames: ManifestTimelineKeyframe[] = [];
  value.forEach((frameValue, index) => {
    if (!isRecord(frameValue)) {
      pushIssue(issues, 'timeline/keyframes/entry/type', 'Keyframes must be objects', [
        ...path,
        index,
      ]);
      return;
    }
    const time = asNumber(frameValue.time);
    if (time === null || time < 0) {
      pushIssue(
        issues,
        'timeline/keyframes/entry/time',
        'Keyframe time must be a non-negative number',
        [...path, index, 'time'],
      );
      return;
    }
    const valuePayload = normalisePrimitive(frameValue.value);
    if (valuePayload === null) {
      pushIssue(issues, 'timeline/keyframes/entry/value', 'Keyframe value must be a primitive', [
        ...path,
        index,
        'value',
      ]);
      return;
    }
    frames.push({
      time,
      value: valuePayload,
      easing: asString(frameValue.easing) ?? undefined,
    });
  });
  return frames;
};

const normaliseClips = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestTimelineClip[] => {
  if (!Array.isArray(value)) {
    pushIssue(issues, 'timeline/clips/type', 'Timeline clips must be an array', path);
    return [];
  }
  const clips: ManifestTimelineClip[] = [];
  const seen = new Set<string>();
  value.forEach((clipValue, index) => {
    if (!isRecord(clipValue)) {
      pushIssue(issues, 'timeline/clip/type', 'Timeline clip must be an object', [...path, index]);
      return;
    }
    const id = asString(clipValue.id);
    if (!id) {
      pushIssue(issues, 'timeline/clip/id', 'Timeline clip must include an id', [
        ...path,
        index,
        'id',
      ]);
      return;
    }
    if (seen.has(id)) {
      pushIssue(issues, 'timeline/clip/duplicate', `Duplicate clip id "${id}"`, [
        ...path,
        index,
        'id',
      ]);
      return;
    }
    seen.add(id);

    const nodeId = asString(clipValue.nodeId);
    const parameterId = asString(clipValue.parameterId);
    if (!nodeId || !parameterId) {
      pushIssue(
        issues,
        'timeline/clip/target',
        'Timeline clip must target a nodeId and parameterId',
        [...path, index],
      );
      return;
    }

    const keyframes = normaliseKeyframes(clipValue.keyframes, issues, [
      ...path,
      index,
      'keyframes',
    ]);
    clips.push({
      id,
      nodeId,
      parameterId,
      keyframes,
      interpolation:
        clipValue.interpolation === 'step' || clipValue.interpolation === 'spline'
          ? clipValue.interpolation
          : 'linear',
    });
  });
  return clips;
};

const normaliseTimeline = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestTimeline | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    pushIssue(issues, 'timeline/type', 'Timeline must be an object', path);
    return undefined;
  }
  const duration = asNumber(value.duration);
  if (duration === null || duration <= 0) {
    pushIssue(issues, 'timeline/duration', 'Timeline duration must be a positive number', [
      ...path,
      'duration',
    ]);
    return undefined;
  }
  const fps = asNumber(value.fps);
  if (fps === null || fps <= 0) {
    pushIssue(issues, 'timeline/fps', 'Timeline fps must be a positive number', [...path, 'fps']);
    return undefined;
  }
  const clips = normaliseClips(value.clips, issues, [...path, 'clips']);
  return {
    duration,
    fps,
    clips,
  };
};

const normaliseMetadata = (
  value: unknown,
  issues: ManifestValidationIssue[],
  path: readonly (string | number)[],
): ManifestMetadata => {
  if (!isRecord(value)) {
    pushIssue(issues, 'metadata/type', 'Manifest metadata must be an object', path);
    return {
      name: 'Untitled Scene',
    };
  }
  const name = asString(value.name);
  if (!name) {
    pushIssue(issues, 'metadata/name', 'Manifest metadata must include a name', [...path, 'name']);
  }
  return {
    name: name ?? 'Untitled Scene',
    description: asString(value.description) ?? undefined,
    author: asString(value.author) ?? undefined,
    version: asString(value.version) ?? undefined,
    createdAt: asString(value.createdAt) ?? undefined,
    updatedAt: asString(value.updatedAt) ?? undefined,
    tags: Array.isArray(value.tags)
      ? value.tags.filter((entry): entry is string => typeof entry === 'string')
      : undefined,
    defaultPreset: asString(value.defaultPreset) ?? undefined,
  };
};

const verifyReferences = (manifest: SceneManifest, issues: ManifestValidationIssue[]) => {
  const nodeIds = new Set(manifest.nodes.map((node: ManifestNode) => node.id));
  manifest.links.forEach((link: ManifestLink, index: number) => {
    if (!nodeIds.has(link.from.nodeId)) {
      pushIssue(
        issues,
        'link/from/missing-node',
        `Link references missing node "${link.from.nodeId}"`,
        toPath('links', index, 'from', 'nodeId'),
      );
    }
    if (!nodeIds.has(link.to.nodeId)) {
      pushIssue(
        issues,
        'link/to/missing-node',
        `Link references missing node "${link.to.nodeId}"`,
        toPath('links', index, 'to', 'nodeId'),
      );
    }
  });

  if (manifest.timeline) {
    manifest.timeline.clips.forEach((clip: ManifestTimelineClip, index: number) => {
      if (!nodeIds.has(clip.nodeId)) {
        pushIssue(
          issues,
          'timeline/clip/missing-node',
          `Timeline clip targets missing node "${clip.nodeId}"`,
          toPath('timeline', 'clips', index, 'nodeId'),
        );
      }
      const node = manifest.nodes.find((candidate) => candidate.id === clip.nodeId);
      const parameterExists =
        node?.parameters?.some((parameter) => parameter.id === clip.parameterId) ?? false;
      if (!parameterExists) {
        pushIssue(
          issues,
          'timeline/clip/missing-parameter',
          `Timeline clip refers to parameter "${clip.parameterId}" not found on node "${clip.nodeId}"`,
          toPath('timeline', 'clips', index, 'parameterId'),
        );
      }
    });
  }
};

export class ManifestValidationError extends Error {
  constructor(
    message: string,
    readonly issues: ManifestValidationIssue[],
  ) {
    super(message);
    this.name = 'ManifestValidationError';
  }
}

export function validateManifest(payload: unknown): ManifestValidationResult {
  const issues: ManifestValidationIssue[] = [];

  if (!isRecord(payload)) {
    pushIssue(issues, 'manifest/type', 'Manifest root must be an object', toPath());
    throw new ManifestValidationError('Manifest root must be an object', issues);
  }

  const schemaVersionValue = asString(payload.schemaVersion);
  if (!schemaVersionValue) {
    pushIssue(
      issues,
      'manifest/schemaVersion',
      'Manifest must supply a schemaVersion string',
      toPath('schemaVersion'),
    );
  }

  const metadata = normaliseMetadata(payload.metadata, issues, toPath('metadata'));
  const nodes = normaliseNodes(payload.nodes, issues, toPath('nodes'));
  const links = normaliseLinks(payload.links, issues, toPath('links'));
  const timeline = normaliseTimeline(payload.timeline, issues, toPath('timeline'));
  const controls = normaliseControls(payload.controls, issues, nodes, toPath('controls'));

  const manifest: SceneManifest = {
    schemaVersion: schemaVersionValue ?? '1.0.0',
    metadata,
    nodes,
    links,
    timeline,
    environment: isRecord(payload.environment) ? payload.environment : undefined,
    controls,
  };

  verifyReferences(manifest, issues);

  const hasFatalIssues = issues.some((issue) => issue.severity === 'error');
  if (hasFatalIssues) {
    throw new ManifestValidationError('Manifest validation failed', issues);
  }

  return {
    manifest,
    issues,
  };
}
