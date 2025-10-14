const cloneParameterValue = (value) => {
  if (typeof value === 'object' && value !== null && 'kind' in value) {
    return {
      kind: value.kind,
      size: value.size,
      values: [...value.values],
    };
  }
  return value;
};
const mapParameters = (parameters) =>
  parameters.map((parameter) => ({
    id: parameter.id,
    label: parameter.label,
    value: cloneParameterValue(parameter.value),
    description: parameter.description,
    control: parameter.control,
    panel: parameter.panel,
  }));
const mergePresets = (controls, library) => {
  const base = controls?.source?.presets ?? [];
  if (!library || library.length === 0) {
    return base.length > 0 ? base : undefined;
  }
  const merged = new Map();
  base.forEach((preset) => merged.set(preset.id, { ...preset }));
  library.forEach((preset) => {
    merged.set(preset.id, {
      id: preset.id,
      label: preset.label,
      description: preset.description,
      thumbnail: preset.thumbnail,
      panels: preset.panels ? { ...preset.panels } : undefined,
    });
  });
  return Array.from(merged.values());
};
export const serializeSceneManifest = (scene, timeline, metadata, controls, presets) => {
  const manifestControls = controls?.source
    ? {
        panels: controls.source.panels.map((panel) => ({
          ...panel,
          schema: { ...panel.schema },
          uiSchema: panel.uiSchema ? { ...panel.uiSchema } : undefined,
          bindings: { ...panel.bindings },
        })),
        presets: mergePresets(controls, presets),
      }
    : undefined;
  return {
    schemaVersion: '1.0.0',
    metadata: { ...metadata },
    nodes: scene.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      label: node.label,
      position: node.position ? { ...node.position } : undefined,
      parameters: mapParameters(node.parameters),
      metadata: node.metadata ? { ...node.metadata } : undefined,
    })),
    links: scene.links.map((link) => ({
      id: link.id,
      from: { ...link.from },
      to: { ...link.to },
    })),
    timeline: {
      duration: timeline.duration,
      fps: timeline.fps,
      clips: timeline.clips.map((clip) => ({
        id: clip.id,
        nodeId: clip.nodeId,
        parameterId: clip.parameterId,
        keyframes: clip.keyframes.map((keyframe) => ({
          time: keyframe.time,
          value: keyframe.value,
          easing: keyframe.easing,
        })),
      })),
    },
    controls: manifestControls,
  };
};
