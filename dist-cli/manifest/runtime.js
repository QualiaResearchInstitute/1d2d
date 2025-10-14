const deepClone = (value) => {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
};
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
const cloneParameterControl = (control) => {
  if (!control) {
    return undefined;
  }
  return deepClone(control);
};
const mapParameters = (parameters) =>
  parameters.map((parameter) => ({
    id: parameter.id,
    label: parameter.label,
    value: cloneParameterValue(parameter.value),
    description: parameter.description,
    control: cloneParameterControl(parameter.control),
    panel: parameter.panel,
  }));
const mapNodes = (manifest) =>
  manifest.nodes.map((node) => ({
    id: node.id,
    type: node.type,
    label: node.label,
    position: node.position ? { ...node.position } : undefined,
    parameters: mapParameters(node.parameters ?? []),
    metadata: node.metadata ? deepClone(node.metadata) : undefined,
  }));
const mapLinks = (manifest) =>
  manifest.links.map((link) => ({
    id: link.id,
    from: { ...link.from },
    to: { ...link.to },
  }));
const mapKeyframe = (frame) => ({
  time: frame.time,
  value: frame.value,
  easing: frame.easing,
});
const mapClip = (clip) => ({
  id: clip.id,
  nodeId: clip.nodeId,
  parameterId: clip.parameterId,
  keyframes: clip.keyframes.map(mapKeyframe),
});
const DEFAULT_TIMELINE_DURATION = 12;
const DEFAULT_TIMELINE_FPS = 60;
const createTimeline = (manifest) => {
  if (!manifest.timeline) {
    return {
      duration: DEFAULT_TIMELINE_DURATION,
      fps: DEFAULT_TIMELINE_FPS,
      clips: [],
      currentTime: 0,
      playing: false,
    };
  }
  return {
    duration: manifest.timeline.duration,
    fps: manifest.timeline.fps,
    clips: manifest.timeline.clips.map(mapClip),
    currentTime: 0,
    playing: false,
  };
};
export function createRuntimeBundle(manifest) {
  const nodes = mapNodes(manifest);
  const links = mapLinks(manifest);
  const scene = {
    nodes,
    links,
    selectedNodeId: nodes[0]?.id,
  };
  const timeline = createTimeline(manifest);
  return {
    scene,
    timeline,
    controls: manifest.controls ? deepClone(manifest.controls) : undefined,
  };
}
