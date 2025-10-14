import type {
  ManifestTimelineClip,
  ManifestTimelineKeyframe,
  SceneManifest,
  ManifestParameterValue,
  ManifestNodeParameter,
  ManifestNode,
  ManifestLink as ManifestLinkRecord,
  ManifestParameterControl,
  ManifestControls,
} from './types.js';
import type {
  SceneGraphState,
  SceneLink,
  SceneNode,
  SceneNodeParameter,
  TimelineClip,
  TimelineKeyframe,
  TimelineState,
} from '../state/types.js';

const deepClone = <T>(value: T): T => {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
};

const cloneParameterValue = (value: ManifestParameterValue): ManifestParameterValue => {
  if (typeof value === 'object' && value !== null && 'kind' in value) {
    return {
      kind: value.kind,
      size: value.size,
      values: [...value.values],
    };
  }
  return value;
};

const cloneParameterControl = (
  control: ManifestParameterControl | undefined,
): ManifestParameterControl | undefined => {
  if (!control) {
    return undefined;
  }
  return deepClone(control);
};

const mapParameters = (parameters: readonly ManifestNodeParameter[]): SceneNodeParameter[] =>
  parameters.map((parameter: ManifestNodeParameter) => ({
    id: parameter.id,
    label: parameter.label,
    value: cloneParameterValue(parameter.value),
    description: parameter.description,
    control: cloneParameterControl(parameter.control),
    panel: parameter.panel,
  }));

const mapNodes = (manifest: SceneManifest): SceneNode[] =>
  manifest.nodes.map((node: ManifestNode) => ({
    id: node.id,
    type: node.type,
    label: node.label,
    position: node.position ? { ...node.position } : undefined,
    parameters: mapParameters(node.parameters ?? []),
    metadata: node.metadata ? deepClone(node.metadata) : undefined,
  }));

const mapLinks = (manifest: SceneManifest): SceneLink[] =>
  manifest.links.map((link: ManifestLinkRecord) => ({
    id: link.id,
    from: { ...link.from },
    to: { ...link.to },
  }));

const mapKeyframe = (frame: ManifestTimelineKeyframe): TimelineKeyframe => ({
  time: frame.time,
  value: frame.value,
  easing: frame.easing,
});

const mapClip = (clip: ManifestTimelineClip): TimelineClip => ({
  id: clip.id,
  nodeId: clip.nodeId,
  parameterId: clip.parameterId,
  keyframes: clip.keyframes.map(mapKeyframe),
});

const DEFAULT_TIMELINE_DURATION = 12;
const DEFAULT_TIMELINE_FPS = 60;

const createTimeline = (manifest: SceneManifest): TimelineState => {
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

export interface ManifestRuntimeBundle {
  readonly scene: SceneGraphState;
  readonly timeline: TimelineState;
  readonly controls?: ManifestControls;
}

export function createRuntimeBundle(manifest: SceneManifest): ManifestRuntimeBundle {
  const nodes = mapNodes(manifest);
  const links = mapLinks(manifest);

  const scene: SceneGraphState = {
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
