import React, { useEffect, useMemo, useReducer } from 'react';
import type {
  ManifestMetadata,
  ManifestValidationIssue,
  ManifestControls,
} from '../manifest/types';
import type {
  AppState,
  LayoutState,
  ManifestState,
  MediaAsset,
  MediaProcessingStatus,
  MediaTelemetryEntry,
  ControlsState,
  HistoryState,
  PanelKey,
  PanelLayoutState,
  PresetDefinition,
  PresetLibraryState,
  SceneHistoryEntry,
  SceneGraphState,
  SceneLink,
  SceneNode,
  SceneNodeParameter,
  TimelineClip,
  TimelineKeyframe,
  TimelineState,
  GpuState,
  BeamSplitterDiagnosticsEntry,
} from './types';
import {
  createControlsState as buildControlsState,
  syncControlPanels,
  applyControlPanelFormData,
  updatePanelCollapsedState,
} from './controlPanelUtils';
import { sceneRuntimeBridge } from './sceneRuntimeBridge';
import { loadUserPresets, saveUserPresets } from './presetStorage';
import { adjustBeamSplitterBranches } from './beamSplitter';

type AppAction =
  | { readonly type: 'layout/togglePanel'; readonly panel: PanelKey }
  | {
      readonly type: 'layout/setPanelVisibility';
      readonly panel: PanelKey;
      readonly visible: boolean;
    }
  | { readonly type: 'layout/setPanelSize'; readonly panel: PanelKey; readonly size: number }
  | { readonly type: 'manifest/loadStart'; readonly path?: string }
  | {
      readonly type: 'manifest/loadSuccess';
      readonly scene: SceneGraphState;
      readonly timeline: TimelineState;
      readonly path?: string;
      readonly metadata: ManifestMetadata;
      readonly issues: ManifestValidationIssue[];
      readonly controls?: ManifestControls;
    }
  | {
      readonly type: 'manifest/loadError';
      readonly message: string;
      readonly path?: string;
      readonly issues?: ManifestValidationIssue[];
    }
  | { readonly type: 'scene/selectNode'; readonly nodeId?: string }
  | { readonly type: 'scene/addNode'; readonly node: SceneNode }
  | {
      readonly type: 'scene/updateNode';
      readonly nodeId: string;
      readonly patch: Partial<SceneNode>;
    }
  | { readonly type: 'scene/removeNode'; readonly nodeId: string }
  | { readonly type: 'scene/addLink'; readonly link: SceneLink }
  | { readonly type: 'scene/removeLink'; readonly linkId: string }
  | { readonly type: 'timeline/setTime'; readonly time: number }
  | { readonly type: 'timeline/setPlaying'; readonly playing: boolean }
  | { readonly type: 'gpu/initializing' }
  | {
      readonly type: 'gpu/ready';
      readonly backend: GpuState['backend'];
      readonly adapterName?: string;
    }
  | { readonly type: 'gpu/error'; readonly message: string }
  | { readonly type: 'gpu/update-refresh-rate'; readonly hz: number }
  | {
      readonly type: 'gpu/update-beam-splitter';
      readonly entries: readonly BeamSplitterDiagnosticsEntry[];
    }
  | { readonly type: 'media/register-asset'; readonly asset: MediaAsset }
  | {
      readonly type: 'media/update-asset';
      readonly assetId: string;
      readonly patch: Partial<MediaAsset>;
    }
  | { readonly type: 'media/remove-asset'; readonly assetId: string }
  | { readonly type: 'media/select'; readonly assetId?: string }
  | { readonly type: 'media/update-processing'; readonly status: MediaProcessingStatus }
  | { readonly type: 'media/record-telemetry'; readonly entry: MediaTelemetryEntry }
  | {
      readonly type: 'controls/updateForm';
      readonly panelId: string;
      readonly formData: Record<string, unknown>;
    }
  | {
      readonly type: 'controls/setCollapsed';
      readonly panelId: string;
      readonly collapsed: boolean;
    }
  | { readonly type: 'presets/apply'; readonly presetId: string }
  | {
      readonly type: 'presets/save';
      readonly label: string;
      readonly description?: string;
    }
  | { readonly type: 'presets/delete'; readonly presetId: string }
  | { readonly type: 'history/undo' }
  | { readonly type: 'history/redo' };

const clamp = (value: number, min: number, max?: number): number => {
  if (Number.isNaN(value)) {
    return min;
  }
  const clamped = Math.max(min, value);
  return typeof max === 'number' ? Math.min(clamped, max) : clamped;
};

const deepClone = <T,>(value: T): T => {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
};

const MAX_HISTORY_ENTRIES = 100;

const createPresetId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `preset-${Math.random().toString(36).slice(2, 10)}`;
};

const isDeepEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) {
    return true;
  }
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch (error) {
    return false;
  }
};

const createSnapshot = (state: AppState): SceneHistoryEntry => ({
  scene: deepClone(state.scene),
  timeline: deepClone(state.timeline),
});

const pushHistoryEntry = (history: HistoryState, snapshot: SceneHistoryEntry): HistoryState => {
  const past =
    history.past.length >= MAX_HISTORY_ENTRIES
      ? [...history.past.slice(-MAX_HISTORY_ENTRIES + 1), snapshot]
      : [...history.past, snapshot];
  return {
    past,
    future: [],
  };
};

const createPresetLibraryState = (
  controls: ManifestControls | undefined,
  metadata: ManifestMetadata | undefined,
): PresetLibraryState => {
  const timestamp = Date.now();
  const presets: PresetDefinition[] = [];
  controls?.presets?.forEach((preset, index) => {
    presets.push({
      id: preset.id,
      label: preset.label,
      description: preset.description,
      thumbnail: preset.thumbnail,
      panels: preset.panels ? deepClone(preset.panels) : undefined,
      kind: 'builtin',
      createdAt: timestamp + index,
    });
  });
  const userPresets = loadUserPresets(metadata);
  userPresets.forEach((preset) => {
    presets.push({
      ...preset,
      panels: preset.panels ? deepClone(preset.panels) : undefined,
    });
  });
  return { presets };
};

const createPanel = (
  key: PanelKey,
  orientation: PanelLayoutState['orientation'],
  options: { minSize: number; maxSize?: number; defaultSize: number; visible?: boolean },
): PanelLayoutState => ({
  key,
  orientation,
  minSize: options.minSize,
  maxSize: options.maxSize,
  visible: options.visible ?? true,
  size: clamp(options.defaultSize, options.minSize, options.maxSize),
});

const initialState: AppState = {
  layout: {
    presets: createPanel('presets', 'vertical', { minSize: 240, maxSize: 420, defaultSize: 280 }),
    inspector: createPanel('inspector', 'vertical', {
      minSize: 240,
      maxSize: 460,
      defaultSize: 320,
    }),
    timeline: createPanel('timeline', 'horizontal', {
      minSize: 160,
      maxSize: 360,
      defaultSize: 200,
    }),
  },
  manifest: {
    status: 'idle',
    issues: undefined,
    controls: undefined,
    presets: { presets: [] },
  },
  scene: {
    nodes: [],
    links: [],
  },
  timeline: {
    duration: 10,
    fps: 60,
    clips: [],
    currentTime: 0,
    playing: false,
  },
  gpu: {
    status: 'idle',
    backend: 'unknown',
    targetFrameRate: 120,
    beamSplitter: {
      entries: [],
    },
  },
  media: {
    assets: [],
    selectedAssetId: undefined,
    processing: {
      stage: 'idle',
      progress: 0,
    },
    telemetry: [],
  },
  history: {
    past: [],
    future: [],
  },
};

const appStateReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'layout/togglePanel': {
      const panel = state.layout[action.panel];
      return {
        ...state,
        layout: {
          ...state.layout,
          [action.panel]: { ...panel, visible: !panel.visible },
        },
      };
    }
    case 'layout/setPanelVisibility': {
      const panel = state.layout[action.panel];
      if (panel.visible === action.visible) {
        return state;
      }
      return {
        ...state,
        layout: {
          ...state.layout,
          [action.panel]: { ...panel, visible: action.visible },
        },
      };
    }
    case 'layout/setPanelSize': {
      const panel = state.layout[action.panel];
      const size = clamp(action.size, panel.minSize, panel.maxSize);
      if (size === panel.size) {
        return state;
      }
      return {
        ...state,
        layout: {
          ...state.layout,
          [action.panel]: { ...panel, size },
        },
      };
    }
    case 'manifest/loadStart':
      return {
        ...state,
        manifest: {
          status: 'loading',
          lastLoadedPath: action.path ?? state.manifest.lastLoadedPath,
          metadata: state.manifest.metadata,
          issues: undefined,
          errorMessage: undefined,
          controls: state.manifest.controls,
          presets: state.manifest.presets,
          activePresetId: state.manifest.activePresetId,
        },
      };
    case 'manifest/loadSuccess': {
      const controlsState = buildControlsState(action.controls, action.scene);
      const presetsState = createPresetLibraryState(action.controls, action.metadata);
      const defaultPresetId = action.metadata.defaultPreset;
      return {
        ...state,
        manifest: {
          status: 'loaded',
          lastLoadedPath: action.path ?? state.manifest.lastLoadedPath,
          metadata: action.metadata,
          issues: action.issues,
          errorMessage: undefined,
          controls: controlsState,
          presets: presetsState,
          activePresetId: defaultPresetId,
        },
        scene: action.scene,
        timeline: action.timeline,
        history: {
          past: [],
          future: [],
        },
      };
    }
    case 'manifest/loadError':
      return {
        ...state,
        manifest: {
          status: 'error',
          errorMessage: action.message,
          lastLoadedPath: action.path ?? state.manifest.lastLoadedPath,
          metadata: state.manifest.metadata,
          issues: action.issues,
          controls: state.manifest.controls,
          presets: state.manifest.presets,
          activePresetId: state.manifest.activePresetId,
        },
      };
    case 'scene/selectNode':
      return {
        ...state,
        scene: {
          ...state.scene,
          selectedNodeId: action.nodeId,
        },
      };
    case 'scene/addNode': {
      const snapshot = createSnapshot(state);
      const scene = {
        ...state.scene,
        nodes: [...state.scene.nodes, action.node],
        selectedNodeId: action.node.id,
      };
      const controls = syncControlPanels(state.manifest.controls, scene) ?? state.manifest.controls;
      return {
        ...state,
        scene,
        manifest: {
          ...state.manifest,
          controls,
          activePresetId: undefined,
        },
        history: pushHistoryEntry(state.history, snapshot),
      };
    }
    case 'scene/updateNode': {
      let changed = false;
      const nodes = state.scene.nodes.map((node) => {
        if (node.id !== action.nodeId) {
          return node;
        }
        const updatedNode = { ...node, ...action.patch };
        if (!changed) {
          changed = Object.keys(action.patch).some((key) => {
            const keyTyped = key as keyof typeof updatedNode;
            return !isDeepEqual(node[keyTyped], updatedNode[keyTyped]);
          });
        }
        return updatedNode;
      });
      if (!changed) {
        return state;
      }
      const snapshot = createSnapshot(state);
      const scene = {
        ...state.scene,
        nodes,
      };
      const controls = syncControlPanels(state.manifest.controls, scene) ?? state.manifest.controls;
      return {
        ...state,
        scene,
        manifest: {
          ...state.manifest,
          controls,
          activePresetId: undefined,
        },
        history: pushHistoryEntry(state.history, snapshot),
      };
    }
    case 'scene/removeNode': {
      const nodes = state.scene.nodes.filter((node) => node.id !== action.nodeId);
      const links = state.scene.links.filter(
        (link) => link.from.nodeId !== action.nodeId && link.to.nodeId !== action.nodeId,
      );
      const selectedNodeId =
        state.scene.selectedNodeId === action.nodeId ? undefined : state.scene.selectedNodeId;
      if (nodes.length === state.scene.nodes.length) {
        return state;
      }
      const snapshot = createSnapshot(state);
      const scene = {
        ...state.scene,
        nodes,
        links,
        selectedNodeId,
      };
      const controls = syncControlPanels(state.manifest.controls, scene) ?? state.manifest.controls;
      return {
        ...state,
        scene,
        manifest: {
          ...state.manifest,
          controls,
          activePresetId: undefined,
        },
        history: pushHistoryEntry(state.history, snapshot),
      };
    }
    case 'scene/addLink': {
      if (state.scene.links.some((link) => link.id === action.link.id)) {
        return state;
      }
      const addLinkSnapshot = createSnapshot(state);
      const sceneWithLink = {
        ...state.scene,
        links: [...state.scene.links, action.link],
      };
      return {
        ...state,
        scene: sceneWithLink,
        manifest: {
          ...state.manifest,
          controls:
            syncControlPanels(state.manifest.controls, sceneWithLink) ?? state.manifest.controls,
          activePresetId: undefined,
        },
        history: pushHistoryEntry(state.history, addLinkSnapshot),
      };
    }
    case 'scene/removeLink': {
      if (!state.scene.links.some((link) => link.id === action.linkId)) {
        return state;
      }
      const removeLinkSnapshot = createSnapshot(state);
      const sceneWithoutLink = {
        ...state.scene,
        links: state.scene.links.filter((link) => link.id !== action.linkId),
      };
      return {
        ...state,
        scene: sceneWithoutLink,
        manifest: {
          ...state.manifest,
          controls:
            syncControlPanels(state.manifest.controls, sceneWithoutLink) ?? state.manifest.controls,
          activePresetId: undefined,
        },
        history: pushHistoryEntry(state.history, removeLinkSnapshot),
      };
    }
    case 'timeline/setTime': {
      const clampedTime = clamp(action.time, 0, state.timeline.duration);
      if (clampedTime === state.timeline.currentTime) {
        return state;
      }
      const snapshot = createSnapshot(state);
      return {
        ...state,
        timeline: {
          ...state.timeline,
          currentTime: clampedTime,
        },
        history: pushHistoryEntry(state.history, snapshot),
      };
    }
    case 'timeline/setPlaying': {
      if (state.timeline.playing === action.playing) {
        return state;
      }
      const snapshot = createSnapshot(state);
      return {
        ...state,
        timeline: {
          ...state.timeline,
          playing: action.playing,
        },
        history: pushHistoryEntry(state.history, snapshot),
      };
    }
    case 'gpu/initializing':
      return {
        ...state,
        gpu: {
          ...state.gpu,
          status: 'initializing',
          backend: 'unknown',
          targetFrameRate: state.gpu.targetFrameRate,
          adapterName: undefined,
          errorMessage: undefined,
        },
      };
    case 'gpu/ready':
      return {
        ...state,
        gpu: {
          ...state.gpu,
          status: 'ready',
          backend: action.backend,
          adapterName: action.adapterName,
          targetFrameRate: state.gpu.targetFrameRate,
          errorMessage: undefined,
        },
      };
    case 'gpu/error':
      return {
        ...state,
        gpu: {
          ...state.gpu,
          status: 'error',
          backend: 'unavailable',
          targetFrameRate: state.gpu.targetFrameRate,
          errorMessage: action.message,
        },
      };
    case 'gpu/update-refresh-rate':
      return {
        ...state,
        gpu: {
          ...state.gpu,
          targetFrameRate: action.hz,
        },
      };
    case 'gpu/update-beam-splitter': {
      const merged = new Map<string, BeamSplitterDiagnosticsEntry>();
      for (const entry of state.gpu.beamSplitter.entries) {
        merged.set(entry.nodeId, entry);
      }
      for (const entry of action.entries) {
        merged.set(entry.nodeId, entry);
      }
      const nextEntries = Array.from(merged.values()).sort((a, b) => b.updatedAt - a.updatedAt);
      return {
        ...state,
        gpu: {
          ...state.gpu,
          beamSplitter: {
            entries: nextEntries,
          },
        },
      };
    }
    case 'media/register-asset': {
      const existingIndex = state.media.assets.findIndex((asset) => asset.id === action.asset.id);
      const assets =
        existingIndex >= 0
          ? state.media.assets.map((asset, index) =>
              index === existingIndex ? action.asset : asset,
            )
          : [...state.media.assets, action.asset];
      return {
        ...state,
        media: {
          ...state.media,
          assets,
          selectedAssetId: action.asset.id,
        },
      };
    }
    case 'media/update-asset': {
      const assets = state.media.assets.map((asset) =>
        asset.id === action.assetId ? { ...asset, ...action.patch } : asset,
      );
      return {
        ...state,
        media: {
          ...state.media,
          assets,
        },
      };
    }
    case 'media/remove-asset': {
      const assets = state.media.assets.filter((asset) => asset.id !== action.assetId);
      const selectedAssetId =
        state.media.selectedAssetId === action.assetId
          ? assets[assets.length - 1]?.id
          : state.media.selectedAssetId;
      return {
        ...state,
        media: {
          ...state.media,
          assets,
          selectedAssetId,
        },
      };
    }
    case 'media/select':
      return {
        ...state,
        media: {
          ...state.media,
          selectedAssetId: action.assetId,
        },
      };
    case 'media/update-processing':
      return {
        ...state,
        media: {
          ...state.media,
          processing: action.status,
        },
      };
    case 'media/record-telemetry': {
      const MAX_ENTRIES = 200;
      const telemetry =
        state.media.telemetry.length >= MAX_ENTRIES
          ? [...state.media.telemetry.slice(1), action.entry]
          : [...state.media.telemetry, action.entry];
      return {
        ...state,
        media: {
          ...state.media,
          telemetry,
        },
      };
    }
    case 'controls/updateForm': {
      const controlsState = state.manifest.controls;
      if (!controlsState) {
        return state;
      }
      const currentPanelState = controlsState.panels.find((panel) => panel.id === action.panelId);
      const snapshot = createSnapshot(state);
      const result = applyControlPanelFormData(
        controlsState,
        action.panelId,
        action.formData,
        state.scene,
      );
      let updatedScene = result.scene;
      const updatedPanels = result.panels;

      if (action.panelId === 'beamSplitterPanel') {
        const previousCount =
          currentPanelState &&
          typeof (currentPanelState.formData as Record<string, unknown>).branchCount === 'number'
            ? ((currentPanelState.formData as Record<string, unknown>).branchCount as number)
            : undefined;
        const rawNextCount = (action.formData as Record<string, unknown>).branchCount;
        const nextCount =
          typeof rawNextCount === 'number' ? Math.max(0, Math.round(rawNextCount)) : previousCount;
        if (typeof nextCount === 'number' && nextCount > 0 && nextCount !== previousCount) {
          updatedScene = adjustBeamSplitterBranches(updatedScene, nextCount);
        }
      }

      const controlsWithPanels: ControlsState = {
        ...controlsState,
        panels: updatedPanels,
      };
      const syncedControls =
        syncControlPanels(controlsWithPanels, updatedScene) ?? controlsWithPanels;
      const history =
        updatedScene !== state.scene ? pushHistoryEntry(state.history, snapshot) : state.history;
      return {
        ...state,
        scene: updatedScene,
        manifest: {
          ...state.manifest,
          controls: syncedControls,
          activePresetId: updatedScene !== state.scene ? undefined : state.manifest.activePresetId,
        },
        history,
      };
    }
    case 'controls/setCollapsed': {
      const controls = state.manifest.controls;
      if (!controls) {
        return state;
      }
      const panels = updatePanelCollapsedState(controls.panels, action.panelId, action.collapsed);
      return {
        ...state,
        manifest: {
          ...state.manifest,
          controls: {
            ...controls,
            panels,
          },
        },
      };
    }
    case 'presets/apply': {
      const controls = state.manifest.controls;
      const library = state.manifest.presets;
      if (!controls || !library) {
        return state;
      }
      const preset = library.presets.find((entry) => entry.id === action.presetId);
      if (!preset || !preset.panels) {
        return {
          ...state,
          manifest: {
            ...state.manifest,
            activePresetId: preset ? action.presetId : state.manifest.activePresetId,
          },
        };
      }
      const snapshot = createSnapshot(state);
      let nextScene = state.scene;
      let nextControls: ControlsState = controls;
      const panelEntries = Object.entries(preset.panels);
      panelEntries.forEach(([panelId, formData]) => {
        const result = applyControlPanelFormData(nextControls, panelId, formData ?? {}, nextScene);
        nextScene = result.scene;
        nextControls = {
          ...nextControls,
          panels: result.panels,
        };
      });
      const syncedControls = syncControlPanels(nextControls, nextScene) ?? nextControls;
      const history =
        nextScene !== state.scene ? pushHistoryEntry(state.history, snapshot) : state.history;
      return {
        ...state,
        scene: nextScene,
        manifest: {
          ...state.manifest,
          controls: syncedControls,
          presets: library,
          activePresetId: action.presetId,
        },
        history,
      };
    }
    case 'presets/save': {
      const controls = state.manifest.controls;
      const library = state.manifest.presets;
      if (!controls || !library) {
        return state;
      }
      const panelsData: Record<string, Record<string, unknown>> = {};
      controls.panels.forEach((panel) => {
        panelsData[panel.id] = deepClone(panel.formData);
      });
      const newPreset: PresetDefinition = {
        id: createPresetId(),
        label: action.label,
        description: action.description,
        panels: panelsData,
        thumbnail: undefined,
        kind: 'user',
        createdAt: Date.now(),
      };
      const presets = [...library.presets, newPreset];
      saveUserPresets(state.manifest.metadata, presets);
      return {
        ...state,
        manifest: {
          ...state.manifest,
          presets: {
            ...library,
            presets,
          },
          activePresetId: newPreset.id,
        },
      };
    }
    case 'presets/delete': {
      const library = state.manifest.presets;
      if (!library) {
        return state;
      }
      const preset = library.presets.find((entry) => entry.id === action.presetId);
      if (!preset || preset.kind !== 'user') {
        return state;
      }
      const presets = library.presets.filter((entry) => entry.id !== action.presetId);
      saveUserPresets(state.manifest.metadata, presets);
      return {
        ...state,
        manifest: {
          ...state.manifest,
          presets: {
            ...library,
            presets,
          },
          activePresetId:
            state.manifest.activePresetId === action.presetId
              ? undefined
              : state.manifest.activePresetId,
        },
      };
    }
    case 'history/undo': {
      if (state.history.past.length === 0) {
        return state;
      }
      const snapshot = state.history.past[state.history.past.length - 1]!;
      const history: HistoryState = {
        past: state.history.past.slice(0, -1),
        future: [createSnapshot(state), ...state.history.future],
      };
      const controls =
        syncControlPanels(state.manifest.controls, snapshot.scene) ?? state.manifest.controls;
      return {
        ...state,
        scene: snapshot.scene,
        timeline: snapshot.timeline,
        manifest: {
          ...state.manifest,
          controls,
          activePresetId: undefined,
        },
        history,
      };
    }
    case 'history/redo': {
      if (state.history.future.length === 0) {
        return state;
      }
      const snapshot = state.history.future[0]!;
      const history: HistoryState = {
        past: [...state.history.past, createSnapshot(state)],
        future: state.history.future.slice(1),
      };
      const controls =
        syncControlPanels(state.manifest.controls, snapshot.scene) ?? state.manifest.controls;
      return {
        ...state,
        scene: snapshot.scene,
        timeline: snapshot.timeline,
        manifest: {
          ...state.manifest,
          controls,
          activePresetId: undefined,
        },
        history,
      };
    }
    default:
      return state;
  }
};

interface AppStateContextValue {
  readonly state: AppState;
  readonly dispatch: React.Dispatch<AppAction>;
}

const AppStateContext = React.createContext<AppStateContextValue | undefined>(undefined);

export interface AppStateProviderProps {
  readonly children: React.ReactNode;
}

export function AppStateProvider({ children }: AppStateProviderProps) {
  const [state, dispatch] = useReducer(appStateReducer, initialState);

  useEffect(() => {
    sceneRuntimeBridge.registerDispatch(dispatch);
  }, [dispatch]);

  useEffect(() => {
    sceneRuntimeBridge.publish({ scene: state.scene, timeline: state.timeline });
  }, [state.scene, state.timeline]);

  const contextValue = useMemo<AppStateContextValue>(
    () => ({
      state,
      dispatch,
    }),
    [state, dispatch],
  );

  return <AppStateContext.Provider value={contextValue}>{children}</AppStateContext.Provider>;
}

function useAppStateContext(): AppStateContextValue {
  const context = React.useContext(AppStateContext);
  if (!context) {
    throw new Error('useAppState must be used within an AppStateProvider');
  }
  return context;
}

export function useAppState(): AppStateContextValue {
  return useAppStateContext();
}

export function usePanelLayout(panel: PanelKey) {
  const { state, dispatch } = useAppStateContext();
  const layout = state.layout[panel];
  return useMemo(
    () => ({
      layout,
      toggle: () => dispatch({ type: 'layout/togglePanel', panel }),
      setVisible: (visible: boolean) =>
        dispatch({ type: 'layout/setPanelVisibility', panel, visible }),
      setSize: (size: number) => dispatch({ type: 'layout/setPanelSize', panel, size }),
    }),
    [dispatch, layout, panel],
  );
}

export function useManifestStatus() {
  const { state } = useAppStateContext();
  return state.manifest;
}

export function useSceneGraph() {
  const { state, dispatch } = useAppStateContext();
  return useMemo(
    () => ({
      scene: state.scene,
      selectNode: (nodeId?: string) => dispatch({ type: 'scene/selectNode', nodeId }),
      addNode: (node: SceneNode) => dispatch({ type: 'scene/addNode', node }),
      updateNode: (nodeId: string, patch: Partial<SceneNode>) =>
        dispatch({ type: 'scene/updateNode', nodeId, patch }),
      removeNode: (nodeId: string) => dispatch({ type: 'scene/removeNode', nodeId }),
      addLink: (link: SceneLink) => dispatch({ type: 'scene/addLink', link }),
      removeLink: (linkId: string) => dispatch({ type: 'scene/removeLink', linkId }),
    }),
    [dispatch, state.scene],
  );
}

export function useControlPanels() {
  const { state, dispatch } = useAppStateContext();
  const controls = state.manifest.controls;
  return useMemo(
    () => ({
      controlsState: controls,
      panels: controls?.panels ?? [],
      source: controls?.source,
      updateForm: (panelId: string, formData: Record<string, unknown>) =>
        dispatch({ type: 'controls/updateForm', panelId, formData }),
      setCollapsed: (panelId: string, collapsed: boolean) =>
        dispatch({ type: 'controls/setCollapsed', panelId, collapsed }),
    }),
    [controls, dispatch],
  );
}

export function usePresetLibrary() {
  const { state, dispatch } = useAppStateContext();
  const library = state.manifest.presets;
  return useMemo(
    () => ({
      presets: library?.presets ?? [],
      activePresetId: state.manifest.activePresetId,
      applyPreset: (presetId: string) => dispatch({ type: 'presets/apply', presetId }),
      savePreset: (label: string, description?: string) =>
        dispatch({ type: 'presets/save', label, description }),
      deletePreset: (presetId: string) => dispatch({ type: 'presets/delete', presetId }),
    }),
    [dispatch, library?.presets, state.manifest.activePresetId],
  );
}

export function useHistoryController() {
  const { state, dispatch } = useAppStateContext();
  return useMemo(
    () => ({
      canUndo: state.history.past.length > 0,
      canRedo: state.history.future.length > 0,
      undo: () => dispatch({ type: 'history/undo' }),
      redo: () => dispatch({ type: 'history/redo' }),
    }),
    [dispatch, state.history.future.length, state.history.past.length],
  );
}

export function useTimelineState() {
  const { state, dispatch } = useAppStateContext();
  return useMemo(
    () => ({
      timeline: state.timeline,
      setTime: (time: number) => dispatch({ type: 'timeline/setTime', time }),
      setPlaying: (playing: boolean) => dispatch({ type: 'timeline/setPlaying', playing }),
    }),
    [dispatch, state.timeline],
  );
}

export function useGpuState() {
  const { state, dispatch } = useAppStateContext();
  return useMemo(
    () => ({
      gpu: state.gpu,
      setTargetFrameRate: (hz: number) => dispatch({ type: 'gpu/update-refresh-rate', hz }),
      setInitializing: () => dispatch({ type: 'gpu/initializing' }),
      setReady: (backend: GpuState['backend'], adapterName?: string) =>
        dispatch({ type: 'gpu/ready', backend, adapterName }),
      setError: (message: string) => dispatch({ type: 'gpu/error', message }),
      updateBeamSplitterDiagnostics: (entries: readonly BeamSplitterDiagnosticsEntry[]) =>
        dispatch({ type: 'gpu/update-beam-splitter', entries }),
    }),
    [dispatch, state.gpu],
  );
}

export function useBeamSplitterDiagnostics(nodeId?: string) {
  const { state } = useAppStateContext();
  return useMemo(() => {
    if (!nodeId) {
      return state.gpu.beamSplitter.entries;
    }
    return state.gpu.beamSplitter.entries.filter((entry) => entry.nodeId === nodeId);
  }, [nodeId, state.gpu.beamSplitter.entries]);
}

export function useMediaLibrary() {
  const { state, dispatch } = useAppStateContext();
  const selectedAsset =
    state.media.selectedAssetId != null
      ? state.media.assets.find((asset) => asset.id === state.media.selectedAssetId)
      : undefined;
  return useMemo(
    () => ({
      media: state.media,
      assets: state.media.assets,
      selectedAsset,
      selectAsset: (assetId?: string) => dispatch({ type: 'media/select', assetId }),
      registerAsset: (asset: MediaAsset) => dispatch({ type: 'media/register-asset', asset }),
      updateAsset: (assetId: string, patch: Partial<MediaAsset>) =>
        dispatch({ type: 'media/update-asset', assetId, patch }),
      removeAsset: (assetId: string) => dispatch({ type: 'media/remove-asset', assetId }),
      setProcessingStatus: (status: MediaProcessingStatus) =>
        dispatch({ type: 'media/update-processing', status }),
      recordTelemetry: (entry: MediaTelemetryEntry) =>
        dispatch({ type: 'media/record-telemetry', entry }),
    }),
    [dispatch, selectedAsset, state.media],
  );
}

export type {
  PanelKey,
  PanelLayoutState,
  LayoutState,
  SceneGraphState,
  SceneNode,
  SceneNodeParameter,
  SceneLink,
  TimelineClip,
  TimelineKeyframe,
  TimelineState,
  ManifestState,
  AppState,
  GpuState,
  MediaAsset,
  MediaAssetKind,
  MediaAssetStatus,
  MediaProcessingStage,
  MediaTelemetryDurations,
  MediaTelemetryMetrics,
  MediaState,
  MediaProcessingStatus,
  MediaTelemetryEntry,
  ControlsState,
  ControlPanelState,
  ControlBindingState,
  PresetDefinition,
  PresetLibraryState,
  HistoryState,
} from './types';

export type { AppAction };
export { sceneRuntimeBridge } from './sceneRuntimeBridge';
