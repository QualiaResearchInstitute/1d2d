import type {
  ManifestMetadata,
  ManifestParameterValue,
  ManifestParameterControl,
  ManifestValidationIssue,
  ManifestPreset,
  ManifestControls,
  ManifestControlBinding,
  JsonSchema,
  JsonUiSchema,
} from '../manifest/types.js';

export type PanelKey = 'presets' | 'inspector' | 'timeline';

export type MediaAssetKind = 'image' | 'video';

export type MediaAssetStatus = 'pending' | 'ready' | 'error';

export type MediaProcessingStage =
  | 'idle'
  | 'loading'
  | 'thumbnail'
  | 'edge'
  | 'phase'
  | 'kuramoto'
  | 'complete'
  | 'error';

export interface MediaAsset {
  readonly id: string;
  readonly name: string;
  readonly kind: MediaAssetKind;
  readonly sourceUrl: string;
  readonly importedAt: number;
  readonly status: MediaAssetStatus;
  readonly width?: number;
  readonly height?: number;
  readonly frameRate?: number;
  readonly frameCount?: number;
  readonly durationSeconds?: number;
  readonly previewUrl?: string;
  readonly metadata?: Record<string, unknown>;
  readonly errorMessage?: string;
}

export interface MediaProcessingStatus {
  readonly assetId?: string;
  readonly stage: MediaProcessingStage;
  readonly progress: number;
  readonly message?: string;
  readonly error?: string;
  readonly startedAt?: number;
  readonly finishedAt?: number;
}

export type MediaTelemetryDurations = Partial<{
  loadMs: number;
  thumbnailMs: number;
  edgeMs: number;
  phaseMs: number;
  kuramotoMs: number;
}>;

export interface MediaTelemetryMetrics {
  readonly edgePixelCount?: number;
  readonly edgeMagnitudeMean?: number;
  readonly phaseVariance?: number;
  readonly coherenceMean?: number;
}

export interface MediaTelemetryEntry {
  readonly assetId: string;
  readonly timestamp: number;
  readonly durations: MediaTelemetryDurations;
  readonly metrics?: MediaTelemetryMetrics;
  readonly notes?: string;
}

export interface MediaState {
  readonly assets: MediaAsset[];
  readonly selectedAssetId?: string;
  readonly processing: MediaProcessingStatus;
  readonly telemetry: MediaTelemetryEntry[];
}

export interface PanelLayoutState {
  readonly key: PanelKey;
  readonly orientation: 'vertical' | 'horizontal';
  readonly minSize: number;
  readonly maxSize?: number;
  visible: boolean;
  size: number;
}

export interface LayoutState {
  readonly presets: PanelLayoutState;
  readonly inspector: PanelLayoutState;
  readonly timeline: PanelLayoutState;
}

export interface SceneNodeParameter {
  readonly id: string;
  readonly label: string;
  readonly value: ManifestParameterValue;
  readonly description?: string;
  readonly control?: ManifestParameterControl;
  readonly panel?: string;
}

export interface SceneNode {
  readonly id: string;
  readonly type: string;
  readonly label: string;
  readonly position?: { readonly x: number; readonly y: number };
  readonly parameters: SceneNodeParameter[];
  readonly metadata?: Record<string, unknown>;
}

export interface SceneLink {
  readonly id: string;
  readonly from: { readonly nodeId: string; readonly port: string };
  readonly to: { readonly nodeId: string; readonly port: string };
}

export interface SceneGraphState {
  readonly nodes: SceneNode[];
  readonly links: SceneLink[];
  readonly selectedNodeId?: string;
}

export interface TimelineKeyframe {
  readonly time: number;
  readonly value: number | string | boolean;
  readonly easing?: string;
}

export interface TimelineClip {
  readonly id: string;
  readonly nodeId: string;
  readonly parameterId: string;
  readonly keyframes: TimelineKeyframe[];
}

export interface TimelineState {
  readonly duration: number;
  readonly fps: number;
  readonly clips: TimelineClip[];
  readonly currentTime: number;
  readonly playing: boolean;
}

export interface BeamSplitterBranchMetrics {
  readonly branchId: string;
  readonly label: string;
  readonly energy: number;
  readonly energyShare: number;
  readonly coverage: number;
  readonly occlusion: number;
  readonly priority?: number;
  readonly source?: string;
  readonly weight?: number;
}

export interface BeamSplitterDiagnosticsEntry {
  readonly nodeId: string;
  readonly frameId: number;
  readonly recombineMode?: string;
  readonly updatedAt: number;
  readonly branches: readonly BeamSplitterBranchMetrics[];
}

export interface BeamSplitterDiagnosticsState {
  readonly entries: readonly BeamSplitterDiagnosticsEntry[];
}

export interface GpuState {
  readonly status: 'idle' | 'initializing' | 'ready' | 'error';
  readonly backend: 'unknown' | 'webgpu' | 'webgl2' | 'unavailable';
  readonly adapterName?: string;
  readonly errorMessage?: string;
  readonly targetFrameRate: number;
  readonly beamSplitter: BeamSplitterDiagnosticsState;
}

export interface ManifestState {
  readonly status: 'idle' | 'loading' | 'loaded' | 'error';
  readonly lastLoadedPath?: string;
  readonly errorMessage?: string;
  readonly metadata?: ManifestMetadata;
  readonly issues?: ManifestValidationIssue[];
  readonly controls?: ControlsState;
  readonly presets?: PresetLibraryState;
  readonly activePresetId?: string;
}

export interface AppState {
  readonly layout: LayoutState;
  readonly manifest: ManifestState;
  readonly scene: SceneGraphState;
  readonly timeline: TimelineState;
  readonly gpu: GpuState;
  readonly media: MediaState;
  readonly history: HistoryState;
}

export interface ControlBindingState {
  readonly pointer: string;
  readonly binding: ManifestControlBinding;
}

export interface ControlPanelState {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly category?: string;
  readonly icon?: string;
  readonly collapsed?: boolean;
  readonly schema: JsonSchema;
  readonly uiSchema?: JsonUiSchema;
  readonly bindings: readonly ControlBindingState[];
  readonly formData: Record<string, unknown>;
}

export interface ControlsState {
  readonly panels: ControlPanelState[];
  readonly source?: ManifestControls;
}

export interface PresetDefinition {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly thumbnail?: string;
  readonly panels?: Record<string, Record<string, unknown>>;
  readonly kind: 'builtin' | 'user';
  readonly createdAt: number;
  readonly manifestPath?: string;
}

export interface PresetLibraryState {
  readonly presets: PresetDefinition[];
}

export interface SceneHistoryEntry {
  readonly scene: SceneGraphState;
  readonly timeline: TimelineState;
}

export interface HistoryState {
  readonly past: SceneHistoryEntry[];
  readonly future: SceneHistoryEntry[];
}
