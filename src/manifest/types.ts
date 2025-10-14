export type ManifestPrimitive = number | string | boolean;

export interface ManifestParameterVector {
  readonly kind: 'vector';
  readonly size: number;
  readonly values: number[];
}

export type ManifestParameterValue = ManifestPrimitive | ManifestParameterVector;

export interface ManifestNodeParameter {
  readonly id: string;
  readonly label: string;
  readonly value: ManifestParameterValue;
  readonly description?: string;
  readonly control?: ManifestParameterControl;
  readonly panel?: string;
}

export interface ManifestNode {
  readonly id: string;
  readonly type: string;
  readonly label: string;
  readonly parameters?: ManifestNodeParameter[];
  readonly position?: {
    readonly x: number;
    readonly y: number;
  };
  readonly metadata?: Record<string, unknown>;
}

export interface ManifestLinkEndpoint {
  readonly nodeId: string;
  readonly port: string;
}

export interface ManifestLink {
  readonly id: string;
  readonly from: ManifestLinkEndpoint;
  readonly to: ManifestLinkEndpoint;
  readonly metadata?: Record<string, unknown>;
}

export interface ManifestTimelineKeyframe {
  readonly time: number;
  readonly value: ManifestPrimitive;
  readonly easing?: string;
}

export interface ManifestTimelineClip {
  readonly id: string;
  readonly nodeId: string;
  readonly parameterId: string;
  readonly keyframes: ManifestTimelineKeyframe[];
  readonly interpolation?: 'step' | 'linear' | 'spline';
}

export interface ManifestTimeline {
  readonly duration: number;
  readonly fps: number;
  readonly clips: ManifestTimelineClip[];
}

export interface ManifestMetadata {
  readonly name: string;
  readonly description?: string;
  readonly author?: string;
  readonly version?: string;
  readonly createdAt?: string;
  readonly updatedAt?: string;
  readonly tags?: string[];
  readonly defaultPreset?: string;
}

export type JsonSchema = {
  readonly [key: string]: unknown;
};

export type JsonUiSchema = {
  readonly [key: string]: unknown;
};

export type ManifestBindingKind = 'nodeParameter' | 'environment' | 'timeline';

export interface ManifestControlBindingBase {
  readonly kind: ManifestBindingKind;
}

export interface ManifestControlNodeBinding extends ManifestControlBindingBase {
  readonly kind: 'nodeParameter';
  readonly nodeId: string;
  readonly parameterId: string;
  readonly valuePath?: string;
  readonly transform?: ManifestControlTransform;
}

export interface ManifestControlEnvironmentBinding extends ManifestControlBindingBase {
  readonly kind: 'environment';
  readonly path: string;
}

export interface ManifestControlTimelineBinding extends ManifestControlBindingBase {
  readonly kind: 'timeline';
  readonly clipId: string;
  readonly parameterId: string;
  readonly valuePath?: string;
}

export type ManifestControlBinding =
  | ManifestControlNodeBinding
  | ManifestControlEnvironmentBinding
  | ManifestControlTimelineBinding;

export interface ManifestControlPanel {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly category?: string;
  readonly icon?: string;
  readonly collapsed?: boolean;
  readonly schema: JsonSchema;
  readonly uiSchema?: JsonUiSchema;
  readonly bindings: Record<string, ManifestControlBinding>;
}

export interface ManifestPreset {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly thumbnail?: string;
  readonly panels?: Record<string, Record<string, unknown>>;
}

export interface ManifestControls {
  readonly panels: ManifestControlPanel[];
  readonly presets?: ManifestPreset[];
}

export type ManifestParameterControl =
  | {
      readonly kind: 'slider' | 'range';
      readonly min?: number;
      readonly max?: number;
      readonly step?: number;
      readonly unit?: string;
    }
  | {
      readonly kind: 'toggle';
      readonly trueLabel?: string;
      readonly falseLabel?: string;
    }
  | {
      readonly kind: 'select';
      readonly options: ReadonlyArray<{
        readonly value: string | number | boolean;
        readonly label?: string;
      }>;
    }
  | {
      readonly kind: 'color';
      readonly format?: 'hex' | 'rgb';
    }
  | {
      readonly kind: 'vector';
      readonly componentLabels?: ReadonlyArray<string>;
      readonly min?: number;
      readonly max?: number;
      readonly step?: number;
    }
  | {
      readonly kind: 'curve';
      readonly description?: string;
    }
  | {
      readonly kind: 'custom';
      readonly widget?: string;
      readonly options?: Record<string, unknown>;
    };

export interface ManifestControlTransform {
  readonly toParameter?:
    | 'degreesToRadians'
    | 'radiansToDegrees'
    | 'percentToUnit'
    | 'unitToPercent';
  readonly fromParameter?:
    | 'degreesToRadians'
    | 'radiansToDegrees'
    | 'percentToUnit'
    | 'unitToPercent';
  readonly scale?: number;
  readonly offset?: number;
}

export interface SceneManifest {
  readonly schemaVersion: string;
  readonly metadata: ManifestMetadata;
  readonly nodes: ManifestNode[];
  readonly links: ManifestLink[];
  readonly timeline?: ManifestTimeline;
  readonly environment?: Record<string, unknown>;
  readonly controls?: ManifestControls;
}

export interface ManifestValidationIssue {
  readonly code: string;
  readonly message: string;
  readonly path: readonly (string | number)[];
  readonly severity: 'error' | 'warning';
}

export interface ManifestValidationResult {
  readonly manifest: SceneManifest;
  readonly issues: ManifestValidationIssue[];
}
