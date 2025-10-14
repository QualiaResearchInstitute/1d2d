import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSceneGraph, useBeamSplitterDiagnostics } from '../../state/AppState';
import {
  parseBranches,
  serialiseBranches,
  createBranch,
  type BeamSplitterBranch,
  type BeamSplitterBranchSource,
  type BeamSplitterTransformStep,
} from '../../state/beamSplitter';
import type { BeamSplitterBranchMetrics, SceneNode } from '../../state/types';
import { useI18n } from '../../i18n/LocalizationProvider';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const clampWeight = (value: number) => {
  if (!Number.isFinite(value)) return 1;
  return Math.max(0, Math.min(8, value));
};

const clampPriority = (value: number) => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.round(value));
};

const GRAPH_WIDTH = 420;
const GRAPH_PADDING = 48;
const GRAPH_ROW_HEIGHT = 112;
const BRANCH_NODE_HEIGHT = 88;
const INPUT_X = 48;
const BRANCH_X = 210;
const OUTPUT_X = 368;

const SOURCE_OPTION_DEFS: ReadonlyArray<{ value: BeamSplitterBranchSource; labelKey: string }> = [
  { value: 'source', labelKey: 'beamSplitter.source.primary' },
  { value: 'edge', labelKey: 'beamSplitter.source.edge' },
  { value: 'phase', labelKey: 'beamSplitter.source.phase' },
  { value: 'oscillator', labelKey: 'beamSplitter.source.oscillator' },
  { value: 'surface', labelKey: 'beamSplitter.source.surface' },
];

const SYMMETRY_OPTION_DEFS: ReadonlyArray<{ value: string; labelKey: string }> = [
  { value: 'identity', labelKey: 'beamSplitter.symmetry.identity' },
  { value: 'rotation', labelKey: 'beamSplitter.symmetry.rotation' },
  { value: 'mirrorX', labelKey: 'beamSplitter.symmetry.mirrorX' },
  { value: 'mirrorY', labelKey: 'beamSplitter.symmetry.mirrorY' },
];

const TRANSFORM_LIBRARY_DEFS: ReadonlyArray<{
  kind: BeamSplitterTransformStep['kind'];
  labelKey: string;
  create: () => BeamSplitterTransformStep;
}> = [
  {
    kind: 'rotate',
    labelKey: 'beamSplitter.transform.kind.rotate',
    create: () => ({ kind: 'rotate', degrees: 0 }),
  },
  {
    kind: 'scale',
    labelKey: 'beamSplitter.transform.kind.scale',
    create: () => ({ kind: 'scale', factor: 1 }),
  },
  {
    kind: 'mirror',
    labelKey: 'beamSplitter.transform.kind.mirror',
    create: () => ({ kind: 'mirror', axis: 'x' }),
  },
];

const RECOMBINE_OPTION_DEFS: ReadonlyArray<{
  value: string;
  labelKey: string;
  descriptionKey: string;
}> = [
  {
    value: 'sum',
    labelKey: 'beamSplitter.recombine.sum',
    descriptionKey: 'beamSplitter.recombine.sum.description',
  },
  {
    value: 'average',
    labelKey: 'beamSplitter.recombine.average',
    descriptionKey: 'beamSplitter.recombine.average.description',
  },
  {
    value: 'energy',
    labelKey: 'beamSplitter.recombine.energy',
    descriptionKey: 'beamSplitter.recombine.energy.description',
  },
  {
    value: 'priority',
    labelKey: 'beamSplitter.recombine.priority',
    descriptionKey: 'beamSplitter.recombine.priority.description',
  },
  {
    value: 'max',
    labelKey: 'beamSplitter.recombine.max',
    descriptionKey: 'beamSplitter.recombine.max.description',
  },
  {
    value: 'phase',
    labelKey: 'beamSplitter.recombine.phase',
    descriptionKey: 'beamSplitter.recombine.phase.description',
  },
];

type BranchDiagnosticsMap = Map<string, BeamSplitterBranchMetrics>;

const ensureTransformStack = (
  stack: readonly BeamSplitterTransformStep[],
): BeamSplitterTransformStep[] => {
  if (stack.length === 0) {
    return [{ kind: 'rotate', degrees: 0 }];
  }
  return stack.map((step) => ({ ...step }));
};

interface BranchGraphProps {
  readonly branches: readonly BeamSplitterBranch[];
  readonly diagnostics: BranchDiagnosticsMap;
  readonly selectedBranchId?: string | null;
  readonly recombineMode: string;
  readonly onSelectBranch: (branchId: string) => void;
  readonly onAddBranch: () => void;
}

function BranchGraph({
  branches,
  diagnostics,
  selectedBranchId,
  recombineMode,
  onSelectBranch,
  onAddBranch,
}: BranchGraphProps) {
  const { t } = useI18n();
  const branchCount = Math.max(branches.length, 1);
  const graphHeight = GRAPH_PADDING * 2 + branchCount * GRAPH_ROW_HEIGHT;
  const inputCenterY = graphHeight / 2;
  const outputCenterY = graphHeight / 2;
  const sourceLabels = useMemo(
    () => new Map(SOURCE_OPTION_DEFS.map((entry) => [entry.value, t(entry.labelKey)])),
    [t],
  );
  const recombineLabels = useMemo(
    () => new Map(RECOMBINE_OPTION_DEFS.map((entry) => [entry.value, t(entry.labelKey)])),
    [t],
  );
  const recombineLabel = recombineLabels.get(recombineMode) ?? t('beamSplitter.recombine.unknown');

  return (
    <div className="beam-graph" style={{ height: `${graphHeight}px` }}>
      <svg
        className="beam-graph__canvas"
        viewBox={`0 0 ${GRAPH_WIDTH} ${graphHeight}`}
        aria-hidden="true"
      >
        {branches.map((branch, index) => {
          const offsetY = GRAPH_PADDING + index * GRAPH_ROW_HEIGHT + BRANCH_NODE_HEIGHT / 2;
          const entry = diagnostics.get(branch.id);
          const weight = branch.weight ?? 1;
          const energyShare = entry ? Math.max(0, Math.min(1, entry.energyShare)) : 0;
          const occlusion = entry ? Math.max(0, Math.min(1, entry.occlusion)) : 0;
          const highlight = selectedBranchId === branch.id;
          const linkClass = highlight
            ? 'beam-graph__link beam-graph__link--active'
            : 'beam-graph__link';
          const fromInput = `M ${INPUT_X} ${inputCenterY} C ${INPUT_X + 60} ${inputCenterY} ${BRANCH_X - 56} ${offsetY} ${BRANCH_X} ${offsetY}`;
          const toOutput = `M ${BRANCH_X + 40} ${offsetY} C ${BRANCH_X + 110} ${offsetY} ${OUTPUT_X - 60} ${outputCenterY} ${OUTPUT_X} ${outputCenterY}`;
          return (
            <g key={branch.id}>
              <path className={linkClass} d={fromInput} />
              <path className={linkClass} d={toOutput} />
              <circle
                className="beam-graph__node-dot"
                cx={BRANCH_X}
                cy={offsetY}
                r={highlight ? 6 : energyShare > 0.05 ? 5 : 3}
                opacity={Math.max(0.15, energyShare)}
              />
              <text className="beam-graph__link-weight" x={BRANCH_X - 70} y={offsetY - 14}>
                {weight.toFixed(2)}×
              </text>
              {entry ? (
                <text className="beam-graph__link-energy" x={BRANCH_X - 70} y={offsetY + 24}>
                  {t('beamSplitter.graph.energyOcclusion', {
                    values: {
                      energy: (energyShare * 100).toFixed(0),
                      occlusion: (occlusion * 100).toFixed(0),
                    },
                  })}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div
        className="beam-graph__node beam-graph__node--input"
        style={{ top: `${inputCenterY - 40}px` }}
      >
        <span className="beam-graph__node-title">{t('beamSplitter.graph.inputTitle')}</span>
        <span className="beam-graph__node-subtitle">{t('beamSplitter.graph.inputSubtitle')}</span>
      </div>
      {branches.map((branch, index) => {
        const offsetY = GRAPH_PADDING + index * GRAPH_ROW_HEIGHT;
        const metrics = diagnostics.get(branch.id);
        const energyShare = metrics ? Math.max(0, Math.min(1, metrics.energyShare)) : 0;
        const occlusion = metrics ? Math.max(0, Math.min(1, metrics.occlusion)) : 0;
        const occluded = occlusion >= 0.85 || energyShare < 0.01;
        const branchSourceLabel =
          sourceLabels.get(branch.source ?? 'source') ?? t('beamSplitter.source.unknown');
        return (
          <button
            key={branch.id}
            type="button"
            className={
              selectedBranchId === branch.id
                ? 'beam-graph__branch beam-graph__branch--selected'
                : 'beam-graph__branch'
            }
            style={{ top: `${offsetY}px` }}
            onClick={() => onSelectBranch(branch.id)}
          >
            <div className="beam-graph__branch-header">
              <span className="beam-graph__branch-label">{branch.label}</span>
              <span className="beam-graph__branch-source">{branchSourceLabel}</span>
            </div>
            <div className="beam-graph__branch-metric">
              <div className="beam-graph__metric-bar" aria-hidden="true">
                <div
                  className="beam-graph__metric-bar-fill"
                  style={{ width: `${Math.max(6, Math.min(100, energyShare * 100))}%` }}
                />
              </div>
              <span className="beam-graph__metric-text">
                {t('beamSplitter.graph.energyValue', {
                  values: { value: (energyShare * 100).toFixed(0) },
                })}
              </span>
            </div>
            {occluded ? (
              <span className="beam-graph__badge beam-graph__badge--warn">
                {t('beamSplitter.graph.badge.muted')}
              </span>
            ) : null}
          </button>
        );
      })}
      <div
        className="beam-graph__node beam-graph__node--output"
        style={{ top: `${outputCenterY - 44}px` }}
      >
        <span className="beam-graph__node-title">{t('beamSplitter.graph.recombineTitle')}</span>
        <span className="beam-graph__node-subtitle">{recombineLabel}</span>
        <button type="button" className="beam-graph__add" onClick={onAddBranch}>
          {t('beamSplitter.graph.addBranch')}
        </button>
      </div>
    </div>
  );
}

interface TransformStackEditorProps {
  readonly stack: readonly BeamSplitterTransformStep[];
  readonly onChange: (stack: BeamSplitterTransformStep[]) => void;
}

function TransformStackEditor({ stack, onChange }: TransformStackEditorProps) {
  const { t } = useI18n();
  const transformLibrary = useMemo(
    () =>
      TRANSFORM_LIBRARY_DEFS.map((entry) => ({
        kind: entry.kind,
        label: t(entry.labelKey),
        create: entry.create,
      })),
    [t],
  );

  const handleUpdate = useCallback(
    (index: number, step: BeamSplitterTransformStep) => {
      const next = stack.slice();
      next[index] = step;
      onChange(ensureTransformStack(next));
    },
    [stack, onChange],
  );

  const handleRemove = useCallback(
    (index: number) => {
      if (stack.length <= 1) {
        onChange([{ kind: 'rotate', degrees: 0 }]);
        return;
      }
      const next = stack.slice();
      next.splice(index, 1);
      onChange(ensureTransformStack(next));
    },
    [stack, onChange],
  );

  const handleMove = useCallback(
    (index: number, direction: -1 | 1) => {
      const target = index + direction;
      if (target < 0 || target >= stack.length) {
        return;
      }
      const next = stack.slice();
      const [step] = next.splice(index, 1);
      next.splice(target, 0, step);
      onChange(next);
    },
    [stack, onChange],
  );

  const handleAdd = useCallback(
    (kind: BeamSplitterTransformStep['kind']) => {
      const libraryEntry = TRANSFORM_LIBRARY_DEFS.find((entry) => entry.kind === kind);
      const next = stack.concat(
        libraryEntry ? libraryEntry.create() : { kind: 'rotate', degrees: 0 },
      );
      onChange(next);
    },
    [stack, onChange],
  );

  return (
    <div className="transform-stack">
      <header className="transform-stack__header">
        <h5>{t('beamSplitter.transform.title')}</h5>
        <div className="transform-stack__actions">
          {transformLibrary.map((entry) => (
            <button
              key={entry.kind}
              type="button"
              className="ghost-button ghost-button--sm"
              onClick={() => handleAdd(entry.kind)}
            >
              {t('beamSplitter.transform.add', { values: { label: entry.label } })}
            </button>
          ))}
        </div>
      </header>
      {stack.length === 0 ? (
        <p className="panel-muted">{t('beamSplitter.transform.empty')}</p>
      ) : (
        <ol className="transform-stack__list">
          {stack.map((step, index) => (
            <li key={`${step.kind}-${index}`} className="transform-stack__item">
              <div className="transform-stack__item-header">
                <span className="transform-stack__item-label">
                  {t('beamSplitter.transform.stepLabel', {
                    values: {
                      index: index + 1,
                      kind: t(
                        step.kind === 'rotate'
                          ? 'beamSplitter.transform.kind.rotate'
                          : step.kind === 'scale'
                            ? 'beamSplitter.transform.kind.scale'
                            : 'beamSplitter.transform.kind.mirror',
                      ),
                    },
                  })}
                </span>
                <div className="transform-stack__item-actions">
                  <button
                    type="button"
                    className="ghost-button ghost-button--sm"
                    onClick={() => handleMove(index, -1)}
                    disabled={index === 0}
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    className="ghost-button ghost-button--sm"
                    onClick={() => handleMove(index, 1)}
                    disabled={index === stack.length - 1}
                  >
                    ↓
                  </button>
                  <button
                    type="button"
                    className="ghost-button ghost-button--sm"
                    onClick={() => handleRemove(index)}
                  >
                    {t('beamSplitter.common.remove')}
                  </button>
                </div>
              </div>
              <div className="transform-stack__controls">
                {step.kind === 'rotate' ? (
                  <label>
                    {t('beamSplitter.transform.degrees')}
                    <input
                      type="number"
                      step={0.5}
                      value={step.degrees}
                      onChange={(event) =>
                        handleUpdate(index, {
                          kind: 'rotate',
                          degrees: Number.parseFloat(event.target.value) || 0,
                        })
                      }
                    />
                  </label>
                ) : null}
                {step.kind === 'scale' ? (
                  <label>
                    {t('beamSplitter.transform.factor')}
                    <input
                      type="number"
                      step={0.05}
                      value={step.factor}
                      onChange={(event) =>
                        handleUpdate(index, {
                          kind: 'scale',
                          factor: Number.parseFloat(event.target.value) || 1,
                        })
                      }
                    />
                  </label>
                ) : null}
                {step.kind === 'mirror' ? (
                  <label>
                    {t('beamSplitter.transform.axis')}
                    <select
                      value={step.axis}
                      onChange={(event) =>
                        handleUpdate(index, {
                          kind: 'mirror',
                          axis: event.target.value === 'y' ? 'y' : 'x',
                        })
                      }
                    >
                      <option value="x">{t('beamSplitter.transform.axisX')}</option>
                      <option value="y">{t('beamSplitter.transform.axisY')}</option>
                    </select>
                  </label>
                ) : null}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

interface BranchDetailProps {
  readonly branch: BeamSplitterBranch;
  readonly diagnostics?: BeamSplitterBranchMetrics;
  readonly onChange: (patch: Partial<BeamSplitterBranch>) => void;
  readonly onTransformChange: (stack: BeamSplitterTransformStep[]) => void;
  readonly onDuplicate: () => void;
  readonly onRemove: () => void;
}

function BranchDetailPanel({
  branch,
  diagnostics,
  onChange,
  onTransformChange,
  onDuplicate,
  onRemove,
}: BranchDetailProps) {
  const { t } = useI18n();
  const sourceOptions = useMemo(
    () => SOURCE_OPTION_DEFS.map((option) => ({ value: option.value, label: t(option.labelKey) })),
    [t],
  );
  const symmetryOptions = useMemo(
    () =>
      SYMMETRY_OPTION_DEFS.map((option) => ({ value: option.value, label: t(option.labelKey) })),
    [t],
  );
  return (
    <div className="beam-splitter-details">
      <header className="beam-splitter-details__header">
        <h4>{branch.label}</h4>
        <div className="beam-splitter-details__actions">
          <button type="button" className="ghost-button" onClick={onDuplicate}>
            {t('beamSplitter.actions.duplicate')}
          </button>
          <button type="button" className="ghost-button ghost-button--danger" onClick={onRemove}>
            {t('beamSplitter.common.remove')}
          </button>
        </div>
      </header>
      <div className="beam-splitter-details__grid">
        <label>
          {t('beamSplitter.fields.label')}
          <input
            type="text"
            value={branch.label}
            onChange={(event) => onChange({ label: event.target.value })}
          />
        </label>
        <label>
          {t('beamSplitter.fields.source')}
          <select
            value={branch.source ?? 'source'}
            onChange={(event) =>
              onChange({ source: event.target.value as BeamSplitterBranchSource })
            }
          >
            {sourceOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t('beamSplitter.fields.symmetry')}
          <select
            value={branch.symmetry ?? 'identity'}
            onChange={(event) => onChange({ symmetry: event.target.value })}
          >
            {symmetryOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t('beamSplitter.fields.weight')}
          <input
            type="number"
            step={0.05}
            min={0}
            value={branch.weight ?? 1}
            onChange={(event) =>
              onChange({ weight: clampWeight(Number.parseFloat(event.target.value)) })
            }
          />
        </label>
        <label>
          {t('beamSplitter.fields.priority')}
          <input
            type="number"
            step={1}
            min={0}
            value={branch.priority ?? 0}
            onChange={(event) =>
              onChange({ priority: clampPriority(Number.parseFloat(event.target.value)) })
            }
          />
        </label>
      </div>
      <TransformStackEditor stack={branch.transformStack} onChange={onTransformChange} />
      {diagnostics ? (
        <section className="beam-splitter-details__metrics">
          <h5>{t('beamSplitter.diagnostics.title')}</h5>
          <dl>
            <div>
              <dt>{t('beamSplitter.diagnostics.energy')}</dt>
              <dd>{diagnostics.energy.toFixed(3)}</dd>
            </div>
            <div>
              <dt>{t('beamSplitter.diagnostics.contribution')}</dt>
              <dd>
                {t('beamSplitter.diagnostics.percent', {
                  values: { value: (diagnostics.energyShare * 100).toFixed(1) },
                })}
              </dd>
            </div>
            <div>
              <dt>{t('beamSplitter.diagnostics.coverage')}</dt>
              <dd>
                {t('beamSplitter.diagnostics.percent', {
                  values: { value: (diagnostics.coverage * 100).toFixed(1) },
                })}
              </dd>
            </div>
            <div>
              <dt>{t('beamSplitter.diagnostics.occlusion')}</dt>
              <dd>
                {t('beamSplitter.diagnostics.percent', {
                  values: { value: (diagnostics.occlusion * 100).toFixed(1) },
                })}
              </dd>
            </div>
          </dl>
        </section>
      ) : (
        <p className="panel-muted">{t('beamSplitter.diagnostics.placeholder')}</p>
      )}
    </div>
  );
}

interface BeamSplitterNodeEditorProps {
  readonly node: SceneNode;
  readonly branches: BeamSplitterBranch[];
  readonly onCommitBranches: (branches: BeamSplitterBranch[]) => void;
  readonly onUpdateNodeParameters: (parameters: SceneNode['parameters']) => void;
}

function BeamSplitterNodeEditor({
  node,
  branches,
  onCommitBranches,
  onUpdateNodeParameters,
}: BeamSplitterNodeEditorProps) {
  const { t } = useI18n();
  const recombineOptions = useMemo(
    () =>
      RECOMBINE_OPTION_DEFS.map((option) => ({ value: option.value, label: t(option.labelKey) })),
    [t],
  );
  const diagnosticsEntries = useBeamSplitterDiagnostics(node.id);
  const diagnosticsMap = useMemo(() => {
    const latest = diagnosticsEntries[0];
    const map: BranchDiagnosticsMap = new Map();
    if (latest) {
      latest.branches.forEach((entry) => {
        map.set(entry.branchId, entry);
      });
    }
    return map;
  }, [diagnosticsEntries]);

  const [selectedBranchId, setSelectedBranchId] = useState<string | null>(
    branches.length > 0 ? branches[0].id : null,
  );

  useEffect(() => {
    if (!selectedBranchId || !branches.some((branch) => branch.id === selectedBranchId)) {
      setSelectedBranchId(branches[0]?.id ?? null);
    }
  }, [branches, selectedBranchId]);

  const recombineParameter = useMemo(
    () => node.parameters.find((parameter) => parameter.id === 'recombine'),
    [node.parameters],
  );
  const recombineMode =
    recombineParameter && typeof recombineParameter.value === 'string'
      ? recombineParameter.value
      : 'sum';
  const recombineLabel = useMemo(() => {
    const entry = recombineOptions.find((option) => option.value === recombineMode);
    return entry ? entry.label : t('beamSplitter.recombine.unknown');
  }, [recombineMode, recombineOptions, t]);

  const handleRecombineChange = useCallback(
    (value: string) => {
      const parameters = node.parameters.map((parameter) =>
        parameter.id === 'recombine' ? { ...parameter, value } : parameter,
      );
      onUpdateNodeParameters(parameters);
    },
    [node.parameters, onUpdateNodeParameters],
  );

  const handleBranchPatch = useCallback(
    (branchId: string, patch: Partial<BeamSplitterBranch>) => {
      const index = branches.findIndex((branch) => branch.id === branchId);
      if (index < 0) {
        return;
      }
      const next = branches.slice();
      next[index] = { ...next[index], ...patch };
      onCommitBranches(next);
      setSelectedBranchId(branchId);
    },
    [branches, onCommitBranches],
  );

  const handleTransformChange = useCallback(
    (branchId: string, stack: BeamSplitterTransformStep[]) => {
      handleBranchPatch(branchId, { transformStack: ensureTransformStack(stack) });
    },
    [handleBranchPatch],
  );

  const handleAddBranch = useCallback(() => {
    const branch = createBranch(branches.length);
    const next = [...branches, branch];
    onCommitBranches(next);
    setSelectedBranchId(branch.id);
  }, [branches, onCommitBranches]);

  const handleDuplicateBranch = useCallback(
    (branchId: string) => {
      const index = branches.findIndex((branch) => branch.id === branchId);
      if (index < 0) return;
      const source = branches[index];
      const clone = createBranch(branches.length);
      const duplicate: BeamSplitterBranch = {
        ...clone,
        label: `${source.label} copy`,
        symmetry: source.symmetry,
        weight: source.weight,
        priority: branches.length,
        source: source.source,
        transformStack: source.transformStack.map((step) => ({ ...step })),
      };
      const next = [...branches.slice(0, index + 1), duplicate, ...branches.slice(index + 1)];
      onCommitBranches(next);
      setSelectedBranchId(duplicate.id);
    },
    [branches, onCommitBranches],
  );

  const handleRemoveBranch = useCallback(
    (branchId: string) => {
      const index = branches.findIndex((branch) => branch.id === branchId);
      if (index < 0) return;
      const next = branches.filter((branch) => branch.id !== branchId);
      const fallback = next.length > 0 ? next[Math.max(0, index - 1)].id : null;
      onCommitBranches(next);
      setSelectedBranchId(fallback ?? null);
    },
    [branches, onCommitBranches],
  );

  const activeBranch =
    selectedBranchId != null
      ? (branches.find((branch) => branch.id === selectedBranchId) ?? null)
      : null;

  return (
    <section className="beam-splitter-editor">
      <header className="beam-splitter-editor__header">
        <div>
          <h4>{node.label}</h4>
          <span className="beam-splitter-editor__subtitle">
            {branches.length === 1
              ? t('beamSplitter.summary.single', {
                  values: { count: branches.length, mode: recombineLabel },
                })
              : t('beamSplitter.summary.plural', {
                  values: { count: branches.length, mode: recombineLabel },
                })}
          </span>
        </div>
        <label className="beam-splitter-editor__recombine">
          {t('beamSplitter.recombine.label')}
          <select
            value={recombineMode}
            onChange={(event) => handleRecombineChange(event.target.value)}
          >
            {recombineOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </header>
      <div className="beam-splitter-editor__grid">
        <BranchGraph
          branches={branches}
          diagnostics={diagnosticsMap}
          selectedBranchId={selectedBranchId}
          recombineMode={recombineMode}
          onSelectBranch={setSelectedBranchId}
          onAddBranch={handleAddBranch}
        />
        {activeBranch ? (
          <BranchDetailPanel
            branch={activeBranch}
            diagnostics={diagnosticsMap.get(activeBranch.id)}
            onChange={(patch) => handleBranchPatch(activeBranch.id, patch)}
            onTransformChange={(stack) => handleTransformChange(activeBranch.id, stack)}
            onDuplicate={() => handleDuplicateBranch(activeBranch.id)}
            onRemove={() => handleRemoveBranch(activeBranch.id)}
          />
        ) : (
          <div className="beam-splitter-details beam-splitter-details--empty">
            <p className="panel-muted">{t('beamSplitter.empty')}</p>
          </div>
        )}
      </div>
    </section>
  );
}

export function BeamSplitterEditor() {
  const { scene, updateNode } = useSceneGraph();
  const { t } = useI18n();
  const splitterNodes = useMemo(
    () => scene.nodes.filter((node) => node.type === 'BeamSplitter'),
    [scene.nodes],
  );

  const commitBranches = useCallback(
    (node: SceneNode, branches: BeamSplitterBranch[]) => {
      const metadata = isRecord(node.metadata) ? { ...node.metadata } : {};
      metadata.branches = serialiseBranches(branches);
      const parameters = node.parameters.map((parameter) =>
        parameter.id === 'branchCount' ? { ...parameter, value: branches.length } : parameter,
      );
      updateNode(node.id, {
        metadata,
        parameters,
      });
    },
    [updateNode],
  );

  const handleCommit = useCallback(
    (node: SceneNode) => (branches: BeamSplitterBranch[]) => {
      commitBranches(node, branches);
    },
    [commitBranches],
  );

  const handleParametersUpdate = useCallback(
    (node: SceneNode) => (parameters: SceneNode['parameters']) => {
      updateNode(node.id, { parameters });
    },
    [updateNode],
  );

  if (splitterNodes.length === 0) {
    return null;
  }

  return (
    <div className="beam-splitter-editor__container">
      <h3>{t('beamSplitter.title')}</h3>
      {splitterNodes.map((node) => {
        const branches = parseBranches(node.metadata ?? {});
        return (
          <BeamSplitterNodeEditor
            key={node.id}
            node={node}
            branches={branches}
            onCommitBranches={handleCommit(node)}
            onUpdateNodeParameters={handleParametersUpdate(node)}
          />
        );
      })}
    </div>
  );
}
