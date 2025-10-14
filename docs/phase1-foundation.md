# Indra PhaseÂ 1 â€“ Foundation Deliverable

This document captures the scaffolding established in PhaseÂ 1 so the team can iterate quickly on upcoming feature work.

## Application Shell

- Vite + React render a four-region workspace (`src/app/layout/AppLayout.tsx`), covering presets, viewport, inspector, and timeline.
- Panels can be shown/hidden via the toolbar and resized with pointer handles; layout state persists in a central store (`src/state/AppState.tsx`).
- A bundled manifest (`public/sample-manifest.json`) exercises the plumbing end to end, populating the node graph, timeline clips, and panel readouts.

## State Management & Manifest Flow

- `AppStateProvider` exposes scene graph, timeline, GPU, and manifest status slices that both the UI and runtime hooks share.
- `src/manifest/schema.ts` validates manifests, enforcing node/link referential integrity and surfacing warnings for non-fatal issues.
- `src/manifest/runtime.ts` normalises validated manifests into runtime objects consumed by the store.
- `PresetsPanel` wires a file picker and a one-click sample loader through `useManifestLoader`, which reuses the same code path as the CLI.
- `tests/manifestLoader.test.ts` guards the loader/validator path with the sample manifest and an invalid-case regression.

## CLI Stub

- `src/cli/indraCli.ts` delivers `indra-cli validate <manifest.json>`.
- Build with `npm run cli:build`; invoke locally via `npm run indra-cli -- validate ./public/sample-manifest.json`.
- `--json` mode emits machine-readable validation details for automation.

## GPU Context & Apple Silicon Notes

- `useViewportRenderer` initialises WebGPU when available and falls back to WebGL2 otherwise. A dummy render pass clears the surface at 120Â Hz so future shader work can slot in without touching the plumbing.
- The status bar reports the active backend, detected adapter, and the configured refresh budget.
- On AppleÂ Silicon (M1/M2), the WebGPU path prefers a high-performance adapter and sizes the canvas with device pixel ratio awareness. This keeps the swap chain aligned with ProMotion (120Â Hz) panels.
- Electron builds should retain the same initialisation path; when packaging, ensure the app bundle has the `com.apple.developer.kernel.extended-virtual-addressing` entitlement if native modules require it.
- Use Xcode Instruments (Metal System Trace + Core Animation) to confirm frame pacing stays under the 8.3Â ms target once the real renderer lands.

## Next Steps

- Layer timeline editing controls over the existing timeline panel.
- Expand the manifest schema with parameter domains and node metadata once node implementations exist.
- Swap the placeholder render loop with the real shader pipeline and bridge telemetry into the inspector panel.
