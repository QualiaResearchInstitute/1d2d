# Developer Guide

This guide targets contributors who need to extend or maintain the Indra codebase. It summarises architecture, build tooling, and extension patterns introduced in Phase� 10.

## Repository layout

```
src/                 — React UI, core physics kernels, rendering pipeline
src/cli/             — CLI entry points and helpers
src/runtime/         — Shared orchestration services used by CLI + REST
src/server/          — Local REST/IPC API implementation
sdk/typescript/      — TypeScript SDK
sdk/python/          — Python SDK
docs/                — Guides, references, tutorials
examples/            — Ready-to-run scripts and notebooks
```

## Build targets

- `npm run build` – UI bundle
- `npm run cli:build` – CLI + server bundle (`dist-cli/`)
- `npm run test` – unit tests
- `npm run coverage` – coverage suite
- `npm run baseline` – regression harness for optics kernels

### CLI bundle

The CLI is compiled via `tsconfig.cli.json`. Inputs are `src/cli/**`, `src/runtime/**`, and supporting modules. The emitted binaries live under `dist-cli/` and are referenced by the `indra-cli` bin entry in `package.json`.

### REST server

`src/server/index.ts` depends exclusively on the runtime orchestrator (`src/runtime/services.ts`). The build process emits `dist-cli/server/index.js`, which can be spawned directly (`node dist-cli/server/index.js`).

## Extensibility

### Adding CLI commands

1. Extend `src/runtime/services.ts` with the core functionality (keep I/O agnostic).
2. Update `src/cli/indraCli.ts` to parse flags and call the new service function.
3. Document the command in `docs/reference/cli.md` and add coverage in `tests/` if applicable.

### REST endpoints

1. Add a new branch in the routing switch inside `src/server/index.ts`.
2. Reuse the same service function used by the CLI to keep behaviour consistent.
3. Document the endpoint in `docs/reference/rest.md`.

### SDKs

Both SDKs are intentionally thin; they pass requests straight to the REST server. To expose new functionality, add the endpoint first, then wrap it inside:

- `sdk/typescript/src/index.ts`
- `sdk/python/indra_sdk/client.py`

## Testing

- Core physics and rendering logic is covered by existing tests under `tests/`.
- Add new tests for CLI/runtime features in `tests/cli/` (create directory as needed).
- SDK smoke tests can be placed in `examples/` and referenced by CI scripts.

## Release checklist

1. Run `npm run build`, `npm run cli:build`, and `npm run test`.
2. Ensure `docs/manual/user-guide.md` and `docs/reference/*` are up to date.
3. Bump SDK versions in their respective manifests if publishing.
4. Tag the release (`v1.0.0-rc` etc.) and attach the generated documentation.

## Support scripts

- `scripts/videoPipeline.ts` remains available for legacy pipelines and can be re-used to prototype new offline workflows.
- `scripts/tools/esm-extension-loader.mjs` provides loader glue for Node when running TypeScript directly.

## Conventions

- Source code uses TypeScript `strict` mode. Prefer pure functions in `src/runtime/` to remain platform agnostic.
- Avoid introducing new dependencies without updating the documentation and licensing sections.
- Documentation is Markdown-first; keep line lengths under 120 characters.
