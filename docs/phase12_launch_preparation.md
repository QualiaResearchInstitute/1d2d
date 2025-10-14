# Phase 12 Launch Preparation & Roadmap

Phase 12 packages the 1d2d Perceptual Synthesis Studio for public release, ships the documentation/marketing stack, and establishes the community + telemetry loops that sustain the project after v1.0. Treat this playbook as the launch gate: it scopes the artefacts to produce, where to store them, and the process checks that keep the first public impression polished.

---

## Deliverables Snapshot

- **Signed desktop installers** – Universal (arm64 + x64) notarised builds staged under `docs/release/phase12/mac/` with signature logs and first-run screenshots.
- **Standalone CLI bundle** – Zipped binaries + shims in `docs/release/phase12/cli/` alongside checksum manifest and smoke-test logs.
- **Public landing site + docs** – Live site (GitHub Pages or static hosting) backed by `docs/site/` source, Quick Start, and linked API/manual references.
- **Tutorial media & launch announcement** – Recorded walkthrough(s), hero imagery, and a written announcement in `docs/launch/phase12/announcement.md`.
- **Support & feedback channels** – GitHub Issues labeling guide, community chat invites, in-app/menu deep links, and FAQ seeded at `docs/support/`.
- **Opt-in telemetry plan** – UX copy, instrumentation toggles, storage pipeline, and monitoring dashboard notes in `docs/telemetry/phase12/`.
- **Post-launch roadmap** – Public backlog primer stored at `docs/roadmap/postlaunch.md` and linked from the site/README.

---

## Release Packaging Pipeline

### Desktop Build (macOS DMG / .pkg)

1. **Prerequisites**
   - Verify Apple Developer ID certificates are installed on the build host (`security find-identity -p codesigning`).
   - Update `scripts/package/mac/package.json` (if applicable) with `version` = `1.0.0`.
2. **Create universal binary**
   - Build both `arm64` and `x64` variants:
     ```bash
     npm run build:desktop -- --arch=arm64
     npm run build:desktop -- --arch=x64
     ```
   - Combine with `lipo` into `dist/mac/1d2d.app`.
3. **Sign + notarise**
   - Codesign app bundle:
     ```bash
     codesign --deep --force --options runtime \
       --sign "Developer ID Application: <Org Name>" \
       dist/mac/1d2d.app
     ```
   - Create DMG via `scripts/package/mac/build_dmg.sh`.
   - Submit for notarisation (`xcrun notarytool submit ... --wait`); archive tickets in `docs/release/phase12/mac/notary/`.
4. **Gatekeeper verification**
   - Mount the notarised DMG on a clean macOS VM.
   - Record first-run (screenshot + screen capture) and stash under `docs/release/phase12/mac/first-run/`.
5. **Distribution**
   - Publish checksums in `docs/release/phase12/mac/checksums.txt`; mirror in the release notes and landing site download table.

### CLI Binary Distribution

1. Build the CLI with static deps:
   ```bash
   npm run build:cli
   pkg dist-cli/index.js --targets node18-macos-arm64,node18-macos-x64,node18-linux-x64 --out-path dist/cli
   ```
2. For each target bundle:
   - Strip symbols (`strip dist/cli/<target>`).
   - Generate SHA256 checksums → `docs/release/phase12/cli/checksums.txt`.
   - Include a `README.md` referencing usage, install instructions, and license.
3. Smoke-test on clean environments (macOS arm64/x64 VMs, Ubuntu runner). Capture command transcripts in `docs/release/phase12/cli/smoke-tests.md`.
4. Optional: create a Homebrew tap definition (`Formula/indra-cli.rb`) and stage it in `docs/release/phase12/cli/homebrew/`.

### Web / Electron Deployment

- Confirm `npm run build` outputs production assets under `dist/`.
- For static hosting:
  - Sync `dist/` to the target bucket (e.g., GitHub Pages via `dist/site/` branch).
  - Run Lighthouse (`npm run audit:lighthouse`) and store the HTML report at `docs/release/phase12/web/lighthouse.html`.
- For Electron wrapper:
  - Ensure auto-update channel points to the tagged GitHub release.
  - Validate that the bundled CLI is accessible (`app.getAppPath()` → `resources/cli`).

---

## Documentation, Website & Launch Content

### Landing Site Structure

- Source under `docs/site/` with build pipeline (`npm run docs:build`) pushing to `gh-pages` or a static host.
- Required sections:
  - Hero value proposition (symmetry + dynamics + optics unification).
  - Download cards for Desktop & CLI with platform badges and checksums.
  - Quick Start (top 3 tasks): load canonical preset, tweak Kuramoto parameter, export a render.
  - Embedded or linked tutorial video.
  - Links to manual (`docs/manual/user-guide.md`), reference (`docs/reference/`), SDK (`docs/reference/sdk-typescript.md`), and REST docs.
- Store final build artefacts in `docs/site/dist/` with `site-build.log`.

### Tutorial Media

- Capture a ≤2 min guided screen recording (macOS QuickTime or OBS).
- Script outline stored at `docs/launch/phase12/tutorial-script.md`.
- Export final video to `docs/launch/phase12/media/launch-demo.mp4` plus GIF excerpts for social (`launch-demo.gif`).
- Generate caption file (`.vtt`) and transcript for accessibility.

### Launch Announcement

- Draft announcement in `docs/launch/phase12/announcement.md` covering:
  - Why the studio matters.
  - Headline features of v1.0.
  - Getting started steps.
  - Links to downloads, docs, tutorial video, support channels.
- When ready, mirror the copy to the project README and publish via blog/newsletter.

---

## Feedback & Support Channels

- **GitHub Issues**
  - Create issue templates (`.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`) if absent.
  - Publish a labeling matrix in `docs/support/labels.md` mapping severity/priority.
  - Link to Issues via in-app menu (`Help → Report an Issue`) calling `shell.openExternal`.
- **Community Chat**
  - Stand up Discord or Slack; document rules + onboarding in `docs/support/community-guide.md`.
  - Add invite link to landing site footer and README.
- **FAQ**
  - Seed `docs/support/faq.md` with troubleshooting topics (Gatekeeper warnings, missing codecs, CLI PATH setup).
- **Support Workflow**
  - Define triage cadence (`docs/support/runbook.md`): response SLA, escalation path, release hotfix protocol.

---

## Telemetry & Analytics (Opt-in)

1. **Consent UX**
   - First-run modal copy stored in `docs/telemetry/phase12/consent.md`.
   - Default to opt-out; persist user choice in app settings (`config/telemetry.json`).
2. **Data Model**
   - Capture anonymised metrics only (session duration, preset usage counts, crash signatures). Document schema in `docs/telemetry/phase12/schema.md`.
3. **Pipelines**
   - Ship data via HTTPS webhook or queued file; ensure retries/backoff.
   - Provide dry-run toggle for dev builds to avoid pollution.
4. **Dashboard**
   - Stand up simple Superset / Metabase (or static report) documented in `docs/telemetry/phase12/dashboard.md`.
   - Include alert thresholds for crash spikes or telemetry drop-offs.
5. **Privacy Compliance**
   - Document data retention policy and removal workflow in `docs/telemetry/phase12/policy.md`.

---

## Post-Launch Operations & Roadmap

- Maintain `docs/roadmap/postlaunch.md`:
  - Section for `v1.1` quick wins (preset library expansion, shader editor UX).
  - Medium-term goals (real-time collaboration, VR viewport, GPU cluster rendering).
  - Stretch targets (plugin marketplace, AR overlay, AI-assisted preset suggestions).
- Link roadmap in README + landing site to set expectations and attract contributors.
- Schedule fortnightly feedback synthesis; log outcomes in `docs/support/feedback-digests/`.
- Track crash/telemetry trends in a `docs/telemetry/phase12/weekly-report-template.md`.

---

## Launch Day Checklist

- Freeze `main` branch; tag `v1.0.0` once smoke + acceptance tests pass.
- Publish release artifacts (DMG, CLI zip, checksums) on GitHub Releases.
- Flip landing site downloads to point at the tagged assets.
- Announce via mailing list, social, community chat, and README update.
- Monitor support channels + telemetry dashboard for the first 72 h; log incidents in `docs/support/launch-watch.md`.

---

## Exit Criteria

- [ ] DMG/pkg notarised, codesign verified (`spctl --assess`) on clean macOS arm64 and x64 machines.
- [ ] CLI binaries hashed, smoke-tested on macOS + Linux; checksums published.
- [ ] Landing site deployed with functioning Quick Start, docs links, and download mirrors.
- [ ] Tutorial video, GIFs, and announcement copy reviewed and staged.
- [ ] Issue templates, community guide, FAQ, and support runbook live.
- [ ] Telemetry opt-in flow implemented, storing consent + pushing anonymised metrics to the dashboard.
- [ ] Post-launch roadmap published and referenced in README/site.
- [ ] Launch watch plan active with owners assigned for support + analytics monitoring.

When every box is ticked, Phase 12 is complete and the project is cleared for the public v1.0 release.
