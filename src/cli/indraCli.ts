#!/usr/bin/env node
import { createWriteStream } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';
import { WebSocketServer } from 'ws';

import { diffValues } from './utils/diff.js';
import { applyFrame, captureVideo, simulate } from '../runtime/services.js';

const exitWithError = (message: string): never => {
  console.error(message);
  process.exit(1);
};

const printMainUsage = () => {
  console.log(`indra-cli – unified CLI for the Indra holographic studio

Commands:
  manifest validate <manifest.json> [--json] [--verbose]
  manifest diff <a.json> <b.json> [--json]
  apply --input <media> --output <image> [--manifest <path>] [--preset <id>]
  simulate --input <image> --frames 120 [--manifest <path>] [--preset <id>] [--output metrics.json]
  capture --input <video> --output <video> [--manifest <path>] [--preset <id>]
  telemetry [--port 8090] [--output metrics.jsonl]

Run "indra-cli <command> --help" to learn more about a command.`);
};

const printManifestUsage = () => {
  console.log(`indra-cli manifest – manifest utilities

Usage:
  indra-cli manifest validate <manifest.json> [--json] [--verbose]
  indra-cli manifest diff <a.json> <b.json> [--json]
`);
};

const printApplyUsage = () => {
  console.log(`indra-cli apply

Process a single image with the Rainbow Rims pipeline and write a processed frame.

Required:
  --input <image>        Input image or video frame (any ffmpeg-supported format)
  --output <image>       Output image path (.png recommended)

Optional:
  --manifest <path>      Manifest JSON to source presets and defaults
  --preset <id>          Preset identifier inside the manifest controls
  --ffmpeg <path>        ffmpeg executable (default "ffmpeg")
  --ffprobe <path>       ffprobe executable (default "ffprobe")
  --bit-depth <8|10|16>  Output bit depth (default 8)
  --json                 Emit metrics JSON instead of human-readable summary
`);
};

const printSimulateUsage = () => {
  console.log(`indra-cli simulate

Run an offline simulation (headless) and emit aggregated metrics.

Required:
  --input <image>             Input image for rim/surface derivation

Optional:
  --manifest <path>           Manifest JSON to source presets
  --preset <id>               Preset identifier from manifest
  --frames <count>            Frame count (default 120)
  --dt <seconds>              Step duration (default 1/60)
  --seed <number>             Seed for Kuramoto noise (default 1337)
  --ffmpeg <path>             ffmpeg executable for input decode (default "ffmpeg")
  --ffprobe <path>            ffprobe executable (default "ffprobe")
  --output <metrics.json>     Write aggregated metrics to file
  --json                      Print metrics JSON to stdout
`);
};

const printCaptureUsage = () => {
  console.log(`indra-cli capture

Render a video by applying the Rainbow Rims pipeline to each frame of an input sequence.

Required:
  --input <video>        Source video (any ffmpeg-supported format)
  --output <video>       Output video path (e.g. out.mp4)

Optional:
  --manifest <path>      Manifest JSON defining presets
  --preset <id>          Preset identifier from the manifest
  --ffmpeg <path>        ffmpeg executable (default "ffmpeg")
  --ffprobe <path>       ffprobe executable (default "ffprobe")
  --frames <count>       Limit processed frames (default: all)
  --keep-temp            Keep intermediate decoded frames for inspection
  --json                 Emit processing summary as JSON
`);
};

const printTelemetryUsage = () => {
  console.log(`indra-cli telemetry

Starts a lightweight WebSocket endpoint that captures telemetry frames and optionally writes them to a JSONL file.

Flags:
  --port <number>     Port to listen on (default 8090)
  --output <path>     Path to append newline-delimited JSON frames

Example:
  indra-cli telemetry --port 8090 --output metrics.jsonl
`);
};

const handleManifestCommand = async (args: string[]) => {
  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    printManifestUsage();
    process.exit(0);
  }

  const [subcommand, ...rest] = args;
  if (subcommand === 'validate') {
    if (rest.length === 0) {
      exitWithError('manifest validate requires a manifest path.');
    }
    const flags = new Set(rest.filter((arg) => arg.startsWith('--')));
    const manifestPath = rest.find((arg) => !arg.startsWith('--')) ?? rest[0];
    if (!manifestPath) {
      exitWithError('manifest validate requires a manifest path.');
    }
    const payload = await readFile(resolve(process.cwd(), manifestPath), 'utf8');
    const result = await (
      await import('../manifest/loader.js')
    ).loadManifestFromJson(payload, manifestPath);
    if (result.kind === 'success') {
      if (flags.has('--json')) {
        console.log(
          JSON.stringify(
            {
              status: 'ok',
              manifest: {
                name: result.manifest.metadata.name,
                schemaVersion: result.manifest.schemaVersion,
                nodes: result.manifest.nodes.length,
                links: result.manifest.links.length,
              },
              warnings: result.issues.filter((issue) => issue.severity === 'warning'),
            },
            null,
            2,
          ),
        );
      } else {
        console.log(`✔ Manifest valid: ${manifestPath}`);
        console.log(`  schema:  ${result.manifest.schemaVersion}`);
        console.log(
          `  scene:   ${result.manifest.nodes.length} nodes, ${result.manifest.links.length} links`,
        );
        if (result.manifest.timeline) {
          console.log(
            `  timeline: ${result.manifest.timeline.duration}s @ ${result.manifest.timeline.fps}fps, ${result.manifest.timeline.clips.length} clips`,
          );
        } else {
          console.log('  timeline: not defined');
        }
        if (result.issues.length > 0 && flags.has('--verbose')) {
          console.warn('Warnings:');
          result.issues
            .filter((issue) => issue.severity === 'warning')
            .forEach((issue) => {
              console.warn(`  • ${issue.message} (${issue.code})`);
            });
        }
      }
      return;
    }
    if (flags.has('--json')) {
      console.log(
        JSON.stringify(
          {
            status: 'error',
            message: result.message,
            issues: result.issues,
          },
          null,
          2,
        ),
      );
    } else {
      console.error(`✖ Manifest invalid: ${manifestPath}`);
      console.error(`  ${result.message}`);
      result.issues?.forEach((issue) => {
        console.error(`   • ${issue.message} (${issue.code} @ ${issue.path.join('.')})`);
      });
    }
    process.exit(1);
  } else if (subcommand === 'diff') {
    if (rest.length < 2) {
      exitWithError('manifest diff requires two manifest paths.');
    }
    const positional = rest.filter((arg) => !arg.startsWith('--'));
    if (positional.length < 2) {
      exitWithError('manifest diff requires two manifest paths.');
    }
    const flags = new Set(rest.filter((arg) => arg.startsWith('--')));
    const [leftPath, rightPath] = positional;
    const leftRaw = await readFile(resolve(process.cwd(), leftPath), 'utf8');
    const rightRaw = await readFile(resolve(process.cwd(), rightPath), 'utf8');
    const left = JSON.parse(leftRaw);
    const right = JSON.parse(rightRaw);
    const diff = diffValues(left, right);
    if (flags.has('--json')) {
      console.log(JSON.stringify({ status: 'ok', changes: diff }, null, 2));
    } else if (diff.length === 0) {
      console.log('No differences detected.');
    } else {
      console.log(`Found ${diff.length} difference(s):`);
      diff.slice(0, 100).forEach((entry) => {
        if (entry.kind === 'added') {
          console.log(` + ${entry.path} = ${JSON.stringify(entry.value)}`);
        } else if (entry.kind === 'removed') {
          console.log(` - ${entry.path} = ${JSON.stringify(entry.value)}`);
        } else {
          console.log(
            ` ~ ${entry.path}: ${JSON.stringify(entry.left)} → ${JSON.stringify(entry.right)}`,
          );
        }
      });
      if (diff.length > 100) {
        console.log(' (truncated)');
      }
    }
    return;
  }

  exitWithError(`Unknown manifest subcommand "${subcommand}".`);
};

const handleApplyCommand = async (args: string[]) => {
  if (args.includes('--help') || args.includes('-h')) {
    printApplyUsage();
    process.exit(0);
  }

  const options: {
    input?: string;
    output?: string;
    manifest?: string;
    preset?: string;
    ffmpeg: string;
    ffprobe: string;
    bitDepth: 8 | 10 | 16;
    json: boolean;
  } = {
    ffmpeg: 'ffmpeg',
    ffprobe: 'ffprobe',
    bitDepth: 8,
    json: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg) continue;
    if (!arg.startsWith('--')) {
      if (!options.input) {
        options.input = arg;
      } else if (!options.output) {
        options.output = arg;
      }
      continue;
    }
    switch (arg) {
      case '--input':
        options.input = args[++i];
        break;
      case '--output':
        options.output = args[++i];
        break;
      case '--manifest':
        options.manifest = args[++i];
        break;
      case '--preset':
        options.preset = args[++i];
        break;
      case '--ffmpeg':
        options.ffmpeg = args[++i] ?? options.ffmpeg;
        break;
      case '--ffprobe':
        options.ffprobe = args[++i] ?? options.ffprobe;
        break;
      case '--bit-depth': {
        const value = Number(args[++i]);
        if (value === 8 || value === 10 || value === 16) {
          options.bitDepth = value;
        } else {
          exitWithError(`Unsupported bit depth "${args[i]}". Use 8, 10, or 16.`);
        }
        break;
      }
      case '--json':
        options.json = true;
        break;
      default:
        exitWithError(`Unknown flag "${arg}"`);
    }
  }

  if (!options.input || !options.output) {
    exitWithError('apply requires --input and --output.');
  }

  const summary = await applyFrame({
    input: options.input,
    output: options.output,
    ffmpeg: options.ffmpeg,
    ffprobe: options.ffprobe,
    manifest: options.manifest,
    preset: options.preset,
    bitDepth: options.bitDepth,
  });

  if (options.json) {
    console.log(
      JSON.stringify(
        {
          status: 'ok',
          ...summary,
        },
        null,
        2,
      ),
    );
  } else {
    console.log(
      `[apply] wrote ${summary.output} (${summary.width}×${summary.height}, rim=${summary.metrics.rimMean.toFixed(3)}, |Z|=${summary.metrics.coherenceMean.toFixed(3)})`,
    );
    if (summary.telemetry.metrics.edgePixelCount != null) {
      console.log(
        `         edge pixels ${summary.telemetry.metrics.edgePixelCount} | durations edge=${summary.telemetry.durations.edgeMs?.toFixed(2)}ms phase=${summary.telemetry.durations.phaseMs?.toFixed(2)}ms`,
      );
    }
  }
};

const handleSimulateCommand = async (args: string[]) => {
  if (args.includes('--help') || args.includes('-h')) {
    printSimulateUsage();
    process.exit(0);
  }

  const options: {
    input?: string;
    manifest?: string;
    preset?: string;
    frames: number;
    dt: number;
    seed: number;
    ffmpeg: string;
    ffprobe: string;
    output?: string;
    json: boolean;
  } = {
    frames: 120,
    dt: 1 / 60,
    seed: 1337,
    ffmpeg: 'ffmpeg',
    ffprobe: 'ffprobe',
    json: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg) continue;
    if (!arg.startsWith('--')) {
      if (!options.input) {
        options.input = arg;
      }
      continue;
    }
    switch (arg) {
      case '--input':
        options.input = args[++i];
        break;
      case '--manifest':
        options.manifest = args[++i];
        break;
      case '--preset':
        options.preset = args[++i];
        break;
      case '--frames':
        options.frames = Math.max(1, Number.parseInt(args[++i] ?? '0', 10));
        break;
      case '--dt':
        options.dt = Number(args[++i] ?? options.dt);
        break;
      case '--seed':
        options.seed = Number(args[++i] ?? options.seed);
        break;
      case '--ffmpeg':
        options.ffmpeg = args[++i] ?? options.ffmpeg;
        break;
      case '--ffprobe':
        options.ffprobe = args[++i] ?? options.ffprobe;
        break;
      case '--output':
        options.output = args[++i];
        break;
      case '--json':
        options.json = true;
        break;
      default:
        exitWithError(`Unknown flag "${arg}"`);
    }
  }

  if (!options.input) {
    exitWithError('simulate requires --input');
  }

  const summary = await simulate({
    input: options.input,
    ffmpeg: options.ffmpeg,
    ffprobe: options.ffprobe,
    manifest: options.manifest,
    preset: options.preset,
    frames: options.frames,
    dt: options.dt,
    seed: options.seed,
  });

  const payload = {
    status: 'ok',
    ...summary,
  };

  if (options.output) {
    await writeFile(options.output, JSON.stringify(payload, null, 2), 'utf8');
  }

  if (options.json) {
    console.log(JSON.stringify(payload, null, 2));
  } else {
    console.log(
      `[simulate] ${summary.frames} frames — rim mean ${summary.metrics.rimMean.toFixed(3)}, |Z| mean ${summary.metrics.cohMean.toFixed(3)}, Indra index ${summary.metrics.indraIndex.toFixed(3)}`,
    );
    if (options.output) {
      console.log(`            metrics written to ${resolve(process.cwd(), options.output)}`);
    }
  }
};

const handleCaptureCommand = async (args: string[]) => {
  if (args.includes('--help') || args.includes('-h')) {
    printCaptureUsage();
    process.exit(0);
  }
  const options: {
    input?: string;
    output?: string;
    manifest?: string;
    preset?: string;
    ffmpeg: string;
    ffprobe: string;
    framesLimit?: number;
    keepTemp: boolean;
    json: boolean;
  } = {
    ffmpeg: 'ffmpeg',
    ffprobe: 'ffprobe',
    keepTemp: false,
    json: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg) continue;
    if (!arg.startsWith('--')) {
      if (!options.input) options.input = arg;
      else if (!options.output) options.output = arg;
      continue;
    }
    switch (arg) {
      case '--input':
        options.input = args[++i];
        break;
      case '--output':
        options.output = args[++i];
        break;
      case '--manifest':
        options.manifest = args[++i];
        break;
      case '--preset':
        options.preset = args[++i];
        break;
      case '--ffmpeg':
        options.ffmpeg = args[++i] ?? options.ffmpeg;
        break;
      case '--ffprobe':
        options.ffprobe = args[++i] ?? options.ffprobe;
        break;
      case '--frames':
        options.framesLimit = Math.max(1, Number.parseInt(args[++i] ?? '0', 10));
        break;
      case '--keep-temp':
        options.keepTemp = true;
        break;
      case '--json':
        options.json = true;
        break;
      default:
        exitWithError(`Unknown flag "${arg}"`);
    }
  }

  if (!options.input || !options.output) {
    exitWithError('capture requires --input and --output');
  }

  const summary = await captureVideo({
    input: options.input,
    output: options.output,
    manifest: options.manifest,
    preset: options.preset,
    ffmpeg: options.ffmpeg,
    ffprobe: options.ffprobe,
    framesLimit: options.framesLimit,
    keepTemp: options.keepTemp,
  });

  if (options.json) {
    console.log(JSON.stringify({ status: 'ok', ...summary }, null, 2));
  } else {
    console.log(
      `[capture] wrote ${summary.output} (${summary.frames} frames ${summary.width}×${summary.height})`,
    );
  }
};

const runTelemetryServer = async (args: string[]) => {
  let port = 8090;
  let outputPath: string | null = null;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (!arg) {
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      printTelemetryUsage();
      process.exit(0);
    }
    if (arg === '--port' && args[index + 1]) {
      const value = Number.parseInt(args[index + 1], 10);
      if (Number.isFinite(value) && value > 0) {
        port = value;
      } else {
        console.warn(`Ignoring invalid port value "${args[index + 1]}"`);
      }
      index += 1;
      continue;
    }
    if (arg === '--output' && args[index + 1]) {
      outputPath = args[index + 1];
      index += 1;
      continue;
    }
    if (arg.startsWith('--')) {
      console.warn(`Unknown flag ${arg}`);
    } else {
      console.warn(`Ignoring positional argument "${arg}"`);
    }
  }

  let writer: ReturnType<typeof createWriteStream> | null = null;
  if (outputPath) {
    const resolved = resolve(process.cwd(), outputPath);
    writer = createWriteStream(resolved, { flags: 'a' });
    console.log(`Appending telemetry frames to ${resolved}`);
  } else {
    console.log('No output file provided; frames will be written to stdout.');
  }

  const server = new WebSocketServer({ port });
  server.on('connection', (socket) => {
    console.log('[telemetry] client connected');
    socket.on('message', (data) => {
      const text = typeof data === 'string' ? data : data.toString('utf8');
      if (writer) {
        writer.write(text);
        writer.write('\n');
      } else {
        console.log(text);
      }
    });
    socket.on('close', () => {
      console.log('[telemetry] client disconnected');
    });
  });

  server.on('listening', () => {
    console.log(`[telemetry] listening on ws://localhost:${port}/telemetry`);
  });

  server.on('error', (error) => {
    console.error('[telemetry] server error', error);
    writer?.close();
    process.exit(1);
  });

  const shutdown = () => {
    console.log('\nShutting down telemetry server…');
    server.close();
    writer?.close();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
};

const main = async () => {
  const [, , ...argv] = process.argv;
  if (argv.length === 0 || argv[0] === '--help' || argv[0] === '-h') {
    printMainUsage();
    process.exit(0);
  }
  const [command, ...rest] = argv;
  switch (command) {
    case 'manifest':
      await handleManifestCommand(rest);
      break;
    case 'apply':
      await handleApplyCommand(rest);
      break;
    case 'simulate':
      await handleSimulateCommand(rest);
      break;
    case 'capture':
      await handleCaptureCommand(rest);
      break;
    case 'telemetry':
      await runTelemetryServer(rest);
      break;
    default:
      exitWithError(`Unknown command "${command}".`);
  }
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
