import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';

import { applyFrame, captureVideo, simulate } from '../runtime/services.js';

type JsonValue = Record<string, unknown>;

const readJsonBody = async (req: IncomingMessage): Promise<JsonValue> => {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  if (chunks.length === 0) {
    return {};
  }
  const payload = Buffer.concat(chunks).toString('utf8');
  try {
    return JSON.parse(payload) as JsonValue;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON payload: ${message}`);
  }
};

const writeJson = (res: ServerResponse, status: number, body: JsonValue) => {
  const payload = JSON.stringify(body, null, 2);
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Content-Length', Buffer.byteLength(payload, 'utf8'));
  res.end(payload);
};

const validateManifestPayload = async (body: JsonValue) => {
  if (typeof body.manifestPath === 'string') {
    const manifestContent = await readFile(resolve(process.cwd(), body.manifestPath), 'utf8');
    const loader = await import('../manifest/loader.js');
    return loader.loadManifestFromJson(manifestContent, body.manifestPath);
  }
  if (typeof body.manifest === 'object' && body.manifest !== null) {
    const loader = await import('../manifest/loader.js');
    return loader.loadManifestFromJson(JSON.stringify(body.manifest), '<inline>');
  }
  throw new Error('manifest/validate requires "manifest" (inline JSON) or "manifestPath".');
};

type ServerOptions = {
  port?: number;
  host?: string;
};

export const startServer = (options: ServerOptions = {}) => {
  const port = options.port ?? 8787;
  const host = options.host ?? '127.0.0.1';

  const server = createServer(async (req, res) => {
    if (!req.url) {
      writeJson(res, 404, { status: 'error', message: 'Not found' });
      return;
    }

    const url = new URL(req.url, `http://${req.headers.host ?? `${host}:${port}`}`);
    try {
      if (req.method === 'GET' && url.pathname === '/health') {
        writeJson(res, 200, { status: 'ok' });
        return;
      }

      if (req.method === 'POST' && url.pathname === '/manifest/validate') {
        const body = await readJsonBody(req);
        const result = await validateManifestPayload(body);
        if (result.kind === 'success') {
          writeJson(res, 200, {
            status: 'ok',
            manifest: {
              name: result.manifest.metadata.name,
              schemaVersion: result.manifest.schemaVersion,
              nodes: result.manifest.nodes.length,
              links: result.manifest.links.length,
            },
            warnings: result.issues.filter((issue) => issue.severity === 'warning'),
          });
        } else {
          writeJson(res, 400, {
            status: 'error',
            message: result.message,
            issues: result.issues,
          });
        }
        return;
      }

      if (req.method === 'POST' && url.pathname === '/render') {
        const body = await readJsonBody(req);
        if (typeof body.input !== 'string' || typeof body.output !== 'string') {
          throw new Error('render requires "input" and "output" paths.');
        }
        const bitDepth = body.bitDepth === 10 || body.bitDepth === 16 ? body.bitDepth : 8;
        const summary = await applyFrame({
          input: body.input,
          output: body.output,
          manifest: typeof body.manifest === 'string' ? body.manifest : undefined,
          preset: typeof body.preset === 'string' ? body.preset : undefined,
          ffmpeg: typeof body.ffmpeg === 'string' ? body.ffmpeg : 'ffmpeg',
          ffprobe: typeof body.ffprobe === 'string' ? body.ffprobe : 'ffprobe',
          bitDepth,
        });
        writeJson(res, 200, { status: 'ok', ...summary });
        return;
      }

      if (req.method === 'POST' && url.pathname === '/simulate') {
        const body = await readJsonBody(req);
        if (typeof body.input !== 'string') {
          throw new Error('simulate requires an "input" image path.');
        }
        const summary = await simulate({
          input: body.input,
          manifest: typeof body.manifest === 'string' ? body.manifest : undefined,
          preset: typeof body.preset === 'string' ? body.preset : undefined,
          frames:
            typeof body.frames === 'number' && Number.isFinite(body.frames) && body.frames > 0
              ? Math.floor(body.frames)
              : 120,
          dt:
            typeof body.dt === 'number' && Number.isFinite(body.dt) && body.dt > 0
              ? body.dt
              : 1 / 60,
          seed:
            typeof body.seed === 'number' && Number.isFinite(body.seed)
              ? Math.floor(body.seed)
              : 1337,
          ffmpeg: typeof body.ffmpeg === 'string' ? body.ffmpeg : 'ffmpeg',
          ffprobe: typeof body.ffprobe === 'string' ? body.ffprobe : 'ffprobe',
        });
        writeJson(res, 200, { status: 'ok', ...summary });
        return;
      }

      if (req.method === 'POST' && url.pathname === '/capture') {
        const body = await readJsonBody(req);
        if (typeof body.input !== 'string' || typeof body.output !== 'string') {
          throw new Error('capture requires "input" and "output" paths.');
        }
        const summary = await captureVideo({
          input: body.input,
          output: body.output,
          manifest: typeof body.manifest === 'string' ? body.manifest : undefined,
          preset: typeof body.preset === 'string' ? body.preset : undefined,
          ffmpeg: typeof body.ffmpeg === 'string' ? body.ffmpeg : 'ffmpeg',
          ffprobe: typeof body.ffprobe === 'string' ? body.ffprobe : 'ffprobe',
          framesLimit:
            typeof body.frames === 'number' && Number.isFinite(body.frames) && body.frames > 0
              ? Math.floor(body.frames)
              : undefined,
          keepTemp: Boolean(body.keepTemp),
        });
        writeJson(res, 200, { status: 'ok', ...summary });
        return;
      }

      writeJson(res, 404, { status: 'error', message: 'Not found' });
    } catch (error) {
      console.error('[server] request failed', error);
      writeJson(res, 500, {
        status: 'error',
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });

  server.listen(port, host, () => {
    console.log(`[server] listening on http://${host}:${port}`);
  });

  return server;
};

if (import.meta.url === `file://${process.argv[1] ?? ''}`) {
  startServer();
}
