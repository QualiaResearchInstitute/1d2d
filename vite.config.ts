import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uploadsDir = path.resolve(__dirname, 'public', 'uploads');
const UPLOAD_ENDPOINT = '/api/upload-image';
const ALLOWED_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);

const ensureUploadsDirectory = () => {
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }
};

const readRequestBody = (req: import('http').IncomingMessage): Promise<string> =>
  new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });

const inferExtension = (name: string | undefined, mime: string | undefined): string => {
  const fallback = '.png';
  if (name) {
    const ext = path.extname(name).toLowerCase();
    if (ext && ALLOWED_EXTENSIONS.has(ext)) {
      return ext;
    }
  }
  if (!mime) return fallback;
  if (mime.includes('png')) return '.png';
  if (mime.includes('jpeg') || mime.includes('jpg')) return '.jpg';
  if (mime.includes('webp')) return '.webp';
  return fallback;
};

const handleUploadRequest = async (
  req: import('http').IncomingMessage,
  res: import('http').ServerResponse,
) => {
  if (req.method !== 'POST') {
    res.statusCode = req.method === 'OPTIONS' ? 204 : 405;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Method Not Allowed' }));
    return;
  }

  try {
    ensureUploadsDirectory();
    const raw = await readRequestBody(req);
    const payload = JSON.parse(raw) as {
      name?: string;
      type?: string;
      data?: string;
    };
    if (!payload?.data) {
      res.statusCode = 400;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: 'Missing data' }));
      return;
    }
    const buffer = Buffer.from(payload.data, 'base64');
    if (!buffer.length) {
      res.statusCode = 400;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: 'Invalid image payload' }));
      return;
    }
    const ext = inferExtension(payload.name, payload.type);
    const fileName = `${Date.now()}-${crypto.randomUUID()}${ext}`;
    const filePath = path.join(uploadsDir, fileName);
    await fs.promises.writeFile(filePath, buffer);
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ path: `/uploads/${fileName}` }));
  } catch (error) {
    console.error('[upload] failed to store image', error);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Upload failed' }));
  }
};

const registerUploadMiddleware = (server: {
  middlewares: {
    use: (path: string, handler: (req: any, res: any, next: () => void) => void) => void;
  };
}) => {
  server.middlewares.use(UPLOAD_ENDPOINT, (req, res, next) => {
    void handleUploadRequest(req, res)
      .catch((error) => {
        console.error('[upload] unexpected error', error);
        if (!res.writableEnded) {
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: 'Upload failed' }));
        }
      })
      .finally(() => {
        if (!res.writableEnded) {
          next();
        }
      });
  });
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    configureServer(server) {
      registerUploadMiddleware(server);
    },
  },
  preview: {
    port: 4173,
    configureServer(server) {
      registerUploadMiddleware(server);
    },
  },
});
