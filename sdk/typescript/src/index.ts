export type FetchLike = typeof fetch;

export interface IndraClientOptions {
  baseUrl?: string;
  fetchImpl?: FetchLike;
}

export interface RenderRequest {
  input: string;
  output: string;
  manifest?: string;
  preset?: string;
  ffmpeg?: string;
  ffprobe?: string;
  bitDepth?: 8 | 10 | 16;
}

export interface RenderResponse {
  output: string;
  width: number;
  height: number;
  manifest: string | null;
  preset: string | null;
  telemetry: Record<string, unknown>;
  metrics: {
    rimMean: number;
    warpMean: number;
    coherenceMean: number;
    indraIndex: number;
  };
}

export interface SimulateRequest {
  input: string;
  manifest?: string;
  preset?: string;
  frames?: number;
  dt?: number;
  seed?: number;
  ffmpeg?: string;
  ffprobe?: string;
}

export interface SimulateResponse {
  frames: number;
  dt: number;
  manifest: string | null;
  preset: string | null;
  metrics: {
    rimMean: number;
    cohMean: number;
    indraIndex: number;
  };
}

export interface CaptureRequest {
  input: string;
  output: string;
  manifest?: string;
  preset?: string;
  ffmpeg?: string;
  ffprobe?: string;
  frames?: number;
  keepTemp?: boolean;
}

export interface CaptureResponse {
  frames: number;
  width: number;
  height: number;
  fps: number | null;
  durationSeconds: number;
  output: string;
}

export interface ManifestValidateRequest {
  manifest?: Record<string, unknown>;
  manifestPath?: string;
}

export interface ManifestValidateResponse {
  manifest: {
    name?: string;
    schemaVersion: string;
    nodes: number;
    links: number;
  };
  warnings: Array<{
    code: string;
    message: string;
    severity: string;
  }>;
}

type ApiSuccess<T> = { status: 'ok' } & T;
type ApiError = { status: 'error'; message: string; issues?: unknown };

export class IndraClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(options: IndraClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? 'http://127.0.0.1:8787';
    const impl = options.fetchImpl ?? globalThis.fetch;
    if (!impl) {
      throw new Error('fetch API unavailable – provide fetchImpl in the client options.');
    }
    this.fetchImpl = impl.bind(globalThis);
  }

  async health(): Promise<boolean> {
    const response = await this.fetchImpl(new URL('/health', this.baseUrl), { method: 'GET' });
    if (!response.ok) {
      return false;
    }
    const payload = (await response.json()) as ApiSuccess<Record<string, unknown>> | ApiError;
    return payload.status === 'ok';
  }

  async render(request: RenderRequest): Promise<RenderResponse> {
    const payload: RenderRequest & { bitDepth: 8 | 10 | 16 } = {
      ...request,
      bitDepth: request.bitDepth ?? 8,
    };
    const response = await this.post<ApiSuccess<RenderResponse>>('/render', payload);
    return response;
  }

  async simulate(request: SimulateRequest): Promise<SimulateResponse> {
    const response = await this.post<ApiSuccess<SimulateResponse>>('/simulate', request);
    return response;
  }

  async capture(request: CaptureRequest): Promise<CaptureResponse> {
    const response = await this.post<ApiSuccess<CaptureResponse>>('/capture', request);
    return response;
  }

  async validateManifest(request: ManifestValidateRequest): Promise<ManifestValidateResponse> {
    const response = await this.post<ApiSuccess<ManifestValidateResponse>>(
      '/manifest/validate',
      request,
    );
    return response;
  }

  private async post<T extends ApiSuccess<unknown>>(
    path: string,
    body: Record<string, unknown>,
  ): Promise<T['status'] extends 'ok' ? T : never> {
    const response = await this.fetchImpl(new URL(path, this.baseUrl), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const json = (await response.json()) as T | ApiError;
    if ('status' in json && json.status === 'error') {
      const error = new Error(json.message);
      (error as any).issues = json.issues;
      throw error;
    }
    return json as T;
  }
}

export default IndraClient;
