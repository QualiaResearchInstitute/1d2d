import type { MediaPipelineResult } from '../media/mediaPipeline';
import type { BeamSplitterBranchSource, BeamSplitterTransformStep } from '../state/beamSplitter';
import type { BeamSplitterBranchMetrics, BeamSplitterDiagnosticsEntry } from '../state/types';

const MAX_BRANCHES = 8;
const DIAGNOSTIC_GRID = 48;

type Mat3 = [number, number, number, number, number, number, number, number, number];

const identityMat3 = (): Mat3 => [1, 0, 0, 0, 1, 0, 0, 0, 1];

const multiplyMat3 = (a: Mat3, b: Mat3): Mat3 => {
  const result: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  result[0] = a[0] * b[0] + a[3] * b[1] + a[6] * b[2];
  result[1] = a[1] * b[0] + a[4] * b[1] + a[7] * b[2];
  result[2] = a[2] * b[0] + a[5] * b[1] + a[8] * b[2];
  result[3] = a[0] * b[3] + a[3] * b[4] + a[6] * b[5];
  result[4] = a[1] * b[3] + a[4] * b[4] + a[7] * b[5];
  result[5] = a[2] * b[3] + a[5] * b[4] + a[8] * b[5];
  result[6] = a[0] * b[6] + a[3] * b[7] + a[6] * b[8];
  result[7] = a[1] * b[6] + a[4] * b[7] + a[7] * b[8];
  result[8] = a[2] * b[6] + a[5] * b[7] + a[8] * b[8];
  return result;
};

const translateMat3 = (tx: number, ty: number): Mat3 => [1, 0, 0, 0, 1, 0, tx, ty, 1];

const scaleMat3 = (sx: number, sy: number): Mat3 => [sx, 0, 0, 0, sy, 0, 0, 0, 1];

const rotateMat3 = (rad: number): Mat3 => {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c, s, 0, -s, c, 0, 0, 0, 1];
};

const CENTER_TRANSLATE = {
  toOrigin: translateMat3(-0.5, -0.5),
  toCenter: translateMat3(0.5, 0.5),
};

const SOURCE_KIND: Record<BeamSplitterBranchSource, number> = {
  source: 0,
  edge: 1,
  phase: 2,
  oscillator: 3,
  surface: 4,
};

const RECOMBINE_KIND: Record<string, number> = {
  sum: 0,
  average: 1,
  energy: 2,
  priority: 3,
  max: 4,
  phase: 5,
};

const VERTEX_SHADER_SRC = `#version 300 es
precision highp float;

layout (location = 0) in vec2 aPosition;
out vec2 vUv;

void main() {
  vUv = (aPosition + 1.0) * 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SRC = `#version 300 es
precision highp float;

const int MAX_BRANCHES = ${MAX_BRANCHES};

uniform sampler2D uSourceTex;
uniform sampler2D uEdgeTex;
uniform sampler2D uPhaseTex;
uniform int uBranchCount;
uniform mat3 uBranchMatrix[MAX_BRANCHES];
uniform float uBranchWeight[MAX_BRANCHES];
uniform int uBranchSource[MAX_BRANCHES];
uniform float uBranchPriority[MAX_BRANCHES];
uniform int uRecombineMode;

in vec2 vUv;
out vec4 fragColor;

float luminance(vec3 rgb) {
  return dot(rgb, vec3(0.299, 0.587, 0.114));
}

vec4 sampleBranch(int index, vec2 uv) {
  mat3 transform = uBranchMatrix[index];
  vec3 coord = transform * vec3(uv, 1.0);
  vec2 sampleUv = clamp(coord.xy, 0.0, 1.0);
  int sourceKind = uBranchSource[index];
  if (sourceKind == 1) {
    float mag = texture(uEdgeTex, sampleUv).r;
    return vec4(vec3(mag), 1.0);
  }
  if (sourceKind == 2) {
    float amp = texture(uPhaseTex, sampleUv).r;
    return vec4(vec3(amp), 1.0);
  }
  if (sourceKind == 3) {
    vec4 phaseSample = texture(uPhaseTex, sampleUv);
    return vec4(vec3(phaseSample.g), 1.0);
  }
  if (sourceKind == 4) {
    vec4 surface = texture(uSourceTex, sampleUv);
    return vec4(surface.rgb * vec3(0.8, 0.9, 1.0), surface.a);
  }
  return texture(uSourceTex, sampleUv);
}

void main() {
  vec3 accumRgb = vec3(0.0);
  float accumAlpha = 0.0;
  float weightSum = 0.0;
  float weightSqSum = 0.0;
  float bestPriority = 1e9;
  float bestIntensity = -1.0;
  vec3 bestRgb = vec3(0.0);
  float bestAlpha = 0.0;

  for (int i = 0; i < MAX_BRANCHES; i++) {
    if (i >= uBranchCount) {
      break;
    }
    vec4 sample = sampleBranch(i, vUv);
    float weight = max(0.0, uBranchWeight[i]);
    weightSum += weight;
    weightSqSum += weight * weight;

    if (uRecombineMode == 3) {
      float priority = uBranchPriority[i];
      if (priority < bestPriority) {
        bestPriority = priority;
        bestRgb = sample.rgb * weight;
        bestAlpha = sample.a * weight;
      }
      continue;
    }

    if (uRecombineMode == 4) {
      float intensity = luminance(sample.rgb) * weight;
      if (intensity > bestIntensity) {
        bestIntensity = intensity;
        bestRgb = sample.rgb * weight;
        bestAlpha = sample.a * weight;
      }
      continue;
    }

    if (uRecombineMode == 5) {
      float amp = luminance(sample.rgb);
      vec3 phased = vec3(amp, amp * 0.7, amp * 1.3);
      accumRgb += phased * weight;
      accumAlpha += sample.a * weight;
    } else {
      accumRgb += sample.rgb * weight;
      accumAlpha += sample.a * weight;
    }
  }

  vec3 finalRgb;
  float finalAlpha;

  if (uRecombineMode == 3 || uRecombineMode == 4) {
    finalRgb = bestRgb;
    finalAlpha = bestAlpha;
  } else {
    if (uRecombineMode == 1 && weightSum > 0.0) {
      accumRgb /= weightSum;
      accumAlpha /= weightSum;
    } else if (uRecombineMode == 2 && weightSqSum > 0.0) {
      float norm = inversesqrt(weightSqSum);
      accumRgb *= norm;
      accumAlpha *= norm;
    }
    finalRgb = accumRgb;
    finalAlpha = accumAlpha;
  }

  fragColor = vec4(finalRgb, clamp(finalAlpha, 0.0, 1.0));
}
`;

export interface BeamSplitterBranchConfig {
  readonly id: string;
  readonly label: string;
  readonly weight: number;
  readonly priority: number;
  readonly source: BeamSplitterBranchSource;
  readonly transformStack: BeamSplitterTransformStep[];
}

export interface BeamSplitterRendererConfig {
  readonly canvasWidth: number;
  readonly canvasHeight: number;
  readonly nodeId?: string;
  readonly asset?: {
    readonly id: string;
    readonly previewUrl: string;
    readonly width: number;
    readonly height: number;
  };
  readonly pipeline?: MediaPipelineResult;
  readonly branches: BeamSplitterBranchConfig[];
  readonly recombine: string;
}

type DiagnosticsPublisher = (entries: readonly BeamSplitterDiagnosticsEntry[]) => void;

const createTexture = (gl: WebGL2RenderingContext): WebGLTexture => {
  const tex = gl.createTexture();
  if (!tex) {
    throw new Error('Failed to allocate texture');
  }
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return tex;
};

const createShader = (gl: WebGL2RenderingContext, type: number, source: string): WebGLShader => {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('Failed to create shader');
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${info ?? 'unknown'}`);
  }
  return shader;
};

const createProgram = (
  gl: WebGL2RenderingContext,
  vertexSrc: string,
  fragmentSrc: string,
): WebGLProgram => {
  const vertex = createShader(gl, gl.VERTEX_SHADER, vertexSrc);
  const fragment = createShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    throw new Error('Failed to create program');
  }
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${info ?? 'unknown'}`);
  }
  return program;
};

const createQuad = (gl: WebGL2RenderingContext) => {
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  if (!vao || !vbo) {
    throw new Error('Failed to allocate quad buffers');
  }
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return { vao, vbo };
};

const toUint8Rgba = (data: Float32Array, width: number, height: number, scale = 1) => {
  const pixels = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const value = Math.max(0, Math.min(255, Math.round(data[i] * scale)));
    const idx = i * 4;
    pixels[idx + 0] = value;
    pixels[idx + 1] = value;
    pixels[idx + 2] = value;
    pixels[idx + 3] = 255;
  }
  return pixels;
};

const toUint8RgbaFromClamped = (data: Uint8ClampedArray): Uint8Array => {
  const copy = new Uint8Array(data.length);
  copy.set(data);
  return copy;
};

const composeTransformMatrix = (stack: BeamSplitterTransformStep[]): Mat3 => {
  let matrix = identityMat3();
  for (const step of stack) {
    if (step.kind === 'rotate') {
      const rotation = multiplyMat3(
        multiplyMat3(CENTER_TRANSLATE.toCenter, rotateMat3(((step.degrees ?? 0) * Math.PI) / 180)),
        CENTER_TRANSLATE.toOrigin,
      );
      matrix = multiplyMat3(rotation, matrix);
    } else if (step.kind === 'scale') {
      const factor = Math.abs(step.factor ?? 1) <= 1e-4 ? 1 : (step.factor ?? 1);
      const scaling = multiplyMat3(
        multiplyMat3(CENTER_TRANSLATE.toCenter, scaleMat3(factor, factor)),
        CENTER_TRANSLATE.toOrigin,
      );
      matrix = multiplyMat3(scaling, matrix);
    } else if (step.kind === 'mirror') {
      const sx = step.axis === 'y' ? 1 : -1;
      const sy = step.axis === 'y' ? -1 : 1;
      const mirror = multiplyMat3(
        multiplyMat3(CENTER_TRANSLATE.toCenter, scaleMat3(sx, sy)),
        CENTER_TRANSLATE.toOrigin,
      );
      matrix = multiplyMat3(mirror, matrix);
    }
  }
  return matrix;
};

const sampleNearest = (
  pixels: Uint8Array,
  width: number,
  height: number,
  uvX: number,
  uvY: number,
): [number, number, number] => {
  const x = Math.min(width - 1, Math.max(0, Math.round(uvX * (width - 1))));
  const y = Math.min(height - 1, Math.max(0, Math.round(uvY * (height - 1))));
  const idx = (y * width + x) * 4;
  return [pixels[idx] / 255, pixels[idx + 1] / 255, pixels[idx + 2] / 255];
};

const sampleFloatField = (
  data: Float32Array,
  width: number,
  height: number,
  uvX: number,
  uvY: number,
): number => {
  const x = Math.min(width - 1, Math.max(0, Math.round(uvX * (width - 1))));
  const y = Math.min(height - 1, Math.max(0, Math.round(uvY * (height - 1))));
  const idx = y * width + x;
  return data[idx];
};

interface CachedAsset {
  id: string;
  previewUrl: string;
  bitmap: ImageBitmap;
  pixels: Uint8Array;
  width: number;
  height: number;
}

export class BeamSplitterRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly quad: { vao: WebGLVertexArrayObject; vbo: WebGLBuffer };
  private readonly sourceTexture: WebGLTexture;
  private readonly edgeTexture: WebGLTexture;
  private readonly phaseTexture: WebGLTexture;
  private readonly emptyTexture: WebGLTexture;
  private readonly updateDiagnostics: DiagnosticsPublisher;

  private branchMatrix = new Float32Array(9 * MAX_BRANCHES);
  private branchWeight = new Float32Array(MAX_BRANCHES);
  private branchPriority = new Float32Array(MAX_BRANCHES);
  private branchSource = new Int32Array(MAX_BRANCHES);
  private branchCount = 0;
  private recombineMode = RECOMBINE_KIND.sum;
  private currentNodeId: string | undefined;
  private branchMeta: BeamSplitterBranchConfig[] = [];

  private asset: CachedAsset | null = null;
  private pipeline: MediaPipelineResult | undefined;
  private rimMagnitudePixels: Uint8Array | null = null;
  private phaseAmpPixels: Uint8Array | null = null;
  private diagnosticsFrame = 0;
  private lastDiagnosticsAt = 0;

  private readonly uniforms: {
    branchCount: WebGLUniformLocation;
    branchMatrix: WebGLUniformLocation;
    branchWeight: WebGLUniformLocation;
    branchSource: WebGLUniformLocation;
    branchPriority: WebGLUniformLocation;
    recombineMode: WebGLUniformLocation;
    sourceTex: WebGLUniformLocation;
    edgeTex: WebGLUniformLocation;
    phaseTex: WebGLUniformLocation;
  };

  constructor(gl: WebGL2RenderingContext, publishDiagnostics: DiagnosticsPublisher) {
    this.gl = gl;
    this.program = createProgram(gl, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC);
    this.quad = createQuad(gl);
    this.sourceTexture = createTexture(gl);
    this.edgeTexture = createTexture(gl);
    this.phaseTexture = createTexture(gl);
    this.emptyTexture = createTexture(gl);
    this.updateDiagnostics = publishDiagnostics;

    gl.bindTexture(gl.TEXTURE_2D, this.emptyTexture);
    const emptyPixels = new Uint8Array([0, 0, 0, 255]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, emptyPixels);

    this.uniforms = {
      branchCount: this.getUniform('uBranchCount'),
      branchMatrix: this.getUniform('uBranchMatrix[0]'),
      branchWeight: this.getUniform('uBranchWeight[0]'),
      branchSource: this.getUniform('uBranchSource[0]'),
      branchPriority: this.getUniform('uBranchPriority[0]'),
      recombineMode: this.getUniform('uRecombineMode'),
      sourceTex: this.getUniform('uSourceTex'),
      edgeTex: this.getUniform('uEdgeTex'),
      phaseTex: this.getUniform('uPhaseTex'),
    };
  }

  private getUniform(name: string): WebGLUniformLocation {
    const location = this.gl.getUniformLocation(this.program, name);
    if (!location) {
      throw new Error(`Uniform ${name} not found`);
    }
    return location;
  }

  dispose() {
    const { gl } = this;
    gl.deleteTexture(this.sourceTexture);
    gl.deleteTexture(this.edgeTexture);
    gl.deleteTexture(this.phaseTexture);
    gl.deleteTexture(this.emptyTexture);
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.quad.vbo);
    gl.deleteVertexArray(this.quad.vao);
    if (this.asset) {
      this.asset.bitmap.close?.();
    }
  }

  async configure(config: BeamSplitterRendererConfig): Promise<void> {
    this.currentNodeId = config.nodeId;
    if (config.asset) {
      if (
        !this.asset ||
        this.asset.id !== config.asset.id ||
        this.asset.previewUrl !== config.asset.previewUrl
      ) {
        await this.loadAsset(config.asset);
      }
    } else if (this.asset) {
      this.asset.bitmap.close?.();
      this.asset = null;
    }

    if (config.pipeline && config.pipeline !== this.pipeline) {
      this.pipeline = config.pipeline;
      this.uploadEdge(config.pipeline);
      this.uploadPhase(config.pipeline);
    }

    this.prepareBranches(config.branches, config.recombine);
  }

  render(viewWidth: number, viewHeight: number) {
    const { gl } = this;
    gl.viewport(0, 0, viewWidth, viewHeight);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.quad.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.asset ? this.sourceTexture : this.emptyTexture);
    gl.uniform1i(this.uniforms.sourceTex, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.rimMagnitudePixels ? this.edgeTexture : this.emptyTexture);
    gl.uniform1i(this.uniforms.edgeTex, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.phaseAmpPixels ? this.phaseTexture : this.emptyTexture);
    gl.uniform1i(this.uniforms.phaseTex, 2);

    gl.uniform1i(this.uniforms.branchCount, this.branchCount);
    gl.uniformMatrix3fv(this.uniforms.branchMatrix, false, this.branchMatrix);
    gl.uniform1fv(this.uniforms.branchWeight, this.branchWeight);
    gl.uniform1iv(this.uniforms.branchSource, this.branchSource);
    gl.uniform1fv(this.uniforms.branchPriority, this.branchPriority);
    gl.uniform1i(this.uniforms.recombineMode, this.recombineMode);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);

    this.maybeEmitDiagnostics();
  }

  private async loadAsset(asset: BeamSplitterRendererConfig['asset']) {
    if (!asset) return;
    const response = await fetch(asset.previewUrl);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to create 2D context for asset texture');
    }
    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = toUint8RgbaFromClamped(imageData.data);

    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.sourceTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      canvas.width,
      canvas.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );

    if (this.asset) {
      this.asset.bitmap.close?.();
    }
    this.asset = {
      id: asset.id,
      previewUrl: asset.previewUrl,
      bitmap,
      pixels,
      width: canvas.width,
      height: canvas.height,
    };
  }

  private uploadEdge(pipeline: MediaPipelineResult) {
    const rim = pipeline.rim;
    const pixels = toUint8Rgba(rim.mag, rim.resolution.width, rim.resolution.height, 255);
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.edgeTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      rim.resolution.width,
      rim.resolution.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    this.rimMagnitudePixels = pixels;
  }

  private uploadPhase(pipeline: MediaPipelineResult) {
    const phaseField = pipeline.phase.field;
    const pixels = toUint8Rgba(
      phaseField.amp,
      phaseField.resolution.width,
      phaseField.resolution.height,
      255,
    );
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.phaseTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      phaseField.resolution.width,
      phaseField.resolution.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    this.phaseAmpPixels = pixels;
  }

  private prepareBranches(branches: BeamSplitterBranchConfig[], recombine: string) {
    const limited = branches.slice(0, MAX_BRANCHES);
    this.branchCount = limited.length;
    this.recombineMode = RECOMBINE_KIND[recombine] ?? RECOMBINE_KIND.sum;
    this.branchMeta = limited.map((branch) => ({
      ...branch,
      transformStack: branch.transformStack?.map((step) => ({ ...step })) ?? [],
    }));
    for (let i = 0; i < MAX_BRANCHES; i++) {
      if (i < limited.length) {
        const branch = limited[i];
        const matrix = composeTransformMatrix(branch.transformStack ?? []);
        this.branchMatrix.set(matrix, i * 9);
        this.branchWeight[i] = branch.weight ?? 1;
        this.branchPriority[i] = branch.priority ?? i;
        this.branchSource[i] = SOURCE_KIND[branch.source ?? 'source'] ?? SOURCE_KIND.source;
      } else {
        this.branchMatrix.set(identityMat3(), i * 9);
        this.branchWeight[i] = 0;
        this.branchPriority[i] = i;
        this.branchSource[i] = SOURCE_KIND.source;
      }
    }
  }

  private maybeEmitDiagnostics() {
    if (!this.currentNodeId || this.branchCount === 0 || !this.asset) {
      return;
    }
    const now = performance.now?.() ?? Date.now();
    if (now - this.lastDiagnosticsAt < 250) {
      return;
    }
    this.lastDiagnosticsAt = now;
    this.diagnosticsFrame += 1;
    const metrics = this.computeDiagnostics();
    this.updateDiagnostics([
      {
        nodeId: this.currentNodeId,
        branches: metrics,
        recombineMode:
          Object.keys(RECOMBINE_KIND).find((key) => RECOMBINE_KIND[key] === this.recombineMode) ??
          'sum',
        frameId: this.diagnosticsFrame,
        updatedAt: now,
      },
    ]);
  }

  private computeDiagnostics(): BeamSplitterBranchMetrics[] {
    const metrics: BeamSplitterBranchMetrics[] = [];
    if (!this.asset) {
      return metrics;
    }
    const width = DIAGNOSTIC_GRID;
    const height = DIAGNOSTIC_GRID;
    const sampleCount = width * height;
    const branchesData: {
      id: string;
      label: string;
      weight: number;
      priority: number;
      sourceKind: number;
      matrix: Mat3;
    }[] = [];

    for (let i = 0; i < this.branchCount; i++) {
      const start = i * 9;
      const meta = this.branchMeta[i];
      branchesData.push({
        id: meta?.id ?? `branch-${i}`,
        label: meta?.label ?? `Branch ${i + 1}`,
        weight: meta?.weight ?? this.branchWeight[i],
        priority: meta?.priority ?? this.branchPriority[i],
        sourceKind: meta
          ? (SOURCE_KIND[meta.source ?? 'source'] ?? this.branchSource[i])
          : this.branchSource[i],
        matrix: [
          this.branchMatrix[start + 0],
          this.branchMatrix[start + 1],
          this.branchMatrix[start + 2],
          this.branchMatrix[start + 3],
          this.branchMatrix[start + 4],
          this.branchMatrix[start + 5],
          this.branchMatrix[start + 6],
          this.branchMatrix[start + 7],
          this.branchMatrix[start + 8],
        ],
      });
    }

    const energy = new Float64Array(this.branchCount);
    const coverage = new Uint32Array(this.branchCount);
    const occluded = new Uint32Array(this.branchCount);

    const sourcePixels = this.asset.pixels;
    const sourceWidth = this.asset.width;
    const sourceHeight = this.asset.height;
    const rim = this.pipeline?.rim;
    const phase = this.pipeline?.phase.field;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const uvX = (x + 0.5) / width;
        const uvY = (y + 0.5) / height;
        let bestPriority = Number.POSITIVE_INFINITY;
        let bestIndex = -1;
        let highestIntensity = -1;
        for (let i = 0; i < branchesData.length; i++) {
          const branch = branchesData[i];
          const coordX = branch.matrix[0] * uvX + branch.matrix[3] * uvY + branch.matrix[6];
          const coordY = branch.matrix[1] * uvX + branch.matrix[4] * uvY + branch.matrix[7];
          const u = Math.min(1, Math.max(0, coordX));
          const v = Math.min(1, Math.max(0, coordY));
          let intensity = 0;
          if (
            branch.sourceKind === SOURCE_KIND.source ||
            branch.sourceKind === SOURCE_KIND.surface
          ) {
            const color = sampleNearest(sourcePixels, sourceWidth, sourceHeight, u, v);
            intensity = (color[0] + color[1] + color[2]) / 3;
          } else if (branch.sourceKind === SOURCE_KIND.edge && rim) {
            intensity = sampleFloatField(
              rim.mag,
              rim.resolution.width,
              rim.resolution.height,
              u,
              v,
            );
          } else if (branch.sourceKind === SOURCE_KIND.phase && phase) {
            intensity = sampleFloatField(
              phase.amp,
              phase.resolution.width,
              phase.resolution.height,
              u,
              v,
            );
          } else if (branch.sourceKind === SOURCE_KIND.oscillator && phase) {
            intensity = sampleFloatField(
              phase.coh,
              phase.resolution.width,
              phase.resolution.height,
              u,
              v,
            );
          }
          const weighted = intensity * branch.weight;
          if (weighted > 1e-3) {
            energy[i] += weighted;
            coverage[i] += 1;
          }
          if (this.recombineMode === RECOMBINE_KIND.priority) {
            if (branch.priority < bestPriority && weighted > 1e-3) {
              bestPriority = branch.priority;
              bestIndex = i;
            }
          } else if (this.recombineMode === RECOMBINE_KIND.max) {
            if (weighted > highestIntensity) {
              highestIntensity = weighted;
              bestIndex = i;
            }
          }
        }
        if (
          this.recombineMode === RECOMBINE_KIND.priority ||
          this.recombineMode === RECOMBINE_KIND.max
        ) {
          for (let i = 0; i < branchesData.length; i++) {
            if (i !== bestIndex && coverage[i] > 0) {
              occluded[i] += 1;
            }
          }
        }
      }
    }

    let totalEnergy = 0;
    for (let i = 0; i < branchesData.length; i++) {
      totalEnergy += energy[i];
    }

    for (let i = 0; i < branchesData.length; i++) {
      const branch = branchesData[i];
      const share = totalEnergy > 0 ? energy[i] / totalEnergy : 0;
      const cover = coverage[i] / sampleCount;
      const occ = coverage[i] > 0 ? Math.min(1, occluded[i] / Math.max(1, coverage[i])) : 0;
      metrics.push({
        branchId: branch.id,
        label: branch.label,
        energy: energy[i],
        energyShare: share,
        coverage: cover,
        occlusion: occ,
        priority: branch.priority,
        source:
          this.branchMeta[i]?.source ??
          Object.keys(SOURCE_KIND).find(
            (key) => SOURCE_KIND[key as BeamSplitterBranchSource] === branch.sourceKind,
          ),
        weight: branch.weight,
      });
    }

    return metrics;
  }
}
