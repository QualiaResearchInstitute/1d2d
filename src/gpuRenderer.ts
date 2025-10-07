/* eslint-disable @typescript-eslint/no-use-before-define */
const VERTEX_SRC = `#version 300 es
precision highp float;

layout (location = 0) in vec2 aPosition;
out vec2 vUv;

void main() {
  // Keep y=0 at the top edge so the shader matches the CPU compositor orientation.
  vUv = vec2((aPosition.x + 1.0) * 0.5, (1.0 - aPosition.y) * 0.5);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const FRAGMENT_SRC = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uBaseTex;
uniform sampler2D uEdgeTex;
uniform sampler2D uKurTex;

uniform vec2 uResolution;
uniform float uTime;

uniform float uEdgeThreshold;
uniform float uEffectiveBlend;
uniform int uDisplayMode;
uniform vec3 uBaseOffsets;
uniform float uSigma;
uniform float uSigmaMin;
uniform float uJitter;
uniform float uJitterPhase;
uniform float uBreath;
uniform float uMuJ;
uniform int uPhasePin;
uniform int uMicrosaccade;
uniform int uPolBins;
uniform int uThetaMode;
uniform float uThetaGlobal;
uniform int uCoupleS2E;
uniform float uAlphaPol;
uniform float uGammaOff;
uniform float uKSigma;
uniform float uGammaSigma;
uniform float uContrast;
uniform float uFrameGain;
uniform float uRimAlpha;
uniform float uWarpAmp;
uniform float uSurfaceBlend;
uniform int uSurfaceRegion;
uniform int uSurfEnabled;
uniform int uCoupleE2S;
uniform float uEtaAmp;
uniform int uKurEnabled;
uniform int uUseWallpaper;
uniform int uOpsCount;
uniform vec4 uOps[8];
uniform int uOrientCount;
uniform float uOrientCos[8];
uniform float uOrientSin[8];
uniform vec2 uCanvasCenter;
uniform float uKernelGain;
uniform float uKernelK0;
uniform float uKernelQ;
uniform float uKernelAniso;
uniform float uKernelChirality;
uniform float uAlive;

const float TAU = 6.283185307179586;

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

float hash2(float x, float y) {
  return fract(sin(x * 127.1 + y * 311.7) * 43758.5453);
}

vec4 fetchEdge(ivec2 coord) {
  return texelFetch(uEdgeTex, coord, 0);
}

vec4 fetchKur(ivec2 coord) {
  return texelFetch(uKurTex, coord, 0);
}

float sampleScalarTexture(sampler2D tex, vec2 coord, int component) {
  vec2 clamped = clamp(coord, vec2(0.0), uResolution - vec2(1.001));
  vec2 base = floor(clamped);
  vec2 frac = clamped - base;
  int width = int(uResolution.x);
  int height = int(uResolution.y);
  int x0 = int(base.x);
  int y0 = int(base.y);
  int x1 = min(x0 + 1, width - 1);
  int y1 = min(y0 + 1, height - 1);
  // Textures are uploaded without UNPACK_FLIP; texelFetch coords follow the same top-left convention as the CPU sampler.
  ivec2 i00 = ivec2(x0, y0);
  ivec2 i10 = ivec2(x1, y0);
  ivec2 i01 = ivec2(x0, y1);
  ivec2 i11 = ivec2(x1, y1);
  vec4 v00 = texelFetch(tex, i00, 0);
  vec4 v10 = texelFetch(tex, i10, 0);
  vec4 v01 = texelFetch(tex, i01, 0);
  vec4 v11 = texelFetch(tex, i11, 0);
  float a = mix(v00[component], v10[component], frac.x);
  float b = mix(v01[component], v11[component], frac.x);
  return mix(a, b, frac.y);
}

vec3 sampleRgbTexture(sampler2D tex, vec2 coord) {
  vec2 clamped = clamp(coord, vec2(0.0), uResolution - vec2(1.001));
  vec2 base = floor(clamped);
  vec2 frac = clamped - base;
  int width = int(uResolution.x);
  int height = int(uResolution.y);
  int x0 = int(base.x);
int y0 = int(base.y);
int x1 = min(x0 + 1, width - 1);
int y1 = min(y0 + 1, height - 1);
ivec2 i00 = ivec2(x0, y0);
ivec2 i10 = ivec2(x1, y0);
ivec2 i01 = ivec2(x0, y1);
ivec2 i11 = ivec2(x1, y1);
  vec4 v00 = texelFetch(tex, i00, 0);
  vec4 v10 = texelFetch(tex, i10, 0);
  vec4 v01 = texelFetch(tex, i01, 0);
  vec4 v11 = texelFetch(tex, i11, 0);
  vec3 a = mix(v00.rgb, v10.rgb, frac.x);
  vec3 b = mix(v01.rgb, v11.rgb, frac.x);
  return mix(a, b, frac.y);
}

vec2 applyOp(vec4 op, vec2 point) {
  int kind = int(op.x + 0.5);
  vec2 p = point - uCanvasCenter;
  if (kind == 0) {
    float ang = op.y;
    float c = cos(ang);
    float s = sin(ang);
    p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
  } else if (kind == 1) {
    p = vec2(-p.x, p.y);
  } else if (kind == 2) {
    p = vec2(p.x, -p.y);
  } else if (kind == 3) {
    p = vec2(p.y, p.x);
  } else if (kind == 4) {
    p = vec2(-p.y, -p.x);
  }
  return p + uCanvasCenter;
}

vec2 wallpaperAt(vec2 relPt) {
  int N = uOrientCount;
  if (N <= 0) return vec2(0.0);
  float gx = 0.0;
  float gy = 0.0;
  for (int j = 0; j < 8; ++j) {
    if (j >= N) break;
    float c = uOrientCos[j];
    float s = uOrientSin[j];
    float phase = uKernelChirality * (float(j) / float(N));
    if (uAlive > 0.5) {
      phase += 0.2 * sin(TAU * 0.3 * uTime + float(j));
    }
    float arg = TAU * uKernelK0 * (relPt.x * c + relPt.y * s) + phase;
    float d = -TAU * uKernelK0 * sin(arg);
    gx += d * c;
    gy += d * s;
  }
  float inv = 1.0 / float(N);
  return vec2(gx, gy) * inv;
}

float gauss(float x, float s) {
  return exp(-(x * x) / (2.0 * s * s + 1e-9));
}

float luma01(vec3 c) {
  return clamp01(dot(c, vec3(0.2126, 0.7152, 0.0722)));
}

void main() {
  vec2 fragCoord = vec2(vUv.x * uResolution.x, vUv.y * uResolution.y);
  vec2 fragClamped = clamp(fragCoord, vec2(0.0), uResolution - vec2(1.0));
  int xPix = int(fragClamped.x);
  int yPix = int(fragClamped.y);
  ivec2 icoord = ivec2(xPix, yPix);
  vec3 baseRgb = texelFetch(uBaseTex, icoord, 0).rgb;
  float magVal = fetchEdge(icoord).b;

  if (uDisplayMode == 1 || uDisplayMode == 2) {
    float yb = clamp01(dot(baseRgb, vec3(0.2126, 0.7152, 0.0722)));
    baseRgb = vec3(yb);
  }

  vec2 flow = vec2(0.0);
  if (uKurEnabled == 1) {
    vec4 kur = fetchKur(icoord);
    flow = kur.xy;
  } else if (uUseWallpaper == 1) {
    vec2 accum = vec2(0.0);
    if (uOpsCount == 0) {
      accum = wallpaperAt((fragCoord - uCanvasCenter));
    } else {
      for (int k = 0; k < 8; ++k) {
        if (k >= uOpsCount) break;
        vec2 pt = applyOp(uOps[k], fragCoord);
        accum += wallpaperAt(pt - uCanvasCenter);
      }
      accum /= float(uOpsCount);
    }
    flow = accum;
  }

  vec2 gradEdge = fetchEdge(icoord).rg;

  vec3 resultRgb = baseRgb;
  float rimEnergy = 0.0;

  if (magVal >= uEdgeThreshold) {
    vec2 normal = gradEdge;
    float nlen = length(normal) + 1e-8;
    normal /= nlen;
    vec2 tangent = vec2(-normal.y, normal.x);

    float thetaRaw = atan(normal.y, normal.x);
    float thetaEdge = thetaRaw;
    if (uThetaMode == 1) {
      thetaEdge = uThetaGlobal;
    } else if (uPolBins > 0) {
      float steps = float(uPolBins);
      thetaEdge = round((thetaRaw / TAU) * steps) * (TAU / steps);
    }

    float gradNorm0 = length(flow);
    float thetaUse = thetaEdge;
    if (uCoupleS2E == 1 && gradNorm0 > 1e-6) {
      vec2 edgeDir = vec2(cos(thetaEdge), sin(thetaEdge));
      vec2 surfDir = flow / gradNorm0;
      vec2 blended = (1.0 - uAlphaPol) * edgeDir + uAlphaPol * surfDir;
      float tBlend = atan(blended.y, blended.x);
      if (uPolBins > 0) {
        float steps = float(uPolBins);
        tBlend = round((tBlend / TAU) * steps) * (TAU / steps);
      }
      thetaUse = tBlend;
    }

    float delta = uKernelAniso * 0.9;
    float rho = uKernelChirality * 0.75;
    float thetaEff = thetaUse + rho * uTime;
    float polL = 0.5 * (1.0 + cos(delta) * cos(2.0 * thetaEff));
    float polM = 0.5 * (1.0 + cos(delta) * cos(2.0 * (thetaEff + 0.3)));
    float polS = 0.5 * (1.0 + cos(delta) * cos(2.0 * (thetaEff + 0.6)));

    float jitterSeed = hash2(floor(fragCoord.x + 0.5), floor(fragCoord.y + 0.5));
    float rawJ = sin(uJitterPhase + jitterSeed * TAU);
    float localJ = uJitter * float(uMicrosaccade) * (float(uPhasePin) == 1.0 ? (rawJ - uMuJ) : rawJ);

    float gradNorm = length(flow);
    float bias = 0.0;
    if (uCoupleS2E == 1 && gradNorm > 1e-6) {
      bias = uGammaOff * dot(flow, normal) / gradNorm;
    }

    float Esurf = 0.0;
    float sigmaEff = uSigma;
    if (uCoupleS2E == 1) {
      Esurf = clamp01(gradNorm * uKSigma);
      sigmaEff = max(uSigmaMin, uSigma * (1.0 - uGammaSigma * Esurf));
    }

    float breath = uBreath;

    float offL = uBaseOffsets.x + localJ * 0.35 + bias;
    float offM = uBaseOffsets.y + localJ * 0.5 + bias;
    float offS = uBaseOffsets.z + localJ * 0.8 + bias;

    vec2 pos = fragCoord;
    vec2 samplePosL = pos + (offL + breath) * normal;
    vec2 samplePosM = pos + (offM + breath) * normal;
    vec2 samplePosS = pos + (offS + breath) * normal;

    float pL = sampleScalarTexture(uEdgeTex, samplePosL, 2);
    float pM = sampleScalarTexture(uEdgeTex, samplePosM, 2);
    float pS = sampleScalarTexture(uEdgeTex, samplePosS, 2);

    float gL = gauss(offL, sigmaEff) * uKernelGain;
    float gM = gauss(offM, sigmaEff) * uKernelGain;
    float gS = gauss(offS, sigmaEff) * uKernelGain;

    float QQ = 1.0 + 0.5 * uKernelQ;
    float modL = pow(0.5 * (1.0 + cos(TAU * uKernelK0 * offL)), QQ);
    float modM = pow(0.5 * (1.0 + cos(TAU * uKernelK0 * offM)), QQ);
    float modS = pow(0.5 * (1.0 + cos(TAU * uKernelK0 * offS)), QQ);

    float chiPhase = TAU * uKernelK0 * dot(vec2(pos.x, pos.y), tangent) * 0.002;
    float chBase = uKernelChirality;
    if (uKurEnabled == 1) {
      chBase += clamp(fetchKur(icoord).z * 0.5, -1.0, 1.0);
    }

    float chiL = 0.5 + 0.5 * sin(chiPhase) * chBase;
    float chiM = 0.5 + 0.5 * sin(chiPhase + 0.8) * chBase;
    float chiS = 0.5 + 0.5 * sin(chiPhase + 1.6) * chBase;

    float cont = uContrast * uFrameGain;
    float Lc = pL * gL * modL * chiL * polL * cont;
    float Mc = pM * gM * modM * chiM * polM * cont;
    float Sc = pS * gS * modS * chiS * polS * cont;

    rimEnergy = (Lc + Mc + Sc) / max(1e-6, cont);

    vec3 rimRgb = vec3(
      clamp01(4.4679 * Lc - 3.5873 * Mc + 0.1193 * Sc),
      clamp01(-1.2186 * Lc + 2.3809 * Mc - 0.1624 * Sc),
      clamp01(0.0497 * Lc - 0.2439 * Mc + 1.2045 * Sc)
    );

    if (uDisplayMode == 2 || uDisplayMode == 3) {
      float yr = luma01(rimRgb);
      rimRgb = vec3(yr);
    }

    rimRgb *= uRimAlpha;
    resultRgb = mix(resultRgb, rimRgb, clamp01(uEffectiveBlend));
  }

  if (uSurfEnabled == 1) {
    float mask = 1.0;
    if (uSurfaceRegion == 0) {
      mask = clamp01((uEdgeThreshold - magVal) / max(1e-6, uEdgeThreshold));
    } else if (uSurfaceRegion == 1) {
      mask = clamp01((magVal - uEdgeThreshold) / max(1e-6, 1.0 - uEdgeThreshold));
    }
    if (mask > 1e-3) {
      vec2 surfFlow = flow;
      if (uCoupleE2S == 1 && magVal >= uEdgeThreshold) {
        vec2 tangent = normalize(vec2(-gradEdge.y, gradEdge.x));
        vec2 flowNorm = normalize(surfFlow);
        float dotVal = dot(flowNorm, tangent);
        dotVal /= (length(surfFlow) * length(tangent) + 1e-6);
        float align = pow(clamp01((dotVal + 1.0) * 0.5), uAlphaPol);
        surfFlow = (1.0 - align) * surfFlow + align * tangent;
      }
      float dirNorm = length(surfFlow) + 1e-6;
      float dirW = 1.0 + 0.5 * uKernelAniso * cos(2.0 * atan(surfFlow.y, surfFlow.x));
      vec2 warp = uWarpAmp * (surfFlow / dirNorm) * dirW;
      vec3 warped = sampleRgbTexture(uBaseTex, fragCoord + warp);
      if (uDisplayMode == 2) {
        float yy = luma01(warped);
        warped = vec3(yy);
      }
      float sb = uSurfaceBlend * mask;
      if (uCoupleE2S == 1) {
        sb *= 1.0 + uEtaAmp * clamp01(rimEnergy);
      }
      if (uKurEnabled == 1) {
        sb *= 0.5 + 0.5 * fetchKur(icoord).w;
      }
      sb = clamp01(sb);
      resultRgb = mix(resultRgb, warped, sb);
    }
  }

  fragColor = vec4(resultRgb, 1.0);
}
`;

export type EdgeTextures = {
  gx: Float32Array;
  gy: Float32Array;
  mag: Float32Array;
};

export type KurFields = {
  gradX: Float32Array | null;
  gradY: Float32Array | null;
  vort: Float32Array | null;
  coh: Float32Array | null;
};

export type OrientationUniforms = {
  cos: Float32Array;
  sin: Float32Array;
};

export type WallpaperOp = {
  kind: number;
  angle: number;
};

export type KernelUniform = {
  gain: number;
  k0: number;
  Q: number;
  anisotropy: number;
  chirality: number;
};

export type RenderUniforms = {
  time: number;
  edgeThreshold: number;
  effectiveBlend: number;
  displayMode: number;
  baseOffsets: [number, number, number];
  sigma: number;
  sigmaMin: number;
  jitter: number;
  jitterPhase: number;
  breath: number;
  muJ: number;
  phasePin: boolean;
  microsaccade: boolean;
  polBins: number;
  thetaMode: number;
  thetaGlobal: number;
  coupleS2E: boolean;
  alphaPol: number;
  gammaOff: number;
  kSigma: number;
  gammaSigma: number;
  contrast: number;
  frameGain: number;
  rimAlpha: number;
  warpAmp: number;
  surfaceBlend: number;
  surfaceRegion: number;
  surfEnabled: boolean;
  coupleE2S: boolean;
  etaAmp: number;
  kurEnabled: boolean;
  useWallpaper: boolean;
  kernel: KernelUniform;
  alive: boolean;
};

export type RenderInputs = {
  orientations: OrientationUniforms;
  ops: WallpaperOp[];
  center: [number, number];
};

export type RenderOptions = RenderUniforms & RenderInputs;

export type GpuRenderer = {
  resize(width: number, height: number): void;
  uploadBase(image: ImageData): void;
  uploadEdge(field: EdgeTextures): void;
  uploadKur(fields: KurFields): void;
  render(options: RenderOptions): void;
  readPixels(target: Uint8Array): void;
  dispose(): void;
};

type TextureInfo = {
  texture: WebGLTexture;
  width: number;
  height: number;
};

const quadBufferData = new Float32Array([
  -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1
]);

export function createGpuRenderer(gl: WebGL2RenderingContext): GpuRenderer {
  const program = createProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
  const attribLoc = gl.getAttribLocation(program, "aPosition");

  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  if (!vao || !vbo) {
    throw new Error("Failed to allocate buffers for GPU renderer");
  }

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, quadBufferData, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(attribLoc);
  gl.vertexAttribPointer(attribLoc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const uniforms = locateUniforms(gl, program);

  const baseTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA8,
    type: gl.UNSIGNED_BYTE
  });
  const edgeTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT
  });
  const kurTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT
  });

  const state = {
    width: 1,
    height: 1,
    program,
    vao,
    textures: {
      base: baseTex,
      edge: edgeTex,
      kur: kurTex
    },
    gl
  };

  const opsUniform = new Float32Array(32);
  const orientCosBuffer = new Float32Array(8);
  const orientSinBuffer = new Float32Array(8);
  let edgeBuffer: Float32Array | null = null;
  let kurBuffer: Float32Array | null = null;

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.disable(gl.BLEND);
  gl.clearColor(0, 0, 0, 1);

  const renderer: GpuRenderer = {
    resize(width, height) {
      state.width = width;
      state.height = height;
      gl.viewport(0, 0, width, height);
    },
    uploadBase(image) {
      uploadImageData(gl, state.textures.base.texture, image);
    },
    uploadEdge(field) {
      const total = state.width * state.height;
      const needed = total * 4;
      if (!edgeBuffer || edgeBuffer.length !== needed) {
        edgeBuffer = new Float32Array(needed);
      }
      for (let i = 0; i < total; i++) {
        edgeBuffer[i * 4 + 0] = field.gx[i];
        edgeBuffer[i * 4 + 1] = field.gy[i];
        edgeBuffer[i * 4 + 2] = field.mag[i];
        edgeBuffer[i * 4 + 3] = 0;
      }
      uploadFloatTexture(gl, state.textures.edge.texture, edgeBuffer, state.width, state.height);
    },
    uploadKur(fields) {
      if (!fields.gradX || !fields.gradY || !fields.vort || !fields.coh) return;
      const total = state.width * state.height;
      const needed = total * 4;
      if (!kurBuffer || kurBuffer.length !== needed) {
        kurBuffer = new Float32Array(needed);
      }
      for (let i = 0; i < total; i++) {
        kurBuffer[i * 4 + 0] = fields.gradX[i];
        kurBuffer[i * 4 + 1] = fields.gradY[i];
        kurBuffer[i * 4 + 2] = fields.vort[i];
        kurBuffer[i * 4 + 3] = fields.coh[i];
      }
      uploadFloatTexture(gl, state.textures.kur.texture, kurBuffer, state.width, state.height);
    },
    render(options) {
      gl.useProgram(program);

      bindTexture(gl, state.textures.base.texture, 0, uniforms.baseTex);
      bindTexture(gl, state.textures.edge.texture, 1, uniforms.edgeTex);
      bindTexture(gl, state.textures.kur.texture, 2, uniforms.kurTex);

      gl.uniform2f(uniforms.resolution, state.width, state.height);
      gl.uniform1f(uniforms.time, options.time);
      gl.uniform1f(uniforms.edgeThreshold, options.edgeThreshold);
      gl.uniform1f(uniforms.effectiveBlend, options.effectiveBlend);
      gl.uniform1i(uniforms.displayMode, options.displayMode);
      gl.uniform3f(uniforms.baseOffsets, options.baseOffsets[0], options.baseOffsets[1], options.baseOffsets[2]);
      gl.uniform1f(uniforms.sigma, options.sigma);
      gl.uniform1f(uniforms.sigmaMin, options.sigmaMin);
      gl.uniform1f(uniforms.jitter, options.jitter);
      gl.uniform1f(uniforms.jitterPhase, options.jitterPhase);
      gl.uniform1f(uniforms.breath, options.breath);
      gl.uniform1f(uniforms.muJ, options.muJ);
      gl.uniform1i(uniforms.phasePin, options.phasePin ? 1 : 0);
      gl.uniform1i(uniforms.microsaccade, options.microsaccade ? 1 : 0);
      gl.uniform1i(uniforms.polBins, options.polBins);
      gl.uniform1i(uniforms.thetaMode, options.thetaMode);
      gl.uniform1f(uniforms.thetaGlobal, options.thetaGlobal);
      gl.uniform1i(uniforms.coupleS2E, options.coupleS2E ? 1 : 0);
      gl.uniform1f(uniforms.alphaPol, options.alphaPol);
      gl.uniform1f(uniforms.gammaOff, options.gammaOff);
      gl.uniform1f(uniforms.kSigma, options.kSigma);
      gl.uniform1f(uniforms.gammaSigma, options.gammaSigma);
      gl.uniform1f(uniforms.contrast, options.contrast);
      gl.uniform1f(uniforms.frameGain, options.frameGain);
      gl.uniform1f(uniforms.rimAlpha, options.rimAlpha);
      gl.uniform1f(uniforms.warpAmp, options.warpAmp);
      gl.uniform1f(uniforms.surfaceBlend, options.surfaceBlend);
      gl.uniform1i(uniforms.surfaceRegion, options.surfaceRegion);
      gl.uniform1i(uniforms.surfEnabled, options.surfEnabled ? 1 : 0);
      gl.uniform1i(uniforms.coupleE2S, options.coupleE2S ? 1 : 0);
      gl.uniform1f(uniforms.etaAmp, options.etaAmp);
      gl.uniform1i(uniforms.kurEnabled, options.kurEnabled ? 1 : 0);
      gl.uniform1i(uniforms.useWallpaper, options.useWallpaper ? 1 : 0);
      gl.uniform2f(uniforms.canvasCenter, options.center[0], options.center[1]);
      gl.uniform1f(uniforms.kernelGain, options.kernel.gain);
      gl.uniform1f(uniforms.kernelK0, options.kernel.k0);
      gl.uniform1f(uniforms.kernelQ, options.kernel.Q);
      gl.uniform1f(uniforms.kernelAniso, options.kernel.anisotropy);
      gl.uniform1f(uniforms.kernelChirality, options.kernel.chirality);
      gl.uniform1f(uniforms.alive, options.alive ? 1 : 0);

      const ops = options.ops;
      gl.uniform1i(uniforms.opsCount, ops.length);
      for (let i = 0; i < 8; i++) {
        const base = i * 4;
        if (i < ops.length) {
          opsUniform[base + 0] = ops[i].kind;
          opsUniform[base + 1] = ops[i].angle;
        } else {
          opsUniform[base + 0] = 0;
          opsUniform[base + 1] = 0;
        }
        opsUniform[base + 2] = 0;
        opsUniform[base + 3] = 0;
      }
      gl.uniform4fv(uniforms.ops, opsUniform);

      gl.uniform1i(uniforms.orientCount, options.orientations.cos.length);
      orientCosBuffer.fill(0);
      orientCosBuffer.set(options.orientations.cos.subarray(0, Math.min(8, options.orientations.cos.length)));
      gl.uniform1fv(uniforms.orientCos, orientCosBuffer);
      orientSinBuffer.fill(0);
      orientSinBuffer.set(options.orientations.sin.subarray(0, Math.min(8, options.orientations.sin.length)));
      gl.uniform1fv(uniforms.orientSin, orientSinBuffer);

      gl.viewport(0, 0, state.width, state.height);
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    },
    readPixels(target) {
      if (target.length < state.width * state.height * 4) {
        throw new Error("readPixels target buffer too small");
      }
      gl.readPixels(0, 0, state.width, state.height, gl.RGBA, gl.UNSIGNED_BYTE, target);
    },
    dispose() {
      gl.deleteProgram(program);
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(vbo);
      gl.deleteTexture(state.textures.base.texture);
      gl.deleteTexture(state.textures.edge.texture);
      gl.deleteTexture(state.textures.kur.texture);
    }
  };

  return renderer;
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${info ?? "unknown error"}`);
  }
  gl.detachShader(program, vertexShader);
  gl.detachShader(program, fragmentShader);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  return program;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${info ?? "unknown error"}`);
  }
  return shader;
}

function createTexture(
  gl: WebGL2RenderingContext,
  opts: { format: number; internalFormat: number; type: number }
): TextureInfo {
  const texture = gl.createTexture();
  if (!texture) {
    throw new Error("Failed to create texture");
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { texture, width: 1, height: 1 };
}

function uploadImageData(gl: WebGL2RenderingContext, texture: WebGLTexture, image: ImageData) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    image.width,
    image.height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    image.data
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function uploadFloatTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  data: Float32Array,
  width: number,
  height: number
) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function bindTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  unit: number,
  location: WebGLUniformLocation
) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(location, unit);
}

function locateUniforms(gl: WebGL2RenderingContext, program: WebGLProgram) {
  return {
    baseTex: getUniform(gl, program, "uBaseTex"),
    edgeTex: getUniform(gl, program, "uEdgeTex"),
    kurTex: getUniform(gl, program, "uKurTex"),
    resolution: getUniform(gl, program, "uResolution"),
    time: getUniform(gl, program, "uTime"),
    edgeThreshold: getUniform(gl, program, "uEdgeThreshold"),
    effectiveBlend: getUniform(gl, program, "uEffectiveBlend"),
    displayMode: getUniform(gl, program, "uDisplayMode"),
    baseOffsets: getUniform(gl, program, "uBaseOffsets"),
    sigma: getUniform(gl, program, "uSigma"),
    sigmaMin: getUniform(gl, program, "uSigmaMin"),
    jitter: getUniform(gl, program, "uJitter"),
    jitterPhase: getUniform(gl, program, "uJitterPhase"),
    breath: getUniform(gl, program, "uBreath"),
    muJ: getUniform(gl, program, "uMuJ"),
    phasePin: getUniform(gl, program, "uPhasePin"),
    microsaccade: getUniform(gl, program, "uMicrosaccade"),
    polBins: getUniform(gl, program, "uPolBins"),
    thetaMode: getUniform(gl, program, "uThetaMode"),
    thetaGlobal: getUniform(gl, program, "uThetaGlobal"),
    coupleS2E: getUniform(gl, program, "uCoupleS2E"),
    alphaPol: getUniform(gl, program, "uAlphaPol"),
    gammaOff: getUniform(gl, program, "uGammaOff"),
    kSigma: getUniform(gl, program, "uKSigma"),
    gammaSigma: getUniform(gl, program, "uGammaSigma"),
    contrast: getUniform(gl, program, "uContrast"),
    frameGain: getUniform(gl, program, "uFrameGain"),
    rimAlpha: getUniform(gl, program, "uRimAlpha"),
    warpAmp: getUniform(gl, program, "uWarpAmp"),
    surfaceBlend: getUniform(gl, program, "uSurfaceBlend"),
    surfaceRegion: getUniform(gl, program, "uSurfaceRegion"),
    surfEnabled: getUniform(gl, program, "uSurfEnabled"),
    coupleE2S: getUniform(gl, program, "uCoupleE2S"),
    etaAmp: getUniform(gl, program, "uEtaAmp"),
    kurEnabled: getUniform(gl, program, "uKurEnabled"),
    useWallpaper: getUniform(gl, program, "uUseWallpaper"),
    opsCount: getUniform(gl, program, "uOpsCount"),
    ops: getUniform(gl, program, "uOps"),
    orientCount: getUniform(gl, program, "uOrientCount"),
    orientCos: getUniform(gl, program, "uOrientCos"),
    orientSin: getUniform(gl, program, "uOrientSin"),
    canvasCenter: getUniform(gl, program, "uCanvasCenter"),
    kernelGain: getUniform(gl, program, "uKernelGain"),
    kernelK0: getUniform(gl, program, "uKernelK0"),
    kernelQ: getUniform(gl, program, "uKernelQ"),
    kernelAniso: getUniform(gl, program, "uKernelAniso"),
    kernelChirality: getUniform(gl, program, "uKernelChirality"),
    alive: getUniform(gl, program, "uAlive")
  };
}

function getUniform(gl: WebGL2RenderingContext, program: WebGLProgram, name: string) {
  const location = gl.getUniformLocation(program, name);
  if (!location) {
    throw new Error(`Uniform ${name} not found`);
  }
  return location;
}
