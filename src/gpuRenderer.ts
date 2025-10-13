/* eslint-disable @typescript-eslint/no-use-before-define */
import {
  FIELD_STRUCTS_GLSL,
  assertPhaseField,
  assertRimField,
  assertVolumeField,
  type PhaseField,
  type RimField,
  type VolumeField,
} from './fields/contracts';
import type { CouplingConfig, CurvatureMode } from './pipeline/rainbowFrame';
import type { HyperbolicAtlasGpuPackage } from './hyperbolic/atlas.js';
import { FLOATS_PER_MATRIX } from './qcd/lattice.js';

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

const clampAtanhInput = (value: number) => Math.min(0.999999, Math.max(-0.999999, value));

const safeAtanh = (value: number) => {
  const clamped = clampAtanhInput(value);
  return 0.5 * Math.log((1 + clamped) / (1 - clamped));
};

const FRAGMENT_SRC = `#version 300 es
precision highp float;

${FIELD_STRUCTS_GLSL}

in vec2 vUv;
layout (location = 0) out vec4 fragColor;
layout (location = 1) out vec4 tracerOut;

uniform sampler2D uBaseTex;
uniform sampler2D uEdgeTex;
uniform sampler2D uKurTex;
uniform sampler2D uKurAmpTex;
uniform sampler2D uVolumeTex;
uniform sampler2D uTracerState;
uniform int uAtlasEnabled;
uniform sampler2D uAtlasSampleTex;
uniform sampler2D uAtlasJacobianTex;
uniform float uAtlasSampleScale;
uniform float uAtlasMaxHyperRadius;
uniform sampler2D uSu7Tex0;
uniform sampler2D uSu7Tex1;
uniform sampler2D uSu7Tex2;
uniform sampler2D uSu7Tex3;
uniform sampler2D uSu7UnitaryTex;

uniform vec2 uResolution;
uniform float uTime;

uniform float uEdgeThreshold;
uniform float uEffectiveBlend;
uniform int uDisplayMode;
uniform vec3 uBaseOffsets;
uniform float uSigma;
uniform float uJitter;
uniform float uJitterPhase;
uniform float uBreath;
uniform float uMuJ;
uniform int uPhasePin;
uniform int uMicrosaccade;
uniform int uPolBins;
uniform int uThetaMode;
uniform float uThetaGlobal;
uniform float uContrast;
uniform float uFrameGain;
uniform int uRimEnabled;
uniform float uRimAlpha;
uniform float uWarpAmp;
uniform float uSurfaceBlend;
uniform int uSurfaceRegion;
uniform int uSurfEnabled;
uniform int uKurEnabled;
uniform int uVolumeEnabled;
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
uniform float uKernelTransparency;
uniform float uAlive;
uniform float uBeta2;
uniform float uCouplingRimSurfaceBlend;
uniform float uCouplingRimSurfaceAlign;
uniform float uCouplingSurfaceRimOffset;
uniform float uCouplingSurfaceRimSigma;
uniform float uCouplingSurfaceRimHue;
uniform float uCouplingKurTransparency;
uniform float uCouplingKurOrientation;
uniform float uCouplingKurChirality;
uniform float uCouplingScale;
uniform float uCouplingVolumePhaseHue;
uniform float uCouplingVolumeDepthWarp;
uniform float uComposerExposure[4];
uniform float uComposerGamma[4];
uniform float uComposerWeight[4];
uniform vec2 uComposerBlendGain;
uniform float uCurvatureStrength;
uniform int uCurvatureMode;
uniform int uTracerEnabled;
uniform float uTracerGain;
uniform float uTracerTau;
uniform float uTracerModDepth;
uniform float uTracerModFrequency;
uniform float uTracerModPhase;
uniform float uTracerDt;
uniform int uSu7Enabled;
uniform float uSu7Gain;
uniform int uSu7DecimationStride;
uniform int uSu7DecimationMode;
uniform float uSu7ProjectorWeight;
uniform int uSu7ProjectorMode;
uniform vec3 uSu7ProjectorMatrix[7];
uniform int uSu7TileCols;
uniform int uSu7TileRows;
uniform int uSu7TileSize;
uniform int uSu7UnitaryTexWidth;
uniform int uSu7UnitaryTexRowsPerTile;
uniform int uSu7Pretransformed;
uniform int uSu7HopfLensCount;
uniform ivec2 uSu7HopfLensAxes[3];
uniform vec2 uSu7HopfLensMix[3];

#define COMPOSER_FIELD_SURFACE 0
#define COMPOSER_FIELD_RIM 1
#define COMPOSER_FIELD_KUR 2
#define COMPOSER_FIELD_VOLUME 3
#define SU7_DECIMATION_HYBRID 0
#define SU7_DECIMATION_STRIDE 1
#define SU7_DECIMATION_EDGES 2
#define SU7_PROJECTOR_IDENTITY 0
#define SU7_PROJECTOR_COMPOSER_WEIGHTS 1
#define SU7_PROJECTOR_OVERLAY_SPLIT 2
#define SU7_PROJECTOR_DIRECT_RGB 3
#define SU7_PROJECTOR_MATRIX 4
#define SU7_PROJECTOR_HOPF_LENS 5
#define SU7_OVERLAY_MAX 6

const float TAU = 6.283185307179586;

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

float hash2(float x, float y) {
  return fract(sin(x * 127.1 + y * 311.7) * 43758.5453);
}

float applyComposerScalar(float value, int fieldIndex) {
  float exposure = uComposerExposure[fieldIndex];
  float gamma = uComposerGamma[fieldIndex];
  float scaled = clamp(value * exposure, 0.0, 1.0);
  float adjusted = pow(scaled, gamma);
  return clamp(adjusted, 0.0, 1.0);
}

vec3 applyComposerVec3(vec3 rgb, int fieldIndex) {
  return vec3(
    applyComposerScalar(rgb.r, fieldIndex),
    applyComposerScalar(rgb.g, fieldIndex),
    applyComposerScalar(rgb.b, fieldIndex)
  );
}

vec4 sampleVec4Texture(sampler2D tex, vec2 coord) {
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
  vec4 a = mix(v00, v10, frac.x);
  vec4 b = mix(v01, v11, frac.x);
  return mix(a, b, frac.y);
}

float sampleScalarTexture(sampler2D tex, vec2 coord, int component) {
  return sampleVec4Texture(tex, coord)[component];
}

vec3 sampleRgbTexture(sampler2D tex, vec2 coord) {
  return sampleVec4Texture(tex, coord).rgb;
}

float srgbToLinearChannel(float value) {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

float linearToSrgbChannel(float value) {
  float clamped = clamp(value, 0.0, 1.0);
  if (clamped <= 0.0031308) {
    return clamped * 12.92;
  }
  return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055;
}

vec3 srgbToLinearVec3(vec3 rgb) {
  return vec3(
    srgbToLinearChannel(rgb.r),
    srgbToLinearChannel(rgb.g),
    srgbToLinearChannel(rgb.b)
  );
}

vec3 linearToSrgbVec3(vec3 rgb) {
  return vec3(
    linearToSrgbChannel(rgb.r),
    linearToSrgbChannel(rgb.g),
    linearToSrgbChannel(rgb.b)
  );
}

vec3 rgbToLms(vec3 rgb) {
  return vec3(
    0.31399022 * rgb.r + 0.63951294 * rgb.g + 0.04649755 * rgb.b,
    0.15537241 * rgb.r + 0.75789446 * rgb.g + 0.08670142 * rgb.b,
    0.01775239 * rgb.r + 0.10944209 * rgb.g + 0.87256922 * rgb.b
  );
}

vec3 lmsToRgb(vec3 lms) {
  return vec3(
    5.47221206 * lms.x - 4.6419601 * lms.y + 0.16963708 * lms.z,
    -1.1252419 * lms.x + 2.29317094 * lms.y - 0.1678952 * lms.z,
    0.02980165 * lms.x - 0.19318073 * lms.y + 1.16364789 * lms.z
  );
}

float linearLuma(vec3 rgb) {
  return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

vec2 complexMul(vec2 a, vec2 b) {
  return vec2(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

vec2 complexAdd(vec2 a, vec2 b) {
  return a + b;
}

float complexAbs(vec2 z) {
  return length(z);
}

int computeSu7TileIndex(ivec2 pixelCoord) {
  int cols = max(uSu7TileCols, 1);
  int rows = max(uSu7TileRows, 1);
  int size = max(uSu7TileSize, 1);
  int tileX = clamp(pixelCoord.x / size, 0, cols - 1);
  int tileY = clamp(pixelCoord.y / size, 0, rows - 1);
  return tileY * cols + tileX;
}

vec2 fetchSu7Unitary(int tileIndex, int row, int col) {
  int texWidth = max(uSu7UnitaryTexWidth, 1);
  int rowsPerTile = max(uSu7UnitaryTexRowsPerTile, 1);
  int complexIndex = row * 7 + col;
  int texelIndex = complexIndex / 2;
  int lane = complexIndex - texelIndex * 2;
  int texX = texelIndex % texWidth;
  int texY = tileIndex * rowsPerTile + texelIndex / texWidth;
  vec4 texel = texelFetch(uSu7UnitaryTex, ivec2(texX, texY), 0);
  return lane == 0 ? texel.xy : texel.zw;
}

vec3 su7TintColor(float share, vec3 tint) {
  float scale = sqrt(clamp01(share));
  vec3 colored = tint * scale;
  return clamp(colored, vec3(0.0), vec3(1.0));
}

vec4 sampleAtlasSample(vec2 screenPos) {
  return sampleVec4Texture(uAtlasSampleTex, screenPos);
}

vec4 sampleAtlasJacobian(vec2 screenPos) {
  return sampleVec4Texture(uAtlasJacobianTex, screenPos);
}

vec2 applyCurvature(vec2 coord);

vec2 mapScreenToSample(vec2 screenPos) {
  if (uAtlasEnabled == 1) {
    return sampleAtlasSample(screenPos).xy;
  }
  return applyCurvature(screenPos);
}

float sampleWarpedMagnitude(vec2 screenPos) {
  vec2 coord = screenPos;
  vec4 jac = vec4(1.0, 0.0, 0.0, 1.0);
  if (uAtlasEnabled == 1) {
    vec4 atlasSampleLocal = sampleAtlasSample(screenPos);
    coord = atlasSampleLocal.xy;
    jac = sampleAtlasJacobian(screenPos);
  } else {
    coord = applyCurvature(screenPos);
  }
  vec2 grad = vec2(
    sampleScalarTexture(uEdgeTex, coord, 0),
    sampleScalarTexture(uEdgeTex, coord, 1)
  );
  if (uAtlasEnabled == 1) {
    float j11 = jac.x * uAtlasSampleScale;
    float j12 = jac.y * uAtlasSampleScale;
    float j21 = jac.z * uAtlasSampleScale;
    float j22 = jac.w * uAtlasSampleScale;
    grad = vec2(
      grad.x * j11 + grad.y * j21,
      grad.x * j12 + grad.y * j22
    );
  }
  return length(grad);
}

vec2 computeLatticeFlow(vec2 coord) {
  if (uOrientCount <= 0) {
    return vec2(0.0);
  }
  float freq = 0.015 + 0.004 * float(uOrientCount);
  float timeShift = uTime * 0.6;
  vec2 flow = vec2(0.0);
  for (int k = 0; k < 8; ++k) {
    if (k >= uOrientCount) break;
    float proj = (coord.x - uCanvasCenter.x) * uOrientCos[k] + (coord.y - uCanvasCenter.y) * uOrientSin[k];
    float phaseVal = proj * freq + timeShift + float(k) * 0.35;
    float magnitude = 0.5 + 0.5 * sin(phaseVal + uBeta2 * 0.25);
    flow += vec2(uOrientCos[k], uOrientSin[k]) * magnitude;
  }
  return flow / max(float(uOrientCount), 1.0);
}

float safeAtanh(float x) {
  float clamped = clamp(x, -0.999, 0.999);
  return 0.5 * log((1.0 + clamped) / (1.0 - clamped));
}

float safeTanh(float x) {
  float limited = clamp(x, -8.0, 8.0);
  float e2 = exp(2.0 * limited);
  return (e2 - 1.0) / (e2 + 1.0);
}

vec2 applyCurvature(vec2 coord) {
  float strength = clamp(uCurvatureStrength, 0.0, 0.95);
  vec2 clampedCoord = clamp(coord, vec2(0.0), uResolution - vec2(1.0));
  if (strength <= 1e-5) {
    return clampedCoord;
  }
  float maxRadius = length(uCanvasCenter);
  if (maxRadius <= 1e-6) {
    return clampedCoord;
  }
  vec2 centered = (coord - uCanvasCenter) / maxRadius;
  float radius = length(centered);
  if (radius <= 1e-6) {
    return clampedCoord;
  }
  float clipped = min(radius, 0.999);
  float rho = safeAtanh(clipped);
  float scale = 1.0 + strength * 3.0;
  float target = safeTanh(rho * scale);
  float denom = clipped > 1e-6 ? clipped : 1.0;
  vec2 disk = centered * (target / denom);
  if (uCurvatureMode == 1) {
    float denomK = 1.0 + dot(disk, disk);
    if (denomK > 1e-6) {
      disk = (2.0 * disk) / denomK;
    }
  }
  vec2 mapped = uCanvasCenter + disk * maxRadius;
  return clamp(mapped, vec2(0.0), uResolution - vec2(1.0));
}

float computeHyperbolicFlowScale(float hyperRadius) {
  float strength = clamp(uCurvatureStrength, 0.0, 0.95);
  if (strength <= 1e-5) {
    return 1.0;
  }
  float limited = min(abs(hyperRadius), 6.0);
  if (limited <= 1e-5) {
    return 1.0;
  }
  float base = sinh(limited) / limited;
  float factor = 1.0 + (base - 1.0) * strength;
  return max(factor, 1.0);
}

float computeHyperbolicTransparencyBoost(
  float hyperScale,
  float radiusNorm,
  float tagAbs,
  float kNorm
) {
  if (hyperScale <= 1.000001 || kNorm <= 1e-6) {
    return 1.0;
  }
  float strength = clamp(0.45 + 0.4 * uKernelTransparency, 0.0, 1.0);
  if (strength <= 1e-6) {
    return 1.0;
  }
  float weight = clamp(radiusNorm, 0.0, 1.0) * clamp(kNorm, 0.0, 1.0) * clamp(tagAbs, 0.0, 1.0);
  float boost = 1.0 + (hyperScale - 1.0) * strength * weight;
  return clamp(boost, 0.4, 2.8);
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
  vec2 translation = vec2(op.z * uResolution.x, op.w * uResolution.y);
  return p + uCanvasCenter + translation;
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

vec3 clamp01(vec3 v) {
  return clamp(v, vec3(0.0), vec3(1.0));
}

float wrapPi(float theta) {
  float twoPi = TAU;
  return theta - floor((theta + 3.141592653589793) / twoPi) * twoPi - 3.141592653589793;
}

float responseSoft(float value, float shaping) {
  float v = clamp(value, 0.0, 1.0);
  float expVal = mix(1.2, 3.5, clamp(shaping, 0.0, 1.0));
  return 1.0 - pow(max(0.0, 1.0 - v), expVal);
}

float responsePow(float value, float shaping) {
  float v = clamp(value, 0.0, 1.0);
  float expVal = mix(1.65, 0.55, clamp(shaping, 0.0, 1.0));
  return pow(v, expVal);
}

void main() {
  vec2 screenCoord = vec2(vUv.x * uResolution.x, vUv.y * uResolution.y);
  bool atlasActive = (uAtlasEnabled == 1);
  vec2 sampleCoord;
  vec4 atlasSample = vec4(0.0);
  vec4 atlasJacobian = vec4(1.0, 0.0, 0.0, 1.0);
  if (atlasActive) {
    atlasSample = sampleAtlasSample(screenCoord);
    sampleCoord = atlasSample.xy;
    atlasJacobian = sampleAtlasJacobian(screenCoord);
  } else {
    sampleCoord = applyCurvature(screenCoord);
  }
  vec2 fragCoord = sampleCoord;
  vec2 pos = fragCoord;

  float hyperFlowScale = 1.0;
  float radiusNorm = 0.0;
  vec2 radialDir = vec2(1.0, 0.0);
  if (atlasActive) {
    float hyperRadius = atlasSample.z;
    hyperFlowScale = computeHyperbolicFlowScale(hyperRadius);
    if (uAtlasMaxHyperRadius > 1e-6) {
      radiusNorm = clamp(hyperRadius / uAtlasMaxHyperRadius, 0.0, 0.999999);
    }
    radialDir = vec2(cos(atlasSample.w), sin(atlasSample.w));
  } else {
    vec2 centered = fragCoord - uCanvasCenter;
    float radialLen = length(centered);
    if (radialLen > 1e-6) {
      radialDir = centered / radialLen;
    }
  }

  vec3 baseRgb = sampleRgbTexture(uBaseTex, sampleCoord);
  vec2 gradEdge = vec2(
    sampleScalarTexture(uEdgeTex, sampleCoord, 0),
    sampleScalarTexture(uEdgeTex, sampleCoord, 1)
  );
  float j11 = atlasJacobian.x * uAtlasSampleScale;
  float j12 = atlasJacobian.y * uAtlasSampleScale;
  float j21 = atlasJacobian.z * uAtlasSampleScale;
  float j22 = atlasJacobian.w * uAtlasSampleScale;
  if (atlasActive) {
    gradEdge = vec2(
      gradEdge.x * j11 + gradEdge.y * j21,
      gradEdge.x * j12 + gradEdge.y * j22
    );
  }
  float magVal = length(gradEdge);
  float edgeLen = magVal;

  float composerWeightSurface = uComposerWeight[COMPOSER_FIELD_SURFACE];
  float composerWeightRim = uComposerWeight[COMPOSER_FIELD_RIM];
  float composerWeightKur = uComposerWeight[COMPOSER_FIELD_KUR];
  float composerWeightVolume = uComposerWeight[COMPOSER_FIELD_VOLUME];
  vec3 baseSrgbOriginal = baseRgb;
  vec3 su7ProjectedRgb = vec3(0.0);
  float su7ProjectedMix = 0.0;
  vec3 su7OverlayRgb[SU7_OVERLAY_MAX];
  float su7OverlayMix[SU7_OVERLAY_MAX];
  int su7OverlayCount = 0;

  if (
    uSu7Enabled == 1 &&
    uSu7TileCols > 0 &&
    uSu7TileRows > 0 &&
    uSu7UnitaryTexWidth > 0 &&
    uSu7UnitaryTexRowsPerTile > 0
  ) {
    ivec2 pixelCoord = ivec2(clamp(floor(screenCoord), vec2(0.0), uResolution - vec2(1.0)));
    vec4 su7Tex0 = texelFetch(uSu7Tex0, pixelCoord, 0);
    vec4 su7Tex1 = texelFetch(uSu7Tex1, pixelCoord, 0);
    vec4 su7Tex2 = texelFetch(uSu7Tex2, pixelCoord, 0);
    vec4 su7Tex3 = texelFetch(uSu7Tex3, pixelCoord, 0);
    float su7Norm = su7Tex3.z;
    float effectiveGain = abs(uSu7Gain);
    int stride = max(uSu7DecimationStride, 1);
    bool strideEligible = stride <= 1 || ((pixelCoord.x % stride == 0) && (pixelCoord.y % stride == 0));
    bool rimEligible = magVal >= uEdgeThreshold;
    bool shouldProject = false;
    if (stride <= 1) {
      shouldProject = true;
    } else if (uSu7DecimationMode == SU7_DECIMATION_STRIDE) {
      shouldProject = strideEligible;
    } else if (uSu7DecimationMode == SU7_DECIMATION_EDGES) {
      shouldProject = rimEligible;
    } else {
      shouldProject = strideEligible || rimEligible;
    }
    if (shouldProject && su7Norm > 1e-6 && effectiveGain > 1e-6) {
    vec2 su7Vec[7];
    su7Vec[0] = su7Tex0.xy;
    su7Vec[1] = su7Tex0.zw;
    su7Vec[2] = su7Tex1.xy;
    su7Vec[3] = su7Tex1.zw;
    su7Vec[4] = su7Tex2.xy;
    su7Vec[5] = su7Tex2.zw;
    su7Vec[6] = su7Tex3.xy;
    vec2 su7Transformed[7];
    if (uSu7Pretransformed == 1) {
      for (int row = 0; row < 7; ++row) {
        su7Transformed[row] = su7Vec[row];
      }
    } else {
      int tileIndex = computeSu7TileIndex(pixelCoord);
      for (int row = 0; row < 7; ++row) {
        vec2 sum = vec2(0.0);
        for (int col = 0; col < 7; ++col) {
          vec2 coeff = fetchSu7Unitary(tileIndex, row, col);
          sum = complexAdd(sum, complexMul(coeff, su7Vec[col]));
        }
        su7Transformed[row] = sum;
      }
    }
      float scale = su7Norm * effectiveGain;
      float toneScale = scale * (uSu7ProjectorWeight > 0.0 ? uSu7ProjectorWeight : 1.0);
      float magnitude0 = complexAbs(su7Transformed[0]);
      float magnitude1 = complexAbs(su7Transformed[1]);
      float magnitude2 = complexAbs(su7Transformed[2]);
      float magnitude3 = complexAbs(su7Transformed[3]);
      float magnitude4 = complexAbs(su7Transformed[4]);
      float magnitude5 = complexAbs(su7Transformed[5]);
      float magnitude6 = complexAbs(su7Transformed[6]);

      vec3 linearRgb;
      if (uSu7ProjectorMode == SU7_PROJECTOR_DIRECT_RGB) {
        linearRgb = vec3(
          magnitude0 * toneScale,
          magnitude1 * toneScale,
          magnitude2 * toneScale
        );
      } else {
        vec3 lms = vec3(
          magnitude0 * toneScale,
          magnitude1 * toneScale,
          magnitude2 * toneScale
        );
        linearRgb = lmsToRgb(lms);
      }
      linearRgb = max(linearRgb * uFrameGain, vec3(0.0));
      vec3 baseLinear = srgbToLinearVec3(baseSrgbOriginal);
      float baseLuma = linearLuma(baseLinear);
      float su7Luma = linearLuma(linearRgb);
      if (su7Luma > 1e-6 && baseLuma > 1e-6) {
        float lumaScale = clamp(baseLuma / su7Luma, 0.25, 4.0);
        linearRgb *= lumaScale;
        su7Luma = linearLuma(linearRgb);
      }
      vec3 su7Srgb = linearToSrgbVec3(linearRgb);
      float mixAmount = clamp(
        uSu7ProjectorWeight * clamp(effectiveGain, 0.0, 1.0),
        0.0,
        1.0
      );

      float rimEnergyShare = (magnitude0 + magnitude1 + magnitude2) / 3.0;
      float surfaceEnergyShare = magnitude3;
      float kurEnergyShare = 0.5 * (magnitude4 + magnitude5);
      float volumeEnergyShare = magnitude6;
      float totalEnergy = rimEnergyShare + surfaceEnergyShare + kurEnergyShare + volumeEnergyShare;
      vec4 shareNorm = vec4(0.0);
      if (totalEnergy > 1e-6) {
        shareNorm = vec4(
          rimEnergyShare,
          surfaceEnergyShare,
          kurEnergyShare,
          volumeEnergyShare
        ) / totalEnergy;
      }

      if (
        (uSu7ProjectorMode == SU7_PROJECTOR_COMPOSER_WEIGHTS ||
          uSu7ProjectorMode == SU7_PROJECTOR_OVERLAY_SPLIT) &&
        totalEnergy > 1e-6
      ) {
        vec4 multipliers = vec4(
          clamp(1.0 + (shareNorm.x - 0.25) * 2.0, 0.2, 2.5),
          clamp(1.0 + (shareNorm.y - 0.25) * 2.0, 0.2, 2.5),
          clamp(1.0 + (shareNorm.z - 0.25) * 2.0, 0.2, 2.5),
          clamp(1.0 + (shareNorm.w - 0.25) * 2.0, 0.2, 2.5)
        );
        composerWeightRim = clamp(composerWeightRim * multipliers.x, 0.05, 3.0);
        composerWeightSurface = clamp(composerWeightSurface * multipliers.y, 0.05, 3.0);
        composerWeightKur = clamp(composerWeightKur * multipliers.z, 0.05, 3.0);
        composerWeightVolume = clamp(composerWeightVolume * multipliers.w, 0.05, 3.0);
      }

      if (uSu7ProjectorMode == SU7_PROJECTOR_OVERLAY_SPLIT && totalEnergy > 1e-6) {
        su7OverlayCount = 0;
        float rimMix = clamp(shareNorm.x * mixAmount, 0.0, 1.0);
        if (rimMix > 1e-6) {
          su7OverlayRgb[su7OverlayCount] = clamp(su7Srgb, vec3(0.0), vec3(1.0));
          su7OverlayMix[su7OverlayCount] = rimMix;
          su7OverlayCount++;
        }
        float surfaceMix = clamp(shareNorm.y * mixAmount, 0.0, 1.0);
        if (surfaceMix > 1e-6 && su7OverlayCount < SU7_OVERLAY_MAX) {
          su7OverlayRgb[su7OverlayCount] = su7TintColor(shareNorm.y, vec3(0.95, 0.72, 0.33));
          su7OverlayMix[su7OverlayCount] = surfaceMix;
          su7OverlayCount++;
        }
        float kurMix = clamp(shareNorm.z * mixAmount, 0.0, 1.0);
        if (kurMix > 1e-6 && su7OverlayCount < SU7_OVERLAY_MAX) {
          su7OverlayRgb[su7OverlayCount] = su7TintColor(shareNorm.z, vec3(0.28, 0.78, 1.0));
          su7OverlayMix[su7OverlayCount] = kurMix;
          su7OverlayCount++;
        }
        float volumeMix = clamp(shareNorm.w * mixAmount, 0.0, 1.0);
        if (volumeMix > 1e-6 && su7OverlayCount < SU7_OVERLAY_MAX) {
          su7OverlayRgb[su7OverlayCount] = su7TintColor(shareNorm.w, vec3(0.42, 0.9, 0.56));
          su7OverlayMix[su7OverlayCount] = volumeMix;
          su7OverlayCount++;
        }
      } else if (uSu7ProjectorMode == SU7_PROJECTOR_HOPF_LENS) {
        su7OverlayCount = 0;
        int lensCount = clamp(uSu7HopfLensCount, 0, 3);
        for (int lensIdx = 0; lensIdx < 3; ++lensIdx) {
          if (lensIdx >= lensCount) {
            break;
          }
          if (su7OverlayCount >= SU7_OVERLAY_MAX) {
            break;
          }
          ivec2 axes = uSu7HopfLensAxes[lensIdx];
          int axisA = clamp(axes.x, 0, 6);
          int axisB = clamp(axes.y, 0, 6);
          if (axisA == axisB) {
            axisB = (axisA + 1) % 7;
          }
          vec2 compA = su7Transformed[axisA];
          vec2 compB = su7Transformed[axisB];
          float magA = length(compA);
          float magB = length(compB);
          float pairMag = sqrt(magA * magA + magB * magB);
          if (pairMag <= 1e-6) {
            continue;
          }
          float share = clamp(pairMag / max(su7Norm, 1e-6), 0.0, 1.0);
          float baseMix = clamp(uSu7HopfLensMix[lensIdx].x * share * mixAmount, 0.0, 1.0);
          float fiberMix = clamp(uSu7HopfLensMix[lensIdx].y * share * mixAmount, 0.0, 1.0);
          if (baseMix <= 1e-6 && fiberMix <= 1e-6) {
            continue;
          }
          vec2 nA = compA / pairMag;
          vec2 nB = compB / pairMag;
          vec2 conjB = vec2(nB.x, -nB.y);
          vec2 prod = complexMul(nA, conjB);
          float baseX = clamp(prod.x * 2.0, -1.0, 1.0);
          float baseY = clamp(prod.y * 2.0, -1.0, 1.0);
          float nzA = clamp(length(nA), 0.0, 1.0);
          float nzB = clamp(length(nB), 0.0, 1.0);
          float baseZ = clamp(nzA * nzA - nzB * nzB, -1.0, 1.0);
          if (baseMix > 1e-6 && su7OverlayCount < SU7_OVERLAY_MAX) {
            vec3 baseColor = clamp(vec3(0.5) + 0.5 * vec3(baseX, baseY, baseZ), vec3(0.0), vec3(1.0));
            su7OverlayRgb[su7OverlayCount] = baseColor;
            su7OverlayMix[su7OverlayCount] = baseMix;
            su7OverlayCount++;
          }
          if (fiberMix > 1e-6 && su7OverlayCount < SU7_OVERLAY_MAX) {
            float fiberAngle = 0.5 * (atan(nA.y, nA.x) + atan(nB.y, nB.x));
            fiberAngle = mod(fiberAngle + 3.141592653589793, TAU) - 3.141592653589793;
            float cosA = cos(fiberAngle);
            float cosB = cos(fiberAngle - 2.094395102f);
            float cosC = cos(fiberAngle + 2.094395102f);
            vec3 fiberColor = vec3(
              clamp01(0.5 + 0.5 * cosA),
              clamp01(0.5 + 0.5 * cosB),
              clamp01(0.5 + 0.5 * cosC)
            );
            su7OverlayRgb[su7OverlayCount] = fiberColor;
            su7OverlayMix[su7OverlayCount] = fiberMix;
            su7OverlayCount++;
          }
        }
      } else {
        su7OverlayCount = 0;
      }

      su7ProjectedRgb = clamp(su7Srgb, vec3(0.0), vec3(1.0));
      su7ProjectedMix = mixAmount;
    }
  }

  vec2 normal = vec2(0.0);
  vec2 tangent = vec2(0.0);
  if (magVal > 1e-8) {
    normal = gradEdge / magVal;
    tangent = vec2(-normal.y, normal.x);
  }

  vec3 volumeSample = vec3(0.0);
  float volumePhaseVal = 0.0;
  float volumeDepthVal = 0.0;
  float volumeIntensityVal = 0.0;
  float volumeDepthGrad = 0.0;
  if (uVolumeEnabled == 1) {
    vec4 volumeVec = sampleVec4Texture(uVolumeTex, sampleCoord);
    volumeSample = volumeVec.rgb;
    volumePhaseVal = volumeVec.r;
    volumeDepthVal = volumeVec.g;
    volumeIntensityVal = volumeVec.b;
    vec2 leftScreen = vec2(max(screenCoord.x - 1.0, 0.0), screenCoord.y);
    vec2 rightScreen = vec2(min(screenCoord.x + 1.0, uResolution.x - 1.0), screenCoord.y);
    vec2 upScreen = vec2(screenCoord.x, max(screenCoord.y - 1.0, 0.0));
    vec2 downScreen = vec2(screenCoord.x, min(screenCoord.y + 1.0, uResolution.y - 1.0));
    float depthLeft = sampleScalarTexture(uVolumeTex, mapScreenToSample(leftScreen), 1);
    float depthRight = sampleScalarTexture(uVolumeTex, mapScreenToSample(rightScreen), 1);
    float depthUp = sampleScalarTexture(uVolumeTex, mapScreenToSample(upScreen), 1);
    float depthDown = sampleScalarTexture(uVolumeTex, mapScreenToSample(downScreen), 1);
    float depthDx = (depthRight - depthLeft) * 0.5;
    float depthDy = (depthDown - depthUp) * 0.5;
    volumeDepthGrad = length(vec2(depthDx, depthDy));
  }

  if (uDisplayMode == 1 || uDisplayMode == 2) {
    float yb = clamp01(dot(baseRgb, vec3(0.2126, 0.7152, 0.0722)));
    baseRgb = vec3(yb);
  }

  vec2 warpFlow = computeLatticeFlow(fragCoord);
  if (uUseWallpaper == 1 && uOrientCount == 0) {
    vec2 accum = vec2(0.0);
    if (uOpsCount == 0) {
      accum = wallpaperAt(fragCoord - uCanvasCenter);
    } else {
      for (int k = 0; k < 8; ++k) {
        if (k >= uOpsCount) break;
        vec2 pt = applyOp(uOps[k], fragCoord);
        accum += wallpaperAt(pt - uCanvasCenter);
      }
      accum /= float(uOpsCount);
    }
    warpFlow += accum;
  }

  if (uVolumeEnabled == 1 && uCouplingVolumeDepthWarp > 0.0) {
    float gradNorm = min(volumeDepthGrad / (0.18 + 0.12 * uKernelK0 + 1e-6), 1.0);
    float gradResponse = responseSoft(gradNorm, uCouplingVolumeDepthWarp);
    float gradSignal = applyComposerScalar(gradResponse, COMPOSER_FIELD_VOLUME);
    float boost =
      1.0 +
      uCouplingScale *
        uCouplingVolumeDepthWarp *
        gradSignal *
        composerWeightVolume;
    warpFlow *= boost;
  }

  if (atlasActive) {
    float det = j11 * j22 - j12 * j21;
    if (abs(det) > 1e-6) {
      float invDet = 1.0 / det;
      warpFlow = vec2(
        (j22 * warpFlow.x - j12 * warpFlow.y) * invDet,
        (-j21 * warpFlow.x + j11 * warpFlow.y) * invDet
      );
    } else {
      warpFlow = vec2(0.0);
    }
  }
  warpFlow *= hyperFlowScale;

  vec2 kurGrad = vec2(0.0);
  float vortVal = 0.0;
  float cohVal = 0.0;
  float ampVal = 0.0;
  if (uKurEnabled == 1) {
    vec4 kurSample = sampleVec4Texture(uKurTex, sampleCoord);
    kurGrad = kurSample.xy;
    if (atlasActive) {
      kurGrad = vec2(
        kurGrad.x * j11 + kurGrad.y * j21,
        kurGrad.x * j12 + kurGrad.y * j22
      );
    }
    kurGrad *= hyperFlowScale;
    vortVal = kurSample.z;
    cohVal = clamp(kurSample.w, 0.0, 1.0);
    ampVal = clamp(sampleScalarTexture(uKurAmpTex, sampleCoord, 0), 0.0, 1.0);
  }

  vec2 combinedFlow = warpFlow + kurGrad;

  float sharedMag = length(combinedFlow);
  float parallaxTransparencyBoost = 1.0;
  if (sharedMag > 1e-6) {
    float normDenom = max(0.12 + 0.85 * uKernelK0, 1e-3);
    float kNorm = clamp(sharedMag / normDenom, 0.0, 1.0);
    float tag = clamp(dot(combinedFlow, radialDir) / sharedMag, -1.0, 1.0);
    parallaxTransparencyBoost = computeHyperbolicTransparencyBoost(
      hyperFlowScale,
      radiusNorm,
      abs(tag),
      kNorm
    );
  }

  vec3 resultRgb = baseRgb;
  float rimEnergy = 0.0;

  float couplingScale = uCouplingScale;
  float sigmaFloorBase = max(0.25, uSigma * 0.25);

  if (uRimEnabled == 1 && magVal >= uEdgeThreshold) {
    float thetaRaw = atan(normal.y, normal.x);
    float thetaEdge = thetaRaw;
    if (uThetaMode == 1) {
      thetaEdge = uThetaGlobal;
    } else if (uPolBins > 0) {
      float steps = float(uPolBins);
      thetaEdge = round((thetaRaw / TAU) * steps) * (TAU / steps);
    }

    float thetaUse = thetaEdge;
    if (uCouplingKurOrientation > 0.0) {
      float kurNorm = length(kurGrad);
      if (kurNorm > 1e-6) {
        vec2 edgeDir = vec2(cos(thetaUse), sin(thetaUse));
        vec2 kurDir = kurGrad / kurNorm;
        float mixW = clamp(uCouplingKurOrientation * couplingScale * composerWeightKur, 0.0, 1.0);
        vec2 blended = mix(edgeDir, kurDir, mixW);
        thetaUse = atan(blended.y, blended.x);
        if (uPolBins > 0) {
          float steps = float(uPolBins);
          thetaUse = round((thetaUse / TAU) * steps) * (TAU / steps);
        }
      }
    }

    float delta = uKernelAniso * 0.9;
    float rho = uKernelChirality * 0.75;
    float thetaEff = thetaUse + rho * uTime;
    float polL = 0.5 * (1.0 + cos(delta) * cos(2.0 * thetaEff));
    float polM = 0.5 * (1.0 + cos(delta) * cos(2.0 * (thetaEff + 0.3)));
    float polS = 0.5 * (1.0 + cos(delta) * cos(2.0 * (thetaEff + 0.6)));

    float jitterSeed = hash2(floor(fragCoord.x + 0.5), floor(fragCoord.y + 0.5));
    float rawJ = sin(uJitterPhase + jitterSeed * TAU);
    float jitterFactor = uJitter * float(uMicrosaccade);
    float localJ = jitterFactor * (float(uPhasePin) == 1.0 ? (rawJ - uMuJ) : rawJ);
    float breath = uBreath;

    float warpNorm = length(warpFlow);
    float bias = 0.0;
    if (uCouplingSurfaceRimOffset > 0.0 && warpNorm > 1e-6) {
      float proj = dot(warpFlow, normal) / (warpNorm + 1e-6);
      float magnitude = responseSoft(
        warpNorm / (1.05 + 0.55 * uKernelK0),
        uCouplingSurfaceRimOffset
      );
      float magnitudeAdjusted = applyComposerScalar(clamp(magnitude, 0.0, 1.0), COMPOSER_FIELD_SURFACE);
      float signedVal = clamp(proj, -1.0, 1.0) * magnitudeAdjusted;
      bias = clamp(
        0.65 *
          couplingScale *
          uCouplingSurfaceRimOffset *
          composerWeightSurface *
          signedVal,
        -0.9,
        0.9
      );
    }

    float sigmaEff = uSigma;
    if (uCouplingSurfaceRimSigma > 0.0 && warpNorm > 1e-6) {
      float sharpness = responsePow(
        warpNorm / (1.15 + 0.6 * uKernelK0),
        uCouplingSurfaceRimSigma
      );
      float sharpnessAdjusted = applyComposerScalar(clamp(sharpness, 0.0, 1.0), COMPOSER_FIELD_SURFACE);
      float drop = clamp(
        0.75 *
          couplingScale *
          uCouplingSurfaceRimSigma *
          composerWeightSurface *
          sharpnessAdjusted,
        0.0,
        1.0
      );
      float sigmaFloor = clamp(sigmaFloorBase, 0.0, uSigma);
      sigmaEff = clamp(uSigma * (1.0 - drop), sigmaFloor, uSigma);
    }

    float offL = uBaseOffsets.x + localJ * 0.35 + bias;
    float offM = uBaseOffsets.y + localJ * 0.5 + bias;
    float offS = uBaseOffsets.z + localJ * 0.8 + bias;

    float hueShift = 0.0;
    if (uVolumeEnabled == 1 && uCouplingVolumePhaseHue > 0.0) {
      float phaseNorm = clamp(volumePhaseVal / 3.141592653589793, -1.0, 1.0);
      float phaseMag = applyComposerScalar(abs(phaseNorm), COMPOSER_FIELD_VOLUME);
      float signedPhase = phaseNorm < 0.0 ? -phaseMag : phaseMag;
      float volHue = clamp(
        uCouplingScale *
          uCouplingVolumePhaseHue *
          composerWeightVolume *
          signedPhase *
          0.6,
        -1.5,
        1.5
      );
      hueShift += volHue;
    }
    if (uCouplingSurfaceRimHue > 0.0 && warpNorm > 1e-6) {
      float latticeAngle = atan(warpFlow.y, warpFlow.x);
      float tangentAngle = atan(tangent.y, tangent.x);
      float deltaAngle = wrapPi(latticeAngle - tangentAngle);
      float hueAmplitude = responseSoft(
        min(abs(deltaAngle) / 3.141592653589793, 1.0),
        uCouplingSurfaceRimHue
      );
      float warpWeight = responsePow(
        warpNorm / (1.1 + 0.6 * uKernelK0),
        uCouplingSurfaceRimHue
      );
      float hueSignal = applyComposerScalar(clamp(hueAmplitude * warpWeight, 0.0, 1.0), COMPOSER_FIELD_SURFACE);
      float signedMag =
        (deltaAngle == 0.0 ? 0.0 : sign(deltaAngle)) * hueSignal;
      hueShift = clamp(
        couplingScale *
          uCouplingSurfaceRimHue *
          composerWeightSurface *
          signedMag *
          0.9,
        -1.4,
        1.4
      );
    }

    vec2 samplePosL = screenCoord + (offL + breath) * normal;
    vec2 samplePosM = screenCoord + (offM + breath) * normal;
    vec2 samplePosS = screenCoord + (offS + breath) * normal;

    float pL = sampleWarpedMagnitude(samplePosL);
    float pM = sampleWarpedMagnitude(samplePosM);
    float pS = sampleWarpedMagnitude(samplePosS);

    float gL = gauss(offL, sigmaEff) * uKernelGain;
    float gM = gauss(offM, sigmaEff) * uKernelGain;
    float gS = gauss(offS, sigmaEff) * uKernelGain;

    float QQ = 1.0 + 0.5 * uKernelQ;
    float modL = pow(0.5 * (1.0 + cos(TAU * uKernelK0 * offL)), QQ);
    float modM = pow(0.5 * (1.0 + cos(TAU * uKernelK0 * offM)), QQ);
    float modS = pow(0.5 * (1.0 + cos(TAU * uKernelK0 * offS)), QQ);

    float chiPhase = TAU * uKernelK0 * dot(pos, tangent) * 0.002 + hueShift;
    float chBase = uKernelChirality;
  if (uKurEnabled == 1 && uCouplingKurChirality > 0.0) {
    float vortClamped = clamp(vortVal, -1.0, 1.0);
    float vortScaled = 0.5 * couplingScale * uCouplingKurChirality * composerWeightKur * vortClamped;
    chBase = clamp(chBase + vortScaled, -3.0, 3.0);
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

    if (uDisplayMode == 4) {
      float rimSum = rimRgb.r + rimRgb.g + rimRgb.b;
      float baseSum = baseRgb.r + baseRgb.g + baseRgb.b;
      vec3 weights = vec3(1.0 / 3.0);
      if (baseSum > 1e-6) {
        weights = baseRgb / baseSum;
      }
      vec3 natural = clamp01(rimSum * weights);
      float hueBlend = 0.7;
      float baseBlend = 0.25;
      rimRgb = mix(rimRgb, natural, hueBlend);
      rimRgb = mix(rimRgb, baseRgb, baseBlend);
    }

    rimRgb *= uRimAlpha;
    rimRgb = applyComposerVec3(rimRgb, COMPOSER_FIELD_RIM);

    float pixelBlend = clamp(uEffectiveBlend * uComposerBlendGain.x, 0.0, 1.0);
    pixelBlend = clamp(pixelBlend * parallaxTransparencyBoost, 0.0, 1.0);
    if (uKurEnabled == 1 && uCouplingKurTransparency > 0.0) {
      float kurMeasure = applyComposerScalar(0.5 * (cohVal + ampVal), COMPOSER_FIELD_KUR);
      float boost = clamp(
        1.0 +
          couplingScale *
            uCouplingKurTransparency *
            composerWeightKur *
            (kurMeasure - 0.5) *
            1.5,
        0.1,
        2.5
      );
      pixelBlend = clamp(pixelBlend * boost, 0.0, 1.0);
    }
    pixelBlend = clamp(pixelBlend * composerWeightRim, 0.0, 1.0);

    resultRgb = mix(resultRgb, rimRgb, pixelBlend);
  }

  if (uSurfEnabled == 1) {
    float mask = 1.0;
    if (uSurfaceRegion == 0) {
      mask = clamp01(
        (uEdgeThreshold - magVal) / max(1e-6, uEdgeThreshold)
      );
    } else if (uSurfaceRegion == 1) {
      mask = clamp01(
        (magVal - uEdgeThreshold) / max(1e-6, 1.0 - uEdgeThreshold)
      );
    }
    if (mask > 1e-3) {
      vec2 surfFlow = combinedFlow;
      if (uCouplingRimSurfaceAlign > 0.0 && magVal >= uEdgeThreshold && edgeLen > 1e-8) {
        vec2 tangentN = tangent;
        float surfLen = length(surfFlow);
        if (surfLen > 1e-6) {
          float dotVal = dot(surfFlow, tangentN) / (surfLen + 1e-9);
          float alignment = responsePow(
            (dotVal + 1.0) * 0.5,
            uCouplingRimSurfaceAlign
          );
          float alignWeight =
            clamp(
              uCouplingRimSurfaceAlign *
                couplingScale *
                composerWeightSurface *
                alignment,
              0.0,
              1.0
            );
          vec2 target = tangentN * surfLen;
          surfFlow = mix(surfFlow, target, alignWeight);
        }
      }
      if (uKurEnabled == 1 && uCouplingKurOrientation > 0.0) {
        float kurNorm = length(kurGrad);
        float surfLen = length(surfFlow);
        if (kurNorm > 1e-6 && surfLen > 1e-6) {
          vec2 target = (kurGrad / kurNorm) * surfLen;
          float weight =
            clamp(uCouplingKurOrientation * couplingScale * composerWeightKur * 0.75, 0.0, 1.0);
          surfFlow = mix(surfFlow, target, weight);
        }
      }
      float dirNorm = length(surfFlow) + 1e-6;
      float dirAngle = atan(surfFlow.y, surfFlow.x);
      float dirW = 1.0 + 0.5 * uKernelAniso * cos(2.0 * dirAngle);
      float amplitude = clamp(uWarpAmp * dirNorm * dirW, 0.0, 1.0);
      float phaseShift = dirAngle + uTime * 0.45;
      vec3 warped = vec3(
        clamp01(0.5 + 0.5 * sin(phaseShift)),
        clamp01(0.5 + 0.5 * sin(phaseShift + 2.094395102f)),
        clamp01(0.5 + 0.5 * sin(phaseShift + 4.188790204f))
      );
      warped *= amplitude;
      if (uDisplayMode == 2) {
        float yy = luma01(warped);
        warped = vec3(yy);
      }
      warped = applyComposerVec3(warped, COMPOSER_FIELD_SURFACE);
      float sb = uSurfaceBlend * mask * uComposerBlendGain.y;
      sb *= parallaxTransparencyBoost;
      if (uCouplingRimSurfaceBlend > 0.0) {
        float energy = responseSoft(
          rimEnergy / (0.75 + 0.25 * uKernelGain),
          uCouplingRimSurfaceBlend
        );
        float rimBoost = clamp(
          1.0 + couplingScale * uCouplingRimSurfaceBlend * composerWeightRim * energy,
          0.2,
          3.0
        );
        sb *= rimBoost;
      }
      if (uKurEnabled == 1 && uCouplingKurTransparency > 0.0) {
        float kurMeasure = applyComposerScalar(0.5 * (cohVal + ampVal), COMPOSER_FIELD_KUR);
        float boost = clamp(
          1.0 +
            couplingScale *
              uCouplingKurTransparency *
              composerWeightKur *
              (kurMeasure - 0.5) *
              1.5,
          0.1,
          3.0
        );
        sb *= boost;
      }
      sb = clamp(sb * composerWeightSurface, 0.0, 1.0);
      resultRgb = mix(resultRgb, warped, sb);
    }
  }

  if (su7ProjectedMix > 1e-6) {
    resultRgb = clamp(mix(resultRgb, su7ProjectedRgb, su7ProjectedMix), vec3(0.0), vec3(1.0));
  }
  for (int overlayIdx = 0; overlayIdx < SU7_OVERLAY_MAX; ++overlayIdx) {
    if (overlayIdx >= su7OverlayCount) {
      break;
    }
    float overlayMix = su7OverlayMix[overlayIdx];
    if (overlayMix > 1e-6) {
      resultRgb = clamp(mix(resultRgb, su7OverlayRgb[overlayIdx], overlayMix), vec3(0.0), vec3(1.0));
    }
  }

  baseRgb = resultRgb;
  vec3 finalRgb = baseRgb;
  vec3 tracerNext = baseRgb;
  if (uTracerEnabled == 1) {
    vec3 prevTracer = texture(uTracerState, vUv).rgb;
    float safeDt = clamp(uTracerDt, 1.0 / 480.0, 0.25);
    float tau = max(uTracerTau, 0.05);
    float decay = exp(-safeDt / tau);
    float sineComponent = 0.0;
    if (uTracerModDepth > 1e-6) {
      float phase = uTracerModPhase;
      if (uTracerModFrequency > 1e-6) {
        phase = TAU * uTracerModFrequency * uTime + uTracerModPhase;
      }
      sineComponent = uTracerModDepth * sin(phase);
    }
    float gain = clamp(uTracerGain * (1.0 + sineComponent), 0.0, 1.2);
    vec3 tail = max(prevTracer - baseRgb, vec3(0.0));
    finalRgb = clamp(baseRgb + tail * gain, vec3(0.0), vec3(1.0));
    vec3 decayed = prevTracer * decay;
    tracerNext = max(baseRgb, decayed);
  }

  fragColor = vec4(finalRgb, 1.0);
  tracerOut = vec4(tracerNext, 1.0);
}
`;

export type OrientationUniforms = {
  cos: Float32Array;
  sin: Float32Array;
};

export type WallpaperOp = {
  kind: number;
  angle: number;
  tx: number;
  ty: number;
};

export type KernelUniform = {
  gain: number;
  k0: number;
  Q: number;
  anisotropy: number;
  chirality: number;
  transparency: number;
};

export type TracerUniforms = {
  enabled: boolean;
  gain: number;
  tau: number;
  modulationDepth: number;
  modulationFrequency: number;
  modulationPhase: number;
  dt: number;
  reset: boolean;
};

export type Su7ProjectorMode =
  | 'identity'
  | 'composerWeights'
  | 'overlaySplit'
  | 'directRgb'
  | 'matrix'
  | 'hopfLens';

export type Su7HopfLensUniform = {
  axes: [number, number];
  baseMix: number;
  fiberMix: number;
};

export type Su7Uniforms = {
  enabled: boolean;
  gain: number;
  decimationStride: number;
  decimationMode: 'hybrid' | 'stride' | 'edges';
  projectorMode: Su7ProjectorMode;
  projectorWeight: number;
  projectorMatrix?: Float32Array | null;
  pretransformed: boolean;
  hopfLenses?: Su7HopfLensUniform[] | null;
};

export type Su7TexturePayload = {
  width: number;
  height: number;
  vectors: [Float32Array, Float32Array, Float32Array, Float32Array];
  tileData: Float32Array;
  tileCols: number;
  tileRows: number;
  tileSize: number;
  tileTexWidth: number;
  tileTexRowsPerTile: number;
  pretransformed?: boolean;
};

export type RenderUniforms = {
  time: number;
  edgeThreshold: number;
  effectiveBlend: number;
  displayMode: number;
  baseOffsets: [number, number, number];
  sigma: number;
  jitter: number;
  jitterPhase: number;
  breath: number;
  muJ: number;
  phasePin: boolean;
  microsaccade: boolean;
  polBins: number;
  thetaMode: number;
  thetaGlobal: number;
  contrast: number;
  frameGain: number;
  rimEnabled: boolean;
  rimAlpha: number;
  warpAmp: number;
  curvatureStrength: number;
  curvatureMode: CurvatureMode;
  surfaceBlend: number;
  surfaceRegion: number;
  surfEnabled: boolean;
  kurEnabled: boolean;
  volumeEnabled: boolean;
  useWallpaper: boolean;
  kernel: KernelUniform;
  alive: boolean;
  beta2: number;
  coupling: CouplingConfig;
  couplingScale: number;
  composerExposure: Float32Array;
  composerGamma: Float32Array;
  composerWeight: Float32Array;
  composerBlendGain: [number, number];
  tracer: TracerUniforms;
  su7: Su7Uniforms;
};

export type RenderInputs = {
  orientations: OrientationUniforms;
  ops: WallpaperOp[];
  center: [number, number];
};

export type RenderOptions = RenderUniforms & RenderInputs;

const QCD_WORKGROUP_SIZE_X = 8;
const QCD_WORKGROUP_SIZE_Y = 8;
const QCD_SEED_ALIGNMENT_BYTES = 256;
const UINT32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const QCD_SEED_BYTES_PER_TEXEL = 4 * UINT32_BYTES;
const QCD_COMPLEX_STRIDE = 2;
const QCD_ROW_STRIDE = 6;
const QCD_UNIFORM_SIZE_BYTES = 64;

type QcdSeedTexturePayload = {
  data: Uint32Array;
  bytesPerRow: number;
  signature: string;
  keyLo: number;
  keyHi: number;
};

export type QcdGpuSweepOptions = {
  lattice: Float32Array;
  width: number;
  height: number;
  siteStride: number;
  linkStride: number;
  rowStride: number;
  complexStride: number;
  beta: number;
  parity: 0 | 1;
  axis: 'x' | 'y';
  sweepIndex: number;
  overRelaxationSteps: number;
  seed?: number;
  scope?: string | number;
};

const toUint32 = (value: number): number => Math.trunc(value) >>> 0;

const hash32 = (value: number): number => {
  let x = toUint32(value);
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
};

const hashScope = (scope: string | number | undefined): number => {
  if (typeof scope === 'number') {
    return hash32(scope);
  }
  if (typeof scope !== 'string' || scope.length === 0) {
    return 0;
  }
  let hash = 0;
  for (let idx = 0; idx < scope.length; idx += 1) {
    hash = (hash << 5) - hash + scope.charCodeAt(idx);
    hash |= 0;
  }
  return hash32(hash);
};

const alignTo = (value: number, alignment: number): number =>
  Math.ceil(value / alignment) * alignment;

const createQcdSeedTexturePayload = (
  width: number,
  height: number,
  seed: number,
  scopeHash: number,
): QcdSeedTexturePayload => {
  const bytesPerRowRaw = width * QCD_SEED_BYTES_PER_TEXEL;
  const bytesPerRow = alignTo(bytesPerRowRaw, QCD_SEED_ALIGNMENT_BYTES);
  const uint32PerRow = bytesPerRow / UINT32_BYTES;
  const data = new Uint32Array(uint32PerRow * height);
  const signature = `${width}x${height}:${toUint32(seed)}:${scopeHash >>> 0}`;
  const baseSeed = hash32(seed ^ scopeHash ^ 0x51ed2705);
  const keyLo = hash32(baseSeed ^ 0x9e3779b9);
  const keyHi = hash32(baseSeed ^ 0x632beb5);
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * uint32PerRow;
    for (let x = 0; x < width; x += 1) {
      const base = rowOffset + x * 4;
      const siteIndex = y * width + x;
      let value = hash32(seed ^ scopeHash ^ siteIndex);
      data[base] = value;
      value = hash32(value ^ 0x9e3779b9);
      data[base + 1] = value;
      value = hash32(value ^ 0x632beb5);
      data[base + 2] = value;
      value = hash32(value ^ 0xa511e9b5);
      data[base + 3] = value;
    }
  }
  return { data, bytesPerRow, signature, keyLo, keyHi };
};

let cachedQcdKernelWgsl: string | null = null;

const loadQcdKernelWgsl = async (): Promise<string> => {
  if (cachedQcdKernelWgsl) {
    return cachedQcdKernelWgsl;
  }
  const maybeGlobal = globalThis as { process?: unknown };
  const nodeProcess = maybeGlobal?.process as { versions?: Record<string, unknown> } | undefined;
  const isNode =
    typeof nodeProcess === 'object' &&
    nodeProcess != null &&
    typeof nodeProcess.versions === 'object';
  if (isNode) {
    const [{ readFile }, { fileURLToPath }, { dirname, resolve }] = await Promise.all([
      import('node:fs/promises'),
      import('node:url'),
      import('node:path'),
    ]);
    const baseDir = dirname(fileURLToPath(import.meta.url));
    const filePath = resolve(baseDir, 'qcd/gpuKernel.wgsl');
    cachedQcdKernelWgsl = await readFile(filePath, 'utf8');
    return cachedQcdKernelWgsl;
  }
  const module = await import('./qcd/gpuKernel.wgsl?raw');
  cachedQcdKernelWgsl = (module as { default: string }).default;
  return cachedQcdKernelWgsl;
};

const GPU_BUFFER_USAGE = (globalThis as { GPUBufferUsage?: Record<string, number> })
  .GPUBufferUsage ?? {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

const GPU_TEXTURE_USAGE = (globalThis as { GPUTextureUsage?: Record<string, number> })
  .GPUTextureUsage ?? {
  COPY_DST: 0x0002,
  TEXTURE_BINDING: 0x0004,
};

const GPU_MAP_MODE = (globalThis as { GPUMapMode?: Record<string, number> }).GPUMapMode ?? {
  READ: 0x0001,
};

type QcdGpuContext = {
  device: any;
  pipeline: any;
  bindGroupLayout: any;
  uniformBuffer: any;
  uniformArray: ArrayBuffer;
  uniformU32: Uint32Array;
  uniformF32: Float32Array;
  latticeBuffer: any | null;
  latticeCapacityFloats: number;
  seedTexture: any | null;
  seedTextureView: any | null;
  seedSignature: string | null;
  seedKey: { lo: number; hi: number } | null;
};

export type GpuRenderer = {
  resize(width: number, height: number): void;
  uploadBase(image: ImageData): void;
  uploadRim(field: RimField | null): void;
  uploadPhase(field: PhaseField | null): void;
  uploadVolume(field: VolumeField | null): void;
  uploadSu7(data: Su7TexturePayload | null): void;
  setHyperbolicAtlas(atlas: HyperbolicAtlasGpuPackage | null): void;
  getHyperbolicAtlas(): HyperbolicAtlasGpuPackage | null;
  render(options: RenderOptions): void;
  readPixels(target: Uint8Array): void;
  dispose(): void;
  runQcdHeatbathSweep?(options: QcdGpuSweepOptions): Promise<boolean>;
};

type TextureInfo = {
  texture: WebGLTexture;
  width: number;
  height: number;
  format: number;
  internalFormat: number;
  type: number;
};

const quadBufferData = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);

export function createGpuRenderer(gl: WebGL2RenderingContext): GpuRenderer {
  const program = createProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
  const attribLoc = gl.getAttribLocation(program, 'aPosition');

  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  if (!vao || !vbo) {
    throw new Error('Failed to allocate buffers for GPU renderer');
  }

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, quadBufferData, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(attribLoc);
  gl.vertexAttribPointer(attribLoc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const uniforms = locateUniforms(gl, program);
  const zeroSu7ProjectorMatrix = new Float32Array(21);
  const hopfAxesScratch = new Int32Array(6);
  const hopfMixScratch = new Float32Array(6);
  let hyperbolicAtlasPackage: HyperbolicAtlasGpuPackage | null = null;

  const baseTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA8,
    type: gl.UNSIGNED_BYTE,
  });
  const edgeTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const kurTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const ampTex = createTexture(gl, {
    format: gl.RED,
    internalFormat: gl.R32F,
    type: gl.FLOAT,
  });
  const volumeTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const tracerTexA = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA8,
    type: gl.UNSIGNED_BYTE,
  });
  const tracerTexB = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA8,
    type: gl.UNSIGNED_BYTE,
  });
  const tracerColorTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA8,
    type: gl.UNSIGNED_BYTE,
  });
  const atlasSampleTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const atlasJacobianTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const su7Tex0 = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const su7Tex1 = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const su7Tex2 = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const su7Tex3 = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const su7UnitaryTex = createTexture(gl, {
    format: gl.RGBA,
    internalFormat: gl.RGBA32F,
    type: gl.FLOAT,
  });
  const tracerFramebuffer = gl.createFramebuffer();
  if (!tracerFramebuffer) {
    throw new Error('Failed to allocate tracer framebuffer');
  }

  const state = {
    width: 1,
    height: 1,
    program,
    vao,
    textures: {
      base: baseTex,
      edge: edgeTex,
      kur: kurTex,
      amp: ampTex,
      volume: volumeTex,
      tracer: [tracerTexA, tracerTexB] as [TextureInfo, TextureInfo],
      composite: tracerColorTex,
      atlasSample: atlasSampleTex,
      atlasJacobian: atlasJacobianTex,
      su7: [su7Tex0, su7Tex1, su7Tex2, su7Tex3] as [
        TextureInfo,
        TextureInfo,
        TextureInfo,
        TextureInfo,
      ],
      su7Unitary: su7UnitaryTex,
    },
    framebuffers: {
      tracer: tracerFramebuffer,
    },
    tracerState: {
      readIndex: 0,
      needsClear: true,
    },
    gl,
  };

  const opsUniform = new Float32Array(32);
  const orientCosBuffer = new Float32Array(8);
  const orientSinBuffer = new Float32Array(8);
  let edgeBuffer: Float32Array | null = null;
  let phaseBuffer: Float32Array | null = null;
  let ampBuffer: Float32Array | null = null;
  let phaseFillState: 'empty' | 'zeros' | 'data' = 'empty';
  let volumeBuffer: Float32Array | null = null;
  let volumeFillState: 'empty' | 'zeros' | 'data' = 'empty';
  let atlasState: {
    width: number;
    height: number;
    sample: Float32Array;
    jacobian: Float32Array;
    sampleScale: number;
    maxHyperRadius: number;
  } | null = null;
  let atlasDirty = false;
  let su7State: {
    tileCols: number;
    tileRows: number;
    tileSize: number;
    tileTexWidth: number;
    tileTexRowsPerTile: number;
    pretransformed: boolean;
  } | null = null;

  let qcdContext: QcdGpuContext | null = null;
  let qcdContextPromise: Promise<QcdGpuContext | null> | null = null;

  const ensureQcdContext = async (): Promise<QcdGpuContext | null> => {
    if (qcdContext) {
      return qcdContext;
    }
    if (qcdContextPromise) {
      qcdContext = await qcdContextPromise;
      return qcdContext;
    }
    if (typeof navigator === 'undefined' || !(navigator as { gpu?: unknown }).gpu) {
      qcdContextPromise = Promise.resolve(null);
      return null;
    }
    qcdContextPromise = (async () => {
      try {
        const gpuNavigator = navigator as { gpu: any };
        const adapter = await gpuNavigator.gpu.requestAdapter?.();
        if (!adapter) {
          return null;
        }
        const device = await adapter.requestDevice?.();
        if (!device) {
          return null;
        }
        const code = await loadQcdKernelWgsl();
        const module = device.createShaderModule({
          label: 'qcd-gpu-kernel',
          code,
        });
        const pipeline =
          device.createComputePipelineAsync != null
            ? await device.createComputePipelineAsync({
                layout: 'auto',
                label: 'qcd-gpu-kernel',
                compute: {
                  module,
                  entryPoint: 'main',
                },
              })
            : device.createComputePipeline({
                layout: 'auto',
                label: 'qcd-gpu-kernel',
                compute: {
                  module,
                  entryPoint: 'main',
                },
              });
        const bindGroupLayout = pipeline.getBindGroupLayout(0);
        const uniformBuffer = device.createBuffer({
          size: QCD_UNIFORM_SIZE_BYTES,
          usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
          label: 'qcd-uniform-buffer',
        });
        const uniformArray = new ArrayBuffer(QCD_UNIFORM_SIZE_BYTES);
        const uniformU32 = new Uint32Array(uniformArray);
        const uniformF32 = new Float32Array(uniformArray);
        return {
          device,
          pipeline,
          bindGroupLayout,
          uniformBuffer,
          uniformArray,
          uniformU32,
          uniformF32,
          latticeBuffer: null,
          latticeCapacityFloats: 0,
          seedTexture: null,
          seedTextureView: null,
          seedSignature: null,
          seedKey: null,
        };
      } catch (error) {
        console.warn('[gpu-renderer] QCD WebGPU initialization failed', error);
        return null;
      }
    })();
    qcdContext = await qcdContextPromise;
    return qcdContext;
  };

  const ensureQcdSeedTexture = (
    ctx: QcdGpuContext,
    width: number,
    height: number,
    seed: number,
    scopeHash: number,
  ): { keyLo: number; keyHi: number } | null => {
    const signature = `${width}x${height}:${toUint32(seed)}:${scopeHash >>> 0}`;
    if (ctx.seedTexture && ctx.seedSignature === signature && ctx.seedKey) {
      return ctx.seedKey;
    }
    const payload = createQcdSeedTexturePayload(width, height, seed, scopeHash);
    ctx.seedTexture?.destroy();
    ctx.seedTexture = ctx.device.createTexture({
      label: 'qcd-seed-texture',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba32uint',
      usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    });
    ctx.seedTextureView = ctx.seedTexture.createView({ dimension: '2d' });
    ctx.device.queue.writeTexture(
      { texture: ctx.seedTexture },
      payload.data,
      { bytesPerRow: payload.bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    ctx.seedSignature = payload.signature;
    ctx.seedKey = { lo: payload.keyLo, hi: payload.keyHi };
    return ctx.seedKey;
  };

  const ensureQcdLatticeBuffer = (ctx: QcdGpuContext, floatCount: number): void => {
    const requiredBytes = floatCount * Float32Array.BYTES_PER_ELEMENT;
    if (ctx.latticeBuffer && ctx.latticeCapacityFloats >= floatCount) {
      return;
    }
    ctx.latticeBuffer?.destroy();
    const size = alignTo(requiredBytes, 256);
    ctx.latticeBuffer = ctx.device.createBuffer({
      size,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.COPY_SRC,
      label: 'qcd-lattice-buffer',
    });
    ctx.latticeCapacityFloats = size / Float32Array.BYTES_PER_ELEMENT;
  };

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.disable(gl.BLEND);
  gl.clearColor(0, 0, 0, 1);

  const renderer: GpuRenderer = {
    resize(width, height) {
      state.width = width;
      state.height = height;
      gl.viewport(0, 0, width, height);
      edgeBuffer = null;
      phaseBuffer = null;
      ampBuffer = null;
      phaseFillState = 'empty';
      volumeBuffer = null;
      volumeFillState = 'empty';
      ensureTextureStorage(gl, state.textures.tracer[0], width, height);
      ensureTextureStorage(gl, state.textures.tracer[1], width, height);
      ensureTextureStorage(gl, state.textures.composite, width, height);
      ensureTextureStorage(gl, state.textures.su7[0], width, height);
      ensureTextureStorage(gl, state.textures.su7[1], width, height);
      ensureTextureStorage(gl, state.textures.su7[2], width, height);
      ensureTextureStorage(gl, state.textures.su7[3], width, height);
      state.tracerState.readIndex = 0;
      state.tracerState.needsClear = true;
      atlasDirty = atlasState !== null;
      su7State = null;
    },
    uploadBase(image) {
      uploadImageData(gl, state.textures.base.texture, image);
    },
    uploadRim(field) {
      const total = state.width * state.height;
      const needed = total * 4;
      if (!edgeBuffer || edgeBuffer.length !== needed) {
        edgeBuffer = new Float32Array(needed);
      }
      if (!field) {
        edgeBuffer.fill(0);
      } else {
        assertRimField(field, 'gpu:rim');
        if (field.resolution.width !== state.width || field.resolution.height !== state.height) {
          throw new Error(
            `[gpuRenderer] rim field resolution ${field.resolution.width}x${field.resolution.height} mismatch renderer ${state.width}x${state.height}`,
          );
        }
        for (let i = 0; i < total; i++) {
          edgeBuffer[i * 4 + 0] = field.gx[i];
          edgeBuffer[i * 4 + 1] = field.gy[i];
          edgeBuffer[i * 4 + 2] = field.mag[i];
          edgeBuffer[i * 4 + 3] = 0;
        }
      }
      uploadFloatTexture(gl, state.textures.edge.texture, edgeBuffer, state.width, state.height);
    },
    uploadPhase(field) {
      const total = state.width * state.height;
      const needed = total * 4;
      if (!phaseBuffer || phaseBuffer.length !== needed) {
        phaseBuffer = new Float32Array(needed);
        phaseFillState = 'empty';
      }
      if (!ampBuffer || ampBuffer.length !== total) {
        ampBuffer = new Float32Array(total);
        phaseFillState = 'empty';
      }
      if (!field) {
        if (phaseFillState !== 'zeros') {
          phaseBuffer!.fill(0);
          ampBuffer!.fill(0);
          uploadFloatTexture(
            gl,
            state.textures.kur.texture,
            phaseBuffer!,
            state.width,
            state.height,
          );
          uploadScalarTexture(
            gl,
            state.textures.amp.texture,
            ampBuffer!,
            state.width,
            state.height,
          );
          phaseFillState = 'zeros';
        }
        return;
      }
      assertPhaseField(field, 'gpu:phase');
      if (field.resolution.width !== state.width || field.resolution.height !== state.height) {
        throw new Error(
          `[gpuRenderer] phase field resolution ${field.resolution.width}x${field.resolution.height} mismatch renderer ${state.width}x${state.height}`,
        );
      }
      for (let i = 0; i < total; i++) {
        phaseBuffer![i * 4 + 0] = field.gradX[i];
        phaseBuffer![i * 4 + 1] = field.gradY[i];
        phaseBuffer![i * 4 + 2] = field.vort[i];
        phaseBuffer![i * 4 + 3] = field.coh[i];
        ampBuffer![i] = field.amp[i];
      }
      uploadFloatTexture(gl, state.textures.kur.texture, phaseBuffer!, state.width, state.height);
      uploadScalarTexture(gl, state.textures.amp.texture, ampBuffer!, state.width, state.height);
      phaseFillState = 'data';
    },
    uploadVolume(field) {
      const total = state.width * state.height;
      const needed = total * 4;
      if (!volumeBuffer || volumeBuffer.length !== needed) {
        volumeBuffer = new Float32Array(needed);
        volumeFillState = 'empty';
      }
      if (!field) {
        if (volumeFillState !== 'zeros') {
          volumeBuffer!.fill(0);
          uploadFloatTexture(
            gl,
            state.textures.volume.texture,
            volumeBuffer!,
            state.width,
            state.height,
          );
          volumeFillState = 'zeros';
        }
        return;
      }
      assertVolumeField(field, 'gpu:volume');
      if (field.resolution.width !== state.width || field.resolution.height !== state.height) {
        throw new Error(
          `[gpuRenderer] volume field resolution ${field.resolution.width}x${field.resolution.height} mismatch renderer ${state.width}x${state.height}`,
        );
      }
      for (let i = 0; i < total; i++) {
        volumeBuffer![i * 4 + 0] = field.phase[i];
        volumeBuffer![i * 4 + 1] = field.depth[i];
        volumeBuffer![i * 4 + 2] = field.intensity[i];
        volumeBuffer![i * 4 + 3] = 0;
      }
      uploadFloatTexture(
        gl,
        state.textures.volume.texture,
        volumeBuffer!,
        state.width,
        state.height,
      );
      volumeFillState = 'data';
    },
    uploadSu7(payload) {
      if (!payload) {
        su7State = null;
        return;
      }
      const {
        width,
        height,
        vectors,
        tileData,
        tileCols,
        tileRows,
        tileSize,
        tileTexWidth,
        tileTexRowsPerTile,
      } = payload;
      if (width !== state.width || height !== state.height) {
        throw new Error(
          `[gpuRenderer] su7 vector resolution ${width}x${height} mismatch renderer ${state.width}x${state.height}`,
        );
      }
      for (let i = 0; i < vectors.length; i++) {
        const buffer = vectors[i];
        if (buffer.length !== width * height * 4) {
          throw new Error(
            `[gpuRenderer] su7 vector buffer ${i} length ${buffer.length} expected ${width * height * 4}`,
          );
        }
        uploadFloatTexture(gl, state.textures.su7[i].texture, buffer, width, height);
      }
      const tileTexHeight = tileRows * tileTexRowsPerTile;
      const expectedTileLength = tileTexWidth * tileTexHeight * 4;
      if (tileData.length !== expectedTileLength) {
        throw new Error(
          `[gpuRenderer] su7 unitary tile buffer length ${tileData.length} expected ${expectedTileLength}`,
        );
      }
      ensureTextureStorage(gl, state.textures.su7Unitary, tileTexWidth, tileTexHeight);
      uploadFloatTexture(
        gl,
        state.textures.su7Unitary.texture,
        tileData,
        tileTexWidth,
        tileTexHeight,
      );
      su7State = {
        tileCols,
        tileRows,
        tileSize,
        tileTexWidth,
        tileTexRowsPerTile,
        pretransformed: Boolean(payload.pretransformed),
      };
    },
    setHyperbolicAtlas(atlas) {
      hyperbolicAtlasPackage = atlas;
      if (!atlas) {
        atlasState = null;
        atlasDirty = false;
        return;
      }
      const { width, height, buffer, layout, metadata } = atlas;
      const texels = width * height;
      const stride = layout.stride;
      if (buffer.length !== texels * stride) {
        throw new Error(
          `[gpuRenderer] hyperbolic atlas buffer length ${buffer.length} does not match expected ${texels * stride}`,
        );
      }
      const sample = new Float32Array(texels * 4);
      const jacobian = new Float32Array(texels * 4);
      for (let i = 0; i < texels; i++) {
        const src = i * stride;
        const dst = i * 4;
        sample[dst + 0] = buffer[src + 0];
        sample[dst + 1] = buffer[src + 1];
        sample[dst + 2] = buffer[src + 2];
        sample[dst + 3] = buffer[src + 3];
        jacobian[dst + 0] = buffer[src + 4];
        jacobian[dst + 1] = buffer[src + 5];
        jacobian[dst + 2] = buffer[src + 6];
        jacobian[dst + 3] = buffer[src + 7];
      }
      const maxHyperRadius = 2 * metadata.curvatureScale * safeAtanh(metadata.diskLimit);
      atlasState = {
        width,
        height,
        sample,
        jacobian,
        sampleScale: metadata.maxRadius,
        maxHyperRadius,
      };
      atlasDirty = true;
    },
    getHyperbolicAtlas() {
      return hyperbolicAtlasPackage;
    },
    render(options) {
      gl.useProgram(program);

      ensureTextureStorage(gl, state.textures.tracer[0], state.width, state.height);
      ensureTextureStorage(gl, state.textures.tracer[1], state.width, state.height);
      ensureTextureStorage(gl, state.textures.composite, state.width, state.height);
      if (state.tracerState.needsClear || options.tracer.reset) {
        clearTexture(gl, state.framebuffers.tracer, state.textures.tracer[0]);
        clearTexture(gl, state.framebuffers.tracer, state.textures.tracer[1]);
        clearTexture(gl, state.framebuffers.tracer, state.textures.composite);
        state.tracerState.readIndex = 0;
        state.tracerState.needsClear = false;
      }
      const tracerReadIndex = state.tracerState.readIndex;
      const tracerRead = state.textures.tracer[tracerReadIndex];
      const tracerWrite = state.textures.tracer[1 - tracerReadIndex];

      bindTexture(gl, state.textures.base.texture, 0, uniforms.baseTex);
      bindTexture(gl, state.textures.edge.texture, 1, uniforms.edgeTex);
      bindTexture(gl, state.textures.kur.texture, 2, uniforms.kurTex);
      bindTexture(gl, state.textures.amp.texture, 3, uniforms.kurAmpTex);
      bindTexture(gl, state.textures.volume.texture, 4, uniforms.volumeTex);
      bindTexture(gl, tracerRead.texture, 5, uniforms.tracerTex);

      const su7Options = options.su7;
      const su7Active = Boolean(su7Options.enabled && su7State);
      if (su7Active && su7State) {
        bindTexture(gl, state.textures.su7[0].texture, 8, uniforms.su7Tex0);
        bindTexture(gl, state.textures.su7[1].texture, 9, uniforms.su7Tex1);
        bindTexture(gl, state.textures.su7[2].texture, 10, uniforms.su7Tex2);
        bindTexture(gl, state.textures.su7[3].texture, 11, uniforms.su7Tex3);
        bindTexture(gl, state.textures.su7Unitary.texture, 12, uniforms.su7UnitaryTex);
        gl.uniform1i(uniforms.su7Enabled, 1);
        gl.uniform1f(uniforms.su7Gain, su7Options.gain);
        const pretransformedActive =
          su7Options.pretransformed != null
            ? su7Options.pretransformed
            : (su7State?.pretransformed ?? false);
        gl.uniform1i(uniforms.su7Pretransformed, pretransformedActive ? 1 : 0);
        gl.uniform1i(uniforms.su7DecimationStride, Math.max(1, su7Options.decimationStride));
        const su7ModeEnum =
          su7Options.decimationMode === 'stride'
            ? 1
            : su7Options.decimationMode === 'edges'
              ? 2
              : 0;
        const projectorModeEnum =
          su7Options.projectorMode === 'composerWeights'
            ? 1
            : su7Options.projectorMode === 'overlaySplit'
              ? 2
              : su7Options.projectorMode === 'directRgb'
                ? 3
                : su7Options.projectorMode === 'matrix'
                  ? 4
                  : su7Options.projectorMode === 'hopfLens'
                    ? 5
                    : 0;
        gl.uniform1i(uniforms.su7DecimationMode, su7ModeEnum);
        gl.uniform1f(uniforms.su7ProjectorWeight, su7Options.projectorWeight);
        gl.uniform1i(uniforms.su7ProjectorMode, projectorModeEnum);
        const hopfLenses = su7Options.hopfLenses ?? [];
        const hopfCount = Math.min(hopfLenses.length, 3);
        if (uniforms.su7HopfLensCount) {
          gl.uniform1i(uniforms.su7HopfLensCount, hopfCount);
        }
        if (uniforms.su7HopfLensAxes) {
          hopfAxesScratch.fill(0);
          for (let i = 0; i < 3; i++) {
            const lens = i < hopfCount ? hopfLenses[i] : undefined;
            const axisA = lens ? Math.max(0, Math.min(6, Math.trunc(lens.axes[0]))) : 0;
            const axisB = lens ? Math.max(0, Math.min(6, Math.trunc(lens.axes[1]))) : 0;
            hopfAxesScratch[i * 2] = axisA;
            hopfAxesScratch[i * 2 + 1] = axisB;
          }
          gl.uniform2iv(uniforms.su7HopfLensAxes, hopfAxesScratch);
        }
        if (uniforms.su7HopfLensMix) {
          hopfMixScratch.fill(0);
          for (let i = 0; i < 3; i++) {
            const lens = i < hopfCount ? hopfLenses[i] : undefined;
            const baseMix = lens ? Math.min(Math.max(lens.baseMix, 0), 1) : 0;
            const fiberMix = lens ? Math.min(Math.max(lens.fiberMix, 0), 1) : 0;
            hopfMixScratch[i * 2] = baseMix;
            hopfMixScratch[i * 2 + 1] = fiberMix;
          }
          gl.uniform2fv(uniforms.su7HopfLensMix, hopfMixScratch);
        }
        gl.uniform1i(uniforms.su7TileCols, su7State.tileCols);
        gl.uniform1i(uniforms.su7TileRows, su7State.tileRows);
        gl.uniform1i(uniforms.su7TileSize, su7State.tileSize);
        gl.uniform1i(uniforms.su7UnitaryTexWidth, su7State.tileTexWidth);
        gl.uniform1i(uniforms.su7UnitaryTexRowsPerTile, su7State.tileTexRowsPerTile);
        if (uniforms.su7ProjectorMatrix) {
          if (su7Options.projectorMatrix && su7Options.projectorMatrix.length >= 21) {
            gl.uniform3fv(uniforms.su7ProjectorMatrix, su7Options.projectorMatrix);
          } else {
            gl.uniform3fv(uniforms.su7ProjectorMatrix, zeroSu7ProjectorMatrix);
          }
        }
      } else {
        gl.uniform1i(uniforms.su7Enabled, 0);
        gl.uniform1f(uniforms.su7Gain, su7Options.gain);
        gl.uniform1i(uniforms.su7DecimationStride, Math.max(1, su7Options.decimationStride));
        gl.uniform1i(uniforms.su7DecimationMode, 0);
        gl.uniform1f(uniforms.su7ProjectorWeight, su7Options.projectorWeight);
        gl.uniform1i(uniforms.su7ProjectorMode, 0);
        if (uniforms.su7HopfLensCount) {
          gl.uniform1i(uniforms.su7HopfLensCount, 0);
        }
        if (uniforms.su7HopfLensAxes) {
          hopfAxesScratch.fill(0);
          gl.uniform2iv(uniforms.su7HopfLensAxes, hopfAxesScratch);
        }
        if (uniforms.su7HopfLensMix) {
          hopfMixScratch.fill(0);
          gl.uniform2fv(uniforms.su7HopfLensMix, hopfMixScratch);
        }
        gl.uniform1i(uniforms.su7TileCols, su7State ? su7State.tileCols : 0);
        gl.uniform1i(uniforms.su7TileRows, su7State ? su7State.tileRows : 0);
        gl.uniform1i(uniforms.su7TileSize, su7State ? su7State.tileSize : 32);
        gl.uniform1i(uniforms.su7Pretransformed, 0);
        gl.uniform1i(uniforms.su7UnitaryTexWidth, su7State ? su7State.tileTexWidth : 1);
        gl.uniform1i(uniforms.su7UnitaryTexRowsPerTile, su7State ? su7State.tileTexRowsPerTile : 1);
        if (uniforms.su7ProjectorMatrix) {
          gl.uniform3fv(uniforms.su7ProjectorMatrix, zeroSu7ProjectorMatrix);
        }
      }
      const atlasAvailable =
        atlasState && atlasState.width === state.width && atlasState.height === state.height;
      const atlasEnabled = atlasAvailable && Math.abs(options.curvatureStrength) > 1e-4;
      if (atlasAvailable) {
        const { width, height, sample, jacobian, sampleScale, maxHyperRadius } = atlasState!;
        if (
          state.textures.atlasSample.width !== width ||
          state.textures.atlasSample.height !== height
        ) {
          ensureTextureStorage(gl, state.textures.atlasSample, width, height);
          ensureTextureStorage(gl, state.textures.atlasJacobian, width, height);
          atlasDirty = true;
        }
        if (atlasDirty) {
          uploadFloatTexture(gl, state.textures.atlasSample.texture, sample, width, height);
          uploadFloatTexture(gl, state.textures.atlasJacobian.texture, jacobian, width, height);
          atlasDirty = false;
        }
        if (atlasEnabled) {
          bindTexture(gl, state.textures.atlasSample.texture, 6, uniforms.atlasSampleTex);
          bindTexture(gl, state.textures.atlasJacobian.texture, 7, uniforms.atlasJacobianTex);
          gl.uniform1i(uniforms.atlasEnabled, 1);
          gl.uniform1f(uniforms.atlasSampleScale, sampleScale);
          gl.uniform1f(uniforms.atlasMaxHyperRadius, maxHyperRadius);
        } else {
          gl.uniform1i(uniforms.atlasEnabled, 0);
          gl.uniform1f(uniforms.atlasSampleScale, sampleScale);
          gl.uniform1f(uniforms.atlasMaxHyperRadius, maxHyperRadius);
        }
      } else {
        gl.uniform1i(uniforms.atlasEnabled, 0);
        gl.uniform1f(uniforms.atlasSampleScale, 1);
        gl.uniform1f(uniforms.atlasMaxHyperRadius, 0);
      }

      gl.uniform2f(uniforms.resolution, state.width, state.height);
      gl.uniform1f(uniforms.time, options.time);
      gl.uniform1f(uniforms.edgeThreshold, options.edgeThreshold);
      gl.uniform1f(uniforms.effectiveBlend, options.effectiveBlend);
      gl.uniform1i(uniforms.displayMode, options.displayMode);
      gl.uniform3f(
        uniforms.baseOffsets,
        options.baseOffsets[0],
        options.baseOffsets[1],
        options.baseOffsets[2],
      );
      gl.uniform1f(uniforms.sigma, options.sigma);
      gl.uniform1f(uniforms.jitter, options.jitter);
      gl.uniform1f(uniforms.jitterPhase, options.jitterPhase);
      gl.uniform1f(uniforms.breath, options.breath);
      gl.uniform1f(uniforms.muJ, options.muJ);
      gl.uniform1i(uniforms.phasePin, options.phasePin ? 1 : 0);
      gl.uniform1i(uniforms.microsaccade, options.microsaccade ? 1 : 0);
      gl.uniform1i(uniforms.polBins, options.polBins);
      gl.uniform1i(uniforms.thetaMode, options.thetaMode);
      gl.uniform1f(uniforms.thetaGlobal, options.thetaGlobal);
      gl.uniform1f(uniforms.contrast, options.contrast);
      gl.uniform1f(uniforms.frameGain, options.frameGain);
      gl.uniform1i(uniforms.rimEnabled, options.rimEnabled ? 1 : 0);
      gl.uniform1f(uniforms.rimAlpha, options.rimAlpha);
      gl.uniform1f(uniforms.warpAmp, options.warpAmp);
      gl.uniform1f(uniforms.surfaceBlend, options.surfaceBlend);
      gl.uniform1i(uniforms.surfaceRegion, options.surfaceRegion);
      gl.uniform1i(uniforms.surfEnabled, options.surfEnabled ? 1 : 0);
      gl.uniform1i(uniforms.kurEnabled, options.kurEnabled ? 1 : 0);
      gl.uniform1i(uniforms.volumeEnabled, options.volumeEnabled ? 1 : 0);
      gl.uniform1i(uniforms.useWallpaper, options.useWallpaper ? 1 : 0);
      gl.uniform2f(uniforms.canvasCenter, options.center[0], options.center[1]);
      gl.uniform1f(uniforms.curvatureStrength, options.curvatureStrength);
      gl.uniform1i(uniforms.curvatureMode, options.curvatureMode === 'klein' ? 1 : 0);
      gl.uniform1f(uniforms.kernelGain, options.kernel.gain);
      gl.uniform1f(uniforms.kernelK0, options.kernel.k0);
      gl.uniform1f(uniforms.kernelQ, options.kernel.Q);
      gl.uniform1f(uniforms.kernelAniso, options.kernel.anisotropy);
      gl.uniform1f(uniforms.kernelChirality, options.kernel.chirality);
      gl.uniform1f(uniforms.kernelTransparency, options.kernel.transparency);
      gl.uniform1f(uniforms.alive, options.alive ? 1 : 0);
      gl.uniform1f(uniforms.beta2, options.beta2);
      gl.uniform1f(uniforms.couplingRimSurfaceBlend, options.coupling.rimToSurfaceBlend);
      gl.uniform1f(uniforms.couplingRimSurfaceAlign, options.coupling.rimToSurfaceAlign);
      gl.uniform1f(uniforms.couplingSurfaceRimOffset, options.coupling.surfaceToRimOffset);
      gl.uniform1f(uniforms.couplingSurfaceRimSigma, options.coupling.surfaceToRimSigma);
      gl.uniform1f(uniforms.couplingSurfaceRimHue, options.coupling.surfaceToRimHue);
      gl.uniform1f(uniforms.couplingKurTransparency, options.coupling.kurToTransparency);
      gl.uniform1f(uniforms.couplingKurOrientation, options.coupling.kurToOrientation);
      gl.uniform1f(uniforms.couplingKurChirality, options.coupling.kurToChirality);
      gl.uniform1f(uniforms.couplingScale, options.couplingScale);
      gl.uniform1f(uniforms.couplingVolumePhaseHue, options.coupling.volumePhaseToHue);
      gl.uniform1f(uniforms.couplingVolumeDepthWarp, options.coupling.volumeDepthToWarp);
      gl.uniform1fv(uniforms.composerExposure, options.composerExposure);
      gl.uniform1fv(uniforms.composerGamma, options.composerGamma);
      gl.uniform1fv(uniforms.composerWeight, options.composerWeight);
      gl.uniform2f(
        uniforms.composerBlendGain,
        options.composerBlendGain[0],
        options.composerBlendGain[1],
      );
      gl.uniform1i(uniforms.tracerEnabled, options.tracer.enabled ? 1 : 0);
      gl.uniform1f(uniforms.tracerGain, options.tracer.gain);
      gl.uniform1f(uniforms.tracerTau, options.tracer.tau);
      gl.uniform1f(uniforms.tracerModDepth, options.tracer.modulationDepth);
      gl.uniform1f(uniforms.tracerModFrequency, options.tracer.modulationFrequency);
      gl.uniform1f(uniforms.tracerModPhase, options.tracer.modulationPhase);
      gl.uniform1f(uniforms.tracerDt, options.tracer.dt);

      const ops = options.ops;
      gl.uniform1i(uniforms.opsCount, ops.length);
      for (let i = 0; i < 8; i++) {
        const base = i * 4;
        if (i < ops.length) {
          const op = ops[i];
          opsUniform[base + 0] = op.kind;
          opsUniform[base + 1] = op.angle;
          opsUniform[base + 2] = op.tx;
          opsUniform[base + 3] = op.ty;
        } else {
          opsUniform[base + 0] = 0;
          opsUniform[base + 1] = 0;
          opsUniform[base + 2] = 0;
          opsUniform[base + 3] = 0;
        }
      }
      gl.uniform4fv(uniforms.ops, opsUniform);

      gl.uniform1i(uniforms.orientCount, options.orientations.cos.length);
      orientCosBuffer.fill(0);
      orientCosBuffer.set(
        options.orientations.cos.subarray(0, Math.min(8, options.orientations.cos.length)),
      );
      gl.uniform1fv(uniforms.orientCos, orientCosBuffer);
      orientSinBuffer.fill(0);
      orientSinBuffer.set(
        options.orientations.sin.subarray(0, Math.min(8, options.orientations.sin.length)),
      );
      gl.uniform1fv(uniforms.orientSin, orientSinBuffer);

      gl.bindFramebuffer(gl.FRAMEBUFFER, state.framebuffers.tracer);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        state.textures.composite.texture,
        0,
      );
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT1,
        gl.TEXTURE_2D,
        tracerWrite.texture,
        0,
      );
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        throw new Error(`[gpuRenderer] framebuffer incomplete: 0x${status.toString(16)}`);
      }

      gl.viewport(0, 0, state.width, state.height);
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);

      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, state.framebuffers.tracer);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      gl.blitFramebuffer(
        0,
        0,
        state.width,
        state.height,
        0,
        0,
        state.width,
        state.height,
        gl.COLOR_BUFFER_BIT,
        gl.LINEAR,
      );
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);

      gl.bindFramebuffer(gl.FRAMEBUFFER, state.framebuffers.tracer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, null, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.readBuffer(gl.BACK);

      state.tracerState.readIndex = 1 - tracerReadIndex;
    },
    readPixels(target) {
      if (target.length < state.width * state.height * 4) {
        throw new Error('readPixels target buffer too small');
      }
      gl.readPixels(0, 0, state.width, state.height, gl.RGBA, gl.UNSIGNED_BYTE, target);
    },
    async runQcdHeatbathSweep(options) {
      try {
        const ctx = await ensureQcdContext();
        if (!ctx) {
          return false;
        }
        const {
          lattice,
          width,
          height,
          siteStride,
          linkStride,
          rowStride,
          complexStride,
          beta,
          parity,
          axis,
          sweepIndex,
          overRelaxationSteps,
          seed,
          scope,
        } = options;
        if (!(lattice instanceof Float32Array)) {
          throw new TypeError('[gpu-renderer] QCD lattice buffer must be a Float32Array');
        }
        const widthInt = Math.max(1, Math.trunc(width));
        const heightInt = Math.max(1, Math.trunc(height));
        const siteStrideInt = Math.max(1, Math.trunc(siteStride));
        const linkStrideInt = Math.max(1, Math.trunc(linkStride));
        const rowStrideInt = Math.max(1, Math.trunc(rowStride));
        const complexStrideInt = Math.max(1, Math.trunc(complexStride));
        const expectedLength = widthInt * heightInt * siteStrideInt;
        if (lattice.length !== expectedLength) {
          throw new RangeError(
            `[gpu-renderer] QCD lattice length mismatch (expected ${expectedLength}, received ${lattice.length})`,
          );
        }
        const seedValue = toUint32(seed ?? 0);
        const scopeHash = hashScope(scope);
        const key = ensureQcdSeedTexture(ctx, widthInt, heightInt, seedValue, scopeHash);
        if (!key || !ctx.seedTextureView) {
          return false;
        }

        ensureQcdLatticeBuffer(ctx, lattice.length);
        if (!ctx.latticeBuffer) {
          return false;
        }

        ctx.device.queue.writeBuffer(
          ctx.latticeBuffer,
          0,
          lattice.buffer,
          lattice.byteOffset,
          lattice.byteLength,
        );

        ctx.uniformU32[0] = toUint32(widthInt);
        ctx.uniformU32[1] = toUint32(heightInt);
        ctx.uniformU32[2] = toUint32(parity & 1);
        ctx.uniformU32[3] = toUint32(Math.max(0, Math.floor(overRelaxationSteps)));
        ctx.uniformU32[4] = toUint32(siteStrideInt);
        ctx.uniformU32[5] = toUint32(linkStrideInt);
        ctx.uniformU32[6] = toUint32(rowStrideInt);
        ctx.uniformU32[7] = toUint32(complexStrideInt);
        ctx.uniformU32[8] = key.lo;
        ctx.uniformU32[9] = key.hi;
        ctx.uniformU32[10] = toUint32(Math.max(0, Math.trunc(sweepIndex)));
        ctx.uniformU32[11] = axis === 'y' ? 1 : 0;
        ctx.uniformF32[12] = Number.isFinite(beta) ? beta : 0;
        ctx.uniformF32[13] = 0;
        ctx.uniformF32[14] = 0;
        ctx.uniformF32[15] = 0;
        ctx.device.queue.writeBuffer(ctx.uniformBuffer, 0, ctx.uniformArray);

        const bindGroup = ctx.device.createBindGroup({
          layout: ctx.bindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: { buffer: ctx.latticeBuffer },
            },
            {
              binding: 1,
              resource: ctx.seedTextureView,
            },
            {
              binding: 2,
              resource: { buffer: ctx.uniformBuffer },
            },
          ],
          label: 'qcd-bind-group',
        });

        const queueAny = ctx.device.queue as {
          readBuffer?: (
            buffer: any,
            offset: number,
            data: ArrayBuffer | ArrayBufferView,
            dataOffset?: number,
            size?: number,
          ) => Promise<void>;
        };
        const supportsReadBuffer = typeof queueAny.readBuffer === 'function';
        let readbackBuffer: GPUBuffer | null = null;
        if (!supportsReadBuffer) {
          const copyBytes = alignTo(lattice.byteLength, 4);
          readbackBuffer = ctx.device.createBuffer({
            size: copyBytes,
            usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ,
            label: 'qcd-readback',
          });
        }

        const workgroupX = Math.ceil(widthInt / QCD_WORKGROUP_SIZE_X);
        const workgroupY = Math.ceil(heightInt / QCD_WORKGROUP_SIZE_Y);
        const encoder = ctx.device.createCommandEncoder({ label: 'qcd-command-encoder' });
        const pass = encoder.beginComputePass({ label: 'qcd-compute-pass' });
        pass.setPipeline(ctx.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroupX, workgroupY);
        pass.end();
        if (!supportsReadBuffer && readbackBuffer) {
          encoder.copyBufferToBuffer(ctx.latticeBuffer, 0, readbackBuffer, 0, lattice.byteLength);
        }
        ctx.device.queue.submit([encoder.finish()]);
        await ctx.device.queue.onSubmittedWorkDone();

        if (supportsReadBuffer && queueAny.readBuffer) {
          await queueAny.readBuffer(
            ctx.latticeBuffer,
            0,
            lattice.buffer,
            lattice.byteOffset,
            lattice.byteLength,
          );
        } else if (readbackBuffer) {
          await readbackBuffer.mapAsync(GPU_MAP_MODE.READ);
          const mapped = readbackBuffer.getMappedRange();
          lattice.set(new Float32Array(mapped, 0, lattice.length));
          readbackBuffer.unmap();
          readbackBuffer.destroy();
        }

        return true;
      } catch (error) {
        console.warn('[gpu-renderer] QCD heatbath sweep failed', error);
        return false;
      }
    },
    dispose() {
      gl.deleteProgram(program);
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(vbo);
      gl.deleteTexture(state.textures.base.texture);
      gl.deleteTexture(state.textures.edge.texture);
      gl.deleteTexture(state.textures.kur.texture);
      gl.deleteTexture(state.textures.amp.texture);
      gl.deleteTexture(state.textures.volume.texture);
      gl.deleteTexture(state.textures.tracer[0].texture);
      gl.deleteTexture(state.textures.tracer[1].texture);
      gl.deleteTexture(state.textures.composite.texture);
      gl.deleteTexture(state.textures.atlasSample.texture);
      gl.deleteTexture(state.textures.atlasJacobian.texture);
      gl.deleteTexture(state.textures.su7[0].texture);
      gl.deleteTexture(state.textures.su7[1].texture);
      gl.deleteTexture(state.textures.su7[2].texture);
      gl.deleteTexture(state.textures.su7[3].texture);
      gl.deleteTexture(state.textures.su7Unitary.texture);
      gl.deleteFramebuffer(state.framebuffers.tracer);
      hyperbolicAtlasPackage = null;
      atlasState = null;
      atlasDirty = false;
      su7State = null;
      if (qcdContext) {
        qcdContext.uniformBuffer.destroy();
        qcdContext.latticeBuffer?.destroy();
        qcdContext.seedTexture?.destroy();
        qcdContext.seedTextureView = null;
        qcdContext.seedSignature = null;
        qcdContext.seedKey = null;
        qcdContext = null;
      }
      qcdContextPromise = null;
    },
  };

  return renderer;
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  if (!program) throw new Error('Failed to create program');
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${info ?? 'unknown error'}`);
  }
  gl.detachShader(program, vertexShader);
  gl.detachShader(program, fragmentShader);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  return program;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${info ?? 'unknown error'}`);
  }
  return shader;
}

function createTexture(
  gl: WebGL2RenderingContext,
  opts: { format: number; internalFormat: number; type: number },
): TextureInfo {
  const texture = gl.createTexture();
  if (!texture) {
    throw new Error('Failed to create texture');
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return {
    texture,
    width: 0,
    height: 0,
    format: opts.format,
    internalFormat: opts.internalFormat,
    type: opts.type,
  };
}

function ensureTextureStorage(
  gl: WebGL2RenderingContext,
  info: TextureInfo,
  width: number,
  height: number,
) {
  if (info.width === width && info.height === height) {
    return;
  }
  gl.bindTexture(gl.TEXTURE_2D, info.texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    info.internalFormat,
    width,
    height,
    0,
    info.format,
    info.type,
    null,
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
  info.width = width;
  info.height = height;
}

function clearTexture(
  gl: WebGL2RenderingContext,
  framebuffer: WebGLFramebuffer,
  info: TextureInfo,
) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, info.texture, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
    image.data,
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function uploadFloatTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  data: Float32Array,
  width: number,
  height: number,
) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function uploadScalarTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  data: Float32Array,
  width: number,
  height: number,
) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function bindTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  unit: number,
  location: WebGLUniformLocation,
) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(location, unit);
}

function locateUniforms(gl: WebGL2RenderingContext, program: WebGLProgram) {
  return {
    baseTex: getUniform(gl, program, 'uBaseTex'),
    edgeTex: getUniform(gl, program, 'uEdgeTex'),
    kurTex: getUniform(gl, program, 'uKurTex'),
    kurAmpTex: getUniform(gl, program, 'uKurAmpTex'),
    volumeTex: getUniform(gl, program, 'uVolumeTex'),
    tracerTex: getUniform(gl, program, 'uTracerState'),
    atlasSampleTex: getUniform(gl, program, 'uAtlasSampleTex'),
    atlasJacobianTex: getUniform(gl, program, 'uAtlasJacobianTex'),
    su7Tex0: getUniform(gl, program, 'uSu7Tex0'),
    su7Tex1: getUniform(gl, program, 'uSu7Tex1'),
    su7Tex2: getUniform(gl, program, 'uSu7Tex2'),
    su7Tex3: getUniform(gl, program, 'uSu7Tex3'),
    su7UnitaryTex: getUniform(gl, program, 'uSu7UnitaryTex'),
    resolution: getUniform(gl, program, 'uResolution'),
    time: getUniform(gl, program, 'uTime'),
    edgeThreshold: getUniform(gl, program, 'uEdgeThreshold'),
    effectiveBlend: getUniform(gl, program, 'uEffectiveBlend'),
    displayMode: getUniform(gl, program, 'uDisplayMode'),
    baseOffsets: getUniform(gl, program, 'uBaseOffsets'),
    sigma: getUniform(gl, program, 'uSigma'),
    jitter: getUniform(gl, program, 'uJitter'),
    jitterPhase: getUniform(gl, program, 'uJitterPhase'),
    breath: getUniform(gl, program, 'uBreath'),
    muJ: getUniform(gl, program, 'uMuJ'),
    phasePin: getUniform(gl, program, 'uPhasePin'),
    microsaccade: getUniform(gl, program, 'uMicrosaccade'),
    polBins: getUniform(gl, program, 'uPolBins'),
    thetaMode: getUniform(gl, program, 'uThetaMode'),
    thetaGlobal: getUniform(gl, program, 'uThetaGlobal'),
    contrast: getUniform(gl, program, 'uContrast'),
    frameGain: getUniform(gl, program, 'uFrameGain'),
    rimEnabled: getUniform(gl, program, 'uRimEnabled'),
    rimAlpha: getUniform(gl, program, 'uRimAlpha'),
    warpAmp: getUniform(gl, program, 'uWarpAmp'),
    surfaceBlend: getUniform(gl, program, 'uSurfaceBlend'),
    surfaceRegion: getUniform(gl, program, 'uSurfaceRegion'),
    surfEnabled: getUniform(gl, program, 'uSurfEnabled'),
    kurEnabled: getUniform(gl, program, 'uKurEnabled'),
    volumeEnabled: getUniform(gl, program, 'uVolumeEnabled'),
    useWallpaper: getUniform(gl, program, 'uUseWallpaper'),
    opsCount: getUniform(gl, program, 'uOpsCount'),
    ops: getUniform(gl, program, 'uOps'),
    orientCount: getUniform(gl, program, 'uOrientCount'),
    orientCos: getUniform(gl, program, 'uOrientCos'),
    orientSin: getUniform(gl, program, 'uOrientSin'),
    canvasCenter: getUniform(gl, program, 'uCanvasCenter'),
    curvatureStrength: getUniform(gl, program, 'uCurvatureStrength'),
    curvatureMode: getUniform(gl, program, 'uCurvatureMode'),
    atlasEnabled: getUniform(gl, program, 'uAtlasEnabled'),
    atlasSampleScale: getUniform(gl, program, 'uAtlasSampleScale'),
    atlasMaxHyperRadius: getUniform(gl, program, 'uAtlasMaxHyperRadius'),
    kernelGain: getUniform(gl, program, 'uKernelGain'),
    kernelK0: getUniform(gl, program, 'uKernelK0'),
    kernelQ: getUniform(gl, program, 'uKernelQ'),
    kernelAniso: getUniform(gl, program, 'uKernelAniso'),
    kernelChirality: getUniform(gl, program, 'uKernelChirality'),
    kernelTransparency: getUniform(gl, program, 'uKernelTransparency'),
    alive: getUniform(gl, program, 'uAlive'),
    beta2: getUniform(gl, program, 'uBeta2'),
    couplingRimSurfaceBlend: getUniform(gl, program, 'uCouplingRimSurfaceBlend'),
    couplingRimSurfaceAlign: getUniform(gl, program, 'uCouplingRimSurfaceAlign'),
    couplingSurfaceRimOffset: getUniform(gl, program, 'uCouplingSurfaceRimOffset'),
    couplingSurfaceRimSigma: getUniform(gl, program, 'uCouplingSurfaceRimSigma'),
    couplingSurfaceRimHue: getUniform(gl, program, 'uCouplingSurfaceRimHue'),
    couplingKurTransparency: getUniform(gl, program, 'uCouplingKurTransparency'),
    couplingKurOrientation: getUniform(gl, program, 'uCouplingKurOrientation'),
    couplingKurChirality: getUniform(gl, program, 'uCouplingKurChirality'),
    couplingScale: getUniform(gl, program, 'uCouplingScale'),
    couplingVolumePhaseHue: getUniform(gl, program, 'uCouplingVolumePhaseHue'),
    couplingVolumeDepthWarp: getUniform(gl, program, 'uCouplingVolumeDepthWarp'),
    composerExposure: getUniform(gl, program, 'uComposerExposure'),
    composerGamma: getUniform(gl, program, 'uComposerGamma'),
    composerWeight: getUniform(gl, program, 'uComposerWeight'),
    composerBlendGain: getUniform(gl, program, 'uComposerBlendGain'),
    tracerEnabled: getUniform(gl, program, 'uTracerEnabled'),
    tracerGain: getUniform(gl, program, 'uTracerGain'),
    tracerTau: getUniform(gl, program, 'uTracerTau'),
    tracerModDepth: getUniform(gl, program, 'uTracerModDepth'),
    tracerModFrequency: getUniform(gl, program, 'uTracerModFrequency'),
    tracerModPhase: getUniform(gl, program, 'uTracerModPhase'),
    tracerDt: getUniform(gl, program, 'uTracerDt'),
    su7Enabled: getUniform(gl, program, 'uSu7Enabled'),
    su7Gain: getUniform(gl, program, 'uSu7Gain'),
    su7Pretransformed: getUniform(gl, program, 'uSu7Pretransformed'),
    su7DecimationStride: getUniform(gl, program, 'uSu7DecimationStride'),
    su7DecimationMode: getUniform(gl, program, 'uSu7DecimationMode'),
    su7ProjectorWeight: getUniform(gl, program, 'uSu7ProjectorWeight'),
    su7ProjectorMode: getUniform(gl, program, 'uSu7ProjectorMode'),
    su7TileCols: getUniform(gl, program, 'uSu7TileCols'),
    su7TileRows: getUniform(gl, program, 'uSu7TileRows'),
    su7TileSize: getUniform(gl, program, 'uSu7TileSize'),
    su7UnitaryTexWidth: getUniform(gl, program, 'uSu7UnitaryTexWidth'),
    su7UnitaryTexRowsPerTile: getUniform(gl, program, 'uSu7UnitaryTexRowsPerTile'),
    su7ProjectorMatrix: getOptionalUniform(gl, program, 'uSu7ProjectorMatrix'),
    su7HopfLensCount: getOptionalUniform(gl, program, 'uSu7HopfLensCount'),
    su7HopfLensAxes: getOptionalUniform(gl, program, 'uSu7HopfLensAxes'),
    su7HopfLensMix: getOptionalUniform(gl, program, 'uSu7HopfLensMix'),
  };
}

function getUniform(gl: WebGL2RenderingContext, program: WebGLProgram, name: string) {
  const location = gl.getUniformLocation(program, name);
  if (!location) {
    throw new Error(`Uniform ${name} not found`);
  }
  return location;
}

function getOptionalUniform(gl: WebGL2RenderingContext, program: WebGLProgram, name: string) {
  return gl.getUniformLocation(program, name);
}
