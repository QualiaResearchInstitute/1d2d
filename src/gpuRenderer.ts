/* eslint-disable @typescript-eslint/no-use-before-define */
import {
  FIELD_STRUCTS_GLSL,
  assertPhaseField,
  assertRimField,
  assertVolumeField,
  type PhaseField,
  type RimField,
  type VolumeField
} from "./fields/contracts";
import type { CouplingConfig } from "./pipeline/rainbowFrame";

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

${FIELD_STRUCTS_GLSL}

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uBaseTex;
uniform sampler2D uEdgeTex;
uniform sampler2D uKurTex;
uniform sampler2D uKurAmpTex;
uniform sampler2D uVolumeTex;

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

#define COMPOSER_FIELD_SURFACE 0
#define COMPOSER_FIELD_RIM 1
#define COMPOSER_FIELD_KUR 2
#define COMPOSER_FIELD_VOLUME 3

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
  vec2 fragCoord = vec2(vUv.x * uResolution.x, vUv.y * uResolution.y);
  vec2 fragClamped = clamp(fragCoord, vec2(0.0), uResolution - vec2(1.0));
  int xPix = int(fragClamped.x);
  int yPix = int(fragClamped.y);
  ivec2 icoord = ivec2(xPix, yPix);
  int width = int(uResolution.x);
  int height = int(uResolution.y);

  vec4 edgeSample = fetchEdge(icoord);
  vec3 baseRgb = texelFetch(uBaseTex, icoord, 0).rgb;
  float magVal = edgeSample.b;

  vec3 volumeSample = vec3(0.0);
  float volumePhaseVal = 0.0;
  float volumeDepthVal = 0.0;
  float volumeIntensityVal = 0.0;
  float volumeDepthGrad = 0.0;
  if (uVolumeEnabled == 1) {
    volumeSample = texelFetch(uVolumeTex, icoord, 0).rgb;
    volumePhaseVal = volumeSample.r;
    volumeDepthVal = volumeSample.g;
    volumeIntensityVal = volumeSample.b;
    int leftX = max(xPix - 1, 0);
    int rightX = min(xPix + 1, width - 1);
    int upY = max(yPix - 1, 0);
    int downY = min(yPix + 1, height - 1);
    float depthLeft = texelFetch(uVolumeTex, ivec2(leftX, yPix), 0).g;
    float depthRight = texelFetch(uVolumeTex, ivec2(rightX, yPix), 0).g;
    float depthUp = texelFetch(uVolumeTex, ivec2(xPix, upY), 0).g;
    float depthDown = texelFetch(uVolumeTex, ivec2(xPix, downY), 0).g;
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
        uComposerWeight[COMPOSER_FIELD_VOLUME];
    warpFlow *= boost;
  }

  vec2 kurGrad = vec2(0.0);
  float vortVal = 0.0;
  float cohVal = 0.0;
  float ampVal = 0.0;
  if (uKurEnabled == 1) {
    vec4 kurSample = fetchKur(icoord);
    kurGrad = kurSample.xy;
    vortVal = kurSample.z;
    cohVal = clamp(kurSample.w, 0.0, 1.0);
    ampVal = clamp(texelFetch(uKurAmpTex, icoord, 0).r, 0.0, 1.0);
  }

  vec2 combinedFlow = warpFlow + kurGrad;
  vec2 gradEdge = edgeSample.rg;

  vec3 resultRgb = baseRgb;
  float rimEnergy = 0.0;

  float couplingScale = uCouplingScale;
  float sigmaFloorBase = max(0.25, uSigma * 0.25);

  vec2 normal = vec2(0.0);
  vec2 tangent = vec2(0.0);
  float edgeLen = length(gradEdge);
  if (edgeLen > 1e-8) {
    normal = gradEdge / edgeLen;
    tangent = vec2(-normal.y, normal.x);
  }

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
        float mixW = clamp(uCouplingKurOrientation * couplingScale * uComposerWeight[COMPOSER_FIELD_KUR], 0.0, 1.0);
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
          uComposerWeight[COMPOSER_FIELD_SURFACE] *
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
          uComposerWeight[COMPOSER_FIELD_SURFACE] *
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
          uComposerWeight[COMPOSER_FIELD_VOLUME] *
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
          uComposerWeight[COMPOSER_FIELD_SURFACE] *
          signedMag *
          0.9,
        -1.4,
        1.4
      );
    }

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

    float chiPhase = TAU * uKernelK0 * dot(pos, tangent) * 0.002 + hueShift;
    float chBase = uKernelChirality;
    if (uKurEnabled == 1 && uCouplingKurChirality > 0.0) {
      float vortClamped = clamp(vortVal, -1.0, 1.0);
      float vortScaled = 0.5 * couplingScale * uCouplingKurChirality * uComposerWeight[COMPOSER_FIELD_KUR] * vortClamped;
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
    if (uKurEnabled == 1 && uCouplingKurTransparency > 0.0) {
      float kurMeasure = applyComposerScalar(0.5 * (cohVal + ampVal), COMPOSER_FIELD_KUR);
      float boost = clamp(
        1.0 +
          couplingScale *
            uCouplingKurTransparency *
            uComposerWeight[COMPOSER_FIELD_KUR] *
            (kurMeasure - 0.5) *
            1.5,
        0.1,
        2.5
      );
      pixelBlend = clamp(pixelBlend * boost, 0.0, 1.0);
    }
    pixelBlend = clamp(pixelBlend * uComposerWeight[COMPOSER_FIELD_RIM], 0.0, 1.0);

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
                uComposerWeight[COMPOSER_FIELD_SURFACE] *
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
            clamp(uCouplingKurOrientation * couplingScale * uComposerWeight[COMPOSER_FIELD_KUR] * 0.75, 0.0, 1.0);
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
      if (uCouplingRimSurfaceBlend > 0.0) {
        float energy = responseSoft(
          rimEnergy / (0.75 + 0.25 * uKernelGain),
          uCouplingRimSurfaceBlend
        );
        float rimBoost = clamp(
          1.0 + couplingScale * uCouplingRimSurfaceBlend * uComposerWeight[COMPOSER_FIELD_RIM] * energy,
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
              uComposerWeight[COMPOSER_FIELD_KUR] *
              (kurMeasure - 0.5) *
              1.5,
          0.1,
          3.0
        );
        sb *= boost;
      }
      sb = clamp(sb * uComposerWeight[COMPOSER_FIELD_SURFACE], 0.0, 1.0);
      resultRgb = mix(resultRgb, warped, sb);
    }
  }

  fragColor = vec4(resultRgb, 1.0);
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
  uploadRim(field: RimField | null): void;
  uploadPhase(field: PhaseField | null): void;
  uploadVolume(field: VolumeField | null): void;
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
  const ampTex = createTexture(gl, {
    format: gl.RED,
    internalFormat: gl.R32F,
    type: gl.FLOAT
  });
  const volumeTex = createTexture(gl, {
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
      kur: kurTex,
      amp: ampTex,
      volume: volumeTex
    },
    gl
  };

  const opsUniform = new Float32Array(32);
  const orientCosBuffer = new Float32Array(8);
  const orientSinBuffer = new Float32Array(8);
  let edgeBuffer: Float32Array | null = null;
  let phaseBuffer: Float32Array | null = null;
  let ampBuffer: Float32Array | null = null;
  let phaseFillState: "empty" | "zeros" | "data" = "empty";
  let volumeBuffer: Float32Array | null = null;
  let volumeFillState: "empty" | "zeros" | "data" = "empty";

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
      phaseFillState = "empty";
      volumeBuffer = null;
      volumeFillState = "empty";
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
        assertRimField(field, "gpu:rim");
        if (field.resolution.width !== state.width || field.resolution.height !== state.height) {
          throw new Error(
            `[gpuRenderer] rim field resolution ${field.resolution.width}x${field.resolution.height} mismatch renderer ${state.width}x${state.height}`
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
        phaseFillState = "empty";
      }
      if (!ampBuffer || ampBuffer.length !== total) {
        ampBuffer = new Float32Array(total);
        phaseFillState = "empty";
      }
      if (!field) {
        if (phaseFillState !== "zeros") {
          phaseBuffer!.fill(0);
          ampBuffer!.fill(0);
          uploadFloatTexture(gl, state.textures.kur.texture, phaseBuffer!, state.width, state.height);
          uploadScalarTexture(gl, state.textures.amp.texture, ampBuffer!, state.width, state.height);
          phaseFillState = "zeros";
        }
        return;
      }
      assertPhaseField(field, "gpu:phase");
      if (field.resolution.width !== state.width || field.resolution.height !== state.height) {
        throw new Error(
          `[gpuRenderer] phase field resolution ${field.resolution.width}x${field.resolution.height} mismatch renderer ${state.width}x${state.height}`
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
      phaseFillState = "data";
    },
    uploadVolume(field) {
      const total = state.width * state.height;
      const needed = total * 4;
      if (!volumeBuffer || volumeBuffer.length !== needed) {
        volumeBuffer = new Float32Array(needed);
        volumeFillState = "empty";
      }
      if (!field) {
        if (volumeFillState !== "zeros") {
          volumeBuffer!.fill(0);
          uploadFloatTexture(gl, state.textures.volume.texture, volumeBuffer!, state.width, state.height);
          volumeFillState = "zeros";
        }
        return;
      }
      assertVolumeField(field, "gpu:volume");
      if (field.resolution.width !== state.width || field.resolution.height !== state.height) {
        throw new Error(
          `[gpuRenderer] volume field resolution ${field.resolution.width}x${field.resolution.height} mismatch renderer ${state.width}x${state.height}`
        );
      }
      for (let i = 0; i < total; i++) {
        volumeBuffer![i * 4 + 0] = field.phase[i];
        volumeBuffer![i * 4 + 1] = field.depth[i];
        volumeBuffer![i * 4 + 2] = field.intensity[i];
        volumeBuffer![i * 4 + 3] = 0;
      }
      uploadFloatTexture(gl, state.textures.volume.texture, volumeBuffer!, state.width, state.height);
      volumeFillState = "data";
    },
    render(options) {
      gl.useProgram(program);

      bindTexture(gl, state.textures.base.texture, 0, uniforms.baseTex);
      bindTexture(gl, state.textures.edge.texture, 1, uniforms.edgeTex);
      bindTexture(gl, state.textures.kur.texture, 2, uniforms.kurTex);
      bindTexture(gl, state.textures.amp.texture, 3, uniforms.kurAmpTex);
      bindTexture(gl, state.textures.volume.texture, 4, uniforms.volumeTex);

      gl.uniform2f(uniforms.resolution, state.width, state.height);
      gl.uniform1f(uniforms.time, options.time);
      gl.uniform1f(uniforms.edgeThreshold, options.edgeThreshold);
      gl.uniform1f(uniforms.effectiveBlend, options.effectiveBlend);
      gl.uniform1i(uniforms.displayMode, options.displayMode);
      gl.uniform3f(uniforms.baseOffsets, options.baseOffsets[0], options.baseOffsets[1], options.baseOffsets[2]);
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
      gl.uniform1f(uniforms.kernelGain, options.kernel.gain);
      gl.uniform1f(uniforms.kernelK0, options.kernel.k0);
      gl.uniform1f(uniforms.kernelQ, options.kernel.Q);
      gl.uniform1f(uniforms.kernelAniso, options.kernel.anisotropy);
      gl.uniform1f(uniforms.kernelChirality, options.kernel.chirality);
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
      gl.uniform2f(uniforms.composerBlendGain, options.composerBlendGain[0], options.composerBlendGain[1]);

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
      gl.deleteTexture(state.textures.amp.texture);
      gl.deleteTexture(state.textures.volume.texture);
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

function uploadScalarTexture(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  data: Float32Array,
  width: number,
  height: number
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
    kurAmpTex: getUniform(gl, program, "uKurAmpTex"),
    volumeTex: getUniform(gl, program, "uVolumeTex"),
    resolution: getUniform(gl, program, "uResolution"),
    time: getUniform(gl, program, "uTime"),
    edgeThreshold: getUniform(gl, program, "uEdgeThreshold"),
    effectiveBlend: getUniform(gl, program, "uEffectiveBlend"),
    displayMode: getUniform(gl, program, "uDisplayMode"),
    baseOffsets: getUniform(gl, program, "uBaseOffsets"),
    sigma: getUniform(gl, program, "uSigma"),
    jitter: getUniform(gl, program, "uJitter"),
    jitterPhase: getUniform(gl, program, "uJitterPhase"),
    breath: getUniform(gl, program, "uBreath"),
    muJ: getUniform(gl, program, "uMuJ"),
    phasePin: getUniform(gl, program, "uPhasePin"),
    microsaccade: getUniform(gl, program, "uMicrosaccade"),
    polBins: getUniform(gl, program, "uPolBins"),
    thetaMode: getUniform(gl, program, "uThetaMode"),
    thetaGlobal: getUniform(gl, program, "uThetaGlobal"),
    contrast: getUniform(gl, program, "uContrast"),
    frameGain: getUniform(gl, program, "uFrameGain"),
    rimEnabled: getUniform(gl, program, "uRimEnabled"),
    rimAlpha: getUniform(gl, program, "uRimAlpha"),
    warpAmp: getUniform(gl, program, "uWarpAmp"),
    surfaceBlend: getUniform(gl, program, "uSurfaceBlend"),
    surfaceRegion: getUniform(gl, program, "uSurfaceRegion"),
    surfEnabled: getUniform(gl, program, "uSurfEnabled"),
    kurEnabled: getUniform(gl, program, "uKurEnabled"),
    volumeEnabled: getUniform(gl, program, "uVolumeEnabled"),
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
    alive: getUniform(gl, program, "uAlive"),
    beta2: getUniform(gl, program, "uBeta2"),
    couplingRimSurfaceBlend: getUniform(gl, program, "uCouplingRimSurfaceBlend"),
    couplingRimSurfaceAlign: getUniform(gl, program, "uCouplingRimSurfaceAlign"),
    couplingSurfaceRimOffset: getUniform(gl, program, "uCouplingSurfaceRimOffset"),
    couplingSurfaceRimSigma: getUniform(gl, program, "uCouplingSurfaceRimSigma"),
    couplingSurfaceRimHue: getUniform(gl, program, "uCouplingSurfaceRimHue"),
    couplingKurTransparency: getUniform(gl, program, "uCouplingKurTransparency"),
    couplingKurOrientation: getUniform(gl, program, "uCouplingKurOrientation"),
    couplingKurChirality: getUniform(gl, program, "uCouplingKurChirality"),
    couplingScale: getUniform(gl, program, "uCouplingScale"),
    couplingVolumePhaseHue: getUniform(gl, program, "uCouplingVolumePhaseHue"),
    couplingVolumeDepthWarp: getUniform(gl, program, "uCouplingVolumeDepthWarp"),
    composerExposure: getUniform(gl, program, "uComposerExposure"),
    composerGamma: getUniform(gl, program, "uComposerGamma"),
    composerWeight: getUniform(gl, program, "uComposerWeight"),
    composerBlendGain: getUniform(gl, program, "uComposerBlendGain")
  };
}

function getUniform(gl: WebGL2RenderingContext, program: WebGLProgram, name: string) {
  const location = gl.getUniformLocation(program, name);
  if (!location) {
    throw new Error(`Uniform ${name} not found`);
  }
  return location;
}
