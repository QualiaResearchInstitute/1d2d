import React, { useEffect, useMemo, useRef, useState } from "react";

// Rainbow Perimeter Lines — KernelSpec + DMT gain, polarization (birefringence/optical activity),
// crystalline bias, chiral bias, transparency composer, image upload, edge thresholding,
// display modes (grayscale base/rims), rim opacity, stability (gauge/phase pinning),
// 2D surface wallpaper/warp, edge↔surface coupling, explicit wallpaper groups, and diagnostics.
//
// Defaults (Lab‑Stable):
// - Spectral phase pinning via lambdaRef (implicit in offsets)
// - Zero-mean microsaccade jitter (intensity-only)
// - Polarization frame quantization (gradient-based smoothing)
// - Normalization pin (running target)
// Optional: Alive microbreath (tiny positional oscillation)

// --------------------------- Types & Params ---------------------------
type KernelSpec = {
gain: number; // amplitude/brightness
k0: number; // preferred spatial frequency (px^-1)
Q: number; // sharpness (higher = more periodic/crystalline)
anisotropy: number; // birefringence strength (Δn proxy)
chirality: number; // optical rotation rate (rad/s proxy)
transparency: number;// overlay friendliness (raises blend)
};

type Preset = {
name: string;
params: {
edgeThreshold: number; // 0..1
blend: number; // 0..1
kernel: KernelSpec;
dmt: number; // 0..1
thetaMode: 'gradient'|'global';
thetaGlobal: number; // radians
beta2: number; // 0..3
jitter: number; // 0..2
sigma: number; // 0.5..4
microsaccade: boolean;
speed: number; // 0..3
contrast: number; // 0.2..3
};
};

const DMT_SENS = { g1: 0.6, k1: 0.35, q1: 0.5, a1: 0.7, c1: 0.6, t1: 0.5 } as const;

// Canonical preset requested by the user
const PRESETS: Preset[] = [
{
name: "Rainbow Rims + DMT Kernel Effects",
params: {
edgeThreshold: 0.08, blend: 0.39,
kernel: { gain: 3.00, k0: 0.200, Q: 4.60, anisotropy: 0.95, chirality: 1.46, transparency: 0.28 },
dmt: 0.20, thetaMode: 'gradient', thetaGlobal: 0.0,
beta2: 1.90, jitter: 1.16, sigma: 4.00,
microsaccade: true, speed: 1.32, contrast: 1.62,
},
},
];

type SurfaceRegion = 'surfaces' | 'edges' | 'both';

type WallpaperGroup =
| 'off' | 'p1' | 'p2' | 'pm' | 'pg' | 'cm'
| 'pmm' | 'pmg' | 'pgg' | 'cmm'
| 'p4' | 'p4g' | 'p4m'
| 'p3' | 'p31m' | 'p3m1'
| 'p6' | 'p6m';

type Op = {
kind: 'rot'|'mirrorX'|'mirrorY'|'diag1'|'diag2';
angle?: number;
tx?: number;
ty?: number;
};

export default function RainbowPerimeterKernelDemo() {
const canvasRef = useRef<HTMLCanvasElement | null>(null);

// Canvas size (adapts to image load up to a max bound)
const [w, setW] = useState(720);
const [h, setH] = useState(480);

// Base rainbow-rim controls
const [beta2, setBeta2] = useState(1.2);
const [jitter, setJitter] = useState(0.6);
const [sigma, setSigma] = useState(1.3);
const [edgeRadius, setEdgeRadius] = useState(140); // fallback demo
const [microsaccade, setMicrosaccade] = useState(true);
const [speed, setSpeed] = useState(1.0);
const [contrast, setContrast] = useState(1.0);

// Image & edge gating
const [imgBitmap, setImgBitmap] = useState<ImageBitmap | null>(null);
const [edgeThreshold, setEdgeThreshold] = useState(0.25);
const [blend, setBlend] = useState(0.85);

// --- KernelSpec + DMT ---
const [kernel, setKernel] = useState<KernelSpec>({
gain: 1.0,
k0: 0.08, // cycles per px along normal (try 0.03–0.15)
Q: 2.0, // 1..6 is a good range
anisotropy: 0.6,
chirality: 0.4,
transparency: 0.2,
});
const [dmt, setDmt] = useState(0.0); // global DMT gain in [0,1]
const [thetaMode, setThetaMode] = useState<'gradient'|'global'>('gradient');
const [thetaGlobal, setThetaGlobal] = useState(0.0); // radians
const [displayMode, setDisplayMode] = useState<'color' | 'grayBaseColorRims' | 'grayBaseGrayRims' | 'colorBaseGrayRims'>('color');
const [rimAlpha, setRimAlpha] = useState(1.0);

// Stability / pinning controls
const [phasePin, setPhasePin] = useState(true); // zero-mean jitter
const [alive, setAlive] = useState(false); // tiny positional breath
const [polBins, setPolBins] = useState(16); // quantization bins for θ
const [normPin, setNormPin] = useState(true); // normalization pin

// Normalization running stats
const normTargetRef = useRef(0.6); // desired average edge energy (unitless)
const lastObsRef = useRef(0.6); // previous frame observed energy

// Presets state + applier
const [presetIndex, setPresetIndex] = useState(0);
const applyPreset = (p: Preset) => {
const v = p.params;
setEdgeThreshold(v.edgeThreshold);
setBlend(v.blend);
setKernel(v.kernel);
setDmt(v.dmt);
setThetaMode(v.thetaMode);
setThetaGlobal(v.thetaGlobal);
setBeta2(v.beta2);
setJitter(v.jitter);
setSigma(v.sigma);
setMicrosaccade(v.microsaccade);
setSpeed(v.speed);
setContrast(v.contrast);
};

// 2D Surface / Wallpaper morph controls
const [surfEnabled, setSurfEnabled] = useState(false);
const [wallGroup, setWallGroup] = useState<WallpaperGroup>('p4');
const [nOrient, setNOrient] = useState(4); // number of plane-wave orientations (2..8)
const [surfaceBlend, setSurfaceBlend] = useState(0.35); // overlay mix onto out (0..1)
const [warpAmp, setWarpAmp] = useState(1.0); // px warp amplitude
const [region, setRegion] = useState<SurfaceRegion>('surfaces');

// Kuramoto Z-field controls
const [kurEnabled, setKurEnabled] = useState(false);
const [K0, setK0] = useState(0.6);
const [alphaKur, setAlphaKur] = useState(0.2);
const [gammaKur, setGammaKur] = useState(0.15);
const [omega0, setOmega0] = useState(0.0);
const [epsKur, setEpsKur] = useState(0.002);
const [qInit, setQInit] = useState(1);

// Kuramoto Z-field controls
const [kurEnabled, setKurEnabled] = useState(false);
const [K0, setK0] = useState(0.6);
const [alphaKur, setAlphaKur] = useState(0.2);
const [gammaKur, setGammaKur] = useState(0.15);
const [omega0, setOmega0] = useState(0.0);
const [epsKur, setEpsKur] = useState(0.002);
const [qInit, setQInit] = useState(1);

// Kuramoto Z-field controls
const [kurEnabled, setKurEnabled] = useState(false);
const [K0, setK0] = useState(0.6);
const [alphaKur, setAlphaKur] = useState(0.2);
const [gammaKur, setGammaKur] = useState(0.15);
const [omega0, setOmega0] = useState(0.0);
const [epsKur, setEpsKur] = useState(0.002);
const [qInit, setQInit] = useState(1);

// Kuramoto Z-field controls
const [kurEnabled, setKurEnabled] = useState(false);
const [K0, setK0] = useState(0.6);
const [alphaKur, setAlphaKur] = useState(0.2);
const [gammaKur, setGammaKur] = useState(0.15);
const [omega0, setOmega0] = useState(0.0);
const [epsKur, setEpsKur] = useState(0.002);
const [qInit, setQInit] = useState(1);

// Kuramoto Z-field controls
const [kurEnabled, setKurEnabled] = useState(false);
const [K0, setK0] = useState(0.6);
const [alphaKur, setAlphaKur] = useState(0.2);
const [gammaKur, setGammaKur] = useState(0.15);
const [omega0, setOmega0] = useState(0.0);
const [epsKur, setEpsKur] = useState(0.002);
const [qInit, setQInit] = useState(1);

// Edge↔Surface coupling controls
const [coupleE2S, setCoupleE2S] = useState(true); // 1.5D → 2D
const [etaAmp, setEtaAmp] = useState(0.6); // rims boost surface blend
const [betaAlign, setBetaAlign] = useState(2.0); // lattice aligns to tangent

const [coupleS2E, setCoupleS2E] = useState(true); // 2D → 1.5D
const [gammaOff, setGammaOff] = useState(0.20); // px bias along n·∇t
const [gammaSigma, setGammaSigma] = useState(0.35); // sharpen rims with |∇t|
const [alphaPol, setAlphaPol] = useState(0.25); // blend edge vs surface orientation
const [kSigma, setKSigma] = useState(0.8); // scale for |∇t| → Esurf

// Edge fields & base pixels
const edgeDataRef = useRef<{ gx: Float32Array; gy: Float32Array; mag: Float32Array } | null>(null);

// ---- Kuramoto (Z-field) buffers ----
const ZrRef = useRef<Float32Array | null>(null);
const ZiRef = useRef<Float32Array | null>(null);
const gradXRef = useRef<Float32Array | null>(null);
const gradYRef = useRef<Float32Array | null>(null);
const vortRef = useRef<Float32Array | null>(null);
const cohRef = useRef<Float32Array | null>(null);

// Kuramoto field buffers (match canvas size)
const ZrRef = useRef<Float32Array | null>(null);
const ZiRef = useRef<Float32Array | null>(null);
const gradXRef = useRef<Float32Array | null>(null);
const gradYRef = useRef<Float32Array | null>(null);
const vortRef = useRef<Float32Array | null>(null);
const cohRef = useRef<Float32Array | null>(null);

// Kuramoto field buffers (match canvas size)
const ZrRef = useRef<Float32Array | null>(null);
const ZiRef = useRef<Float32Array | null>(null);
const gradXRef = useRef<Float32Array | null>(null);
const gradYRef = useRef<Float32Array | null>(null);
const vortRef = useRef<Float32Array | null>(null);
const cohRef = useRef<Float32Array | null>(null);

// Kuramoto field buffers (match canvas size)
const ZrRef = useRef<Float32Array | null>(null);
const ZiRef = useRef<Float32Array | null>(null);
const gradXRef = useRef<Float32Array | null>(null);
const gradYRef = useRef<Float32Array | null>(null);
const vortRef = useRef<Float32Array | null>(null);
const cohRef = useRef<Float32Array | null>(null);

// Kuramoto field buffers (match canvas size)
const ZrRef = useRef<Float32Array | null>(null);
const ZiRef = useRef<Float32Array | null>(null);
const gradXRef = useRef<Float32Array | null>(null);
const gradYRef = useRef<Float32Array | null>(null);
const vortRef = useRef<Float32Array | null>(null);
const cohRef = useRef<Float32Array | null>(null);

// Kuramoto field buffers (match canvas size)
const ZrRef = useRef<Float32Array | null>(null);
const ZiRef = useRef<Float32Array | null>(null);
const gradXRef = useRef<Float32Array | null>(null);
const gradYRef = useRef<Float32Array | null>(null);
const vortRef = useRef<Float32Array | null>(null);
const cohRef = useRef<Float32Array | null>(null);
const baseImagePixelsRef = useRef<ImageData | null>(null);

// LMS setup
const lambdas = useMemo(() => ({ L: 560, M: 530, S: 420 }), []);
const lambdaRef = 520;
const LMS2RGB = useMemo(() => [
[4.4679, -3.5873, 0.1193],
[-1.2186, 2.3809, -0.1624],
[0.0497, -0.2439, 1.2045],
], []);

// Hash noise per pixel
const hash2 = (x: number, y: number) => {
const s = Math.sin(x _ 127.1 + y _ 311.7) \* 43758.5453;
return s - Math.floor(s);
};

// --------------- Image upload + edges ---------------
async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
const f = e.target.files?.[0];
if (!f) return;
const bmp = await createImageBitmap(f);

    const maxW = 1280, maxH = 900;
    let newW = bmp.width, newH = bmp.height;
    const scale = Math.min(maxW / newW, maxH / newH, 1);
    newW = Math.round(newW * scale);
    newH = Math.round(newH * scale);

    setW(newW); setH(newH);
    setImgBitmap(bmp);

    const off = document.createElement('canvas');
    off.width = newW; off.height = newH;
    const octx = off.getContext('2d', { willReadFrequently: true });
    if (!octx) return;
    octx.drawImage(bmp, 0, 0, newW, newH);
    const img = octx.getImageData(0, 0, newW, newH);
    baseImagePixelsRef.current = img;

    const gray = new Float32Array(newW * newH);
    const d = img.data;
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const i = (y * newW + x) * 4;
        gray[y * newW + x] = (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255;
      }
    }
    // Sobel
    const gxK = [-1,0,1,-2,0,2,-1,0,1];
    const gyK = [-1,-2,-1,0,0,0,1,2,1];
    const gx = new Float32Array(newW * newH);
    const gy = new Float32Array(newW * newH);
    const idx = (x: number, y: number) => y * newW + x;
    for (let y = 1; y < newH - 1; y++) {
      for (let x = 1; x < newW - 1; x++) {
        let sx = 0, sy = 0; let k = 0;
        for (let j = -1; j <= 1; j++) {
          for (let i2 = -1; i2 <= 1; i2++) {
            const v = gray[idx(x + i2, y + j)];
            sx += v * gxK[k]; sy += v * gyK[k]; k++;
          }
        }
        gx[idx(x, y)] = sx; gy[idx(x, y)] = sy;
      }
    }
    const mag = new Float32Array(newW * newH);
    let maxMag = 1e-6;
    for (let i2 = 0; i2 < mag.length; i2++) { const m = Math.hypot(gx[i2], gy[i2]); mag[i2] = m; if (m > maxMag) maxMag = m; }
    const inv = 1 / maxMag; for (let i2 = 0; i2 < mag.length; i2++) mag[i2] *= inv;
    edgeDataRef.current = { gx, gy, mag };

}

// --------------- Helpers ---------------
const gauss = (x: number, s: number) => Math.exp(-(x _ x) / (2 _ s _ s));
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const luma01 = (R: number, G: number, B: number) => clamp01(0.2126 _ R + 0.7152 _ G + 0.0722 _ B);

const sampleScalar = (arr: Float32Array, x: number, y: number, W: number, H: number) => {
x = Math.max(0, Math.min(W - 1.001, x)); y = Math.max(0, Math.min(H - 1.001, y));
const x0 = Math.floor(x), y0 = Math.floor(y); const x1 = x0 + 1, y1 = y0 + 1;
const fx = x - x0, fy = y - y0; const i = (yy: number, xx: number) => yy _ W + xx;
const v00 = arr[i(y0, x0)], v10 = arr[i(y0, x1)], v01 = arr[i(y1, x0)], v11 = arr[i(y1, x1)];
const v0 = v00 _ (1 - fx) + v10 _ fx; const v1 = v01 _ (1 - fx) + v11 _ fx; return v0 _ (1 - fy) + v1 \* fy;
};

const sampleRGB = (data: Uint8ClampedArray, x: number, y: number, W: number, H: number) => {
x = Math.max(0, Math.min(W - 1.001, x)); y = Math.max(0, Math.min(H - 1.001, y));
const x0 = Math.floor(x), y0 = Math.floor(y); const x1 = x0 + 1, y1 = y0 + 1;
const fx = x - x0, fy = y - y0; const idx = (yy: number, xx: number) => (yy _ W + xx) _ 4;
const i00 = idx(y0, x0), i10 = idx(y0, x1), i01 = idx(y1, x0), i11 = idx(y1, x1);
const r = (a:number,b:number,t:number)=>a*(1-t)+b*t;
const R0 = r(data[i00+0], data[i10+0], fx), G0 = r(data[i00+1], data[i10+1], fx), B0 = r(data[i00+2], data[i10+2], fx);
const R1 = r(data[i01+0], data[i11+0], fx), G1 = r(data[i01+1], data[i11+1], fx), B1 = r(data[i01+2], data[i11+2], fx);
return { R: r(R0,R1,fy), G: r(G0,G1,fy), B: r(B0,B1,fy) };
};

// -------- Kuramoto helpers (minimal CPU step) --------
const ensureKurBuffers = () => {
const N = w * h;
if (!ZrRef.current || ZrRef.current.length !== N) {
ZrRef.current = new Float32Array(N);
ZiRef.current = new Float32Array(N);
gradXRef.current = new Float32Array(N);
gradYRef.current = new Float32Array(N);
vortRef.current = new Float32Array(N);
cohRef.current = new Float32Array(N);
initKuramoto(qInit);
}
};
const idx2 = (x:number,y:number)=> y*w + x;
const wrapX = (x:number)=> (x<0? x+w : (x>=w? x-w : x));
const wrapY = (y:number)=> (y<0? y+h : (y>=h? y-h : y));
const randn = (()=>{ let spare:number|undefined; return ()=>{
if (spare!==undefined){ const v=spare; spare=undefined; return v; }
let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
const mag=Math.sqrt(-2*Math.log(u)); const z0=mag*Math.cos(2*Math.PI*v); const z1=mag*Math.sin(2*Math.PI*v);
spare=z1; return z0; }; })();
const initKuramoto = (q:number) => {
ensureKurBuffers();
const Zr = ZrRef.current!, Zi = ZiRef.current!;
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const th = 2*Math.PI*q*(x/w);
const i = idx2(x,y);
Zr[i] = Math.cos(th);
Zi[i] = Math.sin(th);
}
}
};
const stepKuramoto = (dt:number) => {
const Zr = ZrRef.current!, Zi = ZiRef.current!;
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i = idx2(x,y);
const iL = idx2(wrapX(x-1),y), iR = idx2(wrapX(x+1),y);
const iU = idx2(x,wrapY(y-1)), iD = idx2(x,wrapY(y+1));
const Hr = 0.2*(Zr[i]+Zr[iL]+Zr[iR]+Zr[iU]+Zr[iD]);
const Hi = 0.2*(Zi[i]+Zi[iL]+Zi[iR]+Zi[iU]+Zi[iD]);
const Zre = Zr[i], Zim = Zi[i];
const Z2r = Zre*Zre - Zim*Zim; const Z2i = 2*Zre*Zim;
const ca = Math.cos(alphaKur), sa = Math.sin(alphaKur);
const H1r = ca*Hr + sa*Hi, H1i = -sa*Hr + ca*Hi; // e^{-iα}H
const Hc*r = Hr, Hc_i = -Hi; // H*
const T_r = Z2r*Hc_r - Z2i*Hc_i, T_i = Z2r*Hc_i + Z2i*Hc_r;
const H2r = ca*T_r - sa*T_i, H2i = sa*T_r + ca*T_i; // e^{iα} Z^2 H*
const dZr = -gammaKur*Zre + ( -omega0*Zim ) + 0.5*K0*( H1r - H2r );
const dZi = -gammaKur*Zim + ( omega0*Zre ) + 0.5*K0*( H1i - H2i );
Zr[i] = Zre + dt*dZr + Math.sqrt(dt*epsKur)*randn();
Zi[i] = Zim + dt*dZi + Math.sqrt(dt*epsKur)*randn();
}
}
};
const wrapAngle = (a:number)=>{ while(a> Math.PI) a -= 2*Math.PI; while(a<=-Math.PI) a += 2*Math.PI; return a; };
const deriveKurFields = () => {
const Zr = ZrRef.current!, Zi = ZiRef.current!;
const gx = gradXRef.current!, gy = gradYRef.current!, vort = vortRef.current!, coh = cohRef.current!;
const theta = (i:number)=> Math.atan2(Zi[i], Zr[i]);
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i = idx2(x,y), iL = idx2(wrapX(x-1),y), iR = idx2(wrapX(x+1),y), iU = idx2(x,wrapY(y-1)), iD = idx2(x,wrapY(y+1));
const tL = theta(iL), tR = theta(iR), tU = theta(iU), tD = theta(iD);
gx[i] = 0.5 * ( wrapAngle(tR - tL) );
gy[i] = 0.5 \_ ( wrapAngle(tD - tU) );
const mag = Math.hypot(Zr[i], Zi[i]); coh[i] = Math.max(0, Math.min(1, mag));
}
}
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i00 = idx2(x,y), i10 = idx2(wrapX(x+1),y), i11 = idx2(wrapX(x+1),wrapY(y+1)), i01 = idx2(x,wrapY(y+1));
const a = wrapAngle(Math.atan2(ZiRef.current![i10], ZrRef.current![i10]) - Math.atan2(ZiRef.current![i00], ZrRef.current![i00]));
const b = wrapAngle(Math.atan2(ZiRef.current![i11], ZrRef.current![i11]) - Math.atan2(ZiRef.current![i10], ZrRef.current![i10]));
const c = wrapAngle(Math.atan2(ZiRef.current![i01], ZrRef.current![i01]) - Math.atan2(ZiRef.current![i11], ZrRef.current![i11]));
const d = wrapAngle(Math.atan2(ZiRef.current![i00], ZrRef.current![i00]) - Math.atan2(ZiRef.current![i01], ZrRef.current![i01]));
vort[i00] = (a+b+c+d)/(2\*Math.PI);
}
}
};
};

// -------- Kuramoto helpers (minimal CPU step) --------
const ensureKurBuffers = () => {
const N = w * h;
if (!ZrRef.current || ZrRef.current.length !== N) {
ZrRef.current = new Float32Array(N);
ZiRef.current = new Float32Array(N);
gradXRef.current = new Float32Array(N);
gradYRef.current = new Float32Array(N);
vortRef.current = new Float32Array(N);
cohRef.current = new Float32Array(N);
// init as q-twist along x
initKuramoto(qInit);
}
};
const idx2 = (x:number,y:number)=> y*w + x;
const wrapX = (x:number)=> (x<0? x+w : (x>=w? x-w : x));
const wrapY = (y:number)=> (y<0? y+h : (y>=h? y-h : y));
const randn = (()=>{ let spare:number|undefined; return ()=>{
if (spare!==undefined){ const v=spare; spare=undefined; return v; }
let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
const mag=Math.sqrt(-2*Math.log(u)); const z0=mag*Math.cos(2*Math.PI*v); const z1=mag*Math.sin(2*Math.PI*v);
spare=z1; return z0; }; })();
const initKuramoto = (q:number) => {
ensureKurBuffers();
const Zr = ZrRef.current!, Zi = ZiRef.current!;
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const th = 2*Math.PI*q*(x/w);
const i = idx2(x,y);
Zr[i] = Math.cos(th);
Zi[i] = Math.sin(th);
}
}
};
const stepKuramoto = (dt:number) => {
const Zr = ZrRef.current!, Zi = ZiRef.current!;
// cheap neighborhood mean for H (5-point stencil)
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i = idx2(x,y);
const iL = idx2(wrapX(x-1),y), iR = idx2(wrapX(x+1),y);
const iU = idx2(x,wrapY(y-1)), iD = idx2(x,wrapY(y+1));
const Hr = 0.2*(Zr[i]+Zr[iL]+Zr[iR]+Zr[iU]+Zr[iD]);
const Hi = 0.2*(Zi[i]+Zi[iL]+Zi[iR]+Zi[iU]+Zi[iD]);
// OA update: dZ = (i*omega0 - gamma)Z + 0.5*K0*(e^{-iα}H - e^{iα} Z^2 H*)
const Zre = Zr[i], Zim = Zi[i];
const Z2r = Zre*Zre - Zim*Zim; const Z2i = 2*Zre*Zim;
const ca = Math.cos(alphaKur), sa = Math.sin(alphaKur);
// e^{-iα}H
const H1r = ca*Hr + sa*Hi, H1i = -sa*Hr + ca*Hi;
// e^{iα} (Z^2 H*)
const Hc_r = Hr, Hc_i = -Hi; // conjugate
const T_r = Z2r*Hc*r - Z2i*Hc*i, T_i = Z2r*Hc_i + Z2i*Hc_r; // Z^2 H*
const H2r = ca*T_r - sa*T_i, H2i = sa*T_r + ca*T_i;
const dZr = -gammaKur*Zre + ( -omega0*Zim ) + 0.5*K0*( H1r - H2r );
const dZi = -gammaKur*Zim + ( omega0*Zre ) + 0.5*K0*( H1i - H2i );
// Euler + noise
Zr[i] = Zre + dt*dZr + Math.sqrt(dt*epsKur)*randn();
Zi[i] = Zim + dt*dZi + Math.sqrt(dt*epsKur)*randn();
}
}
};
const wrapAngle = (a:number)=>{
while(a> Math.PI) a -= 2*Math.PI; while(a<=-Math.PI) a += 2*Math.PI; return a;
};
const deriveKurFields = () => {
const Zr = ZrRef.current!, Zi = ZiRef.current!;
const gx = gradXRef.current!, gy = gradYRef.current!, vort = vortRef.current!, coh = cohRef.current!;
// θ, ∇θ, |Z|, vorticity (plaquette winding)
const theta = (i:number)=> Math.atan2(Zi[i], Zr[i]);
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i = idx2(x,y), iL = idx2(wrapX(x-1),y), iR = idx2(wrapX(x+1),y), iU = idx2(x,wrapY(y-1)), iD = idx2(x,wrapY(y+1));
const tC = theta(i), tL = theta(iL), tR = theta(iR), tU = theta(iU), tD = theta(iD);
gx[i] = 0.5 * ( wrapAngle(tR - tL) );
gy[i] = 0.5 * ( wrapAngle(tD - tU) );
const mag = Math.hypot(Zr[i], Zi[i]); coh[i] = Math.max(0, Math.min(1, mag));
}
}
// vorticity by small loop (x,y)->(x+1,y)->(x+1,y+1)->(x,y+1)
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i00 = idx2(x,y), i10 = idx2(wrapX(x+1),y), i11 = idx2(wrapX(x+1),wrapY(y+1)), i01 = idx2(x,wrapY(y+1));
const a = wrapAngle(Math.atan2(ZiRef.current![i10], ZrRef.current![i10]) - Math.atan2(ZiRef.current![i00], ZrRef.current![i00]));
const b = wrapAngle(Math.atan2(ZiRef.current![i11], ZrRef.current![i11]) - Math.atan2(ZiRef.current![i10], ZrRef.current![i10]));
const c = wrapAngle(Math.atan2(ZiRef.current![i01], ZrRef.current![i01]) - Math.atan2(ZiRef.current![i11], ZrRef.current![i11]));
const d = wrapAngle(Math.atan2(ZiRef.current![i00], ZrRef.current![i00]) - Math.atan2(ZiRef.current![i01], ZrRef.current![i01]));
vort[i00] = (a+b+c+d)/(2*Math.PI); // approx winding (−1..+1)
}
}
};.cos(th);
Zi[i] = Math.sin(th);
}
}
};
const stepKuramoto = (dt:number) => {
const Zr = ZrRef.current!, Zi = ZiRef.current!;
// cheap neighborhood mean for H (5-point stencil)
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i = idx2(x,y);
const iL = idx2(wrapX(x-1),y), iR = idx2(wrapX(x+1),y);
const iU = idx2(x,wrapY(y-1)), iD = idx2(x,wrapY(y+1));
const Hr = 0.2*(Zr[i]+Zr[iL]+Zr[iR]+Zr[iU]+Zr[iD]);
const Hi = 0.2*(Zi[i]+Zi[iL]+Zi[iR]+Zi[iU]+Zi[iD]);
// OA update: dZ = (i*omega0 - gamma)Z + 0.5*K0*(e^{-iα}H - e^{iα} Z^2 H*)
const Zre = Zr[i], Zim = Zi[i];
const Z2r = Zre*Zre - Zim*Zim; const Z2i = 2*Zre*Zim;
const ca = Math.cos(alphaKur), sa = Math.sin(alphaKur);
// e^{-iα}H
const H1r = ca*Hr + sa*Hi, H1i = -sa*Hr + ca*Hi;
// e^{iα} (Z^2 H*)
const Hc*r = Hr, Hc_i = -Hi; // conjugate
const T_r = Z2r*Hc_r - Z2i*Hc_i, T_i = Z2r*Hc_i + Z2i*Hc_r; // Z^2 H*
const H2r = ca*T_r - sa*T_i, H2i = sa*T_r + ca*T_i;
const dZr = -gammaKur*Zre + ( -omega0*Zim ) + 0.5*K0*( H1r - H2r );
const dZi = -gammaKur*Zim + ( omega0*Zre ) + 0.5*K0*( H1i - H2i );
// Euler + noise
Zr[i] = Zre + dt*dZr + Math.sqrt(dt*epsKur)*randn();
Zi[i] = Zim + dt*dZi + Math.sqrt(dt*epsKur)*randn();
}
}
};
const wrapAngle = (a:number)=>{
while(a> Math.PI) a -= 2*Math.PI; while(a<=-Math.PI) a += 2*Math.PI; return a;
};
const deriveKurFields = () => {
const Zr = ZrRef.current!, Zi = ZiRef.current!;
const gx = gradXRef.current!, gy = gradYRef.current!, vort = vortRef.current!, coh = cohRef.current!;
// θ, ∇θ, |Z|, vorticity (plaquette winding)
const theta = (i:number)=> Math.atan2(Zi[i], Zr[i]);
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i = idx2(x,y), iL = idx2(wrapX(x-1),y), iR = idx2(wrapX(x+1),y), iU = idx2(x,wrapY(y-1)), iD = idx2(x,wrapY(y+1));
const tC = theta(i), tL = theta(iL), tR = theta(iR), tU = theta(iU), tD = theta(iD);
gx[i] = 0.5 * ( wrapAngle(tR - tL) );
gy[i] = 0.5 * ( wrapAngle(tD - tU) );
const mag = Math.hypot(Zr[i], Zi[i]); coh[i] = Math.max(0, Math.min(1, mag));
}
}
// vorticity by small loop (x,y)->(x+1,y)->(x+1,y+1)->(x,y+1)
for (let y=0;y<h;y++){
for (let x=0;x<w;x++){
const i00 = idx2(x,y), i10 = idx2(wrapX(x+1),y), i11 = idx2(wrapX(x+1),wrapY(y+1)), i01 = idx2(x,wrapY(y+1));
const a = wrapAngle(Math.atan2(ZiRef.current![i10], ZrRef.current![i10]) - Math.atan2(ZiRef.current![i00], ZrRef.current![i00]));
const b = wrapAngle(Math.atan2(ZiRef.current![i11], ZrRef.current![i11]) - Math.atan2(ZiRef.current![i10], ZrRef.current![i10]));
const c = wrapAngle(Math.atan2(ZiRef.current![i01], ZrRef.current![i01]) - Math.atan2(ZiRef.current![i11], ZrRef.current![i11]));
const d = wrapAngle(Math.atan2(ZiRef.current![i00], ZrRef.current![i00]) - Math.atan2(ZiRef.current![i01], ZrRef.current![i01]));
vort[i00] = (a+b+c+d)/(2*Math.PI); // approx winding (−1..+1)
}
}
}; return { x: cx + dy, y: cy + dx }; // reflect y=x
if (op.kind === 'diag2') return { x: cx - dy, y: cy - dx }; // reflect y=-x
return { x, y };
}
function wallpaperAt(xp: number, yp: number, cosA: Float32Array, sinA: Float32Array, ke: KernelSpec, tSeconds: number) {
// uniform (unweighted) plane-wave sum → gradient only (gx, gy)
const N = cosA.length; const twoPI = Math.PI * 2;
let gx = 0, gy = 0;
for (let j=0;j<N;j++) {
const phi = ke.chirality _ (j _ Math.PI / N) + (alive ? 0.2 _ Math.sin(twoPI _ 0.3 _ tSeconds + j) : 0);
const s = (xp) _ cosA[j] + (yp) _ sinA[j];
const arg = twoPI _ ke.k0 _ s + phi;
const d = -twoPI _ ke.k0 _ Math.sin(arg);
gx += d _ cosA[j];
gy += d \_ sinA[j];
}
gx /= N; gy /= N;
return { gx, gy };
}

// --------------- Draw at time ---------------
const drawAtTime = (ctx: CanvasRenderingContext2D, tSeconds: number) => {
if (kurEnabled) {
ensureKurBuffers();
const substeps = 1;
for (let s=0;s<substeps;s++) stepKuramoto(0.016\*speed);
deriveKurFields();
}
const imgOut = ctx.createImageData(w, h);
const out = imgOut.data;

const drawAtTime = (ctx: CanvasRenderingContext2D, tSeconds: number) => {
// Kuramoto pre-step
if (kurEnabled) {
ensureKurBuffers();
const substeps = 1;
for (let s=0;s<substeps;s++) stepKuramoto(0.016*speed);
deriveKurFields();
} ke.transparency * 0.5); // transparency lifts overlay

    // Normalization pin: frame-level gain based on last observed energy
    const eps = 1e-6;
    const frameGain = normPin ? Math.pow((normTargetRef.current + eps) / (lastObsRef.current + eps), 0.5) : 1.0;

    // Dispersion offsets per LMS (relative to lambdaRef)
    const baseOffsets = {
      L: beta2 * (lambdaRef / lambdas.L - 1),
      M: beta2 * (lambdaRef / lambdas.M - 1),
      S: beta2 * (lambdaRef / lambdas.S - 1),
    } as const;

    const jitterPhase = microsaccade ? tSeconds * 6.0 : 0.0;
    const breath = alive ? 0.15 * Math.sin(2 * Math.PI * 0.55 * tSeconds) : 0.0; // px

    if (imgBitmap && edgeDataRef.current && baseImagePixelsRef.current) {
      const { gx, gy, mag } = edgeDataRef.current;
      const base = baseImagePixelsRef.current.data;

      // Compute zero-mean jitter (μJ) over edge const drawAtTime = (ctx: CanvasRenderingContext2D, tSeconds: number) => {
    // Kuramoto pre-step
    if (kurEnabled) {
      ensureKurBuffers();
      const substeps = 1;
      for (let s=0;s<substeps;s++) stepKuramoto(0.016*speed);
      deriveKurFields();
    }
        const stride = 8;
        for (let yy = 0; yy < h; yy += stride) {
          for (let xx = 0; xx < w; xx += stride) {
            const p = yy * w + xx;
            if (mag[p] >= edgeThreshold) {
              const n = hash2(xx, yy);
              muJ += Math.sin(jitterPhase + n * Math.PI * 2);
              cnt++;
            }
          }
        }
        muJ = cnt ? muJ / cnt : 0;
      }

      let obsSum = 0, obsCount = 0;

      // Precompute orientation cos/sin
      const N = orientations.length;
      const cosA = new Float32Array(N); const sinA = new Float32Array(N);
      for (let j=0;j<N;j++){ cosA[j] = Math.cos(orientations[j]); sinA[j] = Math.sin(orientations[j]); }

      const cx = w * 0.5, cy = h * 0.5;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const p = (y * w + x);
          const i = p * 4;

          // base image copy
          out[i+0] = base[i+0]; out[i+1] = base[i+1]; out[i+2] = base[i+2]; out[i+3] = 255;
          // optional grayscale base
          if (displayMode === 'grayBaseColorRims' || displayMode === 'grayBaseGrayRims') {
            const yb = Math.floor(0.2126 * out[i+0] + 0.7152 * out[i+1] + 0.0722 * out[i+2]);
            out[i+0] = out[i+1] = out[i+2] = yb;
          }

          // --------- Wallpaper gradient for S→E coupling (group-averaged) ---------
          let gxT0 = 0, gyT0 = 0;
          if (kurEnabled && gradXRef.current && gradYRef.current) {
            gxT0 = gradXRef.current[p];
            gyT0 = gradYRef.current[p];
          } else if (surfEnabled || coupleS2E) {
            const ops = groupOps(wallGroup);
            for (let k=0;k<ops.length;k++)let gxT0 = 0, gyT0 = 0;
          if (kurEnabled && gradXRef.current && gradYRef.current) {
            gxT0 = gradXRef.current[p];
            gyT0 = gradYRef.current[p];
          } else if (surfEnabled || coupleS2E) {       const r = wallpaperAt(pt.x - cx, pt.y - cy, cosA, sinA, ke, tSeconds);
              gxT0 += r.gx; gyT0 += r.gy;
            }
            gxT0 /= Math.max(1, groupOps(wallGroup).length);
            gyT0 /= Math.max(1, groupOps(wallGroup).length);
          }

          const m = mag[p];

          // ------------------ 1.5D edge rims ------------------
          let Erim = 0; // will be used by E→S coupling
          if (m >= edgeThreshold) {
            // Local edge geometry
            let nx = gx[p], ny = gy[p];
            const nlen = Math.hypot(nx, ny) + 1e-8; nx /= nlen; ny /= nlen; // normal (gradient dir)
            const tx = -ny, ty = nx; // tangent

            // Polarization orientation θ (with surface nudge)
            const TAU = Math.PI * 2;
            const thetaRaw = Math.atan2(ny, nx); // edge-normal angle
            const thetaEdgeQ = polBins > 0 ? Math.round((thetaRaw / TAU) *let gxT0 = 0, gyT0 = 0;
          if (kurEnabled && gradXRef.current && gradYRef.current) {
            gxT0 = gradXRef.current[p];
            gyT0 = gradYRef.current[p];
          } else if (surfEnabled || coupleS2E) {ge = (thetaMode === 'gradient') ? thetaEdgeQ : thetaGlobal;

            // Blend with surface direction if enabled
            const gradTnorm0 = Math.hypot(gxT0, gyT0);
            let thetaUse = thetaEdge;
            if (coupleS2E && gradTnorm0 > 1e-6) {
              const ex = Math.cos(thetaEdge), ey = Math.sin(thetaEdge);
              const sx = gxT0 / gradTnorm0, sy = gyT0 / gradTnorm0;
              const vx = (1 - alphaPol) * ex + alphaPol * sx;
              const vy = (1 - alphaPol) * ey + alphaPol * sy;
              let tBlend = Math.atan2(vy, vx);
              if (polBins > 0) tBlend = Math.round((tBlend / TAU) * polBins) * (TAU / polBins);
              thetaUse = tBlend;
            }

            // Polarization: approximate via retardance δ and optical rotation ρ
            const delta = ke.anisotropy * 0.9;             // retardance proxy (radians)
            const rho = ke.chirality * 0.75;               // rotation rate proxy (rad/s)
            const thetaEff = thetaUse + rho * tSeconds;    // rotate polarization over time
            const polL = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * thetaEff));
            const polM = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.3)));
            const polS = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.6)));

            // Local jitter (zero-mean if pinned)
            const n = hash2(x, y);
            const rawJ = Math.sin(jitterPhase + n * Math.PI * 2);
            const localJ = jitter * (microsaccade ? (phasePin ? rawJ - muJ : rawJ) : 0);

            // Surface→Edge bias (offset + sigma)
            const gradTnorm = Math.hypot(gxT0, gyT0);
            const bias = coupleS2E && gradTnorm > 1e-6 ? gammaOff * ((gxT0 * nx + gyT0 * ny) / gradTnorm) : 0;
            const Esurf = coupleS2E ? clamp01(gradTnorm * kSigma) : 0;
            const sigmaEff = coupleS2E ? Math.max(0.3, sigma * (1 - gammaSigma * Esurf)) : sigma;

            const offL = baseOffsets.L + localJ * 0.35 + bias;
            const offM = baseOffsets.M + localJ * 0.50 + bias;
            const offS = baseOffsets.S + localJ * 0.80 + bias;

            // Displaced sampling along the normal (+ optional breath)
            const pL = sampleScalar(mag, x + (offL + breath) * nx, y + (offL + breath) * ny, w, h);
            const pM = sampleScalar(mag, x + (offM + breath) * nx, y + (offM + breath) * ny, w, h);
            const pS = sampleScalar(mag, x + (offS + breath) * nx, y + (offS + breath) * ny, w, h);

            // Rim thinness (window around edge) and kernel gain
            const gL = gauss(offL, sigmaEff) * ke.gain;
            const gM = gauss(offM, sigmaEff) * ke.gain;
            const gS = gauss(offS, sigmaEff) * ke.gain;

            // Crystalline modulation: periodic gain along normal using k0 & Q
            const sL = offL; const sM = offM; const sS = offS; // distance along normal (px)
            const QQ = 1 + 0.5 * ke.Q; // tame exponent
            const modL = Math.pow(0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * sL)), QQ);
            const modM = Math.pow(0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * sM)), QQ);
            const modS = Math.pow(0.5 * (1 + Math.cos(2 * Math.PI * ke.k0 * sS)), QQ);

            // Chiral bias along tangent
            const chiPhase = 2 * Math.PI * ke.k0 * (x * tx + y * ty) * 0.002;
            const chLoc = ke.chirality + (kurEnabled && vortRef.current ? Math.max(-1, Math.min(1, vortRef.current[p]*0.5)) : 0);
            const chiL = 0.5 + 0.5 * Math.sin(chiPhase + 0.0) * chLoc;
            const chiM = 0.5 + 0.5 * Math.sin(chiPhase + 0.8) * chLoc;
            const chiS = 0.5 + 0.5 * Math.sin(chiPhase + 1.6) * chLoc;

            // Compose edge rim strengths per LMS (contrast compresses broadband energy)
            const cont = contrast * frameGain;
            const Lc = pL * gL * modL * chiL * polL * cont;
            const Mc = pM * gM * modM * chiM * polM * cont;
            const Sc = pS * gS * modS * chiS * polS * cont;

            // Store rim energy for E→S coupling
            Erim = (Lc + Mc + Sc) / Math.max(1e-6, cont);

            // LMS → RGB
            let R = LMS2RGB[0][0] * Lc + LMS2RGB[0][1] * Mc + LMS2RGB[0][2] * Sc;
            let G = LMS2RGB[1][0] * Lc + LMS2RGB[1][1] * Mc + LMS2RGB[1][2] * Sc;
            let B = LMS2RGB[2][0] * Lc + LMS2RGB[2][1] * Mc + LMS2RGB[2][2] * Sc;
            R = clamp01(R); G = clamp01(G); B = clamp01(B);
            if (displayMode === 'grayBaseGrayRims' || displayMode === 'colorBaseGrayRims') {
              const yr = luma01(R, G, B); R = G = B = yr;
            }
            R *= rimAlpha; G *= rimAlpha; B *= rimAlpha;

            // Traconst chiPhase = 2 * Math.PI * ke.k0 * (x * tx + y * ty) * 0.002;
            const chLoc = ke.chirality + (kurEnabled && vortRef.current ? Math.max(-1, Math.min(1, vortRef.current[p]*0.5)) : 0);
            const chiL = 0.5 + 0.5 * Math.sin(chiPhase + 0.0) * chLoc;
            const chiM = 0.5 + 0.5 * Math.sin(chiPhase + 0.8) * chLoc;
            const chiS = 0.5 + 0.5 * Math.sin(chiPhase + 1.6) * chLoc;[i + 2] * (1 - effectiveBlend) + (B * 255) * effectiveBlend);

            // accumulate observed energy sparsely for normalization pin
            if ((x & 7) === 0 && (y & 7) === 0) { obsSum += (pL + pM + pS) / 3; obsCount++; }
          }

          // ------------------ 2D surface wallpaper/warp ------------------
          if (surfEnabled) {
            // Region mask: where to apply (surfaces/edges/both)
            let mask = 1.0;
            if (region === 'surfaces') mask = clamp01((edgeThreshold - m) / Math.max(1e-6, edgeThreshold));
            else if (region === 'edges') mask = clamp01((m - edgeThreshold) / Math.max(1e-6, 1 - edgeThreshold));

            if (mask > 0.001) {
              // Explicit wallpaper groups — average gradient over group transforms
              const ops = groupOps(wallGroup);
              let gxTw = 0, gyTw = 0;
              for (let k=0;k<ops.length;k++){
                const pt = applyOp(ops[k], x, y, cx, cy);
                const r = wallpaperAt(pt.x - cx, pt.y - cy, cosA, sinA, ke, tSeconds);
                gxTw += r.gx; gyTw += r.gy;
              }
              gxTw /= Math.max(1, ops.length); gyTw /= Math.max(1, ops.length);

              // Optional anisotropy weight (favor one global direction)
              const dirW = 1 + 0.5 * ke.anisotropy * Math.cos(2 * Math.atan2(gyTw, gxTw));

              // Warp sample from base image along gradient of t
              const norm = Math.hypot(gxTw, gyTw) + 1e-6;
              const wx = warpAmp * (gxTw / norm) * dirW;
              const wy = warpAmp * (gyTw / norm) * dirW;
              const sxy = sampleRGB(base, x + wx, y + wy, w, h);
              let rW = sxy.R / 255, gW = sxy.G / 255, bW = sxy.B / 255;
              if (displayMode === 'grayBaseGrayRims') { const yy = luma01(rW,gW,bW); rW = gW = bW = yy; }

              // Surface blend, optionally boosted by rim energy
              let sb = surfaceBlend * mask;
              if (coupleE2S) sb *= (1 + etaAmp * clamp01(Erim));
              if (kurEnabled && cohRef.current) sb *= (0.5 + 0.5*cohRef.current[p]);
              sb = clamp(sb, 0, 1);

              out[i+0] = Math.floor(out[i+0]*(1-sb) + (rW*255)*sb);
              out[i+1] = Math.floor(out[i+1]*(1-sb) + (gW*255)*sb);
              out[i+2] = Math.floor(out[i+2]*(1-sb) + (bW*255)*sb);
            }
          }
        }
      }

      // Update observed edge energy for next frame
      const obs = obsCount ? obsSum / obsCount : lastObsRef.current;
      lastObsRef.current = clamp(obs, 0.001, 10);
    } else {
      // Fallback synthetic circle (kept simple; no 2D surface to keep it fast)
      let muJ = 0, cnt = 0;
      if (phasePin && microsaccade) {
        const stride = 8; const cx = Math.floor(w/2), cy = Math.floor(h/2);
        for (let yy = 0; yy < h; yy += stride) {
          for (let xx = 0; xx < w; xx += stride) {
            const dx = xx - cx, dy = yy - cy; const d = Math.hypot(dx, dy);
            if (Math.abs(d - edgeRadius) < 3) { const n = hash2(xx, y
