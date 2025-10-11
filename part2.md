// --- Effective kernel with DMT (needed by drawAtTime) ---
const kEff = (k: KernelSpec, d: number): KernelSpec => ({
gain: k.gain _ (1 + DMT_SENS.g1 _ d),
k0: k.k0 _ (1 + DMT_SENS.k1 _ d),
Q: k.Q _ (1 + DMT_SENS.q1 _ d),
anisotropy: k.anisotropy + DMT*SENS.a1 * d,
chirality: k.chirality + DMT*SENS.c1 * d,
transparency:k.transparency + DMT_SENS.t1 \* d,
});

// --- Orientations bank for wallpaper (if used) ---
const orientations = useMemo(() => {
const N = Math.max(2, Math.min(8, Math.round(nOrient)));
return Array.from({length: N}, (\_, j) => (j _ 2 _ Math.PI) / N);
}, [nOrient]);

// --- Wallpaper symmetry ops (explicit groups) ---
function groupOps(kind: WallpaperGroup): Op[] {
const HALF = 0.5;
const THIRD = 1 / 3;
const rot = (angle: number, tx = 0, ty = 0): Op => ({ kind:'rot', angle, tx, ty });
const id = (tx = 0, ty = 0): Op => rot(0, tx, ty);
const mirrorX = (tx = 0, ty = 0): Op => ({ kind:'mirrorX', tx, ty });
const mirrorY = (tx = 0, ty = 0): Op => ({ kind:'mirrorY', tx, ty });
const diag1 = (tx = 0, ty = 0): Op => ({ kind:'diag1', tx, ty });
const diag2 = (tx = 0, ty = 0): Op => ({ kind:'diag2', tx, ty });

switch (kind) {
case 'p1':
return [id(), id(HALF,0), id(0,HALF), id(HALF,HALF)];
case 'p2':
return [rot(0), rot(Math.PI), rot(0,HALF,HALF), rot(Math.PI,HALF,HALF)];
case 'pm':
return [id(), mirrorX(), id(HALF,0), mirrorX(HALF,0)];
case 'pg':
return [id(), mirrorX(HALF,0), id(0,HALF), mirrorX(HALF,HALF)];
case 'cm':
return [id(), mirrorY(0,HALF), mirrorX(HALF,0), mirrorY(HALF,HALF)];
case 'pmm':
return [rot(0), rot(Math.PI), mirrorX(), mirrorY()];
case 'pmg':
return [rot(0), mirrorX(), mirrorY(HALF,0), rot(Math.PI,HALF,0)];
case 'pgg':
return [rot(0), rot(Math.PI), mirrorX(HALF,0), mirrorY(0,HALF)];
case 'cmm':
return [rot(0), mirrorX(), mirrorY(), diag1(), diag2(), rot(Math.PI,HALF,HALF)];
case 'p4':
return [0,Math.PI/2,Math.PI,3*Math.PI/2].map(a => rot(a));
case 'p4m':
return [
rot(0), rot(Math.PI/2), rot(Math.PI), rot(3*Math.PI/2),
mirrorX(), mirrorY(), diag1(), diag2()
];
case 'p4g':
return [
rot(0), rot(Math.PI/2), rot(Math.PI), rot(3*Math.PI/2),
mirrorX(HALF,0), mirrorY(0,HALF), diag1(HALF,0), diag2(0,HALF)
];
case 'p3':
return [0, 2*Math.PI/3, 4*Math.PI/3].map(a => rot(a));
case 'p3m1':
return [
rot(0), rot(2*Math.PI/3), rot(4*Math.PI/3),
mirrorX(), diag1(THIRD,0), diag2(0,THIRD)
];
case 'p31m':
return [
rot(0), rot(2*Math.PI/3), rot(4*Math.PI/3),
mirrorY(), diag1(2*THIRD,THIRD), diag2(THIRD,2*THIRD)
];
case 'p6':
return Array.from({length:6},(_,k)=>rot(k\*(Math.PI/3)));
case 'p6m':
return [
...Array.from({length:6},(_,k)=>rot(k*(Math.PI/3))),
mirrorX(), mirrorY()
];
default:
return [rot(0)];
}
}
function applyOp(op: Op, x: number, y: number, cx: number, cy: number) {
const dx = x - cx, dy = y - cy;
let px = dx, py = dy;
if (op.kind === 'rot') {
const c = Math.cos(op.angle || 0), s = Math.sin(op.angle || 0);
px = c*dx - s*dy;
py = s*dx + c*dy;
} else if (op.kind === 'mirrorX') {
px = -dx; py = dy;
} else if (op.kind === 'mirrorY') {
px = dx; py = -dy;
} else if (op.kind === 'diag1') {
px = dy; py = dx;
} else if (op.kind === 'diag2') {
px = -dy; py = -dx;
}
const tx = (op.tx ?? 0) * (cx _ 2);
const ty = (op.ty ?? 0) _ (cy \* 2);
return { x: cx + px + tx, y: cy + py + ty };
}

// --- Complete drawAtTime (close it and put the image) ---
const drawAtTime = (ctx: CanvasRenderingContext2D, tSeconds: number) => {
if (kurEnabled) {
ensureKurBuffers();
stepKuramoto(0.016 _ speed);
deriveKurFields();
}
const imgOut = ctx.createImageData(w, h);
const out = imgOut.data;
const ke = kEff(kernel, dmt);
const effectiveBlend = clamp01(blend + ke.transparency _ 0.5);

const eps = 1e-6;
const frameGain = normPin ? Math.pow((normTargetRef.current + eps) / (lastObsRef.current + eps), 0.5) : 1.0;

const baseOffsets = {
L: beta2 _ (lambdaRef / lambdas.L - 1),
M: beta2 _ (lambdaRef / lambdas.M - 1),
S: beta2 \* (lambdaRef / lambdas.S - 1),
} as const;

const jitterPhase = microsaccade ? tSeconds _ 6.0 : 0.0;
const breath = alive ? 0.15 _ Math.sin(2 _ Math.PI _ 0.55 \* tSeconds) : 0.0;

if (imgBitmap && edgeDataRef.current && baseImagePixelsRef.current) {
const { gx, gy, mag } = edgeDataRef.current;
const base = baseImagePixelsRef.current.data;

    // zero-mean jitter μJ on edge samples
    let muJ = 0, cnt = 0;
    if (phasePin && microsaccade) {
      const stride = 8;
      for (let yy = 0; yy < h; yy += stride) for (let xx = 0; xx < w; xx += stride) {
        const p = yy * w + xx;
        if (mag[p] >= edgeThreshold) { const n = hash2(xx, yy); muJ += Math.sin(jitterPhase + n * Math.PI * 2); cnt++; }
      }
      muJ = cnt ? muJ / cnt : 0;
    }

    const N = orientations.length;
    const cosA = new Float32Array(N), sinA = new Float32Array(N);
    for (let j=0;j<N;j++){ cosA[j] = Math.cos(orientations[j]); sinA[j] = Math.sin(orientations[j]); }

    const cx = w * 0.5, cy = h * 0.5;
    let obsSum = 0, obsCount = 0;

    for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
      const p = y * w + x; const i = p * 4;
      // base
      out[i+0]=base[i+0]; out[i+1]=base[i+1]; out[i+2]=base[i+2]; out[i+3]=255;
      if (displayMode!=='color') {
        const yb = Math.floor(0.2126*out[i+0]+0.7152*out[i+1]+0.0722*out[i+2]);
        if (displayMode==='grayBaseColorRims' || displayMode==='grayBaseGrayRims') out[i+0]=out[i+1]=out[i+2]=yb;
      }

      // drive field (Kuramoto ∇θ if enabled, else wallpaper)
      let gxT0=0, gyT0=0;
      if (kurEnabled && gradXRef.current && gradYRef.current){ gxT0=gradXRef.current[p]; gyT0=gradYRef.current[p]; }
      else if (surfEnabled || coupleS2E){
        const ops = groupOps(wallGroup);
        for (let k=0;k<ops.length;k++){ const pt = applyOp(ops[k], x, y, cx, cy);
          const r = wallpaperAt(pt.x - cx, pt.y - cy, cosA, sinA, ke, tSeconds); gxT0+=r.gx; gyT0+=r.gy; }
        const s = Math.max(1, ops.length); gxT0/=s; gyT0/=s;
      }

      const m = mag[p];
      // --- rims (edge only) ---
      if (m >= edgeThreshold) {
        let nx = gx[p], ny = gy[p]; const nlen = Math.hypot(nx,ny)+1e-8; nx/=nlen; ny/=nlen;
        const tx = -ny, ty = nx;

        const TAU = 2*Math.PI;
        const thetaRaw = Math.atan2(ny, nx);
        const thetaEdge = (thetaMode==='gradient')
          ? (polBins>0 ? Math.round((thetaRaw/TAU)*polBins)*(TAU/polBins) : thetaRaw)
          : thetaGlobal;

        const gradTnorm0 = Math.hypot(gxT0, gyT0);
        let thetaUse = thetaEdge;
        if (coupleS2E && gradTnorm0>1e-6){
          const ex=Math.cos(thetaEdge), ey=Math.sin(thetaEdge);
          const sx=gxT0/gradTnorm0, sy=gyT0/gradTnorm0;
          const vx=(1-alphaPol)*ex + alphaPol*sx, vy=(1-alphaPol)*ey + alphaPol*sy;
          thetaUse = Math.atan2(vy, vx);
          if (polBins>0) thetaUse = Math.round((thetaUse/TAU)*polBins)*(TAU/polBins);
        }

        const delta = ke.anisotropy * 0.9, rho = ke.chirality * 0.75;
        const thetaEff = thetaUse + rho * tSeconds;
        const polL = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * thetaEff));
        const polM = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.3)));
        const polS = 0.5 * (1 + Math.cos(delta) * Math.cos(2 * (thetaEff + 0.6)));

        const n = hash2(x, y);
        const rawJ = Math.sin(jitterPhase + n * Math.PI * 2);
        const localJ = jitter * (microsaccade ? (phasePin ? rawJ - muJ : rawJ) : 0);

        const gradTnorm = Math.hypot(gxT0, gyT0);
        const bias = coupleS2E && gradTnorm>1e-6 ? gammaOff * ((gxT0*nx + gyT0*ny)/gradTnorm) : 0;
        const Esurf = coupleS2E ? clamp01(gradTnorm * kSigma) : 0;
        const sigmaEff = coupleS2E ? Math.max(0.3, sigma * (1 - gammaSigma * Esurf)) : sigma;

        const offL = baseOffsets.L + localJ*0.35 + bias;
        const offM = baseOffsets.M + localJ*0.50 + bias;
        const offS = baseOffsets.S + localJ*0.80 + bias;

        const pL = sampleScalar(mag, x + (offL+breath)*nx, y + (offL+breath)*ny, w, h);
        const pM = sampleScalar(mag, x + (offM+breath)*nx, y + (offM+breath)*ny, w, h);
        const pS = sampleScalar(mag, x + (offS+breath)*nx, y + (offS+breath)*ny, w, h);

        const gL = gauss(offL, sigmaEff) * ke.gain;
        const gM = gauss(offM, sigmaEff) * ke.gain;
        const gS = gauss(offS, sigmaEff) * ke.gain;

        const sL=offL,sM=offM,sS=offS, QQ=1+0.5*ke.Q;
        const modL = Math.pow(0.5*(1+Math.cos(2*Math.PI*ke.k0*sL)), QQ);
        const modM = Math.pow(0.5*(1+Math.cos(2*Math.PI*ke.k0*sM)), QQ);
        const modS = Math.pow(0.5*(1+Math.cos(2*Math.PI*ke.k0*sS)), QQ);

        const chiPhase = 2*Math.PI*ke.k0*(x*tx + y*ty)*0.002;
        const chLoc = ke.chirality + (kurEnabled && vortRef.current ? Math.max(-1, Math.min(1, vortRef.current[p]*0.5)) : 0);
        const chiL = 0.5 + 0.5*Math.sin(chiPhase+0.0)*chLoc;
        const chiM = 0.5 + 0.5*Math.sin(chiPhase+0.8)*chLoc;
        const chiS = 0.5 + 0.5*Math.sin(chiPhase+1.6)*chLoc;

        const cont = contrast * frameGain;
        const Lc = pL*gL*modL*chiL*polL*cont;
        const Mc = pM*gM*modM*chiM*polM*cont;
        const Sc = pS*gS*modS*chiS*polS*cont;

        let R = LMS2RGB[0][0]*Lc + LMS2RGB[0][1]*Mc + LMS2RGB[0][2]*Sc;
        let G = LMS2RGB[1][0]*Lc + LMS2RGB[1][1]*Mc + LMS2RGB[1][2]*Sc;
        let B = LMS2RGB[2][0]*Lc + LMS2RGB[2][1]*Mc + LMS2RGB[2][2]*Sc;
        R=clamp01(R); G=clamp01(G); B=clamp01(B);
        if (displayMode==='grayBaseGrayRims' || displayMode==='colorBaseGrayRims'){ const yr=luma01(R,G,B); R=G=B=yr; }
        R*=rimAlpha; G*=rimAlpha; B*=rimAlpha;

        out[i+0] = Math.floor(out[i+0]*(1-effectiveBlend) + R*255*effectiveBlend);
        out[i+1] = Math.floor(out[i+1]*(1-effectiveBlend) + G*255*effectiveBlend);
        out[i+2] = Math.floor(out[i+2]*(1-effectiveBlend) + B*255*effectiveBlend);

        if ((x & 7)===0 && (y & 7)===0) { obsSum += (pL+pM+pS)/3; obsCount++; }
      }

      // --- 2D surface warp (optional) ---
      if (surfEnabled) {
        let mask = 1.0;
        if (region==='surfaces') mask = clamp01((edgeThreshold - m) / Math.max(1e-6, edgeThreshold));
        else if (region==='edges') mask = clamp01((m - edgeThreshold) / Math.max(1e-6, 1 - edgeThreshold));
        if (mask > 1e-3) {
          const ops = groupOps(wallGroup);
          let gxTw=0, gyTw=0;
          for (let k=0;k<ops.length;k++){
            const pt = applyOp(ops[k], x, y, cx, cy);
            const r = wallpaperAt(pt.x - cx, pt.y - cy, cosA, sinA, ke, tSeconds);
            gxTw += r.gx; gyTw += r.gy;
          }
          const s=Math.max(1,ops.length); gxTw/=s; gyTw/=s;
          const dirW = 1 + 0.5 * ke.anisotropy * Math.cos(2 * Math.atan2(gyTw, gxTw));
          const norm = Math.hypot(gxTw, gyTw) + 1e-6;
          const wx = warpAmp * (gxTw / norm) * dirW;
          const wy = warpAmp * (gyTw / norm) * dirW;
          const base = baseImagePixelsRef.current.data;
          const sxy = sampleRGB(base, x + wx, y + wy, w, h);
          let rW = sxy.R/255, gW = sxy.G/255, bW = sxy.B/255;
          if (displayMode==='grayBaseGrayRims'){ const yy=luma01(rW,gW,bW); rW=gW=bW=yy; }
          let sb = surfaceBlend * mask;
          if (coupleE2S) sb *= (1 + etaAmp * clamp01(0.6)); // (optionally use Erim carried in a tmp)
          if (kurEnabled && cohRef.current) sb *= (0.5 + 0.5*cohRef.current[p]);
          sb = clamp(sb, 0, 1);
          out[i+0] = Math.floor(out[i+0]*(1-sb) + (rW*255)*sb);
          out[i+1] = Math.floor(out[i+1]*(1-sb) + (gW*255)*sb);
          out[i+2] = Math.floor(out[i+2]*(1-sb) + (bW*255)*sb);
        }
      }
    }
    const obs = obsCount ? obsSum / obsCount : lastObsRef.current;
    lastObsRef.current = clamp(obs, 0.001, 10);

} else {
// fallback synthetic (omit here for brevity)
}

ctx.putImageData(imgOut, 0, 0);
};

// --- RAF loop ---
useEffect(() => {
const canvas = canvasRef.current; if (!canvas) return;
const ctx = canvas.getContext('2d', { willReadFrequently: true }); if (!ctx) return;
let anim = true, t0 = performance.now();
const frame = () => { if (!anim) return;
const t = ((performance.now() - t0) \* 0.001);
drawAtTime(ctx, t);
requestAnimationFrame(frame);
};
frame();
return () => { anim = false; };
}, [w,h, imgBitmap, edgeThreshold, blend, kernel, dmt, thetaMode, thetaGlobal, beta2, jitter, sigma, edgeRadius, microsaccade, speed, contrast, displayMode, rimAlpha, phasePin, alive, polBins, normPin, surfEnabled, wallGroup, nOrient, surfaceBlend, warpAmp, region, coupleE2S, etaAmp, betaAlign, coupleS2E, gammaOff, gammaSigma, alphaPol, kSigma, kurEnabled, K0, alphaKur, gammaKur, omega0, epsKur]);

// --- Minimal JSX return ---
return (

  <div className="w-full min-h-screen bg-black text-white p-4">
    <div className="mb-2 flex gap-3 items-center">
      <input type="file" accept="image/*" onChange={onFile} />
      <select value={presetIndex} onChange={e=>setPresetIndex(parseInt(e.target.value))}>
        {PRESETS.map((p,i)=>(<option key={p.name} value={i}>{p.name}</option>))}
      </select>
      <button onClick={()=>applyPreset(PRESETS[presetIndex])} className="px-3 py-1 bg-white/10 rounded">Apply preset</button>
    </div>
    <canvas ref={canvasRef} width={w} height={h} className="rounded-xl ring-1 ring-white/10" />
  </div>
);
} // <-- end component
