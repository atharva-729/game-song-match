/* Song analysis + matcher. Mirrors scripts/song_energy.py and scripts/match.py step for step
   (sequential sums, same interpolation, same DTW tie-breaking) so Python and JS agree.
   Runs on the page, inside the Web Worker, and under Node for the parity test. */
(function (root) {
  const N = 200, W = 10, FAMOUS_BONUS = 0.4, FRAME = 2048, HOP = 1024;

  // ---- radix-2 FFT (in place), magnitudes of bins 0..n/2 ----
  function fftMag(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
    const mag = new Float64Array(n / 2 + 1);
    for (let k = 0; k <= n / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
    return mag;
  }

  // ---- energy per frame: 0.6*loudness + 0.4*spectral flux, ~3 s moving average, silence trimmed ----
  function energyCurve(x, sr) {
    const nf = x.length >= FRAME ? Math.floor((x.length - FRAME) / HOP) + 1 : 0;
    if (nf < 4) return null;
    const hann = new Float64Array(FRAME);
    for (let i = 0; i < FRAME; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FRAME);
    const loud = new Float64Array(nf), flux = new Float64Array(nf);
    let prev = null;
    const re = new Float64Array(FRAME), im = new Float64Array(FRAME);
    for (let f = 0; f < nf; f++) {
      const s = f * HOP;
      let ss = 0;
      for (let i = 0; i < FRAME; i++) { const v = x[s + i]; ss += v * v; re[i] = v * hann[i]; im[i] = 0; }
      const db = 20 * Math.log10(Math.sqrt(ss / FRAME) + 1e-10);
      loud[f] = (Math.min(0, Math.max(-60, db)) + 60) / 60;
      const mag = fftMag(re, im);
      let fl = 0;
      if (prev) for (let k = 0; k < mag.length; k++) { const d = mag[k] - prev[k]; if (d > 0) fl += d; }
      flux[f] = fl; prev = mag;
    }
    const sorted = Array.from(flux).sort((a, b) => a - b);
    const p95 = sorted[Math.floor(0.95 * (nf - 1))] || 1;
    const e = new Float64Array(nf);
    for (let f = 0; f < nf; f++) e[f] = 0.6 * loud[f] + 0.4 * Math.min(flux[f] / p95, 1);
    // centred moving average over ~3 s (edges average what is available)
    const half = Math.floor(Math.round(3 * sr / HOP) / 2);
    const csum = new Float64Array(nf + 1);
    for (let f = 0; f < nf; f++) csum[f + 1] = csum[f] + e[f];
    const sm = new Float64Array(nf);
    for (let f = 0; f < nf; f++) {
      const a = Math.max(0, f - half), b = Math.min(nf, f + half + 1);
      sm[f] = (csum[b] - csum[a]) / (b - a);
    }
    let mx = 0;
    for (let f = 0; f < nf; f++) if (sm[f] > mx) mx = sm[f];
    let first = 0, last = nf - 1;
    while (first < nf && sm[first] < 0.05 * mx) first++;
    while (last > first && sm[last] < 0.05 * mx) last--;
    const curve = Array.from(sm.subarray(first, last + 1));
    const t0 = (first * HOP + FRAME / 2) / sr, t1 = (last * HOP + FRAME / 2) / sr;
    return { curve, t0, t1, frames: nf };
  }

  // ---- normalise: resample to N, z-score ----
  function resample(x, n = N) {
    const m = x.length, out = new Array(n);
    if (m === 1) return out.fill(x[0]);
    for (let k = 0; k < n; k++) {
      const t = k * (m - 1) / (n - 1), i = Math.floor(t);
      out[k] = i >= m - 1 ? x[m - 1] : x[i] + (x[i + 1] - x[i]) * (t - i);
    }
    return out;
  }
  function zscore(x) {
    const n = x.length;
    let s = 0; for (const v of x) s += v;
    const mean = s / n;
    let ss = 0; for (const v of x) ss += (v - mean) * (v - mean);
    const std = Math.sqrt(ss / n);
    if (std < 1e-6) return null;
    return x.map(v => (v - mean) / std);
  }
  const normalise = x => zscore(resample(x));

  // ---- banded DTW, squared difference, divided by path length ----
  const K = 2 * W + 1;
  function dtw(a, Q, off, scale) {
    let pD = new Float64Array(K).fill(Infinity), pL = new Float64Array(K);
    let cD = new Float64Array(K), cL = new Float64Array(K);
    for (let i = 0; i < N; i++) {
      cD.fill(Infinity); cL.fill(0);
      for (let k = 0; k < K; k++) {
        const j = i + k - W;
        if (j < 0 || j >= N) continue;
        const d = a[i] - Q[off + j] / scale, c = d * d;
        if (i === 0 && j === 0) { cD[k] = c; cL[k] = 1; continue; }
        let bD = Infinity, bL = 0;
        if (i > 0 && j > 0) { bD = pD[k]; bL = pL[k]; }
        if (i > 0 && k + 1 < K && pD[k + 1] < bD) { bD = pD[k + 1]; bL = pL[k + 1]; }
        if (j > 0 && k - 1 >= 0 && cD[k - 1] < bD) { bD = cD[k - 1]; bL = cL[k - 1]; }
        cD[k] = c + bD; cL[k] = bL + 1;
      }
      let t = pD; pD = cD; cD = t; t = pL; pL = cL; cL = t;
    }
    return pD[W] / pL[W];
  }

  function dtwPath(a, b) {
    const D = [], L = [], P = [];
    for (let i = 0; i < N; i++) { D.push(new Float64Array(K).fill(Infinity)); L.push(new Float64Array(K)); P.push(new Int8Array(K).fill(-1)); }
    for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) {
      const j = i + k - W;
      if (j < 0 || j >= N) continue;
      const c = (a[i] - b[j]) * (a[i] - b[j]);
      if (i === 0 && j === 0) { D[i][k] = c; L[i][k] = 1; continue; }
      let bd = Infinity, bl = 0, bp = -1;
      if (i > 0 && j > 0) { bd = D[i - 1][k]; bl = L[i - 1][k]; bp = 0; }
      if (i > 0 && k + 1 < K && D[i - 1][k + 1] < bd) { bd = D[i - 1][k + 1]; bl = L[i - 1][k + 1]; bp = 1; }
      if (j > 0 && k - 1 >= 0 && D[i][k - 1] < bd) { bd = D[i][k - 1]; bl = L[i][k - 1]; bp = 2; }
      D[i][k] = c + bd; L[i][k] = bl + 1; P[i][k] = bp;
    }
    const path = [];
    let i = N - 1, k = W;
    for (;;) {
      path.push([i, i + k - W]);
      const p = P[i][k];
      if (p === -1) break;
      if (p === 0) i--; else if (p === 1) { i--; k++; } else k--;
    }
    return { dist: D[N - 1][W] / L[N - 1][W], path: path.reverse() };
  }

  // ---- search: score = dist * (1 - 0.4*famous); ties broken by index order ----
  function search(a, Q, rows, scale, scope = "both", top = 5) {
    const M = rows.length, score = new Float64Array(M), dist = new Float64Array(M);
    for (let m = 0; m < M; m++) {
      dist[m] = dtw(a, Q, m * N, scale);
      score[m] = dist[m] * (1 - FAMOUS_BONUS * (rows[m].famous ? 1 : 0));
    }
    const order = Array.from({ length: M }, (_, m) => m).sort((p, q) => score[p] - score[q] || p - q);
    const ok = m => scope === "both" || rows[m].league === scope;
    const pick = m => ({ i: m, score: score[m], dist: dist[m] });
    const res = { top: order.filter(ok).slice(0, top).map(pick) };
    for (const lg of ["NFL", "IPL"]) {
      const b = order.find(m => rows[m].league === lg);
      res["best_" + lg.toLowerCase()] = b === undefined ? null : pick(b);
    }
    return res;
  }

  const api = { N, W, FRAME, HOP, fftMag, energyCurve, resample, zscore, normalise, dtw, dtwPath, search };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SongMatch = api;
})(typeof self !== "undefined" ? self : this);
