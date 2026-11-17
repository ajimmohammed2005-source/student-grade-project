/* Overfit Lab - engine.
 * Everything the game shows is computed live here: seeded random data,
 * least-squares polynomial fitting (Chebyshev basis for numerical stability), and RMSE.
 * Works in the browser (window.Engine) and in Node (module.exports) so it can be tested.
 */
(function (root) {
  "use strict";

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // The hidden "truth" the learner is trying to discover (a gentle wave with a slope).
  const TRUE_F = (x) => Math.sin(2.6 * x) + 0.3 * x;
  const NOISE = 0.35;

  function makeSynthetic(seed, nTrain, nTest) {
    const rng = mulberry32(seed);
    const draw = (n) => {
      const x = [], y = [];
      for (let i = 0; i < n; i++) {
        const xi = rng() * 2 - 1;
        x.push(xi);
        y.push(TRUE_F(xi) + NOISE * gauss(rng));
      }
      return { x, y };
    };
    return { train: draw(nTrain), test: draw(nTest), yScale: 1, noise: NOISE };
  }

  // Real students: predict final grade G3 from first-period grade G1 using only a few students.
  function makeReal(seed, pts, nFit, nVal) {
    const rng = mulberry32(seed);
    const idx = pts.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const take = (a, b) => {
      const x = [], y = [];
      for (const i of idx.slice(a, b)) { x.push((pts[i][0] - 11) / 8); y.push((pts[i][1] - 10) / 4); }
      return { x, y };
    };
    return { train: take(0, nFit), val: take(nFit, nFit + nVal), test: take(nFit + nVal, idx.length), yScale: 4 };
  }

  // Chebyshev basis T_0..T_deg evaluated at x in [-1, 1]
  function basis(x, deg) {
    const t = new Array(deg + 1);
    t[0] = 1;
    if (deg >= 1) t[1] = x;
    for (let k = 2; k <= deg; k++) t[k] = 2 * x * t[k - 1] - t[k - 2];
    return t;
  }

  function solve(A, b) { // Gaussian elimination with partial pivoting
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let p = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < n; r++) {
        const f = A[r][i] / A[i][i];
        for (let c = i; c < n; c++) A[r][c] -= f * A[i][c];
        b[r] -= f * b[i];
      }
    }
    const w = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      for (let c = i + 1; c < n; c++) s -= A[i][c] * w[c];
      w[i] = s / A[i][i];
    }
    return w;
  }

  function fitPoly(xs, ys, deg, lambda) {
    lambda = lambda === undefined ? 1e-8 : lambda; // tiny ridge only to keep the solver stable
    const m = deg + 1;
    const AtA = Array.from({ length: m }, () => new Array(m).fill(0));
    const Atb = new Array(m).fill(0);
    for (let i = 0; i < xs.length; i++) {
      const t = basis(xs[i], deg);
      for (let r = 0; r < m; r++) {
        Atb[r] += t[r] * ys[i];
        for (let c = 0; c < m; c++) AtA[r][c] += t[r] * t[c];
      }
    }
    for (let r = 0; r < m; r++) AtA[r][r] += lambda;
    return solve(AtA, Atb);
  }

  function predict(w, x) {
    const t = basis(x, w.length - 1);
    let s = 0;
    for (let k = 0; k < t.length; k++) s += w[k] * t[k];
    return s;
  }

  function rmse(w, set, yScale) {
    if (!set || set.x.length === 0) return NaN;
    let s = 0;
    for (let i = 0; i < set.x.length; i++) { const e = predict(w, set.x[i]) - set.y[i]; s += e * e; }
    return Math.sqrt(s / set.x.length) * (yScale || 1);
  }

  // Train / validation / test error for every degree 1..maxDeg (used for the reveal chart and the score)
  function errorsByDegree(data, maxDeg, nFit) {
    const fitSet = nFit ? { x: data.train.x.slice(0, nFit), y: data.train.y.slice(0, nFit) } : data.train;
    const out = [];
    for (let d = 1; d <= maxDeg; d++) {
      const w = fitPoly(fitSet.x, fitSet.y, d);
      out.push({ deg: d, train: rmse(w, fitSet, data.yScale), val: rmse(w, data.val, data.yScale), test: rmse(w, data.test, data.yScale) });
    }
    return out;
  }

  // Score = how close your test error is to the best test error achievable by any degree (0-100)
  function scoreFromErrors(errs, chosenDeg) {
    const best = Math.min(...errs.map((e) => e.test));
    const mine = errs.find((e) => e.deg === chosenDeg).test;
    return { best, mine, score: Math.max(0, Math.min(100, Math.round((100 * best) / mine))), bestDeg: errs.find((e) => e.test === best).deg };
  }

  const api = { mulberry32, gauss, TRUE_F, NOISE, makeSynthetic, makeReal, fitPoly, predict, rmse, errorsByDegree, scoreFromErrors };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof window !== "undefined" ? window : globalThis);
