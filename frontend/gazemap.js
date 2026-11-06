// Per-person gaze mapping: ridge regression from iris/head features to screen position.
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };

export function solve(A, b) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-9;
    for (let j = c; j <= n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (!f) continue;
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((r) => r[n]);
}

export function fitRidge(rows, tx, ty, lambda = 1) {
  const p = rows[0].length, n = rows.length;
  const mu = Array(p).fill(0), sd = Array(p).fill(0);
  rows.forEach((r) => r.forEach((v, j) => { mu[j] += v / n; }));
  rows.forEach((r) => r.forEach((v, j) => { sd[j] += (v - mu[j]) ** 2 / n; }));
  for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  const q = p + 1;
  const A = Array.from({ length: q }, () => Array(q).fill(0));
  const bx = Array(q).fill(0), by = Array(q).fill(0);
  rows.forEach((r, i) => {
    const x = [1, ...r.map((v, j) => (v - mu[j]) / sd[j])];
    for (let a = 0; a < q; a++) {
      bx[a] += x[a] * tx[i]; by[a] += x[a] * ty[i];
      for (let c = 0; c < q; c++) A[a][c] += x[a] * x[c];
    }
  });
  for (let a = 1; a < q; a++) A[a][a] += lambda;
  return { mu, sd, wx: solve(A, bx), wy: solve(A, by) };
}

export function predictGaze(m, row) {
  const x = [1, ...row.map((v, j) => (v - m.mu[j]) / m.sd[j])];
  return { sx: x.reduce((s, v, i) => s + v * m.wx[i], 0), sy: x.reduce((s, v, i) => s + v * m.wy[i], 0) };
}

// Leave-one-point-out validation: median distance error (fraction of screen) on a held-out calibration point.
export function lopoError(samples, lambda = 1) {
  const ids = [...new Set(samples.map((s) => s.pi))];
  const errs = [];
  for (const id of ids) {
    const tr = samples.filter((s) => s.pi !== id), te = samples.filter((s) => s.pi === id);
    if (tr.length < 10 || !te.length) continue;
    const m = fitRidge(tr.map((s) => s.row), tr.map((s) => s.tx), tr.map((s) => s.ty), lambda);
    errs.push(mean(te.map((s) => { const p = predictGaze(m, s.row); return Math.hypot(p.sx - s.tx, p.sy - s.ty); })));
  }
  return errs.length ? median(errs) : Infinity;
}