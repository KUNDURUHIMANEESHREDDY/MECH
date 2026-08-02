import { NeuronPoint, ProjectedPoint } from './types';

const PROJECTION_CACHE = new Map<string, ProjectedPoint[]>();

function jacobiEigenvaluesSymmetric(a: number[][], maxIter = 60): { values: number[]; vectors: number[][] } {
  const n = a.length;
  const v = Array.from({ length: n }, (_, i) => {
    const row = new Array(n).fill(0);
    row[i] = 1;
    return row;
  });
  const aWork = a.map(row => [...row]);

  for (let iter = 0; iter < maxIter; iter++) {
    let p = 0;
    let q = 1;
    let maxOff = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const off = Math.abs(aWork[i][j]);
        if (off > maxOff) {
          maxOff = off;
          p = i;
          q = j;
        }
      }
    }
    if (maxOff < 1e-12) break;

    const app = aWork[p][p];
    const aqq = aWork[q][q];
    const apq = aWork[p][q];
    const theta = (aqq - app) / (2 * apq);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;

    for (let k = 0; k < n; k++) {
      const akp = aWork[k][p];
      const akq = aWork[k][q];
      aWork[k][p] = c * akp - s * akq;
      aWork[k][q] = s * akp + c * akq;
    }
    for (let k = 0; k < n; k++) {
      const apk = aWork[p][k];
      const aqk = aWork[q][k];
      aWork[p][k] = c * apk - s * aqk;
      aWork[q][k] = s * apk + c * aqk;
    }

    for (let k = 0; k < n; k++) {
      const vkp = v[k][p];
      const vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }
  }

  const values = aWork.map((row, i) => row[i]);
  const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i]);
  return {
    values: order.map(i => values[i]),
    vectors: order.map(i => v.map(row => row[i])),
  };
}

function pcaProject(embeddings: number[][], dims: number): ProjectedPoint[] {
  const n = embeddings.length;
  const d = embeddings[0].length;
  if (dims >= d) dims = Math.max(1, d - 1);
  if (dims < 1) dims = 1;

  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) mean[j] += embeddings[i][j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;

  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const centered = new Array(d);
    for (let j = 0; j < d; j++) centered[j] = embeddings[i][j] - mean[j];
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        cov[a][b] += centered[a] * centered[b];
      }
    }
  }
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      cov[a][b] /= n;
      cov[b][a] = cov[a][b];
    }
  }

  const eig = jacobiEigenvaluesSymmetric(cov);
  const axes = eig.vectors.slice(0, dims);

  return embeddings.map(emb => {
    const out = new Array(dims).fill(0);
    for (let a = 0; a < dims; a++) {
      const axis = axes[a];
      let s = 0;
      for (let j = 0; j < d; j++) s += (emb[j] - mean[j]) * axis[j];
      out[a] = s;
    }
    return { x: out[0] ?? 0, y: out[1] ?? out[0] ?? 0 };
  });
}

function normalize(points: ProjectedPoint[]): ProjectedPoint[] {
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const pad = 0.04;
  const range = Math.max(xMax - xMin, yMax - yMin, 1e-9);
  return points.map(p => ({
    x: pad + ((p.x - xMin) / range) * (1 - 2 * pad),
    y: pad + ((p.y - yMin) / range) * (1 - 2 * pad),
  }));
}

function projectionKey(points: NeuronPoint[]): string {
  const dim = points[0]?.embedding.length ?? 0;
  return `${points.length}|${dim}|${points[0]?.layer ?? 0}-${points[points.length - 1]?.layer ?? 0}`;
}

export function projectNeurons(points: NeuronPoint[]): ProjectedPoint[] {
  if (points.length === 0) return [];
  const key = projectionKey(points);
  const cached = PROJECTION_CACHE.get(key);
  if (cached && cached.length === points.length) return cached;

  let projected: ProjectedPoint[];
  const dim = points[0].embedding.length;

  if (dim >= 3) {
    projected = pcaProject(points.map(p => p.embedding), 2);
  } else if (dim === 2) {
    projected = points.map(p => ({ x: p.embedding[0], y: p.embedding[1] }));
  } else {
    const jittered = points.map((p, i) => {
      const r = (i * 0.618033988749895) % 1;
      const base = p.embedding[0] ?? p.activation;
      return { x: base + r * 0.001, y: (i * 0.3141592653589793) % 1 };
    });
    projected = jittered;
  }

  const normalized = normalize(projected);
  PROJECTION_CACHE.set(key, normalized);
  return normalized;
}

export function clearProjectionCache(): void {
  PROJECTION_CACHE.clear();
}
