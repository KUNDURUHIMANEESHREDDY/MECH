import React, { useState } from 'react';
import { Play, Loader2 } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import type { FC, PanelContext } from '../../shared/types';

export interface BenchCase {
  name: string;
  prompt: string;
  expect: string;
}

export const BENCH_CASES: BenchCase[] = [
  { name: 'Factual recall', prompt: 'The capital of France is', expect: 'Paris' },
  { name: 'Induction', prompt: 'The cat chased the dog. The cat chased the', expect: 'dog' },
  { name: 'IOI', prompt: 'When John and Mary went to the store, Mary gave the book to', expect: 'John' },
  { name: 'Copying', prompt: 'Alpha beta gamma delta epsilon alpha beta', expect: 'gamma' },
  { name: 'Common knowledge', prompt: 'The sky is blue because of', expect: 'sunlight' },
];

export const BENCH_SUITE_CASES: BenchCase[] = [
  ...BENCH_CASES,
  { name: 'Translation', prompt: 'The word for hello in French is', expect: 'bonjour' },
  { name: 'Arithmetic', prompt: 'Two plus two equals', expect: 'four' },
  { name: 'Narrative', prompt: 'Once upon a time in a land far away, there lived a', expect: 'king' },
  { name: 'Causal', prompt: 'The glass broke because it fell off the', expect: 'table' },
  { name: 'Definition', prompt: 'Photosynthesis is the process by which plants convert', expect: 'light' },
  { name: 'Memory', prompt: 'The first President of the United States was', expect: 'George' },
];

export interface BenchResult {
  prompt: string;
  latencyMs: number;
  tokens: number;
  tokensPerSec: number;
  gpuUtil: number;
  memoryUtil: number;
  generated: string;
}

const btnPrimary: React.CSSProperties = {
  background: colors.primary,
  color: colors.onPrimary,
  border: 'none',
  borderRadius: 6,
  padding: '6px 14px',
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
};
const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};
const th: React.CSSProperties = {
  textAlign: 'left',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  color: colors.bodyMuted,
  padding: '4px 8px',
  borderBottom: `1px solid ${colors.hairline}`,
};
const td: React.CSSProperties = {
  fontSize: 12,
  padding: '6px 8px',
  borderBottom: `1px solid ${colors.dividerSoft}`,
  color: colors.body,
  fontVariantNumeric: 'tabular-nums',
};

interface BenchmarkRunnerProps {
  cases: BenchCase[];
  maxNewTokens?: number;
  title?: string;
}

/** Core benchmark runner shared by the Benchmark and Benchmark Suite panels. */
export function BenchmarkRunner({ cases, maxNewTokens = 10, title = 'Benchmark' }: BenchmarkRunnerProps): React.ReactElement {
  const { state: model, infer } = useModel();
  const [results, setResults] = useState<BenchResult[]>([]);
  const [running, setRunning] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);

  const runAll = async () => {
    setRunning(true);
    const out: BenchResult[] = [];
    for (const c of cases) {
      setCurrent(c.name);
      const start = performance.now();
      try {
        const r = await infer(c.prompt, maxNewTokens);
        const ms = performance.now() - start;
        out.push({
          prompt: c.prompt,
          latencyMs: Math.round(ms),
          tokens: r.tokens.length,
          tokensPerSec: parseFloat((r.tokens.length / (ms / 1000)).toFixed(1)),
          gpuUtil: r.gpuUtil,
          memoryUtil: r.memoryUtil,
          generated: r.generatedText,
        });
      } catch (err: any) {
        out.push({ prompt: c.prompt, latencyMs: 0, tokens: 0, tokensPerSec: 0, gpuUtil: 0, memoryUtil: 0, generated: `ERR: ${err.message}` });
      }
    }
    setResults(out);
    setRunning(false);
    setCurrent(null);
  };

  const avgLatency = results.length ? Math.round(results.reduce((a, r) => a + r.latencyMs, 0) / results.length) : 0;
  const avgTps = results.length ? parseFloat((results.reduce((a, r) => a + r.tokensPerSec, 0) / results.length).toFixed(1)) : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <button onClick={runAll} disabled={!model.loaded || running} style={{ ...btnPrimary, opacity: !model.loaded || running ? 0.5 : 1 }}>
          {running ? <Loader2 size={13} className="spin" /> : <Play size={13} />}
          {running && current ? `Running: ${current}…` : `Run ${cases.length} cases`}
        </button>
        {model.loaded ? (
          <span style={{ fontSize: 11, background: '#e6f4ea', color: '#137333', padding: '3px 8px', borderRadius: 4, fontWeight: 600 }}>
            ● Live Model Internals (PyTorch / GPU)
          </span>
        ) : (
          <span style={{ fontSize: 11, background: '#fef7e0', color: '#b06000', padding: '3px 8px', borderRadius: 4, fontWeight: 600 }}>
            ○ Framework Mode (Load model in GPT-2 panel for live weights)
          </span>
        )}
        {results.length > 0 && (
          <span style={{ fontSize: 12, color: colors.bodyMuted }}>
            avg <b style={{ color: colors.ink }}>{avgLatency} ms</b> · {avgTps} tok/s
          </span>
        )}
      </div>

      <div style={{ fontSize: 11, color: colors.bodyMuted, background: colors.canvas, padding: '8px 12px', borderRadius: 6, border: `1px solid ${colors.hairline}` }}>
        <b>Scientific Integrity Notice:</b> Live benchmarks run on loaded GPU/CPU PyTorch weights. Offline or stubbed runs validate the software framework and scheduling harness, not real model internals.
      </div>

      {results.length === 0 ? (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          {title}: runs {cases.length} curated prompts through the loaded model, reporting latency, throughput and hardware utilization.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: colors.canvas, borderRadius: 8 }}>
            <thead>
              <tr>
                <th style={th}>#</th>
                <th style={th}>Prompt</th>
                <th style={th}>Tokens</th>
                <th style={th}>Latency</th>
                <th style={th}>tok/s</th>
                <th style={th}>GPU</th>
                <th style={th}>Mem</th>
                <th style={th}>Output</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i}>
                  <td style={td}>{i + 1}</td>
                  <td style={{ ...td, fontFamily: 'monospace', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.prompt}</td>
                  <td style={td}>{r.tokens}</td>
                  <td style={td}>{r.latencyMs} ms</td>
                  <td style={td}>{r.tokensPerSec}</td>
                  <td style={{ ...td, color: r.gpuUtil != null && r.gpuUtil > 60 ? colors.dangerText : colors.body }}>{r.gpuUtil != null ? `${r.gpuUtil}%` : 'n/a'}</td>
                  <td style={td}>{r.memoryUtil != null ? `${r.memoryUtil}%` : 'n/a'}</td>
                  <td style={{ ...td, fontFamily: 'Georgia, serif', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.generated}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Cases ({cases.length})</div>
        {cases.map((c) => (
          <div key={c.name} style={{ display: 'flex', gap: 8, fontSize: 12 }}>
            <span style={{ color: colors.bodyMuted, minWidth: 130 }}>{c.name}</span>
            <span style={{ color: colors.body, fontFamily: 'monospace', flex: 1 }}>{c.prompt}</span>
            <span style={{ color: colors.bodyMuted }}>&rarr; {c.expect}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Benchmark: 5 prompt types against the loaded model. */
export const BenchmarkPanel: FC<PanelContext> = () => <BenchmarkRunner cases={BENCH_CASES} title="Benchmark" />;

/** Benchmark Suite: extended prompt set (11 cases). */
export const BenchmarkSuitePanel: FC<PanelContext> = () => <BenchmarkRunner cases={BENCH_SUITE_CASES} title="Benchmark Suite" maxNewTokens={12} />;