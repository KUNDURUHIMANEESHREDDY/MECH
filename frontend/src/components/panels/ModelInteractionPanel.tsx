import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronRight, FlaskConical, GitCompare, Loader2, MessageSquare, Play, Send, Trash2 } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import type { FC, PanelContext } from '../../shared/types';
import { runInteraction, InteractResponse } from '../../services/interactService';

const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};
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
const btnGhost: React.CSSProperties = {
  ...btnPrimary,
  background: colors.surfacePearl,
  color: colors.ink,
  border: `1px solid ${colors.hairline}`,
};
const selectStyle: React.CSSProperties = {
  padding: '5px 8px',
  borderRadius: 6,
  border: `1px solid ${colors.hairline}`,
  background: colors.canvas,
  fontSize: 12,
  color: colors.ink,
};

const GPT2_MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large'] as const;

interface HistoryEntry {
  id: number;
  backend: string;
  model: string;
  prompt: string;
  response: string;
  status: string;
  nGenerated?: number;
  latency?: number;
  ts: string;
}

let idCounter = 1;

/** Model Interaction: send prompts, capture full-answer responses, run experiments, compare. */
export const ModelInteractionPanel: FC<PanelContext> = () => {
  const [model, setModel] = useState<string>('gpt2');
  const [prompt, setPrompt] = useState('The capital of France is');
  const [maxNewTokens, setMaxNewTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.7);
  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<InteractResponse | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [batchText, setBatchText] = useState('');
  const [batchResults, setBatchResults] = useState<HistoryEntry[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [showBatch, setShowBatch] = useState(false);

  const toEntry = (res: InteractResponse, p: string): HistoryEntry => ({
    id: idCounter++,
    backend: 'gpt2',
    model: res.model ?? model,
    prompt: p,
    response: res.response ?? res.error ?? '',
    status: res.status,
    nGenerated: res.n_generated,
    latency: res.latency_ms,
    ts: new Date().toLocaleTimeString(),
  });

  const runSingle = async (p: string) => {
    setRunning(true);
    setResponse(null);
    try {
      const res = await runInteraction({
        prompt: p,
        backend: 'gpt2',
        model,
        max_new_tokens: maxNewTokens,
        temperature,
      });
      setResponse(res);
      setHistory(prev => [toEntry(res, p), ...prev]);
    } finally {
      setRunning(false);
    }
  };

  const runBatch = async () => {
    const prompts = batchText.split('\n').map(s => s.trim()).filter(Boolean);
    if (prompts.length === 0 || batchRunning) return;
    setBatchRunning(true);
    setBatchResults([]);
    const results: HistoryEntry[] = [];
    for (const p of prompts) {
      try {
        const res = await runInteraction({
          prompt: p,
          backend: 'gpt2',
          model,
          max_new_tokens: maxNewTokens,
          temperature,
        });
        results.push(toEntry(res, p));
      } catch (e: unknown) {
        results.push({
          id: idCounter++,
          backend: 'gpt2',
          model,
          prompt: p,
          response: e instanceof Error ? e.message : 'Failed',
          status: 'error',
          ts: new Date().toLocaleTimeString(),
        });
      }
    }
    setBatchResults(results);
    setBatchRunning(false);
  };

  /** Same prompt across all GPT-2 model sizes — one comparison. */
  const compare = async () => {
    if (!prompt.trim() || running) return;
    setRunning(true);
    setResponse(null);
    const results: HistoryEntry[] = [];
    for (const m of GPT2_MODELS) {
      try {
        const res = await runInteraction({
          prompt,
          backend: 'gpt2',
          model: m,
          max_new_tokens: maxNewTokens,
          temperature,
        });
        results.push(toEntry(res, prompt));
      } catch (e: unknown) {
        results.push({
          id: idCounter++,
          backend: 'gpt2',
          model: m,
          prompt,
          response: e instanceof Error ? e.message : 'Failed',
          status: 'error',
          ts: new Date().toLocaleTimeString(),
        });
      }
    }
    setBatchResults(results);
    setBatchRunning(false);
    setShowBatch(true);
  };

  const clearHistory = () => setHistory([]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 12, padding: 4 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <MessageSquare size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, fontSize: 13, color: colors.ink }}>Model Interaction</span>
        <span style={{ color: colors.bodyMuted, fontSize: 11 }}>
          Send prompts · capture whole-answer responses · run experiments · compare models
        </span>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: colors.bodyMuted, fontWeight: 600 }}>Model</span>
          <select style={selectStyle} value={model} onChange={e => setModel(e.target.value)}>
            {GPT2_MODELS.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: colors.bodyMuted, fontWeight: 600 }}>Max tokens</span>
          <input
            type="number"
            min={1}
            max={512}
            value={maxNewTokens}
            onChange={e => setMaxNewTokens(Math.min(512, Math.max(1, Number(e.target.value))))}
            style={{ ...selectStyle, width: 64 }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: colors.bodyMuted, fontWeight: 600 }}>Temp</span>
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={e => setTemperature(Math.min(2, Math.max(0, Number(e.target.value))))}
            style={{ ...selectStyle, width: 56 }}
          />
        </div>
      </div>

      {/* Prompt + Run */}
      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Prompt</div>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={3}
          placeholder="Enter a prompt for the model..."
          style={{
            width: '100%',
            boxSizing: 'border-box',
            padding: '8px 10px',
            borderRadius: 6,
            border: `1px solid ${colors.hairline}`,
            fontSize: 12,
            fontFamily: 'inherit',
            resize: 'vertical',
            color: colors.ink,
            background: colors.canvas,
          }}
        />
        <div style={{ display: 'flex', gap: 8 }}>
          <button style={btnPrimary} onClick={() => runSingle(prompt)} disabled={running || !prompt.trim()}>
            {running ? <Loader2 size={14} className="spin" /> : <Send size={14} />}
            {running ? 'Generating whole answer…' : 'Run prompt'}
          </button>
          <button style={btnGhost} onClick={compare} disabled={running || !prompt.trim()}>
            <GitCompare size={14} /> Compare GPT-2 sizes
          </button>
        </div>
      </div>

      {/* Response */}
      {response && (
        <div style={{ ...card, background: colors.canvasParchment }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Brain size={14} color={colors.primary} />
            <span style={{ fontWeight: 700, color: colors.ink }}>
              {response.model} ({response.backend})
            </span>
            {response.status === 'demo' && (
              <span style={{ color: colors.warningText, fontSize: 10 }}>demo mode — backend offline</span>
            )}
            {response.status === 'error' && (
              <span style={{ color: colors.dangerText, fontSize: 10 }}>error</span>
            )}
            {response.n_generated !== undefined && (
              <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{response.n_generated} tokens</span>
            )}
            {response.latency_ms !== undefined && (
              <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{response.latency_ms}ms</span>
            )}
          </div>
          <div style={{ color: colors.body, lineHeight: 1.55, whiteSpace: 'pre-wrap', fontFamily: 'Georgia, serif', fontSize: 13 }}>
            {response.response}
          </div>
        </div>
      )}

      {/* Toggles */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button style={{ ...btnGhost, fontSize: 11 }} onClick={() => setShowHistory(v => !v)}>
          {showHistory ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          History ({history.length})
        </button>
        <button style={{ ...btnGhost, fontSize: 11 }} onClick={() => setShowBatch(v => !v)}>
          {showBatch ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <FlaskConical size={12} /> Experiments / Compare
        </button>
        {history.length > 0 && (
          <button style={{ ...btnGhost, fontSize: 11, color: colors.dangerText }} onClick={clearHistory}>
            <Trash2 size={12} /> Clear
          </button>
        )}
      </div>

      {/* History — captured responses across inputs */}
      {showHistory && history.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 260, overflowY: 'auto' }}>
          {history.map(h => (
            <div key={h.id} style={{ ...card, padding: 10, background: colors.canvasParchment }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Brain size={12} color={colors.primary} />
                <span style={{ fontWeight: 700, color: colors.ink, fontSize: 11 }}>
                  {h.model} ({h.backend})
                </span>
                <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{h.ts}</span>
                {h.status === 'demo' && <span style={{ color: colors.warningText, fontSize: 10 }}>demo</span>}
                {h.status === 'error' && <span style={{ color: colors.dangerText, fontSize: 10 }}>error</span>}
                {h.nGenerated !== undefined && (
                  <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{h.nGenerated} tok</span>
                )}
                {h.latency !== undefined && (
                  <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{h.latency}ms</span>
                )}
              </div>
              <div style={{ color: colors.bodyMuted, fontSize: 11, fontStyle: 'italic', maxHeight: 40, overflow: 'hidden' }}>
                {h.prompt}
              </div>
              <div style={{ color: colors.body, fontSize: 11, lineHeight: 1.5, maxHeight: 80, overflow: 'hidden' }}>
                {h.response}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Batch experiments + compare */}
      {showBatch && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink, fontSize: 12 }}>
            <FlaskConical size={13} style={{ verticalAlign: -2, marginRight: 6 }} color={colors.primary} />
            Controlled experiments — one prompt per line
          </div>
          <textarea
            value={batchText}
            onChange={e => setBatchText(e.target.value)}
            rows={4}
            placeholder={'The capital of France is\nThe quick brown fox jumps\n2 + 2 equals\nWrite a short poem about the sea'}
            style={{
              width: '100%',
              boxSizing: 'border-box',
              padding: '8px 10px',
              borderRadius: 6,
              border: `1px solid ${colors.hairline}`,
              fontSize: 12,
              fontFamily: 'inherit',
              resize: 'vertical',
              color: colors.ink,
              background: colors.canvas,
            }}
          />
          <div style={{ display: 'flex', gap: 8 }}>
            <button style={btnPrimary} onClick={runBatch} disabled={batchRunning || !batchText.trim()}>
              {batchRunning ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
              {batchRunning ? 'Running batch…' : 'Run experiment batch'}
            </button>
          </div>
          {batchResults.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 300, overflowY: 'auto' }}>
              {batchResults.map(r => (
                <div key={r.id} style={{ ...card, padding: 10, background: colors.canvasParchment }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontWeight: 700, color: colors.ink, fontSize: 11 }}>{r.prompt}</span>
                    <span style={{ color: colors.bodyMuted, fontSize: 10 }}>→ {r.model} ({r.backend})</span>
                    {r.nGenerated !== undefined && (
                      <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{r.nGenerated} tok</span>
                    )}
                    {r.latency !== undefined && (
                      <span style={{ color: colors.bodyMuted, fontSize: 10 }}>{r.latency}ms</span>
                    )}
                  </div>
                  <div style={{ color: colors.body, fontSize: 11, lineHeight: 1.5 }}>{r.response}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};