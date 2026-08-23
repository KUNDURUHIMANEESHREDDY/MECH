import React, { useEffect, useState } from 'react';
import { Play, Cpu, Database, Loader2, AlertTriangle, Settings2, ChevronDown, ChevronUp } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import type { FC, PanelContext } from '../../shared/types';
import { TokenViewer } from '../visualizations/panels/TokenViewer';

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
const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};

/** GPT-2 Live: load a model, run a prompt, and inspect tokens + metrics. */
export const GPT2LivePanel: FC<PanelContext> = () => {
  const { state: model, listModels, load, infer, clearError } = useModel();
  const [selectedModel, setSelectedModel] = useState('gpt2');
  const [prompt, setPrompt] = useState('The capital of France is');
  const [outputTokens, setOutputTokens] = useState<{ text: string; id: number }[]>([]);
  const [generated, setGenerated] = useState('');
  const [hovered, setHovered] = useState<number | null>(null);
  const [showParams, setShowParams] = useState(false);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [maxNewTokens, setMaxNewTokens] = useState(10);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) {
      void listModels();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const demoMode = model.modelInfo?.status === 'demo';

  const handleLoad = async () => {
    clearError();
    await load(selectedModel);
  };

  const handleRun = async () => {
    clearError();
    const result = await infer({
      prompt,
      maxNewTokens,
      temperature,
      topK,
      topP,
    });
    if (result) {
      setOutputTokens(result.tokens);
      setGenerated(result.generatedText);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {/* Model controls */}
      <div style={card}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <Cpu size={14} color={colors.primary} />
          <span style={{ fontWeight: 700, color: colors.ink }}>Model</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ flex: 1, minWidth: 120, padding: '5px 8px', borderRadius: 6, border: `1px solid ${colors.hairline}`, background: colors.canvas, fontSize: 12 }}
          >
            {model.availableModels.length === 0 ? (
              <option value="gpt2">gpt2</option>
            ) : (
              model.availableModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))
            )}
          </select>
          <button onClick={handleLoad} disabled={model.loading} style={{ ...btnGhost, opacity: model.loading ? 0.5 : 1 }}>
            {model.loading ? <Loader2 size={13} className="spin" /> : null}
            {model.loaded && model.modelInfo?.model_name === selectedModel ? 'Reload' : 'Load'}
          </button>
        </div>
        {model.loading && (
          <div style={{ fontSize: 12, color: colors.bodyMuted }}>Loading model…</div>
        )}
        {model.loaded && (
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 12 }}>
            <span style={{ color: colors.bodyMuted }}>
              Loaded: <b style={{ color: colors.ink }}>{model.modelInfo?.model_name}</b>
            </span>
            <span style={{ color: colors.bodyMuted }}>
              Layers <b style={{ color: colors.ink }}>{model.modelInfo?.num_layers ?? '–'}</b>
            </span>
            <span style={{ color: colors.bodyMuted }}>
              Heads <b style={{ color: colors.ink }}>{model.modelInfo?.num_heads ?? '–'}</b>
            </span>
            <span style={{ color: colors.bodyMuted }}>
              Hidden <b style={{ color: colors.ink }}>{model.modelInfo?.hidden_dim ?? '–'}</b>
            </span>
            {demoMode && <span style={{ color: colors.warningText }}>Demo dataset (sidecar offline)</span>}
          </div>
        )}
      </div>

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <AlertTriangle size={14} /> {model.error}
        </div>
      )}

      {/* Prompt runner */}
      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Prompt</div>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, resize: 'vertical', background: colors.canvas, color: colors.ink }}
        />
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button onClick={handleRun} disabled={!model.loaded || model.running} style={{ ...btnPrimary, opacity: !model.loaded || model.running ? 0.5 : 1 }}>
            {model.running ? <Loader2 size={13} className="spin" /> : <Play size={13} />}
            Run Inference
          </button>
          {!model.loaded && <span style={{ fontSize: 11, color: colors.bodyMuted }}>Load a model first</span>}
          {model.running && <span style={{ fontSize: 11, color: colors.bodyMuted }}>Generating…</span>}
        </div>
      </div>

      {/* Generation Parameters */}
      <div style={card}>
        <button
          onClick={() => setShowParams(!showParams)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: 0,
            color: colors.ink,
            fontWeight: 700,
            fontSize: 13,
          }}
        >
          <Settings2 size={14} color={colors.primary} />
          Generation Parameters
          {showParams ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
        {showParams && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
            {/* Temperature */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: colors.bodyMuted, minWidth: 80 }}>Temperature</label>
              <input
                type="range"
                min={0.1}
                max={2.0}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ fontSize: 12, color: colors.ink, minWidth: 30, textAlign: 'right' }}>{temperature.toFixed(1)}</span>
            </div>
            {/* Top-K */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: colors.bodyMuted, minWidth: 80 }}>Top-K</label>
              <input
                type="range"
                min={1}
                max={100}
                step={1}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ fontSize: 12, color: colors.ink, minWidth: 30, textAlign: 'right' }}>{topK}</span>
            </div>
            {/* Top-P */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: colors.bodyMuted, minWidth: 80 }}>Top-P</label>
              <input
                type="range"
                min={0.0}
                max={1.0}
                step={0.05}
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ fontSize: 12, color: colors.ink, minWidth: 30, textAlign: 'right' }}>{topP.toFixed(2)}</span>
            </div>
            {/* Max New Tokens */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: colors.bodyMuted, minWidth: 80 }}>Max Tokens</label>
              <input
                type="range"
                min={1}
                max={200}
                step={1}
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ fontSize: 12, color: colors.ink, minWidth: 30, textAlign: 'right' }}>{maxNewTokens}</span>
            </div>
          </div>
        )}
      </div>

      {/* Output */}
      {model.result && (
        <>
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink }}>Generated text</div>
            <div style={{ lineHeight: 1.6, fontFamily: 'Georgia, serif', fontSize: 14, color: colors.ink, whiteSpace: 'pre-wrap' }}>
              {model.result.generatedText}
            </div>
          </div>
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', justifyContent: 'space-between' }}>
              <span>Tokens ({model.result.tokens.length})</span>
              <span style={{ display: 'flex', gap: 14, fontWeight: 500, fontSize: 11, alignItems: 'center' }}>
                <span style={{ color: colors.bodyMuted }}>GPU <b style={{ color: colors.ink }}>{model.result.gpuUtil != null ? `${model.result.gpuUtil}%` : 'n/a'}</b></span>
                <span style={{ color: colors.bodyMuted }}>Mem <b style={{ color: colors.ink }}>{model.result.memoryUtil != null ? `${model.result.memoryUtil}%` : 'n/a'}</b></span>
              </span>
            </div>
            <TokenViewer
              tokens={outputTokens.map((t) => t.text).length ? outputTokens.map((t) => t.text) : model.result.tokens.map((t) => t.text)}
              tokenIds={model.result.tokens.map((t) => t.id)}
              selectedToken={hovered}
              onHoverToken={setHovered}
            />
          </div>
        </>
      )}
    </div>
  );
};