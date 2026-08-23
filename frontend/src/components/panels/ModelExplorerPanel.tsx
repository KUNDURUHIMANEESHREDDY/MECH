import React, { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import { Layers, Flame, Loader2, Play, AlertTriangle, Download, Columns, X } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import type { FC, PanelContext, ProcessedResult } from '../../shared/types';
import { AttentionHeatmap } from '../visualizations/panels/AttentionHeatmap';
import { ActivationHeatmap } from '../visualizations/panels/ActivationHeatmap';
import { TokenViewer } from '../visualizations/panels/TokenViewer';

const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};
const btn: React.CSSProperties = {
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
const selectStyle: React.CSSProperties = {
  padding: '5px 8px',
  borderRadius: 6,
  border: `1px solid ${colors.hairline}`,
  background: colors.canvas,
  fontSize: 12,
  color: colors.ink,
};

/**
 * Model Explorer / Transformer Visualizer.
 * Layer + head pickers drive the shared selection store; attention heatmap,
 * activation heatmap and token strip all render from the shared model result.
 */
export const ModelExplorerPanel: FC<PanelContext> = () => {
  const { state: model, listModels, load, infer, clearError } = useModel();
  const sel = useSelectionStore();

  const [prompt, setPrompt] = useState('The capital of France is');
  const [hovered, setHovered] = useState<number | null>(null);
  const [layerNum, setLayerNum] = useState(0);
  const [headNum, setHeadNum] = useState(0);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [secondResult, setSecondResult] = useState<ProcessedResult | null>(null);
  const [secondPrompt, setSecondPrompt] = useState('The quick brown fox');
  const [loadingSecond, setLoadingSecond] = useState(false);
  const [selectedTokens, setSelectedTokens] = useState<Set<number>>(new Set());
  const heatmapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const layers = model.result?.layers ?? [];
  const tokens = model.result?.tokens.map((t) => t.text) ?? [];
  const layer = layerNum < layers.length ? layers[layerNum] : layers[0];
  const head = layer?.heads[headNum < (layer?.heads.length ?? 0) ? headNum : 0];

  // Neuron list for the selected layer, aggregated across heads (dedup by index)
  const layerNeurons = useMemo(() => {
    if (!layer) return [];
    const map = new Map<number, { index: number; activation: number; tokenActivations?: number[] }>();
    for (const h of layer.heads) {
      for (const n of h.neurons) {
        const prev = map.get(n.index);
        if (!prev || n.activation > prev.activation) {
          map.set(n.index, { index: n.index, activation: n.activation, tokenActivations: n.tokenActivations });
        }
      }
    }
    return Array.from(map.values());
  }, [layer]);

  const activations = useMemo(() => layerNeurons.map((n) => n.activation), [layerNeurons]);
  const neuronTokenActivations = useMemo(
    () => layerNeurons.map((n) => n.tokenActivations ?? null),
    [layerNeurons],
  );

  const selectHead = (l: number, h: number) => {
    setLayerNum(l);
    setHeadNum(h);
    sel.setSelectedHead(l, h);
  };

  const handleRun = async () => {
    clearError();
    await infer(prompt);
  };

  const handleRunComparison = async () => {
    setLoadingSecond(true);
    try {
      const result = await infer(secondPrompt);
      setSecondResult(result);
    } finally {
      setLoadingSecond(false);
    }
  };

  const toggleTokenSelection = (index: number) => {
    setSelectedTokens((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const exportHeatmap = useCallback(() => {
    if (!heatmapRef.current) return;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const svgElements = heatmapRef.current.querySelectorAll('svg');
    if (svgElements.length === 0) return;
    
    const svg = svgElements[0];
    const svgData = new XMLSerializer().serializeToString(svg);
    const img = new Image();
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      
      const link = document.createElement('a');
      link.download = `attention-heatmap-L${layerNum}-H${headNum}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };
    img.src = url;
  }, [layerNum, headNum]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {!model.loaded && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Model</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <select
              value={model.availableModels[0] ?? 'gpt2'}
              onChange={(e) => load(e.target.value)}
              style={selectStyle}
            >
              {(model.availableModels.length ? model.availableModels : ['gpt2']).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button onClick={() => load(model.availableModels[0] ?? 'gpt2')} disabled={model.loading} style={{ ...btn, opacity: model.loading ? 0.5 : 1 }}>
              {model.loading ? <Loader2 size={13} className="spin" /> : <Play size={13} />} Load Model
            </button>
          </div>
        </div>
      )}

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <AlertTriangle size={14} /> {model.error}
        </div>
      )}

      {!model.result && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Run inference to inspect internals</div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, background: colors.canvas, color: colors.ink }}
          />
          <div>
            <button onClick={handleRun} disabled={!model.loaded || model.running} style={{ ...btn, opacity: !model.loaded || model.running ? 0.5 : 1 }}>
              {model.running ? <Loader2 size={13} className="spin" /> : <Play size={13} />} Run
            </button>
          </div>
        </div>
      )}

      {model.result && layer && (
        <>
          {/* Layer / head selection */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: colors.bodyMuted }}>
              <Flame size={13} color={colors.primary} /> Layer
              <select
                value={layerNum < layers.length ? layerNum : 0}
                onChange={(e) => selectHead(Number(e.target.value), headNum)}
                style={selectStyle}
              >
                {layers.map((l) => (
                  <option key={l.index} value={l.index}>L{l.index}</option>
                ))}
              </select>
            </label>
            <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: colors.bodyMuted }}>
              Head
              <select value={headNum < (layer.heads.length ?? 1) ? headNum : 0} onChange={(e) => selectHead(layerNum, Number(e.target.value))} style={selectStyle}>
                {layer.heads.map((h) => (
                  <option key={h.index} value={h.index}>H{h.index}</option>
                ))}
              </select>
            </label>
            <span style={{ fontSize: 11, color: colors.bodyMuted }}>L{layer.index}·H{head?.index}</span>
            
            <button
              onClick={() => setComparisonMode(!comparisonMode)}
              style={{
                ...btn,
                background: comparisonMode ? colors.primary : colors.surfacePearl,
                color: comparisonMode ? colors.onPrimary : colors.ink,
                border: `1px solid ${comparisonMode ? colors.primary : colors.hairline}`,
              }}
            >
              <Columns size={13} />
              Compare
            </button>
            
            <button onClick={exportHeatmap} style={{ ...btn, background: colors.surfacePearl, color: colors.ink, border: `1px solid ${colors.hairline}` }}>
              <Download size={13} />
              Export
            </button>
          </div>

          {/* Attention */}
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Layers size={14} color={colors.primary} /> BertViz Attention · L{layer.index} H{head?.index}
            </div>
            <div ref={heatmapRef}>
              <AttentionHeatmap
                matrix={head?.attentionMatrix ?? []}
                tokens={tokens}
                hoveredToken={hovered}
                onHoverToken={setHovered}
                allLayers={model.result?.layers ?? []}
                selectedLayer={layer.index}
                selectedHead={head?.index ?? 0}
                onSelectLayerHead={selectHead}
              />
            </div>
          </div>

          {/* Comparison Mode */}
          {comparisonMode && (
            <div style={card}>
              <div style={{ fontWeight: 700, color: colors.ink }}>Comparison</div>
              <textarea
                value={secondPrompt}
                onChange={(e) => setSecondPrompt(e.target.value)}
                rows={2}
                placeholder="Enter second prompt for comparison..."
                style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, background: colors.canvas, color: colors.ink }}
              />
              <button
                onClick={handleRunComparison}
                disabled={loadingSecond}
                style={{ ...btn, opacity: loadingSecond ? 0.5 : 1 }}
              >
                {loadingSecond ? <Loader2 size={13} className="spin" /> : <Play size={13} />}
                Run Comparison
              </button>
              
              {secondResult && (
                <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 4 }}>Prompt 1</div>
                    <div style={{ fontSize: 12, color: colors.ink }}>{prompt}</div>
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 4 }}>Prompt 2</div>
                    <div style={{ fontSize: 12, color: colors.ink }}>{secondPrompt}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Token Selection */}
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Sequence</span>
              {selectedTokens.size > 0 && (
                <span style={{ fontSize: 11, color: colors.primary }}>{selectedTokens.size} selected</span>
              )}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {tokens.map((t, i) => (
                <div
                  key={i}
                  onClick={() => toggleTokenSelection(i)}
                  style={{
                    padding: '3px 6px',
                    borderRadius: 4,
                    fontSize: 10,
                    background: selectedTokens.has(i) ? colors.primary : colors.canvas,
                    color: selectedTokens.has(i) ? colors.onPrimary : colors.ink,
                    border: `1px solid ${selectedTokens.has(i) ? colors.primary : colors.hairline}`,
                    cursor: 'pointer',
                    transition: 'all 0.1s ease',
                  }}
                >
                  {t}
                </div>
              ))}
            </div>
          </div>

          {/* Activations */}
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink }}>MLP Activations · L{layer.index}</div>
            <ActivationHeatmap
              activations={activations}
              neuronIndex={sel.neuron}
              onSelectNeuron={(i) => {
                const neuron = layerNeurons[i];
                if (neuron) sel.setSelectedNeuron(layer.index, neuron.index);
              }}
              tokens={tokens}
              neuronTokenActivations={neuronTokenActivations}
            />
          </div>

          {/* Token strip */}
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink }}>Sequence</div>
            <TokenViewer
              tokens={tokens}
              tokenIds={model.result?.tokens.map((t) => t.id) ?? []}
              selectedToken={hovered}
              onHoverToken={setHovered}
            />
            <div style={{ fontSize: 11, color: colors.bodyMuted }}>
              GPU {model.result?.gpuUtil}% · Mem {model.result?.memoryUtil}%
            </div>
          </div>
        </>
      )}
    </div>
  );
};