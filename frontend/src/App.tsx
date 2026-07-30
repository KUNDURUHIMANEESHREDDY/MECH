import React, { useEffect, useState } from 'react';
import { useModel } from './hooks/useModel';
import { AttentionHeatmap } from './components/visualizations/panels/AttentionHeatmap';
import { ActivationHeatmap } from './components/visualizations/panels/ActivationHeatmap';
import { TokenViewer } from './components/visualizations/panels/TokenViewer';
import { LayerSidebar } from './components/LayerSidebar';
import { NeuronPanel } from './components/NeuronPanel';
import { StatusBar } from './components/StatusBar';
import { ErrorUI } from './components/ErrorUI';
import { CommandPalette } from './components/CommandPalette';
import { DockManager } from './layout/DockManager';
import { TokenPanel } from './panels/TokenPanel';
import { LayerPanel } from './panels/LayerPanel';
import { PredictionPanel } from './panels/PredictionPanel';
import { useAppStore } from './store/useAppStore';
import { PanelState } from './types';

export default function App() {
  const { state: model, listModels, load, infer } = useModel();
  const [appState, setAppState] = useAppStore();

  const [panel, setPanel] = useState<PanelState>({
    selectedLayer: 0, selectedHead: 0, selectedNeuron: null,
    hoveredToken: null, error: null, darkMode: appState.darkMode,
  });

  useEffect(() => { listModels(); }, []);

  const setError = (e: string | null) => setPanel(s => ({ ...s, error: e }));
  const toggleDarkMode = () => {
    const nextDark = !appState.darkMode;
    setAppState({ darkMode: nextDark });
    setPanel(s => ({ ...s, darkMode: nextDark }));
  };

  const data = model.result;
  const layer = data?.layers[panel.selectedLayer];
  const head = layer?.heads[panel.selectedHead];

  const bg = appState.darkMode ? '#0d0d14' : '#f0f0f5';
  const fg = appState.darkMode ? '#e0e0e0' : '#222';
  const panelBg = appState.darkMode ? '#16161e' : '#ffffff';
  const borderColor = appState.darkMode ? '#2a2a3a' : '#d0d0d0';
  const headerFg = appState.darkMode ? '#fff' : '#111';
  const inputBg = appState.darkMode ? '#1a1a2a' : '#eee';

  const [prompt, setPrompt] = useState('Hello world');

  const handleRun = () => { if (prompt.trim()) infer(prompt); };
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRun(); }
  };

  const handleLoadModel = (name: string) => {
    load(name);
    setPanel(s => ({ ...s, selectedLayer: 0, selectedHead: 0, selectedNeuron: null }));
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: bg, color: fg, fontFamily: 'system-ui, sans-serif', transition: 'all 0.2s' }}>
      <CommandPalette onLoadModel={handleLoadModel} onRunPrompt={handleRun} darkMode={appState.darkMode} />

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 16px', borderBottom: `1px solid ${borderColor}`, background: panelBg }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 18, fontWeight: 700, color: headerFg }}>Model Explorer</span>
          {model.modelInfo && (
            <span style={{ background: '#3b82f6', color: '#fff', fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 10 }}>
              {model.modelInfo.model_name}
            </span>
          )}
          <span style={{ fontSize: 11, color: '#888', background: appState.darkMode ? '#2a2a3a' : '#e0e0e0', padding: '2px 6px', borderRadius: 4 }}>
            Press Ctrl+Shift+P for Command Palette
          </span>
        </div>
        <button
          onClick={toggleDarkMode}
          style={{
            background: appState.darkMode ? '#2a2a3a' : '#ddd', color: appState.darkMode ? '#ffd700' : '#555',
            border: 'none', borderRadius: 6, padding: '6px 12px', cursor: 'pointer', fontSize: 12, fontWeight: 600,
          }}
        >
          {appState.darkMode ? '☀ Light' : '🌙 Dark'}
        </button>
      </div>

      {/* Model selection (shown before load) */}
      {!model.loaded && !model.loading && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 16 }}>
          <div style={{ color: '#888', fontSize: 13 }}>Choose a model to load</div>
          {model.availableModels.length === 0 && model.error
            ? <div style={{ color: '#ff6666', fontSize: 12 }}>Cannot reach runtime at localhost:8000. Start the backend first.</div>
            : <div style={{ display: 'flex', gap: 8 }}>
                {model.availableModels.map(name => (
                  <button key={name} onClick={() => handleLoadModel(name)}
                    style={{
                      background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 8,
                      padding: '12px 24px', cursor: 'pointer', fontSize: 14, fontWeight: 600,
                      textTransform: 'capitalize',
                    }}>
                    Load {name}
                  </button>
                ))}
              </div>
          }
        </div>
      )}

      {/* Loading indicator */}
      {model.loading && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 12 }}>
          <div style={{ color: '#888', fontSize: 13 }}>Loading {model.modelInfo?.model_name ?? 'model'}...</div>
          <div style={{ width: 200, height: 4, background: '#2a2a3a', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ width: '60%', height: '100%', background: '#3b82f6', borderRadius: 2 }} />
          </div>
        </div>
      )}

      {/* Inference UI (shown after load) */}
      {model.loaded && (
        <>
          {/* Prompt input */}
          <div style={{ display: 'flex', gap: 8, padding: '8px 16px', borderBottom: `1px solid ${borderColor}`, background: panelBg, alignItems: 'center' }}>
            <input value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={handleKeyDown}
              placeholder="Enter prompt..."
              style={{
                flex: 1, padding: '8px 12px', borderRadius: 6, border: `1px solid ${borderColor}`,
                background: inputBg, color: fg, fontSize: 13, outline: 'none',
              }} />
            <button onClick={handleRun} disabled={model.running}
              style={{
                background: model.running ? '#555' : '#3b82f6', color: '#fff', border: 'none', borderRadius: 6,
                padding: '8px 16px', cursor: model.running ? 'not-allowed' : 'pointer', fontSize: 12, fontWeight: 600,
              }}>
              {model.running ? 'Running...' : 'Run'}
            </button>
          </div>

          {/* Main layout */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            {!data ? (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontSize: 14 }}>
                {model.running ? 'Running inference...' : 'Enter a prompt and click Run'}
              </div>
            ) : (
              <>
                <div style={{ width: 160, borderRight: `1px solid ${borderColor}`, padding: 12, overflow: 'auto', background: panelBg, flexShrink: 0 }}>
                  <LayerSidebar layers={data.layers}
                    selectedLayer={panel.selectedLayer} selectedHead={panel.selectedHead}
                    onSelectLayer={l => setPanel(s => ({ ...s, selectedLayer: l, selectedHead: 0, selectedNeuron: null }))}
                    onSelectHead={h => setPanel(s => ({ ...s, selectedHead: h, selectedNeuron: null }))} />
                </div>

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <DockManager darkMode={appState.darkMode}>
                    {{
                      token_viewer: (
                        <TokenViewer tokens={data.tokens.map(t => t.text)} tokenIds={data.tokens.map(t => t.id)}
                          selectedToken={panel.hoveredToken}
                          onHoverToken={t => setPanel(s => ({ ...s, hoveredToken: t }))} />
                      ),
                      attention_heatmap: head ? (
                        <AttentionHeatmap matrix={head.attentionMatrix} tokens={data.tokens.map(t => t.text)}
                          hoveredToken={panel.hoveredToken}
                          onHoverToken={t => setPanel(s => ({ ...s, hoveredToken: t }))} />
                      ) : <div style={{ color: '#666' }}>Select a layer and head</div>,
                      activation_heatmap: head ? (
                        <ActivationHeatmap activations={head.neurons.map(n => n.activation)}
                          neuronIndex={panel.selectedNeuron}
                          onSelectNeuron={n => setPanel(s => ({ ...s, selectedNeuron: n }))} />
                      ) : <div style={{ color: '#666' }}>Select a layer and head</div>,
                      neuron_panel: (
                        <NeuronPanel neurons={head?.neurons ?? []} selectedNeuron={panel.selectedNeuron}
                          onSelectNeuron={n => setPanel(s => ({ ...s, selectedNeuron: n }))} />
                      ),
                      token_inspector: (
                        <TokenPanel tokens={data.tokens.map(t => t.text)}
                          selectedTokenIdx={appState.selection.selectedTokenIdx}
                          onSelectToken={idx => setAppState(prev => ({ selection: { ...prev.selection, selectedTokenIdx: idx } }))}
                          darkMode={appState.darkMode} />
                      ),
                      layer_inspector: (
                        <LayerPanel layerIdx={panel.selectedLayer} numHeads={data.layers[panel.selectedLayer]?.heads.length ?? 12} darkMode={appState.darkMode} />
                      ),
                      prediction_inspector: (
                        <PredictionPanel tokens={data.tokens.map(t => t.text)} darkMode={appState.darkMode} />
                      ),
                    }}
                  </DockManager>
                </div>
              </>
            )}
          </div>
        </>
      )}

      <ErrorUI message={panel.error || model.error} onDismiss={() => { setError(null); }} />
      <StatusBar modelName={model.modelInfo?.model_name ?? 'GPT2'}
        gpuUtil={data?.gpuUtil ?? 0} memoryUtil={data?.memoryUtil ?? 0}
        tokenCount={data?.tokens.length ?? 0} darkMode={appState.darkMode} />
    </div>
  );
}
