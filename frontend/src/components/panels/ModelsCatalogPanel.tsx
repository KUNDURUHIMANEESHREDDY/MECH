import React, { useEffect } from 'react';
import { Cpu, Loader2, CheckCircle2, Layers } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import type { FC, PanelContext } from '../../shared/types';

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

/** Models Catalog: browse available models from the sidecar and load one. */
export const ModelsCatalogPanel: FC<PanelContext> = () => {
  const { state: model, listModels, load } = useModel();

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const models = model.availableModels.length ? model.availableModels : ['gpt2', 'gpt2-medium', 'gpt2-large'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Layers size={14} color={colors.primary} /> Model catalog
        </div>
        <div style={{ fontSize: 12, color: colors.bodyMuted }}>
          {model.loaded
            ? `Loaded: ${model.modelInfo?.model_name} (${model.modelInfo?.num_layers} layers · ${model.modelInfo?.num_heads} heads · d_model ${model.modelInfo?.hidden_dim})`
            : 'No model loaded yet. Pick one below.'}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {models.map((m) => {
          const isCurrent = model.loaded && model.modelInfo?.model_name === m;
          return (
            <div key={m} style={{ ...card, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Cpu size={14} color={isCurrent ? colors.success : colors.bodyMuted} />
                <span style={{ fontWeight: 600, color: colors.ink, fontFamily: 'monospace' }}>{m}</span>
                {isCurrent && (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: colors.successText, fontSize: 11 }}>
                    <CheckCircle2 size={12} /> active
                  </span>
                )}
              </div>
              <button
                onClick={() => load(m)}
                disabled={model.loading || isCurrent}
                style={{ ...btn, opacity: model.loading || isCurrent ? 0.5 : 1, background: isCurrent ? colors.success : colors.primary }}
              >
                {model.loading ? <Loader2 size={13} /> : null}
                {isCurrent ? 'Loaded' : 'Load'}
              </button>
            </div>
          );
        })}
      </div>

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', fontSize: 12 }}>
          {model.error}
        </div>
      )}

      {model.modelInfo?.status === 'demo' && (
        <div style={{ background: colors.warningSoft, color: colors.warningText, border: `1px solid ${colors.warningBorder}`, borderRadius: 8, padding: '8px 12px', fontSize: 12 }}>
          Demo dataset active — the Python sidecar is unreachable, so visualization data is simulated.
        </div>
      )}
    </div>
  );
};