import React from 'react';
import {
  FileCode2,
  X,
  Layers,
  Cpu,
  Database,
  Sliders,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface MethodologyData {
  title: string;
  model: string;
  modelVersion: string;
  tokenizer: string;
  dataset: string;
  datasetVersion: string;
  promptConstruction: string;
  interventionMethod: string;
  controlMethod: string;
  controlComponent: string;
  controlRationale: string;
  primaryMetric: string;
  aggregation: string;
  sampleSize: number;
  seed: number;
  normalization: string;
  computationalDevice: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  methodology?: MethodologyData;
}

export const MethodologyDrawer: React.FC<Props> = ({
  isOpen,
  onClose,
  methodology = {
    title: 'IOI Head Ablation Experiment Protocol',
    model: 'gpt2',
    modelVersion: 'openai-community/gpt2 (124M params, 12 layers, 12 heads)',
    tokenizer: 'GPT2TokenizerFast (BPE, vocab size: 50,257)',
    dataset: 'IOI Clean vs Corrupted Name-Swap Prompts',
    datasetVersion: '1.0 (Frozen JSONL)',
    promptConstruction: 'ABBA / BABA templates with flipped indirect-object targets',
    interventionMethod: 'Forward-hook Zero Ablation on Head Output Projection',
    controlMethod: 'Contrastive Negative Control Head Ablation',
    controlComponent: 'L0H0 (Layer 0 Early Head)',
    controlRationale: 'Isolates target-specific causal logit shift from general sentence degradation',
    primaryMetric: 'Target Indirect Object Logit Difference (Δlogit)',
    aggregation: 'Sample Mean across template instances',
    sampleSize: 100,
    seed: 42,
    normalization: 'Raw logit units and normalized probability difference',
    computationalDevice: 'PyTorch CUDA / CPU Fallback (Deterministic Seed 42)',
  },
}) => {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        bottom: 0,
        width: 440,
        backgroundColor: colors.canvas,
        borderLeft: `1px solid ${colors.hairline}`,
        boxShadow: '-8px 0 24px rgba(0,0,0,0.3)',
        zIndex: 1500,
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
      }}
    >
      {/* Drawer Header */}
      <div
        style={{
          padding: '16px 20px',
          borderBottom: `1px solid ${colors.hairline}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: colors.surfaceTile1,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <FileCode2 size={18} color={colors.primary} />
          <div>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: colors.ink }}>
              Experimental Methodology
            </h3>
            <div style={{ fontSize: 11, color: colors.bodyMuted }}>{methodology.title}</div>
          </div>
        </div>
        <button onClick={onClose} style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}>
          <X size={18} color={colors.bodyMuted} />
        </button>
      </div>

      {/* Drawer Body */}
      <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Model & Architecture */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Cpu size={14} color={colors.primary} /> Model & Tokenizer
          </div>
          <div style={{ backgroundColor: colors.surfaceTile1, padding: 10, borderRadius: 6, fontSize: 11, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div><strong style={{ color: colors.ink }}>Model:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.modelVersion}</span></div>
            <div><strong style={{ color: colors.ink }}>Tokenizer:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.tokenizer}</span></div>
            <div><strong style={{ color: colors.ink }}>Execution Runtime:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.computationalDevice}</span></div>
          </div>
        </div>

        {/* Dataset & Prompts */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Database size={14} color={colors.primary} /> Dataset & Prompt Construction
          </div>
          <div style={{ backgroundColor: colors.surfaceTile1, padding: 10, borderRadius: 6, fontSize: 11, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div><strong style={{ color: colors.ink }}>Dataset:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.dataset}</span></div>
            <div><strong style={{ color: colors.ink }}>Prompt Template:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.promptConstruction}</span></div>
            <div><strong style={{ color: colors.ink }}>Sample Size (N):</strong> <span style={{ color: colors.bodyMuted }}>{methodology.sampleSize} independent prompt pairs</span></div>
          </div>
        </div>

        {/* Intervention & Control */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Sliders size={14} color={colors.primary} /> Intervention & Negative Control
          </div>
          <div style={{ backgroundColor: colors.surfaceTile1, padding: 10, borderRadius: 6, fontSize: 11, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div><strong style={{ color: colors.ink }}>Intervention:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.interventionMethod}</span></div>
            <div><strong style={{ color: colors.ink }}>Negative Control:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.controlComponent}</span></div>
            <div><strong style={{ color: colors.ink }}>Control Rationale:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.controlRationale}</span></div>
          </div>
        </div>

        {/* Statistical Estimation */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <CheckCircle2 size={14} color={colors.successText} /> Metrics & Estimation
          </div>
          <div style={{ backgroundColor: colors.surfaceTile1, padding: 10, borderRadius: 6, fontSize: 11, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div><strong style={{ color: colors.ink }}>Primary Metric:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.primaryMetric}</span></div>
            <div><strong style={{ color: colors.ink }}>Aggregation:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.aggregation}</span></div>
            <div><strong style={{ color: colors.ink }}>Random Seed:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.seed}</span></div>
            <div><strong style={{ color: colors.ink }}>Normalization:</strong> <span style={{ color: colors.bodyMuted }}>{methodology.normalization}</span></div>
          </div>
        </div>
      </div>
    </div>
  );
};
