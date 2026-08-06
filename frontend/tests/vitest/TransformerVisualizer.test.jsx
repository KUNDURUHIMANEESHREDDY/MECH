import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/services/api', () => ({
  api: {
    gpt2Architecture: vi.fn(),
    gpt2AttentionHead: vi.fn(),
    gpt2Neuron: vi.fn(),
    gpt2Layer: vi.fn(),
    gpt2RunPrompt: vi.fn(),
    gpt2Load: vi.fn(),
  },
}));

const mockArch = {
  status: 'ok',
  model_name: 'gpt2',
  n_heads: 12,
  d_model: 768,
  d_mlp: 3072,
  d_head: 64,
  n_params: 124439808,
  vocab_size: 50257,
  layers: [
    {
      label: 'block_0',
      n_params: 70411,
      components: [
        { id: 'ln_1', type: 'layernorm', dim: 768 },
        { id: 'attn', type: 'attention', n_heads: 12, d_head: 64 },
        { id: 'ln_2', type: 'layernorm', dim: 768 },
        { id: 'mlp', type: 'mlp', num_neurons: 3072 },
      ],
    },
    {
      label: 'block_1',
      n_params: 70411,
      components: [
        { id: 'ln_1', type: 'layernorm', dim: 768 },
        { id: 'attn', type: 'attention', n_heads: 12, d_head: 64 },
        { id: 'ln_2', type: 'layernorm', dim: 768 },
        { id: 'mlp', type: 'mlp', num_neurons: 3072 },
      ],
    },
  ],
  modules: [
    { id: 'wte', n_params: 39321600, shape: [50257, 768] },
    { id: 'wpe', n_params: 589824, shape: [1024, 768] },
    { id: 'ln_f', n_params: 1536, dim: 768 },
    { id: 'lm_head', n_params: 39321600, shape: [768, 50257] },
  ],
  tokens: ['The', ' capital', ' of', ' France', ' is'],
};

const mockAttentionHeadDetail = {
  status: 'ok',
  layer: 0,
  head: 0,
  str_tokens: ['The', ' capital', ' of', ' France', ' is'],
  matrix: [
    [0.1, 0.2, 0.3, 0.2, 0.2],
    [0.2, 0.4, 0.1, 0.1, 0.2],
    [0.3, 0.1, 0.3, 0.1, 0.2],
    [0.1, 0.1, 0.1, 0.6, 0.1],
    [0.15, 0.15, 0.15, 0.15, 0.4],
  ],
};

describe('TransformerVisualizer (left-to-right card layout)', () => {
  let api;
  beforeEach(async () => {
    vi.clearAllMocks();
    ({ api } = await import('@/services/api'));
    api.gpt2Architecture.mockResolvedValue(mockArch);
  });

  it('renders the architecture header and block cards after loading', async () => {
    const { default: TransformerVisualizer } = await import('../../src/components/TransformerVisualizer');

    render(<TransformerVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('GPT-2 Transformer Architecture')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('gpt2 · 2 layers · 12 heads')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('WTE')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('LM Head')).toBeInTheDocument();
    });

    expect(screen.getByText('Block 0')).toBeInTheDocument();
    expect(screen.getByText('Block 1')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Layers: 2')).toBeInTheDocument();
    });
    expect(screen.getByText('Parameters: 124.4M')).toBeInTheDocument();
  });

  it('shows the initial placeholder when no node is selected', async () => {
    const { default: TransformerVisualizer } = await import('../../src/components/TransformerVisualizer');

    render(<TransformerVisualizer />);

    await waitFor(() => {
      expect(screen.getByText(/Click an attention head, MLP neuron/)).toBeInTheDocument();
    });
  });

  it('renders 24 attention head squares (12 per block × 2 blocks)', async () => {
    const { default: TransformerVisualizer } = await import('../../src/components/TransformerVisualizer');

    render(<TransformerVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('Block 0')).toBeInTheDocument();
    });

    const allHeads = await screen.findAllByTitle(/Head \d/);
    expect(allHeads).toHaveLength(24);
  });

  it('clicking a head div triggers detail panel with attention pattern', async () => {
    const { default: TransformerVisualizer } = await import('../../src/components/TransformerVisualizer');
    api.gpt2AttentionHead.mockResolvedValue(mockAttentionHeadDetail);

    render(<TransformerVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('Block 0')).toBeInTheDocument();
    });

    const headDivs = await screen.findAllByTitle('Head 3', {}, { timeout: 3000 });
    fireEvent.click(headDivs[0]);

    await waitFor(() => {
      expect(screen.getByText('Attention Pattern')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(api.gpt2AttentionHead).toHaveBeenCalledWith(0, 3);
    });
  });

  it('clicking an MLP neuron div triggers detail panel', async () => {
    const { default: TransformerVisualizer } = await import('../../src/components/TransformerVisualizer');
    api.gpt2Neuron.mockResolvedValue({
      status: 'ok',
      id: 'L0N42',
      path: 'blocks.0.mlp',
      component: 'mlp',
      bias: 0.05,
      in_weight_l2: 0.123,
      out_weight_l2: 0.456,
    });

    render(<TransformerVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('Block 0')).toBeInTheDocument();
    });

    const mlpDivs = await screen.findAllByTitle('N3', {}, { timeout: 3000 });
    fireEvent.click(mlpDivs[0]);

    await waitFor(() => {
      expect(screen.getByText('L0N42')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(api.gpt2Neuron).toHaveBeenCalledWith(0, 3, 'mlp', 10);
    });
  });

  it('shows inline error when backend is unreachable', async () => {
    const { default: TransformerVisualizer } = await import('../../src/components/TransformerVisualizer');
    api.gpt2Architecture.mockRejectedValue(new Error('Network error 500'));

    render(<TransformerVisualizer />);

    await waitFor(() => {
      expect(screen.getByText(/Cannot reach backend/)).toBeInTheDocument();
    });
  });
});
