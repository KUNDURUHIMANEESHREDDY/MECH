import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { AttentionHeatmap } from '../../src/components/visualizations/panels/AttentionHeatmap';
import { attentionMapsToLayers } from '../../src/services/attentionService';

describe('BertViz Attention Visualizer & Data Pipeline', () => {
  const sampleTokens = ['The', ' capital', ' of', ' France', ' is'];
  // Real 5x5 normalized attention matrix
  const liveAttentionMatrix = [
    [0.80, 0.05, 0.05, 0.05, 0.05],
    [0.10, 0.70, 0.10, 0.05, 0.05],
    [0.05, 0.15, 0.60, 0.10, 0.10],
    [0.10, 0.10, 0.20, 0.50, 0.10],
    [0.05, 0.10, 0.15, 0.40, 0.30],
  ];

  it('correctly maps live attention tensors in attentionMapsToLayers without synthetic fallback', () => {
    const maps = [
      { layer: 0, head: 0, matrix: liveAttentionMatrix },
      { layer: 0, head: 1, matrix: [] },
    ];
    const layers = attentionMapsToLayers(maps, 1, 2, sampleTokens);
    expect(layers.length).toBe(1);
    expect(layers[0].heads.length).toBe(2);
    // Head 0 has exact live matrix
    expect(layers[0].heads[0].attentionMatrix).toEqual(liveAttentionMatrix);
    // Head 1 with no data has empty matrix, NOT a synthetic fabricated matrix
    expect(layers[0].heads[1].attentionMatrix).toEqual([]);
  });

  it('renders BertViz Head View when live attention tensors are provided', () => {
    render(
      <AttentionHeatmap
        matrix={liveAttentionMatrix}
        tokens={sampleTokens}
        selectedLayer={0}
        selectedHead={0}
      />
    );

    expect(screen.getByText(/BertViz Head View/i)).toBeInTheDocument();
    expect(screen.getByText(/Query \(Source\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Key \(Target\)/i)).toBeInTheDocument();
    // Tokens are rendered in query and key columns
    expect(screen.getAllByText('France').length).toBeGreaterThan(0);
  });

  it('enforces fail-closed state when attention matrix is empty or uninitialized', () => {
    render(
      <AttentionHeatmap
        matrix={[]}
        tokens={sampleTokens}
        selectedLayer={0}
        selectedHead={0}
      />
    );

    expect(screen.getByText(/EXECUTION REQUIRED: No Live Attention Tensors/i)).toBeInTheDocument();
    expect(screen.queryByText(/Query \(Source\)/i)).not.toBeInTheDocument();
  });

  it('switches between Head View, Model View, and Heatmap Matrix', () => {
    render(
      <AttentionHeatmap
        matrix={liveAttentionMatrix}
        tokens={sampleTokens}
        selectedLayer={0}
        selectedHead={0}
      />
    );

    // Switch to Model View
    const modelBtn = screen.getByRole('button', { name: /Model View/i });
    fireEvent.click(modelBtn);
    expect(screen.getByText(/Model View \(12 Layers × 12 Heads Matrix Grid\)/i)).toBeInTheDocument();

    // Switch to Heatmap Matrix
    const heatmapBtn = screen.getByRole('button', { name: /Heatmap Matrix/i });
    fireEvent.click(heatmapBtn);
    expect(screen.getByRole('button', { name: /Heatmap Matrix/i })).toBeInTheDocument();
  });

  it('rejects stale attention cache when new prompt tokens and matrices are passed', () => {
    const promptATokens = ['France', 'Paris'];
    const promptAMatrix = [[0.9, 0.1], [0.2, 0.8]];

    const { rerender } = render(
      <AttentionHeatmap
        matrix={promptAMatrix}
        tokens={promptATokens}
        selectedLayer={0}
        selectedHead={0}
      />
    );

    expect(screen.getAllByText('Paris').length).toBeGreaterThan(0);

    const promptBTokens = ['Germany', 'Berlin'];
    const promptBMatrix = [[0.85, 0.15], [0.10, 0.90]];

    rerender(
      <AttentionHeatmap
        matrix={promptBMatrix}
        tokens={promptBTokens}
        selectedLayer={0}
        selectedHead={0}
      />
    );

    // Prompt B tokens must be present, Prompt A tokens must NOT remain
    expect(screen.getAllByText('Berlin').length).toBeGreaterThan(0);
    expect(screen.queryByText('Paris')).not.toBeInTheDocument();
  });
});
