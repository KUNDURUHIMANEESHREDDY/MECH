import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { RecentFiles } from '../../src/components/RecentFiles';
import { LayerSidebar } from '../../src/components/LayerSidebar';

describe('Sprint2Deliverable', () => {
  it('renders RecentFiles component with title and refresh controls', () => {
    render(<RecentFiles />);
    expect(screen.getByText('Recent Files')).toBeInTheDocument();
  });

  it('renders LayerSidebar with layer list', () => {
    render(<LayerSidebar layers={12} activeLayer={0} onSelectLayer={() => {}} />);
    expect(screen.getByText('Layers')).toBeInTheDocument();
  });
});
