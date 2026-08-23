import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ModelExplorerPanel } from '../../src/components/panels/ModelExplorerPanel';
import { ModelsCatalogPanel } from '../../src/components/panels/ModelsCatalogPanel';

describe('Visualization', () => {
  it('renders ModelExplorerPanel with architecture views', () => {
    render(<ModelExplorerPanel />);
    expect(screen.getByText(/Load Model/i)).toBeInTheDocument();
  });

  it('renders ModelsCatalogPanel with model list', () => {
    render(<ModelsCatalogPanel />);
    expect(screen.getByText(/Model catalog/i)).toBeInTheDocument();
  });
});
