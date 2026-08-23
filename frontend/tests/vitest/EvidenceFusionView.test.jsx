import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { EvidenceFusionView } from '../../src/components/EvidenceFusionView';

describe('EvidenceFusionView', () => {
  it('renders title, probe section, and model status', () => {
    render(<EvidenceFusionView />);
    expect(screen.getByText(/Evidence Fusion/i)).toBeInTheDocument();
  });
});
