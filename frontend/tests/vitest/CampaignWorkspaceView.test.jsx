import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { CampaignWorkspaceView } from '../../src/components/CampaignWorkspaceView';

describe('CampaignWorkspaceView', () => {
  it('renders campaign workspace title and controls', () => {
    render(<CampaignWorkspaceView />);
    expect(screen.getAllByText(/Campaigns/i)[0]).toBeInTheDocument();
  });
});
