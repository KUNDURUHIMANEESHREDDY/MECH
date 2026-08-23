import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ExtensionMarketplaceModal } from '../../src/components/ExtensionMarketplaceModal';
import { PublicationExportModal } from '../../src/components/PublicationExportModal';
import { WorkspaceSharingModal } from '../../src/components/WorkspaceSharingModal';

describe('Sprint6Deliverable', () => {
  it('renders ExtensionMarketplaceModal with extension items', () => {
    render(<ExtensionMarketplaceModal isOpen={true} />);
    expect(screen.getByText('Scientific Extension Marketplace')).toBeInTheDocument();
    expect(screen.getByText('SAE Visual Feature Inspector')).toBeInTheDocument();
  });

  it('renders PublicationExportModal with conference templates', () => {
    render(<PublicationExportModal isOpen={true} />);
    expect(screen.getByText('Export Scientific Publication')).toBeInTheDocument();
  });

  it('renders WorkspaceSharingModal with collaboration token generator', () => {
    render(<WorkspaceSharingModal isOpen={true} />);
    expect(screen.getByText('Share Research Workspace')).toBeInTheDocument();
    expect(screen.getByTestId('generate-share-link-btn')).toBeInTheDocument();
  });
});
