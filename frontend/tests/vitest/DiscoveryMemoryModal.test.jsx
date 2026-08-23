import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DiscoveryMemoryModal } from '../../src/components/DiscoveryMemoryModal';

describe('DiscoveryMemoryModal', () => {
  it('renders discovery cards and title', () => {
    render(<DiscoveryMemoryModal isOpen={true} />);
    expect(screen.getByText('Scientific Discovery Memory')).toBeInTheDocument();
    expect(screen.getByText('Induction Heads Circuit in Layer 5 & 6')).toBeInTheDocument();
    expect(screen.getByText('Indirect Object Identification (IOI) Name Mover Circuit')).toBeInTheDocument();
  });

  it('filters discoveries by domain and search term', () => {
    render(<DiscoveryMemoryModal isOpen={true} />);
    const domainSelect = screen.getByTestId('discovery-domain-select');
    fireEvent.change(domainSelect, { target: { value: 'SAE' } });
    expect(screen.getByText(/Sparse Autoencoder Monosemantic Feature 402/i)).toBeInTheDocument();
    expect(screen.queryByText(/Induction Heads Circuit/i)).not.toBeInTheDocument();

    const searchInput = screen.getByTestId('discovery-search-input');
    fireEvent.change(searchInput, { target: { value: 'non-existent query' } });
    expect(screen.getByText(/No discoveries matching query/i)).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(<DiscoveryMemoryModal isOpen={true} onClose={onClose} />);
    const closeBtn = screen.getByTestId('modal-close-btn');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });
});
