import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import DiscoveryMemoryModal from '../../src/components/DiscoveryMemoryModal.jsx';

describe('DiscoveryMemoryModal Component', () => {
  it('renders modal header when isOpen is true', () => {
    render(<DiscoveryMemoryModal isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByText(/Discovery Memory Search/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Type research question/i)).toBeInTheDocument();
  });

  it('does not render when isOpen is false', () => {
    render(<DiscoveryMemoryModal isOpen={false} onClose={vi.fn()} />);
    expect(screen.queryByText(/Discovery Memory Search/i)).not.toBeInTheDocument();
  });

  it('filters multi-modal memory results by category', () => {
    render(<DiscoveryMemoryModal isOpen={true} onClose={vi.fn()} />);
    const claimsCategoryBtn = screen.getByText(/^claims$/i);
    fireEvent.click(claimsCategoryBtn);
    expect(screen.getByText(/Induction Head Sequence Repeater/i)).toBeInTheDocument();
  });

  it('updates results when user types a new search query', () => {
    render(<DiscoveryMemoryModal isOpen={true} onClose={vi.fn()} />);
    const input = screen.getByPlaceholderText(/Type research question/i);
    fireEvent.change(input, { target: { value: 'IOI Name Mover' } });
    expect(screen.getByText(/IOI Name Mover Circuit/i)).toBeInTheDocument();
  });
});
