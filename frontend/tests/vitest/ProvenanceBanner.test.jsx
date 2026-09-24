import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ProvenanceBanner } from '../../src/components/ProvenanceBanner.tsx';

describe('ProvenanceBanner', () => {
  it('renders nothing for live data', () => {
    const { container } = render(<ProvenanceBanner kind="live" />);
    expect(container.firstChild).toBeNull();
    render(<ProvenanceBanner kind={null} />);
    expect(screen.queryByTestId('provenance-banner')).toBeNull();
  });

  it('warns on seeded data with backend note', () => {
    render(<ProvenanceBanner kind="seeded" note="torch missing" />);
    const el = screen.getByTestId('provenance-banner');
    expect(el).toBeInTheDocument();
    expect(el.getAttribute('data-provenance')).toBe('seeded');
    expect(el.textContent).toMatch(/Seeded demo data/);
    expect(el.textContent).toMatch(/torch missing/);
  });

  it('warns when backend is offline', () => {
    render(<ProvenanceBanner kind="offline" />);
    const el = screen.getByTestId('provenance-banner');
    expect(el.textContent).toMatch(/Backend offline/);
    expect(el.textContent).toMatch(/localhost:8000/);
  });
});
