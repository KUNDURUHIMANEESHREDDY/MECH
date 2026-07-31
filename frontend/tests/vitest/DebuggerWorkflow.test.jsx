import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import InferenceTimeline from '../../src/components/panels/InferenceTimeline.jsx';
import { selectionManager } from '../../src/utils/selectionManager.js';

describe('Debugger Integration Workflow', () => {
  it('executes full debugger flow: Step -> Breakpoint -> Patch -> Continue', async () => {
    render(<InferenceTimeline />);

    // 1. Initial State
    expect(screen.getByTestId('inference-timeline')).toBeInTheDocument();

    // 2. Step forward
    const stepBtn = screen.getByText(/Step Layer/i);
    fireEvent.click(stepBtn);
    expect(selectionManager.getSelection().layer).toBe(1);

    // 3. Apply Patch
    const patchBtn = screen.getByText(/\+ Apply Patch at L1/i);
    fireEvent.click(patchBtn);

    // 4. Continue to Breakpoint
    const continueBtn = screen.getByText(/Continue to Breakpoint/i);
    fireEvent.click(continueBtn);
    expect(selectionManager.getSelection().layer).toBe(5);
  });
});
