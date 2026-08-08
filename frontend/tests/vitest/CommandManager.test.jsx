import { describe, it, expect, vi } from 'vitest';
import { commandManager, Commands } from '../../src/shared/managers/commandManager';

describe('CommandManager & SelectionStore', () => {
  it('publishes neuron.selected and notifies multiple subscribers', () => {
    commandManager.clear();

    const sub1 = vi.fn();
    const sub2 = vi.fn();
    const sub3 = vi.fn();

    commandManager.subscribe('neuron.selected', sub1);
    commandManager.subscribe('neuron.selected', sub2);
    commandManager.subscribe('neuron.selected', sub3);

    Commands.neuronSelected(5, 0, 128);

    expect(sub1).toHaveBeenCalledTimes(1);
    expect(sub2).toHaveBeenCalledTimes(1);
    expect(sub3).toHaveBeenCalledTimes(1);

    expect(sub1).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'neuron.selected',
        payload: { layer: 5, head: 0, neuron: 128 },
      })
    );
  });
});
