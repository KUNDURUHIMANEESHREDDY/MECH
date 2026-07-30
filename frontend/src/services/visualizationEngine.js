/**
 * Central Visualization Engine (Sprint 4).
 * Coordinates layouts, camera states, playback, selections, and exports across panels.
 */

import { sharedWorkspaceState } from '../utils/sharedWorkspaceState.js';
import { visualizationEventBus } from '../utils/visualizationEventBus.js';

class VisualizationEngine {
  constructor() {
    this.playbackState = { isPlaying: false, currentFrame: 0, totalFrames: 12, speed: 1.0 };
    this.layoutMode = 'Hierarchical'; // Hierarchical, Force, Radial, Timeline
  }

  setSelection(patch) {
    sharedWorkspaceState.updateState(patch);
    visualizationEventBus.emit('SelectionChanged', patch);
  }

  setLayoutMode(mode) {
    this.layoutMode = mode;
    visualizationEventBus.emit('GraphUpdated', { layoutMode: mode });
  }

  startPlayback() {
    this.playbackState.isPlaying = true;
    visualizationEventBus.emit('PlaybackStarted', this.playbackState);
  }

  pausePlayback() {
    this.playbackState.isPlaying = false;
    visualizationEventBus.emit('PlaybackPaused', this.playbackState);
  }

  exportCurrentView(format = 'SVG') {
    const payload = { format, timestamp: new Date().toISOString() };
    visualizationEventBus.emit('ExportFinished', payload);
    return payload;
  }
}

export const visualizationEngine = new VisualizationEngine();
