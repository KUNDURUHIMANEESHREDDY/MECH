// Frontend Config Module - Main Export
// Centralized configuration for MECH Platform Frontend

// API Configuration
export * from './api';

// Feature Flags & Experiments
export * from './features';

// Theme System
export * from './theme';

// Shared Constants
export * from './constants';

// Initialize theme on import (browser only)
import { initTheme } from './theme';

if (typeof window !== 'undefined') {
  initTheme();
}

// Default export with all config namespaces
import * as api from './api';
import * as features from './features';
import * as theme from './theme';
import * as constants from './constants';

export default {
  api,
  features,
  theme,
  constants,
};