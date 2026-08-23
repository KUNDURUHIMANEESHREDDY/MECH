// Theme Configuration for MECH Platform Frontend
// Replaces append_themes.py - Centralized design tokens

export interface ColorTokens {
  // Primary colors
  primary: string;
  primaryFocus: string;
  primaryOnDark: string;

  // Ink/Text colors
  ink: string;
  body: string;
  bodyMuted: string;
  inkMuted80: string;
  inkMuted48: string;

  // Divider/Border colors
  dividerSoft: string;
  hairline: string;

  // Canvas/Surface colors
  canvas: string;
  canvasParchment: string;
  surfacePearl: string;

  // On-primary
  onPrimary: string;

  // Legacy aliases
  bg: string;
  bgSidebar: string;
  bgActivity: string;
  bgElev: string;
  bgElev2: string;
  bgHover: string;
  bgActive: string;
  border: string;
  borderLight: string;
  text: string;
  textDim: string;
  textMuted: string;
  accent: string;
  accentHover: string;
  accentSoft: string;
  accentStrong: string;
  danger: string;
  success: string;
  warning: string;
  purple: string;
  pink: string;
}

export interface SpacingTokens {
  radius: string;
  radiusLg: string;
  activityWidth: string;
  sidebarWidth: string;
  topbarHeight: string;
  statusbarHeight: string;
  consoleHeight: string;
}

export interface ShadowTokens {
  sm: string;
  lg: string;
}

export interface FontTokens {
  font: string;
  fontMono: string;
}

export interface GradientTokens {
  gradientAccent: string;
  accentGlow: string;
}

export interface ThemeTokens extends ColorTokens, SpacingTokens, ShadowTokens, FontTokens, GradientTokens {}

// Light theme (default)
export const lightTheme: ThemeTokens = {
  // Primary colors
  primary: '#0066cc',
  primaryFocus: '#0071e3',
  primaryOnDark: '#2997ff',

  // Ink/Text colors
  ink: '#1d1d1f',
  body: '#1d1d1f',
  bodyMuted: '#7a7a7a',
  inkMuted80: '#333333',
  inkMuted48: '#7a7a7a',

  // Divider/Border colors
  dividerSoft: '#f0f0f0',
  hairline: '#e0e0e0',

  // Canvas/Surface colors
  canvas: '#ffffff',
  canvasParchment: '#f5f5f7',
  surfacePearl: '#fafafc',

  // On-primary
  onPrimary: '#ffffff',

  // Legacy aliases
  bg: '#f5f5f7',
  bgSidebar: '#fafafa',
  bgActivity: '#fafafa',
  bgElev: '#ffffff',
  bgElev2: '#f5f5f5',
  bgHover: '#f0f0f0',
  bgActive: '#e5e5e5',
  border: '#e0e0e0',
  borderLight: '#f0f0f0',
  text: '#1d1d1f',
  textDim: '#333333',
  textMuted: '#7a7a7a',
  accent: '#0066cc',
  accentHover: '#0071e3',
  accentSoft: 'rgba(0, 102, 204, 0.08)',
  accentStrong: '#0066cc',
  danger: '#d93025',
  success: '#1e8e3e',
  warning: '#f9ab00',
  purple: '#9334e6',
  pink: '#d91d82',

  // Spacing
  radius: '6px',
  radiusLg: '12px',
  activityWidth: '48px',
  sidebarWidth: '240px',
  topbarHeight: '44px',
  statusbarHeight: '24px',
  consoleHeight: '180px',

  // Shadows
  sm: '0 1px 3px rgba(0, 0, 0, 0.08)',
  lg: '0 4px 16px rgba(0, 0, 0, 0.1)',

  // Fonts
  font: '"Inter", system-ui, -apple-system, sans-serif',
  fontMono: '"SF Mono", "Fira Code", "JetBrains Mono", monospace',

  // Gradients
  gradientAccent: 'none',
  accentGlow: 'none',
};

// Dark theme
export const darkTheme: ThemeTokens = {
  ...lightTheme,
  // Primary colors (adjusted for dark)
  primary: '#2997ff',
  primaryFocus: '#4dafff',
  primaryOnDark: '#66bfff',

  // Ink/Text colors (inverted)
  ink: '#ffffff',
  body: '#ffffff',
  bodyMuted: '#a0a0a0',
  inkMuted80: '#cccccc',
  inkMuted48: '#a0a0a0',

  // Divider/Border colors (darker)
  dividerSoft: '#2a2a2a',
  hairline: '#3a3a3a',

  // Canvas/Surface colors (dark)
  canvas: '#1a1a1a',
  canvasParchment: '#1e1e1e',
  surfacePearl: '#222222',

  // Legacy aliases (dark)
  bg: '#1a1a1a',
  bgSidebar: '#1e1e1e',
  bgActivity: '#1e1e1e',
  bgElev: '#222222',
  bgElev2: '#2a2a2a',
  bgHover: '#2a2a2a',
  bgActive: '#333333',
  border: '#3a3a3a',
  borderLight: '#2a2a2a',
  text: '#ffffff',
  textDim: '#cccccc',
  textMuted: '#a0a0a0',
  accent: '#2997ff',
  accentHover: '#4dafff',
  accentSoft: 'rgba(41, 151, 255, 0.15)',
  accentStrong: '#2997ff',
  danger: '#ff6b6b',
  success: '#6bff6b',
  warning: '#ffd93d',
  purple: '#c084fc',
  pink: '#f472b6',

  // Shadows (darker)
  sm: '0 1px 3px rgba(0, 0, 0, 0.3)',
  lg: '0 4px 16px rgba(0, 0, 0, 0.4)',
};

// Theme variants
export type ThemeVariant = 'light' | 'dark' | 'system';

// Current theme state
let currentTheme: ThemeVariant = 'light';
let currentTokens: ThemeTokens = lightTheme;

// Apply theme to CSS custom properties
export const applyTheme = (theme: ThemeVariant): void => {
  const tokens = theme === 'dark' ? darkTheme : lightTheme;
  currentTokens = tokens;
  currentTheme = theme;

  const root = document.documentElement;
  for (const [key, value] of Object.entries(tokens)) {
    const cssVar = key.replace(/([A-Z])/g, '-$1').toLowerCase();
    root.style.setProperty(`--${cssVar}`, value);
  }

  // Update color-scheme
  root.style.setProperty('color-scheme', theme === 'dark' ? 'dark' : 'light');
};

// Initialize theme from localStorage or system preference
export const initTheme = (): void => {
  // Check localStorage first
  const saved = localStorage.getItem('mech-theme') as ThemeVariant | null;
  if (saved && ['light', 'dark', 'system'].includes(saved)) {
    currentTheme = saved;
  }

  // Apply theme
  if (currentTheme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(prefersDark ? 'dark' : 'light');

    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (currentTheme === 'system') {
        applyTheme(e.matches ? 'dark' : 'light');
      }
    });
  } else {
    applyTheme(currentTheme);
  }
};

// Set theme and persist
export const setTheme = (theme: ThemeVariant): void => {
  currentTheme = theme;
  localStorage.setItem('mech-theme', theme);

  if (theme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(prefersDark ? 'dark' : 'light');
  } else {
    applyTheme(theme);
  }
};

// Get current theme
export const getTheme = (): ThemeVariant => currentTheme;

// Get current tokens
export const getTokens = (): ThemeTokens => currentTokens;

// CSS-in-JS helpers for React components
export const themeStyles = {
  // Container styles
  container: {
    background: 'var(--bg)',
    color: 'var(--text)',
    minHeight: '100vh',
    fontFamily: 'var(--font)',
  },

  // Card styles
  card: {
    background: 'var(--bg-elev)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    boxShadow: 'var(--shadow-sm)',
  },

  // Elevated card
  cardElevated: {
    background: 'var(--bg-elev)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    boxShadow: 'var(--shadow-lg)',
  },

  // Button styles
  buttonPrimary: {
    background: 'var(--accent)',
    color: 'var(--on-primary)',
    border: 'none',
    borderRadius: 'var(--radius)',
    cursor: 'pointer',
    transition: 'background 140ms ease',
  },

  buttonSecondary: {
    background: 'var(--bg-elev)',
    color: 'var(--text)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    cursor: 'pointer',
  },

  buttonDanger: {
    background: 'var(--danger)',
    color: 'white',
    border: 'none',
    borderRadius: 'var(--radius)',
    cursor: 'pointer',
  },

  // Input styles
  input: {
    background: 'var(--bg-elev)',
    color: 'var(--text)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '8px 12px',
    fontFamily: 'var(--font)',
  },

  // Sidebar styles
  sidebar: {
    background: 'var(--bg-sidebar)',
    borderRight: '1px solid var(--border)',
    width: 'var(--sidebar-width)',
  },

  // Activity bar styles
  activityBar: {
    background: 'var(--bg-activity)',
    borderRight: '1px solid var(--border)',
    width: 'var(--activity-width)',
  },

  // Console panel
  console: {
    background: 'var(--bg-elev)',
    borderTop: '1px solid var(--border)',
    height: 'var(--console-height)',
  },

  // Topbar
  topbar: {
    background: 'var(--bg-elev)',
    borderBottom: '1px solid var(--border)',
    height: 'var(--topbar-h)',
  },

  // Status bar
  statusbar: {
    background: 'var(--bg-elev2)',
    borderTop: '1px solid var(--border)',
    height: 'var(--statusbar-h)',
  },

  // Scrollbar
  scrollbar: {
    width: '8px',
    height: '8px',
    track: 'transparent',
    thumb: 'var(--border-light)',
    thumbHover: 'var(--text-muted)',
  },

  // Selection
  selection: {
    background: 'var(--accent-soft)',
    color: 'inherit',
  },

  // Focus ring
  focusRing: {
    outline: 'none',
    boxShadow: 'var(--accent-glow)',
  },

  // Typography
  typography: {
    fontFamily: 'var(--font)',
    fontSize: '14px',
    fontMono: 'var(--font-mono)',
  },
};

// Utility to create CSS variables object for inline styles
export const createCssVariables = (tokens: Partial<ThemeTokens> = {}): Record<string, string> => {
  const vars: Record<string, string> = {};
  const merged = { ...currentTokens, ...tokens };
  for (const [key, value] of Object.entries(merged)) {
    const cssVar = key.replace(/([A-Z])/g, '-$1').toLowerCase();
    vars[`--${cssVar}`] = value;
  }
  return vars;
};

export default {
  lightTheme,
  darkTheme,
  applyTheme,
  initTheme,
  setTheme,
  getTheme,
  getTokens,
  themeStyles,
  createCssVariables,
};