/**
 * MECH Platform - Design System Tokens
 * Single import point for all design tokens
 */

export { colors, colorTokens, darkColors, lightColors } from './colors';
export type { ThemeColors, ColorTokens } from './colors';
export { typography } from './typography';
export { spacing } from './spacing';
export { radii } from './radii';
export { elevation } from './elevation';

// Export theme as default for backward compatibility
export { darkColors as theme } from './colors';
