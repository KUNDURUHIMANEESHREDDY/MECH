/**
 * MECH Platform - Modern Design System Colors
 * Professional, accessible color palette with dark/light theme support
 */

export interface ColorTokens {
  // Primary brand colors
  primary: string;
  primaryLight: string;
  primaryDark: string;
  primary50: string;
  primary100: string;
  primary200: string;
  primary300: string;
  primary400: string;
  primary500: string;
  primary600: string;
  primary700: string;
  primary800: string;
  primary900: string;
  
  // Secondary colors
  secondary: string;
  secondaryLight: string;
  secondaryDark: string;
  
  // Accent colors
  accent: string;
  accentLight: string;
  accentDark: string;
  
  // Semantic colors
  success: string;
  successLight: string;
  successDark: string;
  warning: string;
  warningLight: string;
  warningDark: string;
  error: string;
  errorLight: string;
  errorDark: string;
  info: string;
  infoLight: string;
  infoDark: string;
  
  // Neutral colors
  gray50: string;
  gray100: string;
  gray200: string;
  gray300: string;
  gray400: string;
  gray500: string;
  gray600: string;
  gray700: string;
  gray800: string;
  gray900: string;
  
  // Background colors
  background: string;
  backgroundSecondary: string;
  backgroundTertiary: string;
  surface: string;
  surfaceElevated: string;
  surfaceSunken: string;
  
  // Text colors
  textPrimary: string;
  textSecondary: string;
  textTertiary: string;
  textOnPrimary: string;
  textOnDark: string;
  textInverse: string;
  
  // Border colors
  borderPrimary: string;
  borderSecondary: string;
  borderTertiary: string;
  
  // Gradient colors
  gradientPrimary: string;
  gradientSecondary: string;
  gradientSurface: string;
  
  // Shadow colors
  shadowSm: string;
  shadowMd: string;
  shadowLg: string;
  shadowXl: string;
  
  // Overlay colors
  overlayLight: string;
  overlayMedium: string;
  overlayDark: string;
  
  // Status colors
  statusOnline: string;
  statusOffline: string;
  statusBusy: string;
  statusIdle: string;
}

export interface ThemeColors {
  // Background
  bgPrimary: string;
  bgSecondary: string;
  bgTertiary: string;
  bgElevated: string;
  bgSunken: string;
  bgHover: string;
  bgActive: string;
  bgSelected: string;
  bgDisabled: string;
  
  // Text
  textPrimary: string;
  textSecondary: string;
  textTertiary: string;
  textPlaceholder: string;
  textOnPrimary: string;
  textOnDark: string;
  textInverse: string;
  
  // Borders
  borderPrimary: string;
  borderSecondary: string;
  borderTertiary: string;
  borderHover: string;
  borderActive: string;
  
  // Surfaces
  surfacePrimary: string;
  surfaceSecondary: string;
  surfaceElevated: string;
  surfaceSunken: string;
  
  // Semantic
  success: string;
  successLight: string;
  successDark: string;
  successText: string;
  warning: string;
  warningLight: string;
  warningDark: string;
  warningText: string;
  error: string;
  errorLight: string;
  errorDark: string;
  errorText: string;
  info: string;
  infoLight: string;
  infoDark: string;
  infoText: string;
  
  // Brand
  primary: string;
  primaryLight: string;
  primaryDark: string;
  primaryText: string;
  primaryBorder: string;
  primaryBg: string;
  
  // Accent
  accent: string;
  accentLight: string;
  accentDark: string;
  accentText: string;
  
  // Gradients
  gradientPrimary: string;
  gradientSecondary: string;
  gradientSuccess: string;
  gradientWarning: string;
  gradientError: string;
  
  // Shadows
  shadowColor: string;
  shadowColorPrimary: string;
  
  // Status
  statusOnline: string;
  statusOffline: string;
  statusBusy: string;
  statusIdle: string;
}

// Base color tokens (design system foundation)
export const colorTokens: ColorTokens = {
  // Primary brand colors - Deep blue/purple palette
  primary: '#4F46E5',
  primaryLight: '#818CF8',
  primaryDark: '#3730A3',
  primary50: '#EEF2FF',
  primary100: '#E0E7FF',
  primary200: '#C7D2FE',
  primary300: '#A5B4FC',
  primary400: '#818CF8',
  primary500: '#6366F1',
  primary600: '#4F46E5',
  primary700: '#4338CA',
  primary800: '#3730A3',
  primary900: '#312E81',
  
  // Secondary colors - Teal/Cyan palette
  secondary: '#06B6D4',
  secondaryLight: '#22D3EE',
  secondaryDark: '#0891B2',
  
  // Accent colors - Purple/Pink palette
  accent: '#8B5CF6',
  accentLight: '#A78BFA',
  accentDark: '#7C3AED',
  
  // Semantic colors
  success: '#10B981',
  successLight: '#34D399',
  successDark: '#059669',
  warning: '#F59E0B',
  warningLight: '#FBBF24',
  warningDark: '#D97706',
  error: '#EF4444',
  errorLight: '#F87171',
  errorDark: '#DC2626',
  info: '#3B82F6',
  infoLight: '#60A5FA',
  infoDark: '#2563EB',
  
  // Neutral colors (gray scale)
  gray50: '#F9FAFB',
  gray100: '#F3F4F6',
  gray200: '#E5E7EB',
  gray300: '#D1D5DB',
  gray400: '#9CA3AF',
  gray500: '#6B7280',
  gray600: '#4B5563',
  gray700: '#374151',
  gray800: '#1F2937',
  gray900: '#111827',
  
  // Background colors
  background: '#0F172A',
  backgroundSecondary: '#1E293B',
  backgroundTertiary: '#334155',
  surface: '#1E293B',
  surfaceElevated: '#334155',
  surfaceSunken: '#0F172A',
  
  // Text colors
  textPrimary: '#F8FAFC',
  textSecondary: '#CBD5E1',
  textTertiary: '#94A3B8',
  textOnPrimary: '#FFFFFF',
  textOnDark: '#FFFFFF',
  textInverse: '#0F172A',
  
  // Border colors
  borderPrimary: '#334155',
  borderSecondary: '#475569',
  borderTertiary: '#64748B',
  
  // Gradient colors
  gradientPrimary: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
  gradientSecondary: 'linear-gradient(135deg, #06B6D4 0%, #22D3EE 100%)',
  gradientSurface: 'linear-gradient(180deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%)',
  
  // Shadow colors
  shadowSm: 'rgba(0, 0, 0, 0.1)',
  shadowMd: 'rgba(0, 0, 0, 0.2)',
  shadowLg: 'rgba(0, 0, 0, 0.3)',
  shadowXl: 'rgba(0, 0, 0, 0.4)',
  
  // Overlay colors
  overlayLight: 'rgba(15, 23, 42, 0.3)',
  overlayMedium: 'rgba(15, 23, 42, 0.6)',
  overlayDark: 'rgba(15, 23, 42, 0.9)',
  
  // Status colors
  statusOnline: '#10B981',
  statusOffline: '#6B7280',
  statusBusy: '#F59E0B',
  statusIdle: '#94A3B8',
};

// Light theme colors
export const lightColors: ThemeColors = {
  // Background
  bgPrimary: '#F8FAFC',
  bgSecondary: '#F1F5F9',
  bgTertiary: '#E2E8F0',
  bgElevated: '#FFFFFF',
  bgSunken: '#F8FAFC',
  bgHover: '#F1F5F9',
  bgActive: '#E2E8F0',
  bgSelected: '#CBD5E1',
  bgDisabled: '#F8FAFC',
  
  // Text
  textPrimary: '#0F172A',
  textSecondary: '#475569',
  textTertiary: '#64748B',
  textPlaceholder: '#94A3B8',
  textOnPrimary: '#FFFFFF',
  textOnDark: '#FFFFFF',
  textInverse: '#F8FAFC',
  
  // Borders
  borderPrimary: '#E2E8F0',
  borderSecondary: '#CBD5E1',
  borderTertiary: '#94A3B8',
  borderHover: '#475569',
  borderActive: '#64748B',
  
  // Surfaces
  surfacePrimary: '#FFFFFF',
  surfaceSecondary: '#F8FAFC',
  surfaceElevated: '#FFFFFF',
  surfaceSunken: '#F1F5F9',
  
  // Semantic colors
  success: '#10B981',
  successLight: '#D1FAE5',
  successDark: '#059669',
  successText: '#065F46',
  warning: '#F59E0B',
  warningLight: '#FEF3C7',
  warningDark: '#D97706',
  warningText: '#92400E',
  error: '#EF4444',
  errorLight: '#FEE2E2',
  errorDark: '#DC2626',
  errorText: '#991B1B',
  info: '#3B82F6',
  infoLight: '#DBEAFE',
  infoDark: '#2563EB',
  infoText: '#1E40AF',
  
  // Brand colors
  primary: '#4F46E5',
  primaryLight: '#818CF8',
  primaryDark: '#3730A3',
  primaryText: '#FFFFFF',
  primaryBorder: '#C7D2FE',
  primaryBg: '#E0E7FF',
  
  // Accent colors
  accent: '#8B5CF6',
  accentLight: '#A78BFA',
  accentDark: '#7C3AED',
  accentText: '#FFFFFF',
  
  // Gradients
  gradientPrimary: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
  gradientSecondary: 'linear-gradient(135deg, #06B6D4 0%, #22D3EE 100%)',
  gradientSuccess: 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
  gradientWarning: 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)',
  gradientError: 'linear-gradient(135deg, #EF4444 0%, #F87171 100%)',
  
  // Shadows
  shadowColor: 'rgba(15, 23, 42, 0.1)',
  shadowColorPrimary: 'rgba(79, 70, 229, 0.2)',
  
  // Status
  statusOnline: '#10B981',
  statusOffline: '#6B7280',
  statusBusy: '#F59E0B',
  statusIdle: '#94A3B8',
};

// Dark theme colors (default for MECH)
export const darkColors: ThemeColors = {
  // Background
  bgPrimary: '#0F172A',
  bgSecondary: '#1E293B',
  bgTertiary: '#334155',
  bgElevated: '#1E293B',
  bgSunken: '#0F172A',
  bgHover: '#334155',
  bgActive: '#475569',
  bgSelected: '#475569',
  bgDisabled: '#1E293B',
  
  // Text
  textPrimary: '#F8FAFC',
  textSecondary: '#CBD5E1',
  textTertiary: '#94A3B8',
  textPlaceholder: '#64748B',
  textOnPrimary: '#FFFFFF',
  textOnDark: '#FFFFFF',
  textInverse: '#0F172A',
  
  // Borders
  borderPrimary: '#334155',
  borderSecondary: '#475569',
  borderTertiary: '#64748B',
  borderHover: '#475569',
  borderActive: '#64748B',
  
  // Surfaces
  surfacePrimary: '#1E293B',
  surfaceSecondary: '#0F172A',
  surfaceElevated: '#334155',
  surfaceSunken: '#1E293B',
  
  // Semantic colors
  success: '#10B981',
  successLight: '#10B98126',
  successDark: '#059669',
  successText: '#D1FAE5',
  warning: '#F59E0B',
  warningLight: '#F59E0B26',
  warningDark: '#D97706',
  warningText: '#FEF3C7',
  error: '#EF4444',
  errorLight: '#EF444426',
  errorDark: '#DC2626',
  errorText: '#FEE2E2',
  info: '#3B82F6',
  infoLight: '#3B82F626',
  infoDark: '#2563EB',
  infoText: '#DBEAFE',
  
  // Brand colors
  primary: '#6366F1',
  primaryLight: '#818CF8',
  primaryDark: '#4F46E5',
  primaryText: '#FFFFFF',
  primaryBorder: '#4F46E5',
  primaryBg: '#4F46E526',
  
  // Accent colors
  accent: '#8B5CF6',
  accentLight: '#A78BFA',
  accentDark: '#7C3AED',
  accentText: '#FFFFFF',
  
  // Gradients
  gradientPrimary: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
  gradientSecondary: 'linear-gradient(135deg, #06B6D4 0%, #22D3EE 100%)',
  gradientSuccess: 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
  gradientWarning: 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)',
  gradientError: 'linear-gradient(135deg, #EF4444 0%, #F87171 100%)',
  
  // Shadows
  shadowColor: 'rgba(0, 0, 0, 0.3)',
  shadowColorPrimary: 'rgba(79, 70, 229, 0.3)',
  
  // Status
  statusOnline: '#10B981',
  statusOffline: '#6B7280',
  statusBusy: '#F59E0B',
  statusIdle: '#94A3B8',
};

// Export colors with backward compatibility
export const colors = {
  ...colorTokens,
  // Legacy color names for backward compatibility
  primary: darkColors.primary,
  primaryFocus: darkColors.primaryDark,
  primaryOnDark: darkColors.primaryLight,
  accentOnDark: darkColors.accentLight,
  ink: darkColors.textPrimary,
  body: darkColors.textSecondary,
  bodyOnDark: darkColors.textOnPrimary,
  bodyMuted: darkColors.textTertiary,
  inkMuted80: darkColors.textSecondary,
  inkMuted48: darkColors.textTertiary,
  dividerSoft: darkColors.borderPrimary,
  hairline: darkColors.borderSecondary,
  canvas: darkColors.bgPrimary,
  canvasParchment: darkColors.bgSecondary,
  surfacePearl: darkColors.bgTertiary,
  surfaceTile1: darkColors.surfacePrimary,
  surfaceTile2: darkColors.surfaceSecondary,
  surfaceTile3: darkColors.surfaceElevated,
  surfaceBlack: darkColors.bgPrimary,
  surfaceChipTranslucent: darkColors.primaryBg,
  onPrimary: darkColors.textOnPrimary,
  onDark: darkColors.textOnDark,
  border: darkColors.borderPrimary,
  bgSidebar: darkColors.bgSecondary,
  accentSoft: darkColors.primaryBg,
  success: darkColors.success,
  successSoft: darkColors.successLight,
  successBorder: darkColors.success,
  successText: darkColors.successText,
  danger: darkColors.error,
  dangerSoft: darkColors.errorLight,
  dangerBorder: darkColors.error,
  dangerText: darkColors.errorText,
  warning: darkColors.warning,
  warningSoft: darkColors.warningLight,
  warningBorder: darkColors.warning,
  warningText: darkColors.warningText,
  infoSoft: darkColors.infoLight,
  infoBorder: darkColors.info,
  infoText: darkColors.infoText,
  purple: darkColors.accent,
  purpleSoft: darkColors.accentLight,
  purpleBorder: darkColors.accent,
  purpleText: darkColors.accentText,
  pink: '#EC4899',
  pinkSoft: '#FCE7F3',
  pinkBorder: '#F9A8D4',
  pinkText: '#9F1239',
};

export { darkColors as theme };
