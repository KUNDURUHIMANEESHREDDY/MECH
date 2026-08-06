/**
 * MECH Platform - Spacing System
 * 8px base grid with consistent spacing scale
 */

export interface SpacingTokens {
  // Base spacing (8px grid)
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  xxl: string;
  
  // Fractional spacing
  '0.5': string;
  '1': string;
  '1.5': string;
  '2': string;
  '2.5': string;
  '3': string;
  '3.5': string;
  '4': string;
  '4.5': string;
  '5': string;
  
  // Large spacing
  '6': string;
  '7': string;
  '8': string;
  '9': string;
  '10': string;
  '11': string;
  '12': string;
  '14': string;
  '16': string;
  '20': string;
  '24': string;
  '28': string;
  '32': string;
  '36': string;
  '40': string;
  '44': string;
  '48': string;
  '52': string;
  '56': string;
  '60': string;
  '64': string;
  '72': string;
  '80': string;
  '96': string;
  
  // Special spacing
  none: string;
  auto: string;
  full: string;
  screen: string;
  
  // Component-specific spacing
  navbarHeight: string;
  sidebarWidth: string;
  sidebarCollapsedWidth: string;
  activityBarWidth: string;
  statusBarHeight: string;
  consolePanelHeight: string;
  
  // Layout spacing
  pageMargin: string;
  pageMarginSm: string;
  cardPadding: string;
  cardPaddingSm: string;
  cardPaddingLg: string;
  
  // Gap sizes
  gapXs: string;
  gapSm: string;
  gapMd: string;
  gapLg: string;
  gapXl: string;
}

export const spacing: SpacingTokens = {
  // Base spacing (8px grid)
  xs: '8px',
  sm: '12px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  xxl: '48px',
  
  // Fractional spacing (4px base for fine adjustments)
  '0.5': '2px',
  '1': '4px',
  '1.5': '6px',
  '2': '8px',
  '2.5': '10px',
  '3': '12px',
  '3.5': '14px',
  '4': '16px',
  '4.5': '18px',
  '5': '20px',
  
  // Large spacing
  '6': '24px',
  '7': '28px',
  '8': '32px',
  '9': '36px',
  '10': '40px',
  '11': '44px',
  '12': '48px',
  '14': '56px',
  '16': '64px',
  '20': '80px',
  '24': '96px',
  '28': '112px',
  '32': '128px',
  '36': '144px',
  '40': '160px',
  '44': '176px',
  '48': '192px',
  '52': '208px',
  '56': '224px',
  '60': '240px',
  '64': '256px',
  '72': '288px',
  '80': '320px',
  '96': '384px',
  
  // Special spacing
  none: '0',
  auto: 'auto',
  full: '100%',
  screen: '100vh',
  
  // Component-specific spacing
  navbarHeight: '48px',
  sidebarWidth: '280px',
  sidebarCollapsedWidth: '64px',
  activityBarWidth: '56px',
  statusBarHeight: '28px',
  consolePanelHeight: '200px',
  
  // Layout spacing
  pageMargin: '24px',
  pageMarginSm: '16px',
  cardPadding: '20px',
  cardPaddingSm: '16px',
  cardPaddingLg: '28px',
  
  // Gap sizes
  gapXs: '6px',
  gapSm: '8px',
  gapMd: '12px',
  gapLg: '16px',
  gapXl: '24px',
};
