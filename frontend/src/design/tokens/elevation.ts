/**
 * MECH Platform - Elevation System
 * Consistent shadows and depth levels
 */

export interface ElevationTokens {
  // Shadow levels
  none: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  xxl: string;
  
  // Shadow colors
  shadowColor: string;
  shadowColorPrimary: string;
  shadowColorSuccess: string;
  shadowColorWarning: string;
  shadowColorError: string;
  
  // Shadow sizes
  shadowSizeSm: string;
  shadowSizeMd: string;
  shadowSizeLg: string;
  shadowSizeXl: string;
  
  // Blur levels
  blurSm: string;
  blurMd: string;
  blurLg: string;
  blurXl: string;
  
  // Elevation levels (z-index)
  elevation0: number;
  elevation1: number;
  elevation2: number;
  elevation3: number;
  elevation4: number;
  elevation5: number;
  elevationModal: number;
  elevationTooltip: number;
  elevationDropdown: number;
  
  // Glow effects
  glowSm: string;
  glowMd: string;
  glowLg: string;
  glowPrimary: string;
  glowSuccess: string;
  glowWarning: string;
  glowError: string;
}

export const elevation: ElevationTokens = {
  // Shadow levels
  none: 'none',
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  xxl: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  
  // Shadow colors
  shadowColor: 'rgba(0, 0, 0, 0.1)',
  shadowColorPrimary: 'rgba(79, 70, 229, 0.2)',
  shadowColorSuccess: 'rgba(16, 185, 129, 0.2)',
  shadowColorWarning: 'rgba(245, 158, 11, 0.2)',
  shadowColorError: 'rgba(239, 68, 68, 0.2)',
  
  // Shadow sizes
  shadowSizeSm: '0 1px 2px',
  shadowSizeMd: '0 4px 6px',
  shadowSizeLg: '0 10px 15px',
  shadowSizeXl: '0 20px 25px',
  
  // Blur levels
  blurSm: '4px',
  blurMd: '8px',
  blurLg: '12px',
  blurXl: '16px',
  
  // Elevation levels (z-index)
  elevation0: 0,
  elevation1: 10,
  elevation2: 20,
  elevation3: 30,
  elevation4: 40,
  elevation5: 50,
  elevationModal: 1000,
  elevationTooltip: 1100,
  elevationDropdown: 1000,
  
  // Glow effects
  glowSm: '0 0 0 2px rgba(79, 70, 229, 0.2)',
  glowMd: '0 0 0 4px rgba(79, 70, 229, 0.3)',
  glowLg: '0 0 0 8px rgba(79, 70, 229, 0.4)',
  glowPrimary: '0 0 0 4px rgba(79, 70, 229, 0.3)',
  glowSuccess: '0 0 0 4px rgba(16, 185, 129, 0.3)',
  glowWarning: '0 0 0 4px rgba(245, 158, 11, 0.3)',
  glowError: '0 0 0 4px rgba(239, 68, 68, 0.3)',
};
