/**
 * MECH Platform - Border Radius System
 * Consistent rounded corners with modern aesthetic
 */

export interface RadiiTokens {
  // Base radii
  none: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  xxl: string;
  full: string;
  
  // Special radii
  pill: string;
  circle: string;
  
  // Component-specific radii
  button: string;
  card: string;
  input: string;
  panel: string;
  modal: string;
  toolbar: string;
  badge: string;
  chip: string;
  avatar: string;
  
  // Layout radii
  layout: string;
  section: string;
  
  // Status radii
  statusSm: string;
  statusMd: string;
}

export const radii: RadiiTokens = {
  // Base radii
  none: '0',
  sm: '4px',
  md: '8px',
  lg: '12px',
  xl: '16px',
  xxl: '24px',
  full: '9999px',
  
  // Special radii
  pill: '9999px',
  circle: '50%',
  
  // Component-specific radii
  button: '8px',
  card: '12px',
  input: '8px',
  panel: '12px',
  modal: '16px',
  toolbar: '6px',
  badge: '9999px',
  chip: '9999px',
  avatar: '50%',
  
  // Layout radii
  layout: '12px',
  section: '8px',
  
  // Status radii
  statusSm: '2px',
  statusMd: '4px',
};
