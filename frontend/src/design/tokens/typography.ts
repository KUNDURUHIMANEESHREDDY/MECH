/**
 * MECH Platform - Typography System
 * Modern, readable typography with consistent scale
 */

export interface TypographyTokens {
  // Font families
  fontFamilySans: string;
  fontFamilyMono: string;
  fontFamilyDisplay: string;
  
  // Font sizes
  fontSizeXs: string;
  fontSizeSm: string;
  fontSizeBase: string;
  fontSizeLg: string;
  fontSizeXl: string;
  fontSize2xl: string;
  fontSize3xl: string;
  fontSize4xl: string;
  fontSize5xl: string;
  fontSize6xl: string;
  fontSize7xl: string;
  
  // Font weights
  fontWeightThin: number;
  fontWeightExtralight: number;
  fontWeightLight: number;
  fontWeightNormal: number;
  fontWeightMedium: number;
  fontWeightSemibold: number;
  fontWeightBold: number;
  fontWeightExtrabold: number;
  fontWeightBlack: number;
  
  // Line heights
  lineHeightNone: number;
  lineHeightTight: number;
  lineHeightSnug: number;
  lineHeightNormal: number;
  lineHeightRelaxed: number;
  lineHeightLoose: number;
  
  // Letter spacing
  letterSpacingTighter: string;
  letterSpacingTight: string;
  letterSpacingNormal: string;
  letterSpacingWide: string;
  letterSpacingWider: string;
  letterSpacingWidest: string;
  
  // Text styles
  display: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  displayLg: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  displayMd: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  lead: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  leadAiry: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  tagline: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  body: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  bodyStrong: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  denseLink: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  caption: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  captionStrong: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  buttonLarge: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  buttonUtility: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  finePrint: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  microLegal: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  navLink: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
  code: {
    fontFamily: string;
    fontSize: string;
    fontWeight: number;
    lineHeight: number;
    letterSpacing: string;
  };
}

export const typography: TypographyTokens = {
  // Font families
  fontFamilySans: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  fontFamilyMono: '"Fira Code", "JetBrains Mono", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
  fontFamilyDisplay: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  
  // Font sizes
  fontSizeXs: '12px',
  fontSizeSm: '14px',
  fontSizeBase: '16px',
  fontSizeLg: '18px',
  fontSizeXl: '20px',
  fontSize2xl: '24px',
  fontSize3xl: '30px',
  fontSize4xl: '36px',
  fontSize5xl: '48px',
  fontSize6xl: '60px',
  fontSize7xl: '72px',
  
  // Font weights
  fontWeightThin: 100,
  fontWeightExtralight: 200,
  fontWeightLight: 300,
  fontWeightNormal: 400,
  fontWeightMedium: 500,
  fontWeightSemibold: 600,
  fontWeightBold: 700,
  fontWeightExtrabold: 800,
  fontWeightBlack: 900,
  
  // Line heights
  lineHeightNone: 1,
  lineHeightTight: 1.25,
  lineHeightSnug: 1.375,
  lineHeightNormal: 1.5,
  lineHeightRelaxed: 1.625,
  lineHeightLoose: 2,
  
  // Letter spacing
  letterSpacingTighter: '-0.05em',
  letterSpacingTight: '-0.025em',
  letterSpacingNormal: '0',
  letterSpacingWide: '0.025em',
  letterSpacingWider: '0.05em',
  letterSpacingWidest: '0.1em',
  
  // Text styles
  display: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '72px',
    fontWeight: 700,
    lineHeight: 1.07,
    letterSpacing: '-0.028em',
  },
  displayLg: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '40px',
    fontWeight: 600,
    lineHeight: 1.1,
    letterSpacing: '0',
  },
  displayMd: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '34px',
    fontWeight: 600,
    lineHeight: 1.47,
    letterSpacing: '-0.0374em',
  },
  lead: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '28px',
    fontWeight: 400,
    lineHeight: 1.14,
    letterSpacing: '0.0196em',
  },
  leadAiry: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '24px',
    fontWeight: 300,
    lineHeight: 1.5,
    letterSpacing: '0',
  },
  tagline: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '21px',
    fontWeight: 600,
    lineHeight: 1.19,
    letterSpacing: '0.0231em',
  },
  body: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '16px',
    fontWeight: 400,
    lineHeight: 1.47,
    letterSpacing: '-0.0374em',
  },
  bodyStrong: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '16px',
    fontWeight: 600,
    lineHeight: 1.24,
    letterSpacing: '-0.0374em',
  },
  denseLink: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '16px',
    fontWeight: 400,
    lineHeight: 2.41,
    letterSpacing: '0',
  },
  caption: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '14px',
    fontWeight: 400,
    lineHeight: 1.43,
    letterSpacing: '-0.0224em',
  },
  captionStrong: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '14px',
    fontWeight: 600,
    lineHeight: 1.29,
    letterSpacing: '-0.0224em',
  },
  buttonLarge: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '18px',
    fontWeight: 300,
    lineHeight: 1,
    letterSpacing: '0',
  },
  buttonUtility: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '14px',
    fontWeight: 400,
    lineHeight: 1.29,
    letterSpacing: '-0.0224em',
  },
  finePrint: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '12px',
    fontWeight: 400,
    lineHeight: 1,
    letterSpacing: '-0.012em',
  },
  microLegal: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '10px',
    fontWeight: 400,
    lineHeight: 1.3,
    letterSpacing: '-0.008em',
  },
  navLink: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '12px',
    fontWeight: 400,
    lineHeight: 1,
    letterSpacing: '0',
  },
  code: {
    fontFamily: '"Fira Code", "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace',
    fontSize: '14px',
    fontWeight: 400,
    lineHeight: 1.5,
    letterSpacing: '0',
  },
};
