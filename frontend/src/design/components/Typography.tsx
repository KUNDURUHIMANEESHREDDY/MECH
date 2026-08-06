import React from 'react';
import { darkColors } from '../tokens/colors';
import { typography } from '../tokens/typography';
import { spacing } from '../tokens/spacing';

export interface TypographyProps extends React.HTMLAttributes<HTMLElement> {
  variant?: 
    | 'display'
    | 'display-lg'
    | 'display-md'
    | 'lead'
    | 'lead-airy'
    | 'tagline'
    | 'body'
    | 'body-strong'
    | 'caption'
    | 'caption-strong'
    | 'fine-print'
    | 'micro-legal';
  color?: 'primary' | 'secondary' | 'tertiary' | 'success' | 'warning' | 'error' | 'info' | 'muted';
  weight?: 'light' | 'normal' | 'medium' | 'semibold' | 'bold';
  align?: 'left' | 'center' | 'right';
  truncate?: boolean;
  mono?: boolean;
}

export const Typography: React.FC<TypographyProps> = ({
  children,
  variant = 'body',
  color = 'primary',
  weight,
  align = 'left',
  truncate = false,
  mono = false,
  className = '',
  ...props
}) => {
  const colorMap = {
    primary: darkColors.textPrimary,
    secondary: darkColors.textSecondary,
    tertiary: darkColors.textTertiary,
    success: darkColors.success,
    warning: darkColors.warning,
    error: darkColors.error,
    info: darkColors.info,
    muted: darkColors.textPlaceholder,
  };

  const weightMap = {
    light: typography.fontWeightLight,
    normal: typography.fontWeightNormal,
    medium: typography.fontWeightMedium,
    semibold: typography.fontWeightSemibold,
    bold: typography.fontWeightBold,
  };

  const variantStyles = {
    display: {
      fontFamily: typography.fontFamilyDisplay,
      fontSize: typography.fontSize5xl,
      fontWeight: typography.fontWeightBold,
      lineHeight: typography.lineHeightTight,
      letterSpacing: typography.letterSpacingTighter,
    },
    'display-lg': {
      fontFamily: typography.fontFamilyDisplay,
      fontSize: typography.fontSize4xl,
      fontWeight: typography.fontWeightBold,
      lineHeight: typography.lineHeightTight,
      letterSpacing: typography.letterSpacingTight,
    },
    'display-md': {
      fontFamily: typography.fontFamilyDisplay,
      fontSize: typography.fontSize3xl,
      fontWeight: typography.fontWeightBold,
      lineHeight: typography.lineHeightSnug,
      letterSpacing: typography.letterSpacingTight,
    },
    lead: {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSize2xl,
      fontWeight: typography.fontWeightNormal,
      lineHeight: typography.lineHeightRelaxed,
      letterSpacing: typography.letterSpacingNormal,
    },
    'lead-airy': {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeXl,
      fontWeight: typography.fontWeightLight,
      lineHeight: typography.lineHeightLoose,
      letterSpacing: typography.letterSpacingNormal,
    },
    tagline: {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeLg,
      fontWeight: typography.fontWeightSemibold,
      lineHeight: typography.lineHeightTight,
      letterSpacing: typography.letterSpacingWide,
    },
    body: {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeBase,
      fontWeight: typography.fontWeightNormal,
      lineHeight: typography.lineHeightNormal,
      letterSpacing: typography.letterSpacingNormal,
    },
    'body-strong': {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeBase,
      fontWeight: typography.fontWeightSemibold,
      lineHeight: typography.lineHeightSnug,
      letterSpacing: typography.letterSpacingTight,
    },
    caption: {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeSm,
      fontWeight: typography.fontWeightNormal,
      lineHeight: typography.lineHeightNormal,
      letterSpacing: typography.letterSpacingNormal,
    },
    'caption-strong': {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeSm,
      fontWeight: typography.fontWeightSemibold,
      lineHeight: typography.lineHeightSnug,
      letterSpacing: typography.letterSpacingTight,
    },
    'fine-print': {
      fontFamily: typography.fontFamilySans,
      fontSize: typography.fontSizeXs,
      fontWeight: typography.fontWeightNormal,
      lineHeight: typography.lineHeightTight,
      letterSpacing: typography.letterSpacingNormal,
    },
    'micro-legal': {
      fontFamily: typography.fontFamilySans,
      fontSize: '10px',
      fontWeight: typography.fontWeightNormal,
      lineHeight: typography.lineHeightTight,
      letterSpacing: typography.letterSpacingTight,
    },
  };

  const Component = variant.includes('display') ? 'h1' : 
                   variant.includes('lead') ? 'p' :
                   variant.includes('tagline') ? 'h2' :
                   variant.includes('body') ? 'p' :
                   variant.includes('caption') ? 'span' :
                   variant.includes('fine') || variant.includes('micro') ? 'span' : 'p';

  return (
    <Component
      className={`typography typography-${variant} ${className}`}
      style={{
        color: colorMap[color],
        fontWeight: weight ? weightMap[weight] : variantStyles[variant].fontWeight,
        textAlign: align,
        fontFamily: mono ? typography.fontFamilyMono : variantStyles[variant].fontFamily,
        fontSize: variantStyles[variant].fontSize,
        lineHeight: variantStyles[variant].lineHeight,
        letterSpacing: variantStyles[variant].letterSpacing,
        overflow: truncate ? 'hidden' : undefined,
        textOverflow: truncate ? 'ellipsis' : undefined,
        whiteSpace: truncate ? 'nowrap' : undefined,
      }}
      {...props}
    >
      {children}
    </Component>
  );
};

// Convenience components
export const H1: React.FC<TypographyProps> = (props) => (
  <Typography as="h1" variant="display" {...props} />
);

export const H2: React.FC<TypographyProps> = (props) => (
  <Typography as="h2" variant="display-md" {...props} />
);

export const H3: React.FC<TypographyProps> = (props) => (
  <Typography as="h3" variant="lead" {...props} />
);

export const H4: React.FC<TypographyProps> = (props) => (
  <Typography as="h4" variant="tagline" {...props} />
);

export const P: React.FC<TypographyProps> = (props) => (
  <Typography as="p" variant="body" {...props} />
);

export const Span: React.FC<TypographyProps> = (props) => (
  <Typography as="span" variant="body" {...props} />
);

export const Code: React.FC<TypographyProps> = ({ children, ...props }) => (
  <Typography as="code" mono variant="caption" color="secondary" {...props}>
    {children}
  </Typography>
);

export default Typography;
