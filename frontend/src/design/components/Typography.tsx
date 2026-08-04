import React from 'react';
import { colors, typography } from '../../design/tokens';

type Variant =
  | 'hero-display'
  | 'display-lg'
  | 'display-md'
  | 'lead'
  | 'lead-airy'
  | 'tagline'
  | 'body-strong'
  | 'body'
  | 'dense-link'
  | 'caption'
  | 'caption-strong'
  | 'button-large'
  | 'button-utility'
  | 'fine-print'
  | 'micro-legal'
  | 'nav-link';

interface TypographyProps extends React.HTMLAttributes<HTMLHeadingElement> {
  variant?: Variant;
  color?: keyof typeof colors;
}

const variantMap: Record<Variant, keyof typeof typography> = {
  'hero-display': 'heroDisplay',
  'display-lg': 'displayLg',
  'display-md': 'displayMd',
  lead: 'lead',
  'lead-airy': 'leadAiry',
  tagline: 'tagline',
  'body-strong': 'bodyStrong',
  body: 'body',
  'dense-link': 'denseLink',
  caption: 'caption',
  'caption-strong': 'captionStrong',
  'button-large': 'buttonLarge',
  'button-utility': 'buttonUtility',
  'fine-print': 'finePrint',
  'micro-legal': 'microLegal',
  'nav-link': 'navLink',
};

export const Typography: React.FC<TypographyProps> = ({
  variant = 'body',
  color = 'ink',
  children,
  style,
  ...rest
}) => {
  const t = typography[variantMap[variant]];
  const colorVal = (colors as Record<string, string>)[color] ?? colors.ink;

  return (
    <span
      style={{
        fontFamily: t.fontFamily,
        fontSize: t.fontSize,
        fontWeight: t.fontWeight,
        lineHeight: t.lineHeight,
        letterSpacing: t.letterSpacing,
        color: colorVal,
        ...style,
      }}
      {...rest}
    >
      {children}
    </span>
  );
};
