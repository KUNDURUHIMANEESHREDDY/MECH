import React from 'react';
import { colors, radii, spacing, typography } from '../../design/tokens';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'utility' | 'product-tile-light' | 'product-tile-parchment' | 'product-tile-dark' | 'environment-quote';
}

const variantStyles: Record<Required<CardProps>['variant'], React.CSSProperties> = {
  utility: {
    backgroundColor: colors.canvas,
    border: `1px solid ${colors.hairline}`,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  'product-tile-light': {
    backgroundColor: colors.canvas,
    color: colors.ink,
    borderRadius: radii.none,
    padding: spacing.section,
  },
  'product-tile-parchment': {
    backgroundColor: colors.canvasParchment,
    color: colors.ink,
    borderRadius: radii.none,
    padding: spacing.section,
  },
  'product-tile-dark': {
    backgroundColor: colors.canvasParchment,
    color: colors.ink,
    borderRadius: radii.none,
    padding: spacing.section,
  },
  'environment-quote': {
    backgroundColor: colors.canvasParchment,
    color: colors.ink,
    borderRadius: radii.none,
    padding: spacing.section,
  },
};

export const Card: React.FC<CardProps> = ({
  variant = 'utility',
  children,
  style,
  ...rest
}) => {
  return (
    <div style={{ ...variantStyles[variant], ...style }} {...rest}>
      {children}
    </div>
  );
};
