import React from 'react';
import { colors, radii, spacing, typography } from '../../design/tokens';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary-pill' | 'dark-utility' | 'pearl-capsule' | 'store-hero';
  size?: 'sm' | 'md' | 'lg';
}

const baseStyle: React.CSSProperties = {
  fontFamily: typography.buttonUtility.fontFamily,
  fontSize: typography.buttonUtility.fontSize,
  fontWeight: typography.buttonUtility.fontWeight,
  lineHeight: typography.buttonUtility.lineHeight,
  letterSpacing: typography.buttonUtility.letterSpacing,
  border: 'none',
  cursor: 'pointer',
  transition: 'transform 0.1s ease',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: spacing.xs,
};

const variantStyles: Record<Required<ButtonProps>['variant'], React.CSSProperties> = {
  primary: {
    backgroundColor: colors.primary,
    color: colors.onPrimary,
    borderRadius: radii.pill,
    padding: `${spacing.sm} ${spacing.lg}`,
  },
  'secondary-pill': {
    backgroundColor: colors.canvas,
    color: colors.primary,
    border: `1px solid ${colors.primary}`,
    borderRadius: radii.pill,
    padding: `${spacing.sm} ${spacing.lg}`,
  },
  'dark-utility': {
    backgroundColor: colors.ink,
    color: colors.onDark,
    borderRadius: radii.sm,
    padding: `${spacing.xs} ${spacing.md}`,
  },
  'pearl-capsule': {
    backgroundColor: colors.surfacePearl,
    color: colors.inkMuted80,
    border: `3px solid ${colors.dividerSoft}`,
    borderRadius: radii.md,
    padding: `${spacing.xs} ${spacing.sm}`,
  },
  'store-hero': {
    backgroundColor: colors.primary,
    color: colors.onPrimary,
    borderRadius: radii.pill,
    padding: `${spacing.md} ${spacing.xl}`,
    fontFamily: typography.buttonLarge.fontFamily,
    fontSize: typography.buttonLarge.fontSize,
    fontWeight: typography.buttonLarge.fontWeight,
    lineHeight: typography.buttonLarge.lineHeight,
    letterSpacing: typography.buttonLarge.letterSpacing,
  },
};

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  children,
  style,
  ...rest
}) => {
  const variantStyle = variantStyles[variant];
  return (
    <button
      style={{ ...baseStyle, ...variantStyle, ...style }}
      onMouseDown={(e) => {
        e.currentTarget.style.transform = 'scale(0.95)';
      }}
      onMouseUp={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
      }}
      {...rest}
    >
      {children}
    </button>
  );
};
