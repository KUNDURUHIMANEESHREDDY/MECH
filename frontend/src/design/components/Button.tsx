import React from 'react';
import { darkColors } from '../tokens/colors';
import { spacing } from '../tokens/spacing';
import { radii } from '../tokens/radii';
import { typography } from '../tokens/typography';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'success' | 'outline';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  isLoading?: boolean;
  fullWidth?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'secondary',
  size = 'md',
  leftIcon,
  rightIcon,
  isLoading = false,
  fullWidth = false,
  className = '',
  disabled,
  ...props
}) => {
  const baseStyles: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing[2],
    fontFamily: typography.fontFamilySans,
    fontWeight: typography.fontWeightMedium,
    lineHeight: 1,
    border: 'none',
    borderRadius: radii.button,
    cursor: 'pointer',
    transition: 'all 150ms ease',
    whiteSpace: 'nowrap',
    position: 'relative',
    overflow: 'hidden',
  };

  const variants: Record<ButtonProps['variant'], React.CSSProperties> = {
    primary: {
      background: darkColors.gradientPrimary,
      color: darkColors.textOnPrimary,
      boxShadow: '0 2px 4px rgba(79, 70, 229, 0.2)',
    },
    secondary: {
      background: darkColors.bgTertiary,
      color: darkColors.textPrimary,
      border: `1px solid ${darkColors.borderPrimary}`,
    },
    ghost: {
      background: 'transparent',
      color: darkColors.textSecondary,
    },
    danger: {
      background: darkColors.error,
      color: darkColors.textOnPrimary,
    },
    success: {
      background: darkColors.success,
      color: darkColors.textOnPrimary,
    },
    outline: {
      background: 'transparent',
      color: darkColors.textPrimary,
      border: `1px solid ${darkColors.borderPrimary}`,
    },
  };

  const sizes: Record<ButtonProps['size'], React.CSSProperties> = {
    sm: {
      padding: `${spacing[1.5]} ${spacing[3]}`,
      fontSize: typography.fontSizeSm,
    },
    md: {
      padding: `${spacing[2]} ${spacing[4]}`,
      fontSize: typography.fontSizeSm,
    },
    lg: {
      padding: `${spacing[3]} ${spacing[6]}`,
      fontSize: typography.fontSizeBase,
    },
    icon: {
      width: spacing[9],
      height: spacing[9],
      padding: 0,
    },
  };

  const hoverStyles: Record<ButtonProps['variant'], React.CSSProperties> = {
    primary: {
      boxShadow: '0 4px 8px rgba(79, 70, 229, 0.3)',
      transform: 'translateY(-1px)',
    },
    secondary: {
      background: darkColors.bgHover,
      borderColor: darkColors.borderHover,
    },
    ghost: {
      background: darkColors.bgHover,
      color: darkColors.textPrimary,
    },
    danger: {
      background: darkColors.errorDark,
    },
    success: {
      background: darkColors.successDark,
    },
    outline: {
      background: darkColors.bgHover,
      borderColor: darkColors.borderHover,
    },
  };

  const mergedStyles: React.CSSProperties = {
    ...baseStyles,
    ...variants[variant],
    ...sizes[size],
    width: fullWidth ? '100%' : undefined,
    opacity: disabled || isLoading ? 0.5 : 1,
    cursor: disabled || isLoading ? 'not-allowed' : 'pointer',
  };

  return (
    <button
      className={`btn btn-${variant} btn-${size} ${className}`}
      style={mergedStyles}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <span
          style={{
            width: sizes[size]?.height || sizes[size]?.width || 16,
            height: sizes[size]?.height || sizes[size]?.width || 16,
            border: '2px solid rgba(255, 255, 255, 0.3)',
            borderTopColor: 'transparent',
            borderRadius: '50%',
            animation: 'spin 0.6s linear infinite',
          }}
        />
      ) : (
        <>
          {leftIcon && <span>{leftIcon}</span>}
          {children}
          {rightIcon && <span>{rightIcon}</span>}
        </>
      )}
    </button>
  );
};

export default Button;
