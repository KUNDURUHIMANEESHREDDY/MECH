import React from 'react';
import { darkColors } from '../tokens/colors';
import { spacing } from '../tokens/spacing';
import { radii } from '../tokens/radii';
import { typography } from '../tokens/typography';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  variant?: 'default' | 'filled' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  error?: boolean;
  hint?: string;
}

export const Input: React.FC<InputProps> = ({
  variant = 'default',
  size = 'md',
  leftIcon,
  rightIcon,
  error = false,
  hint,
  className = '',
  disabled,
  ...props
}) => {
  const sizes = {
    sm: {
      padding: `${spacing[2]} ${spacing[3]}`,
      fontSize: typography.fontSizeSm,
      height: 32,
    },
    md: {
      padding: `${spacing[2.5]} ${spacing[3]}`,
      fontSize: typography.fontSizeSm,
      height: 36,
    },
    lg: {
      padding: `${spacing[3]} ${spacing[4]}`,
      fontSize: typography.fontSizeBase,
      height: 42,
    },
  };

  const variants = {
    default: {
      background: darkColors.bgSecondary,
      border: `1px solid ${darkColors.borderPrimary}`,
      color: darkColors.textPrimary,
    },
    filled: {
      background: darkColors.bgTertiary,
      border: `1px solid ${darkColors.borderPrimary}`,
      color: darkColors.textPrimary,
    },
    outline: {
      background: 'transparent',
      border: `1px solid ${darkColors.borderPrimary}`,
      color: darkColors.textPrimary,
    },
    ghost: {
      background: 'transparent',
      border: 'none',
      color: darkColors.textPrimary,
    },
  };

  const errorStyles = error
    ? {
        borderColor: darkColors.error,
        background: darkColors.errorLight,
      }
    : {};

  const disabledStyles = disabled
    ? {
        opacity: 0.5,
        cursor: 'not-allowed',
      }
    : {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing[1] }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          background: variants[variant].background,
          border: variants[variant].border,
          borderRadius: radii.input,
          transition: 'all 150ms ease',
          ...errorStyles,
          ...disabledStyles,
        }}
      >
        {leftIcon && (
          <span
            style={{
              paddingLeft: spacing[2.5],
              color: darkColors.textTertiary,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {leftIcon}
          </span>
        )}
        <input
          className={`input input-${variant} input-${size} ${className}`}
          style={{
            flex: 1,
            border: 'none',
            background: 'transparent',
            outline: 'none',
            fontFamily: typography.fontFamilySans,
            fontSize: sizes[size].fontSize,
            color: variants[variant].color,
            padding: sizes[size].padding,
            height: sizes[size].height,
          }}
          disabled={disabled}
          {...props}
        />
        {rightIcon && (
          <span
            style={{
              paddingRight: spacing[2.5],
              color: darkColors.textTertiary,
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
            }}
          >
            {rightIcon}
          </span>
        )}
      </div>
      {hint && (
        <span
          style={{
            fontSize: typography.fontSizeXs,
            color: error ? darkColors.error : darkColors.textTertiary,
          }}
        >
          {hint}
        </span>
      )}
    </div>
  );
};

export const Textarea: React.FC<InputProps & { rows?: number }> = ({
  rows = 4,
  ...props
}) => {
  return <Input as="textarea" rows={rows} {...props} />;
};

export default Input;
