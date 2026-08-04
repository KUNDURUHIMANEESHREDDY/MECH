import React from 'react';
import { colors, radii, spacing, typography } from '../../design/tokens';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  style,
  ...rest
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: spacing.xs }}>
      {label && (
        <label
          style={{
            fontFamily: typography.body.fontFamily,
            fontSize: typography.caption.fontSize,
            fontWeight: typography.caption.fontWeight,
            color: colors.ink,
          }}
        >
          {label}
        </label>
      )}
      <input
        style={{
          fontFamily: typography.body.fontFamily,
          fontSize: typography.body.fontSize,
          color: colors.ink,
          backgroundColor: colors.canvas,
          border: `1px solid rgba(0, 0, 0, 0.08)`,
          borderRadius: radii.pill,
          padding: `${spacing.sm} ${spacing.lg}`,
          height: '44px',
          outline: 'none',
          width: '100%',
          ...style,
        }}
        {...rest}
      />
      {error && (
        <span style={{ fontSize: typography.finePrint.fontSize, color: colors.inkMuted48 }}>
          {error}
        </span>
      )}
    </div>
  );
};
