import React from 'react';
import { darkColors } from '../tokens/colors';
import { spacing } from '../tokens/spacing';
import { radii } from '../tokens/radii';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'bordered' | 'glass';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hoverable?: boolean;
  clickable?: boolean;
  selected?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'default',
  padding = 'md',
  hoverable = false,
  clickable = false,
  selected = false,
  className = '',
  ...props
}) => {
  const paddingMap = {
    none: 0,
    sm: spacing[3],
    md: spacing[4],
    lg: spacing[6],
  };

  const variants = {
    default: {
      background: darkColors.surfacePrimary,
      border: `1px solid ${darkColors.borderPrimary}`,
    },
    elevated: {
      background: darkColors.surfaceElevated,
      border: `1px solid ${darkColors.borderPrimary}`,
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    },
    bordered: {
      background: darkColors.surfacePrimary,
      border: `1px solid ${darkColors.borderSecondary}`,
    },
    glass: {
      background: 'rgba(30, 41, 59, 0.8)',
      border: `1px solid ${darkColors.borderPrimary}`,
      backdropFilter: 'blur(10px)',
    },
  };

  const baseStyles: React.CSSProperties = {
    borderRadius: radii.card,
    transition: 'all 150ms ease',
    cursor: clickable ? 'pointer' : 'default',
    padding: paddingMap[padding],
    ...variants[variant],
  };

  const hoverStyles: React.CSSProperties = hoverable
    ? {
        ':hover': {
          borderColor: darkColors.borderHover,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          transform: 'translateY(-2px)',
        },
      }
    : {};

  const selectedStyles: React.CSSProperties = selected
    ? {
        borderColor: darkColors.primary,
        boxShadow: '0 0 0 2px rgba(79, 70, 229, 0.3)',
      }
    : {};

  return (
    <div
      className={`card card-${variant} ${className}`}
      style={{
        ...baseStyles,
        ...hoverStyles,
        ...selectedStyles,
      }}
      {...props}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = '',
  ...props
}) => (
  <div
    className={`card-header ${className}`}
    style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingBottom: spacing[3],
      marginBottom: spacing[3],
      borderBottom: `1px solid ${darkColors.borderPrimary}`,
    }}
    {...props}
  >
    {children}
  </div>
);

export const CardBody: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = '',
  ...props
}) => (
  <div className={`card-body ${className}`} {...props}>
    {children}
  </div>
);

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = '',
  ...props
}) => (
  <div
    className={`card-footer ${className}`}
    style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'flex-end',
      gap: spacing[2],
      paddingTop: spacing[3],
      marginTop: spacing[3],
      borderTop: `1px solid ${darkColors.borderPrimary}`,
    }}
    {...props}
  >
    {children}
  </div>
);

export default Card;
