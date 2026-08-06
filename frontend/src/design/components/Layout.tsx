import React from 'react';
import { colors, spacing } from '../../design/tokens';

export interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  maxWidth?: 'narrow' | 'medium' | 'wide' | 'full';
}

const maxWidthMap = {
  narrow: '980px',
  medium: '1200px',
  wide: '1440px',
  full: '100%',
};

export const Container: React.FC<ContainerProps> = ({
  maxWidth = 'wide',
  children,
  style,
  ...rest
}) => {
  return (
    <div
      style={{
        maxWidth: maxWidthMap[maxWidth],
        margin: '0 auto',
        paddingLeft: spacing.lg,
        paddingRight: spacing.lg,
        width: '100%',
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  );
};

export interface SectionProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'light' | 'parchment' | 'dark' | 'dark-2' | 'dark-3';
}

const sectionBgMap = {
  light: colors.canvas,
  parchment: colors.canvasParchment,
  dark: colors.canvasParchment,
  'dark-2': colors.surfacePearl,
  'dark-3': colors.canvas,
};

export const Section: React.FC<SectionProps> = ({
  variant = 'light',
  children,
  style,
  ...rest
}) => {
  return (
    <section
      style={{
        backgroundColor: sectionBgMap[variant],
        color: colors.ink,
        paddingTop: spacing.section,
        paddingBottom: spacing.section,
        width: '100%',
        ...style,
      }}
      {...rest}
    >
      {children}
    </section>
  );
};
