import React from 'react';
import { spacing } from '../tokens/spacing';

export interface StackProps extends React.HTMLAttributes<HTMLDivElement> {
  direction?: 'row' | 'column';
  gap?: keyof typeof spacing | number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly';
  wrap?: boolean;
}

export const Stack: React.FC<StackProps> = ({
  children,
  direction = 'column',
  gap = 3,
  align = 'start',
  justify = 'start',
  wrap = false,
  className = '',
  ...props
}) => {
  const gapValue = typeof gap === 'number' ? gap : spacing[gap];

  const alignMap = {
    start: 'flex-start',
    center: 'center',
    end: 'flex-end',
    stretch: 'stretch',
  };

  const justifyMap = {
    start: 'flex-start',
    center: 'center',
    end: 'flex-end',
    between: 'space-between',
    around: 'space-around',
    evenly: 'space-evenly',
  };

  return (
    <div
      className={`stack stack-${direction} ${className}`}
      style={{
        display: 'flex',
        flexDirection: direction,
        gap: gapValue,
        alignItems: alignMap[align],
        justifyContent: justifyMap[justify],
        flexWrap: wrap ? 'wrap' : 'nowrap',
      }}
      {...props}
    >
      {children}
    </div>
  );
};

export interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  columns?: number | string;
  gap?: keyof typeof spacing | number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly';
}

export const Grid: React.FC<GridProps> = ({
  children,
  columns = 1,
  gap = 3,
  align = 'stretch',
  justify = 'start',
  className = '',
  ...props
}) => {
  const gapValue = typeof gap === 'number' ? gap : spacing[gap];
  const columnsValue = typeof columns === 'number' ? `repeat(${columns}, 1fr)` : columns;

  const alignMap = {
    start: 'start',
    center: 'center',
    end: 'end',
    stretch: 'stretch',
  };

  const justifyMap = {
    start: 'start',
    center: 'center',
    end: 'end',
    between: 'space-between',
    around: 'space-around',
    evenly: 'space-evenly',
  };

  return (
    <div
      className={`grid ${className}`}
      style={{
        display: 'grid',
        gridTemplateColumns: columnsValue,
        gap: gapValue,
        alignItems: alignMap[align],
        justifyItems: justifyMap[justify],
      }}
      {...props}
    >
      {children}
    </div>
  );
};

export interface BoxProps extends React.HTMLAttributes<HTMLDivElement> {
  padding?: keyof typeof spacing | number;
  margin?: keyof typeof spacing | number;
  width?: string | number;
  height?: string | number;
  display?: 'block' | 'inline-block' | 'flex' | 'inline-flex' | 'grid' | 'inline-grid';
  overflow?: 'visible' | 'hidden' | 'auto' | 'scroll';
}

export const Box: React.FC<BoxProps> = ({
  children,
  padding,
  margin,
  width,
  height,
  display = 'block',
  overflow,
  className = '',
  ...props
}) => {
  const paddingValue = padding ? (typeof padding === 'number' ? padding : spacing[padding]) : undefined;
  const marginValue = margin ? (typeof margin === 'number' ? margin : spacing[margin]) : undefined;

  return (
    <div
      className={`box ${className}`}
      style={{
        display,
        padding: paddingValue,
        margin: marginValue,
        width,
        height,
        overflow,
      }}
      {...props}
    >
      {children}
    </div>
  );
};

export interface DividerProps extends React.HTMLAttributes<HTMLDivElement> {
  orientation?: 'horizontal' | 'vertical';
  thickness?: number;
  color?: string;
}

export const Divider: React.FC<DividerProps> = ({
  orientation = 'horizontal',
  thickness = 1,
  color = 'var(--color-border-primary)',
  className = '',
  ...props
}) => {
  return (
    <div
      className={`divider divider-${orientation} ${className}`}
      style={{
        background: color,
        height: orientation === 'horizontal' ? thickness : 'auto',
        width: orientation === 'vertical' ? thickness : '100%',
        margin: orientation === 'horizontal' ? `${spacing[3]} 0` : `0 ${spacing[3]}`,
      }}
      {...props}
    />
  );
};

export interface SpacerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: keyof typeof spacing | number;
}

export const Spacer: React.FC<SpacerProps> = ({
  size = 3,
  className = '',
  ...props
}) => {
  const sizeValue = typeof size === 'number' ? size : spacing[size];

  return (
    <div
      className={`spacer ${className}`}
      style={{ width: sizeValue, height: sizeValue }}
      {...props}
    />
  );
};

export default Stack;
