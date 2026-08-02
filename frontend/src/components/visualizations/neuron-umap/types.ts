export interface NeuronPoint {
  id: string;
  neuronIndex: number;
  layer: number;
  head?: number;
  activation: number;
  embedding: number[];
  tokenActivations?: number[];
  tokens?: string[];
  topToken?: string;
  weightStats?: { mean: number; std: number; min: number; max: number };
}

export interface NeuronUMapTheme {
  bg: string;
  grid: string;
  text: string;
  muted: string;
  cardBg: string;
  border: string;
  green: string;
  red: string;
  gray: string;
  accent: string;
  hover: string;
  tooltipBg: string;
}

export interface Viewport {
  x: number;
  y: number;
  k: number;
}

export interface SelectionRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface NeuronStats {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
  active: number;
  avgActivation: number;
  clusters: number;
}

export interface ProjectedPoint {
  x: number;
  y: number;
}

export interface NeuronUMapProps {
  points: NeuronPoint[];
  tokens?: string[];
  selectedId?: string | null;
  onSelectNeuron?: (id: string | null) => void;
  darkMode: boolean;
  height?: number;
  loading?: boolean;
  title?: string;
}
