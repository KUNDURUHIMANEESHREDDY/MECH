// Feature Flags Configuration for MECH Platform Frontend
// Mirrors backend feature flags from backend/core/features/flags.py

export type FeatureState = 'enabled' | 'disabled' | 'rollout';

export interface FeatureFlag {
  name: string;
  description: string;
  state: FeatureState;
  rolloutPercentage: number; // 0-100
  dependencies: string[];
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface FeatureConfig {
  enabled: boolean;
  state: FeatureState;
  rolloutPercentage: number;
}

export interface ExperimentVariant {
  name: string;
  description: string;
  config: Record<string, unknown>;
  weight: number;
}

export interface Experiment {
  name: string;
  description: string;
  variants: ExperimentVariant[];
  status: 'draft' | 'running' | 'paused' | 'completed' | 'archived';
  assignmentStrategy: 'random' | 'user_id' | 'context' | 'sticky';
  targetingRules: Record<string, unknown>;
}

// Feature flag definitions (mirrored from backend)
export const featureFlags: FeatureFlag[] = [
  {
    name: 'model_integrity_gate',
    description: 'Enforce model integrity gate before experiments',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['validation', 'integrity'],
    metadata: {},
  },
  {
    name: 'behavioral_validation',
    description: 'Run behavioral sanity checks on model load',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['validation', 'behavioral'],
    metadata: {},
  },
  {
    name: 'circuit_verification',
    description: 'Enable ACDC circuit verification pipeline',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['circuits', 'verification'],
    metadata: {},
  },
  {
    name: 'hallucination_pipeline',
    description: 'Enable hallucination competition experiment',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['causal', 'hallucination'],
    metadata: {},
  },
  {
    name: 'semantic_falsification',
    description: 'Enable semantic falsification probes',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['verification', 'falsification'],
    metadata: {},
  },
  {
    name: 'scientific_validation_suite',
    description: 'Enable 5-pillar scientific validation',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['validation', 'scientific'],
    metadata: {},
  },
  {
    name: 'backup_circuit_discovery',
    description: 'Enable redundant backup circuit discovery',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['redundancy', 'backup'],
    metadata: {},
  },
  {
    name: 'async_runtime',
    description: 'Enable async runtime execution',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['runtime', 'async'],
    metadata: {},
  },
  {
    name: 'dynamic_prompt_sampling',
    description: 'Enable dynamic prompt sampling on startup',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['sampling', 'startup'],
    metadata: {},
  },
  {
    name: 'plugin_system',
    description: 'Enable new plugin system for tools',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['plugins', 'tools'],
    metadata: {},
  },
  {
    name: 'capability_definitions',
    description: 'Use new capability definition system',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['capabilities', 'definitions'],
    metadata: {},
  },
  {
    name: 'factory_system',
    description: 'Use factory system for engines/runtimes',
    state: 'enabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['factories', 'engines'],
    metadata: {},
  },
  {
    name: 'unified_registry',
    description: 'Use unified registry for all capabilities',
    state: 'rollout',
    rolloutPercentage: 100,
    dependencies: [],
    tags: ['registry', 'unified'],
    metadata: {},
  },
  {
    name: 'redis_rate_limiting',
    description: 'Use Redis for distributed rate limiting',
    state: 'rollout',
    rolloutPercentage: 100,
    dependencies: ['redis_available'],
    tags: ['rate_limiting', 'redis'],
    metadata: {},
  },
  {
    name: 'new_dispatcher',
    description: 'Use new API dispatcher v2',
    state: 'rollout',
    rolloutPercentage: 50,
    dependencies: [],
    tags: ['api', 'dispatcher'],
    metadata: {},
  },
  {
    name: 'cross_model_analysis',
    description: 'Enable cross-model circuit comparison',
    state: 'rollout',
    rolloutPercentage: 50,
    dependencies: [],
    tags: ['comparative', 'cross_model'],
    metadata: {},
  },
  {
    name: 'live_intervention',
    description: 'Enable live tensor intervention',
    state: 'rollout',
    rolloutPercentage: 75,
    dependencies: [],
    tags: ['causal', 'live'],
    metadata: {},
  },
  {
    name: 'out_of_core_runtime',
    description: 'Enable out-of-core runtime for large models',
    state: 'rollout',
    rolloutPercentage: 25,
    dependencies: [],
    tags: ['runtime', 'memory'],
    metadata: {},
  },
  {
    name: 'multi_precision',
    description: 'Enable multi-precision runtime support',
    state: 'rollout',
    rolloutPercentage: 50,
    dependencies: [],
    tags: ['runtime', 'precision'],
    metadata: {},
  },
  {
    name: 'experiment_tracking',
    description: 'Enable experiment tracking and A/B testing',
    state: 'disabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['experiments', 'tracking'],
    metadata: {},
  },
  {
    name: 'distributed_execution',
    description: 'Enable distributed cluster execution',
    state: 'disabled',
    rolloutPercentage: 0,
    dependencies: [],
    tags: ['distributed', 'cluster'],
    metadata: {},
  },
];

// Built-in experiments (mirrored from backend)
export const experiments: Experiment[] = [
  {
    name: 'dispatcher_v2_rollout',
    description: 'Gradual rollout of new API dispatcher v2',
    variants: [
      { name: 'control', description: 'Legacy dispatcher', config: { dispatcher_version: 'v1' }, weight: 0.5 },
      { name: 'treatment', description: 'New dispatcher v2', config: { dispatcher_version: 'v2' }, weight: 0.5 },
    ],
    status: 'running',
    assignmentStrategy: 'user_id',
    targetingRules: {},
  },
  {
    name: 'runtime_memory_optimization',
    description: 'Test out-of-core runtime vs in-memory for large models',
    variants: [
      { name: 'control', description: 'In-memory runtime', config: { runtime_type: 'in_memory' }, weight: 0.7 },
      { name: 'treatment', description: 'Out-of-core runtime', config: { runtime_type: 'out_of_core' }, weight: 0.3 },
    ],
    status: 'draft',
    assignmentStrategy: 'context',
    targetingRules: { model_size: 'large' },
  },
  {
    name: 'circuit_verification_threshold',
    description: 'Test different ACDC faithfulness thresholds',
    variants: [
      { name: 'control', description: 'Standard threshold (0.05)', config: { acdc_threshold: 0.05 }, weight: 0.5 },
      { name: 'treatment', description: 'Stricter threshold (0.01)', config: { acdc_threshold: 0.01 }, weight: 0.5 },
    ],
    status: 'draft',
    assignmentStrategy: 'user_id',
    targetingRules: {},
  },
  {
    name: 'rate_limit_algorithm',
    description: 'Compare rate limiting algorithms',
    variants: [
      { name: 'control', description: 'Sliding window', config: { rate_limit_algorithm: 'sliding_window' }, weight: 0.5 },
      { name: 'treatment', description: 'Token bucket', config: { rate_limit_algorithm: 'token_bucket' }, weight: 0.5 },
    ],
    status: 'draft',
    assignmentStrategy: 'user_id',
    targetingRules: {},
  },
];

// Feature flag utilities
export const getFeatureFlag = (name: string): FeatureFlag | undefined => {
  return featureFlags.find(f => f.name === name);
};

export const isFeatureEnabled = (name: string, userId?: string): boolean => {
  const flag = getFeatureFlag(name);
  if (!flag) return false;

  if (flag.state === 'enabled') return true;
  if (flag.state === 'disabled') return false;

  // Rollout: deterministic hash-based assignment
  if (flag.state === 'rollout') {
    const identifier = userId || 'anonymous';
    const hash = simpleHash(`${name}:${identifier}`);
    const percentage = (hash % 10000) / 100; // 0.00-99.99
    return percentage < flag.rolloutPercentage;
  }

  return false;
};

export const getExperimentVariant = (experimentName: string, userId?: string, context?: Record<string, unknown>): ExperimentVariant | null => {
  const experiment = experiments.find(e => e.name === experimentName);
  if (!experiment || experiment.status !== 'running') return null;

  // Check targeting rules
  if (experiment.targetingRules && context) {
    for (const [key, value] of Object.entries(experiment.targetingRules)) {
      if (context[key] !== value) return null;
    }
  }

  // Assign variant
  let rand: number;
  if (experiment.assignmentStrategy === 'random') {
    rand = Math.random();
  } else if (experiment.assignmentStrategy === 'user_id') {
    if (!userId) return experiment.variants[0];
    rand = simpleHash(`${experimentName}:${userId}`) / 10000;
  } else if (experiment.assignmentStrategy === 'context') {
    if (!context) return experiment.variants[0];
    const ctxStr = JSON.stringify(context);
    rand = simpleHash(`${experimentName}:${ctxStr}`) / 10000;
  } else { // sticky
    const identifier = userId || (context ? JSON.stringify(context) : 'anonymous');
    rand = simpleHash(`${experimentName}:${identifier}`) / 10000;
  }

  let cumulative = 0;
  for (const variant of experiment.variants) {
    cumulative += variant.weight;
    if (rand < cumulative) return variant;
  }
  return experiment.variants[experiment.variants.length - 1];
};

// Simple hash function for deterministic assignment
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

// Get merged config for a feature (includes experiment overrides)
export const getFeatureConfig = (featureName: string, userId?: string, context?: Record<string, unknown>): Record<string, unknown> => {
  const config: Record<string, unknown> = { feature_enabled: isFeatureEnabled(featureName, userId) };

  // Apply experiment overrides
  for (const experiment of experiments) {
    if (experiment.status === 'running') {
      const variant = getExperimentVariant(experiment.name, userId, context);
      if (variant) {
        Object.assign(config, variant.config);
      }
    }
  }

  return config;
};

// Get all feature configs for a user
export const getAllFeatureConfigs = (userId?: string, context?: Record<string, unknown>): Record<string, FeatureConfig> => {
  const configs: Record<string, FeatureConfig> = {};
  for (const flag of featureFlags) {
    configs[flag.name] = {
      enabled: isFeatureEnabled(flag.name, userId),
      state: flag.state,
      rolloutPercentage: flag.rolloutPercentage,
    };
  }
  return configs;
};

export default {
  featureFlags,
  experiments,
  getFeatureFlag,
  isFeatureEnabled,
  getExperimentVariant,
  getFeatureConfig,
  getAllFeatureConfigs,
};