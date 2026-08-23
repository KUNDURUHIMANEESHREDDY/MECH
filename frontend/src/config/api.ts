// API Configuration for MECH Platform Frontend

export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retryPolicy: RetryPolicy;
  headers: Record<string, string>;
}

export interface RetryPolicy {
  maxRetries: number;
  retryDelay: number; // ms
  retryStatusCodes: number[];
  exponentialBackoff: boolean;
}

export interface EndpointConfig {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  requiresAuth: boolean;
  timeout?: number;
}

// Environment detection - works with Vite's import.meta.env
const isDevelopment = typeof import.meta !== 'undefined' && import.meta.env?.DEV === true;
const isProduction = typeof import.meta !== 'undefined' && import.meta.env?.PROD === true;

// Get environment variable safely
const getEnv = (key: string): string | undefined => {
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    return import.meta.env[key];
  }
  return undefined;
};

// API base URL - can be overridden via VITE_API_URL env var
const getBaseUrl = (): string => {
  const envUrl = getEnv('VITE_API_URL');
  if (envUrl) {
    return envUrl;
  }
  if (isDevelopment) {
    return 'http://127.0.0.1:8000';
  }
  // In production (Electron), the backend runs on the same machine
  return 'http://127.0.0.1:8000';
};

// Default API configuration
export const apiConfig: ApiConfig = {
  baseUrl: getBaseUrl(),
  timeout: 30000, // 30 seconds
  retryPolicy: {
    maxRetries: 3,
    retryDelay: 1000, // 1 second base delay
    retryStatusCodes: [408, 429, 500, 502, 503, 504],
    exponentialBackoff: true,
  },
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
};

// API endpoints
export const endpoints: Record<string, EndpointConfig> = {
  // Health & Root
  health: { path: '/health', method: 'GET', requiresAuth: false },
  root: { path: '/', method: 'GET', requiresAuth: false },

  // Authentication
  authStatus: { path: '/api/auth/status', method: 'GET', requiresAuth: true },

  // Tools & Capabilities
  listTools: { path: '/api/tools', method: 'GET', requiresAuth: true },
  executeTool: { path: '/api/tools/execute', method: 'POST', requiresAuth: true },
  getToolManifest: { path: '/api/tools/:name/manifest', method: 'GET', requiresAuth: true },

  // Capabilities
  listCapabilities: { path: '/api/capabilities', method: 'GET', requiresAuth: true },
  checkCapability: { path: '/api/capabilities/:name', method: 'GET', requiresAuth: true },

  // Models
  listModels: { path: '/api/models', method: 'GET', requiresAuth: true },
  getModelInfo: { path: '/api/models/:name', method: 'GET', requiresAuth: true },
  loadModel: { path: '/api/models/:name/load', method: 'POST', requiresAuth: true },

  // Experiments
  listExperiments: { path: '/api/experiments', method: 'GET', requiresAuth: true },
  createExperiment: { path: '/api/experiments', method: 'POST', requiresAuth: true },
  getExperiment: { path: '/api/experiments/:id', method: 'GET', requiresAuth: true },
  runExperiment: { path: '/api/experiments/:id/run', method: 'POST', requiresAuth: true },

  // Circuits
  discoverCircuit: { path: '/api/circuits/discover', method: 'POST', requiresAuth: true },
  validateCircuit: { path: '/api/circuits/validate', method: 'POST', requiresAuth: true },
  listCircuits: { path: '/api/circuits', method: 'GET', requiresAuth: true },

  // SAE Features
  extractSaeFeatures: { path: '/api/sae/features', method: 'POST', requiresAuth: true },

  // Activation Patching
  runActivationPatching: { path: '/api/patching/activation', method: 'POST', requiresAuth: true },
  runPathPatching: { path: '/api/patching/path', method: 'POST', requiresAuth: true },

  // Logit Lens
  runLogitLens: { path: '/api/lens/logit', method: 'POST', requiresAuth: true },

  // Live Intervention
  runLiveIntervention: { path: '/api/intervention/live', method: 'POST', requiresAuth: true },

  // Hallucination Pipeline
  runHallucinationExperiment: { path: '/api/hallucination/experiment', method: 'POST', requiresAuth: true },

  // Semantic Falsification
  runSemanticFalsification: { path: '/api/falsification/semantic', method: 'POST', requiresAuth: true },

  // Scientific Validation
  runScientificValidation: { path: '/api/validation/scientific', method: 'POST', requiresAuth: true },

  // Cross Model
  runCrossModelSweep: { path: '/api/cross-model/sweep', method: 'POST', requiresAuth: true },

  // Backup Circuits
  discoverBackupCircuits: { path: '/api/redundancy/backup', method: 'POST', requiresAuth: true },

  // Feature Flags
  getFeatureFlags: { path: '/api/features', method: 'GET', requiresAuth: true },
  getFeatureConfig: { path: '/api/features/:name', method: 'GET', requiresAuth: true },

  // Storage
  listArtifacts: { path: '/api/storage/artifacts', method: 'GET', requiresAuth: true },
  getArtifact: { path: '/api/storage/artifacts/:id', method: 'GET', requiresAuth: true },
  uploadArtifact: { path: '/api/storage/artifacts', method: 'POST', requiresAuth: true },

  // Runtime
  getRuntimeInfo: { path: '/api/runtime/info', method: 'GET', requiresAuth: true },
  createRuntime: { path: '/api/runtime/create', method: 'POST', requiresAuth: true },

  // WebSocket
  wsEndpoint: { path: '/ws', method: 'GET', requiresAuth: true },
};

// Helper to build full URL
export const buildUrl = (endpointKey: string, params?: Record<string, string>): string => {
  const endpoint = endpoints[endpointKey];
  if (!endpoint) {
    throw new Error(`Unknown endpoint: ${endpointKey}`);
  }
  let path = endpoint.path;
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      path = path.replace(`:${key}`, value);
    }
  }
  return `${apiConfig.baseUrl}${path}`;
};

// Helper to get headers with auth
export const getAuthHeaders = (apiKey?: string): Record<string, string> => {
  const headers = { ...apiConfig.headers };
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
};

// Default API key for development
export const DEFAULT_API_KEY = 'mech_dev_key_default';

export default apiConfig;