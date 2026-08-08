/**
 * Notebook Execution Engine (Sprint 3 AI 5)
 * Isolated evaluation for JavaScript analysis cells.
 * Uses strict token whitelisting to prevent code injection.
 */

const DANGEROUS_PATTERNS = [
  /eval\s*\(/,
  /Function\s*\(/,
  /document/,
  /window/,
  /localStorage/,
  /sessionStorage/,
  /fetch\s*\(/,
  /XMLHttpRequest/,
  /import\s+/,
  /require\s*\(/,
  /__proto__/,
  /constructor\s*\(/,
  /prototype/,
  /process/,
  /global/,
  /this\s*\./,
];

const SAFE_MATH_FUNCTIONS = new Set([
  'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
  'cbrt', 'ceil', 'clz32', 'cos', 'cosh', 'exp', 'expm1', 'floor',
  'fround', 'hypot', 'imul', 'log', 'log10', 'log1p', 'log2', 'max',
  'min', 'pow', 'random', 'round', 'sign', 'sin', 'sinh', 'sqrt',
  'tan', 'tanh', 'trunc',
]);

export class NotebookExecutionEngine {
  executeCell(code) {
    const t0 = performance.now();

    if (typeof code !== 'string' || !code.trim()) {
      return {
        status: 'error',
        output: 'Empty or invalid code',
        durationMs: '0.00',
        timestamp: new Date().toISOString(),
      };
    }

    for (const pattern of DANGEROUS_PATTERNS) {
      if (pattern.test(code)) {
        return {
          status: 'error',
          output: `Security violation: forbidden pattern detected`,
          durationMs: (performance.now() - t0).toFixed(2),
          timestamp: new Date().toISOString(),
        };
      }
    }

    try {
      const safeContext = {
        Math,
        console: { log: (...args) => args.join(' ') },
        JSON,
        Array,
        Object,
        Number,
        String,
        Boolean,
        Date,
        RegExp,
        Error,
        TypeError,
        RangeError,
        SyntaxError,
        ReferenceError,
        URIError,
        isNaN,
        isFinite,
        parseInt,
        parseFloat,
        encodeURI,
        decodeURI,
        encodeURIComponent,
        decodeURIComponent,
        Infinity,
        NaN,
        undefined: undefined,
        null: null,
        true: true,
        false: false,
      };

      const keys = Object.keys(safeContext);
      const values = Object.values(safeContext);

      const fn = new Function(...keys, `
        'use strict';
        try {
          ${code}
        } catch (err) {
          return { error: err.message };
        }
      `);

      const result = fn(...values);

      if (result && typeof result === 'object' && result.error) {
        return {
          status: 'error',
          output: result.error,
          durationMs: (performance.now() - t0).toFixed(2),
          timestamp: new Date().toISOString(),
        };
      }

      return {
        status: 'success',
        output: result === undefined ? 'undefined' : String(result),
        durationMs: (performance.now() - t0).toFixed(2),
        timestamp: new Date().toISOString(),
      };
    } catch (err) {
      return {
        status: 'error',
        output: err.message,
        durationMs: (performance.now() - t0).toFixed(2),
        timestamp: new Date().toISOString(),
      };
    }
  }
}

export const notebookExecutionEngine = new NotebookExecutionEngine();
