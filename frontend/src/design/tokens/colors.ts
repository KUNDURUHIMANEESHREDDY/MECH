/**
 * MECH Platform - runtime access to the design tokens.
 *
 * The design tokens live in exactly one place: the `:root` block of
 * `src/styles.css`. This module does NOT declare a second palette - it reads
 * the CSS custom properties off the document root at call time, so a token
 * change in the stylesheet is immediately visible to the chart views and to
 * the HTML report exporter.
 *
 * The previous version of this file was a second, independent palette
 * (`darkColors` / `lightColors` / `colorTokens`, 507 lines) that disagreed
 * with the stylesheet: it exported a dark navy canvas while the application
 * shell renders light. Only three of its 57 keys were ever read.
 *
 * FALLBACK is used when there is no document (node scripts, unit tests).
 * `tests/vitest/designSystem.test.js` asserts that every fallback equals the
 * value declared in `src/styles.css`, so the two cannot drift apart.
 */

const FALLBACK: Record<string, string> = {
  primary: '#2563eb',
  canvas: '#f4f6f8',
  ink: '#182230',
};

function readToken(name: string): string {
  const fallback = FALLBACK[name] ?? '';
  if (typeof document === 'undefined' || !document.documentElement) return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
  return value || fallback;
}

export const colors = {
  /** Accent colour, used for chart series and heatmap ramps. */
  get primary(): string {
    return readToken('primary');
  },
  /** Page background, used for the exported HTML report. */
  get canvas(): string {
    return readToken('canvas');
  },
  /** Primary text colour, used for the exported HTML report. */
  get ink(): string {
    return readToken('ink');
  },
};

export default colors;
