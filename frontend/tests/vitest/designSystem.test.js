import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createApp, h } from 'vue';

import UiPanel from '../../src/components/ui/UiPanel.vue';
import UiButton from '../../src/components/ui/UiButton.vue';
import UiStatusPill from '../../src/components/ui/UiStatusPill.vue';
import UiEmptyState from '../../src/components/ui/UiEmptyState.vue';
import UiErrorState from '../../src/components/ui/UiErrorState.vue';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const srcRoot = path.join(frontendRoot, 'src');

// ---------------------------------------------------------------- helpers
function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

const srcFiles = walk(srcRoot);
const codeFiles = srcFiles.filter((f) => ['.vue', '.ts', '.js', '.html'].includes(path.extname(f)));
const corpus = codeFiles.map((f) => readFileSync(f, 'utf8')).join('\n');

/** Comments are blanked out (same length) so braces inside them cannot confuse the parser. */
function maskComments(css) {
  const out = css.split('');
  let i = 0;
  while (i < css.length) {
    if (css[i] === '/' && css[i + 1] === '*') {
      const end = css.indexOf('*/', i + 2);
      const stop = end === -1 ? css.length : end + 2;
      for (let j = i; j < stop; j++) if (out[j] !== '\n') out[j] = ' ';
      i = stop;
      continue;
    }
    i++;
  }
  return out.join('');
}

/** Top-level nodes of a stylesheet: {prelude, inner, isAt}. */
function topLevel(css) {
  const m = maskComments(css);
  const nodes = [];
  let i = 0;
  let selStart = 0;
  while (i < m.length) {
    if (m[i] === '{') {
      const prelude = m.slice(selStart, i).trim();
      const bodyStart = i + 1;
      let d = 1;
      let j = bodyStart;
      for (; j < m.length && d > 0; j++) {
        if (m[j] === '{') d++;
        else if (m[j] === '}') d--;
      }
      nodes.push({ prelude, inner: css.slice(bodyStart, j - 1), isAt: /^\s*@/.test(prelude) });
      i = j;
      selStart = j;
      continue;
    }
    i++;
  }
  return nodes;
}

function splitSelectors(sel) {
  const out = [];
  let depth = 0;
  let cur = '';
  for (const ch of sel) {
    if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') depth--;
    if (ch === ',' && depth === 0) { out.push(cur.trim()); cur = ''; }
    else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** Classes a selector REQUIRES. Classes inside :not(...) are optional. */
function requiredClasses(sel) {
  const cleaned = sel.replace(/:not\([^)]*\)/g, ' ');
  const names = new Set();
  const re = /\.((?:[A-Za-z0-9_-]|\\.)+)/g;
  let m;
  while ((m = re.exec(cleaned))) names.add(m[1].replace(/\\(.)/g, '$1'));
  return [...names];
}

/** Every style rule in a stylesheet, with the classes each selector needs. */
function styleRules(css) {
  const rules = [];
  for (const node of topLevel(css)) {
    if (node.isAt) continue;
    for (const sel of splitSelectors(node.prelude)) {
      rules.push({ selector: sel, classes: requiredClasses(sel) });
    }
  }
  return rules;
}

// A class is referenced when the source mentions it literally, or when a
// `:class` binding builds it at runtime (`status-${state}`, `${tone}-badge`).
function classBindings(text) {
  const out = [];
  const bindingRe = /:class\s*=\s*"([^"]*)"/g;
  let m;
  while ((m = bindingRe.exec(text))) {
    const tplRe = /`([^`$]*)\$\{[^`]*`/g;
    let t;
    while ((t = tplRe.exec(m[1]))) {
      const raw = t[0];
      const close = raw.lastIndexOf('}');
      out.push({ prefix: t[1], suffix: close === -1 ? '' : raw.slice(close + 1, -1) });
    }
  }
  return out;
}

const bindings = classBindings(corpus);

function isReferenced(name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (new RegExp(`(?<![\\w-])${escaped}(?![\\w-])`).test(corpus)) return true;
  return bindings.some(({ prefix, suffix }) =>
    (prefix === '' || name.startsWith(prefix)) && (suffix === '' || name.endsWith(suffix)));
}

const stylesCss = readFileSync(path.join(srcRoot, 'styles.css'), 'utf8');
const desktopCss = readFileSync(path.join(srcRoot, 'desktop.css'), 'utf8');

// ------------------------------------------------- 1. one token source
describe('design system: one token source', () => {
  it('declares the tokens exactly once, at the top level of src/styles.css', () => {
    const owners = [];
    for (const file of srcFiles.filter((f) => f.endsWith('.css'))) {
      const roots = topLevel(readFileSync(file, 'utf8')).filter((n) => /(^|[\s;]):root\s*$/.test(n.prelude));
      if (roots.length) owners.push({ file: path.relative(frontendRoot, file), count: roots.length });
    }
    expect(owners).toEqual([{ file: path.join('src', 'styles.css'), count: 1 }]);
  });

  it('keeps src/desktop.css free of token declarations', () => {
    const declarations = [...maskComments(desktopCss).matchAll(/(--[A-Za-z0-9_-]+)\s*:/g)].map((m) => m[1]);
    expect(declarations).toEqual([]);
  });

  it('has removed the dead design system', () => {
    expect(existsSync(path.join(srcRoot, 'design', 'styles', 'global.css'))).toBe(false);
    for (const gone of ['typography.ts', 'spacing.ts', 'radii.ts', 'elevation.ts', 'index.ts']) {
      expect(existsSync(path.join(srcRoot, 'design', 'tokens', gone))).toBe(false);
    }
    const tokenFiles = readdirSync(path.join(srcRoot, 'design', 'tokens')).filter((f) => f.endsWith('.ts'));
    expect(tokenFiles).toEqual(['colors.ts']);
  });

  it('has no import of a deleted token path', () => {
    const deleted = ['design/styles/global.css', 'design/tokens/typography', 'design/tokens/spacing', 'design/tokens/radii', 'design/tokens/elevation', 'design/tokens/index'];
    const offenders = codeFiles.filter((f) => {
      const text = readFileSync(f, 'utf8');
      return deleted.some((d) => text.includes(d));
    });
    expect(offenders.map((f) => path.relative(frontendRoot, f))).toEqual([]);
  });

  it('keeps the runtime colour accessor in sync with the stylesheet', () => {
    const rootBlock = stylesCss.slice(stylesCss.indexOf(':root {'), stylesCss.indexOf('\n}', stylesCss.indexOf(':root {')));
    const cssTokens = Object.fromEntries(
      [...rootBlock.matchAll(/(--[A-Za-z0-9_-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]),
    );
    const accessor = readFileSync(path.join(srcRoot, 'design', 'tokens', 'colors.ts'), 'utf8');
    const fallbackBlock = accessor.slice(accessor.indexOf('const FALLBACK'), accessor.indexOf('};', accessor.indexOf('const FALLBACK')));
    const fallback = Object.fromEntries(
      [...fallbackBlock.matchAll(/([A-Za-z0-9_-]+)\s*:\s*'([^']+)'/g)].map((m) => [m[1], m[2]]),
    );
    expect(Object.keys(fallback).length).toBeGreaterThan(0);
    for (const [name, value] of Object.entries(fallback)) {
      expect(cssTokens[`--${name}`], `token --${name} must exist in styles.css`).toBeDefined();
      expect(value, `fallback for --${name} must equal the stylesheet value`).toBe(cssTokens[`--${name}`]);
    }
  });
});

// ------------------------------------------------- 2. no dead class tokens
describe('design system: no dead class tokens', () => {
  for (const [label, css] of [['src/styles.css', stylesCss], ['src/desktop.css', desktopCss]]) {
    it(`${label} only defines classes that something references`, () => {
      const dead = [];
      for (const rule of styleRules(css)) {
        for (const cls of rule.classes) {
          if (!isReferenced(cls)) dead.push(`${cls} (${rule.selector})`);
        }
      }
      expect(dead).toEqual([]);
    });
  }
});

// ------------------------------------------------- 3. the shared primitives
function render(component, props = {}, slots = {}) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const app = createApp({ render: () => h(component, props, slots) });
  app.mount(host);
  return { app, host, html: host.innerHTML };
}

describe('design system: shared primitives render', () => {
  it('UiPanel renders its title, body and actions', () => {
    const { app, host } = render(UiPanel, { title: 'Models', subtitle: 'Loaded from the backend' }, {
      default: () => 'panel body',
      actions: () => h('button', { class: 'act' }, 'Refresh'),
    });
    expect(host.querySelector('.ui-panel')).not.toBeNull();
    expect(host.querySelector('.ui-panel__title').textContent).toBe('Models');
    expect(host.querySelector('.ui-panel__subtitle').textContent).toBe('Loaded from the backend');
    expect(host.querySelector('.ui-panel__body').textContent).toContain('panel body');
    expect(host.querySelector('.ui-panel__actions .act').textContent).toBe('Refresh');
    app.unmount();
  });

  it('UiButton renders its variant and disables itself while busy', () => {
    const { app, host } = render(UiButton, { variant: 'primary', busy: true }, { default: () => 'Run' });
    const button = host.querySelector('button.ui-button');
    expect(button.classList.contains('ui-button--primary')).toBe(true);
    expect(button.getAttribute('aria-busy')).toBe('true');
    expect(button.disabled).toBe(true);
    expect(button.textContent).toContain('Run');
    app.unmount();
  });

  it('UiButton is clickable and not busy by default', () => {
    let clicks = 0;
    const { app, host } = render(UiButton, { onClick: () => { clicks += 1; } }, { default: () => 'Run' });
    const button = host.querySelector('button.ui-button');
    expect(button.disabled).toBe(false);
    expect(button.getAttribute('aria-busy')).toBeNull();
    button.click();
    expect(clicks).toBe(1);
    app.unmount();
  });

  it('UiStatusPill renders its tone and label', () => {
    const { app, host } = render(UiStatusPill, { tone: 'success', label: 'COMPLETED' });
    const pill = host.querySelector('.ui-status-pill');
    expect(pill.classList.contains('ui-status-pill--success')).toBe(true);
    expect(pill.textContent).toContain('COMPLETED');
    app.unmount();
  });

  it('UiEmptyState renders the given title, never an invented one', () => {
    const { app, host } = render(UiEmptyState, { title: 'No experiments yet', message: 'Create one to get started' });
    expect(host.querySelector('.ui-empty-state__title').textContent).toBe('No experiments yet');
    expect(host.querySelector('.ui-empty-state__message').textContent).toBe('Create one to get started');
    app.unmount();
  });

  it('UiErrorState shows the caller-supplied reason', () => {
    const { app, host } = render(UiErrorState, { message: 'Backend returned 503' });
    expect(host.querySelector('.ui-error-state__message').textContent).toBe('Backend returned 503');
    expect(host.querySelector('.ui-error-state').getAttribute('role')).toBe('alert');
    app.unmount();
  });

  it('styles every primitive from the token stylesheet', () => {
    const classes = new Set();
    for (const rule of styleRules(stylesCss)) {
      for (const cls of rule.classes) {
        if (cls.startsWith('ui-')) classes.add(cls);
      }
    }
    for (const cls of ['ui-panel', 'ui-button', 'ui-status-pill', 'ui-empty-state', 'ui-error-state']) {
      expect(classes, `${cls} must be styled in src/styles.css`).toContain(cls);
    }
  });
});
