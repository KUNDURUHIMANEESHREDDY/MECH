/**
 * Static analyser for the capture script -- the thing that makes its guarantees
 * checkable instead of merely claimed.
 *
 * Reads a `.cjs` file and reports violations of the rules below as JSON on
 * stdout. `tests/pytest/test_capture_screenshots.py` runs it over the real
 * script and over deliberately-broken fixtures; the fixtures matter, because a
 * checker that reports nothing is indistinguishable from a checker that is
 * broken.
 *
 * Why an AST and not grep
 * -----------------------
 * Every rule below is about structure -- is there an assertion between two
 * statements, does a `catch` swallow an `await`, is a `close()` inside a
 * `finally`. `grep` cannot see any of that, and a naive regex is worse than
 * useless: it flags the prose in the file's own comments, which is how the
 * earlier Python checks in this repository ended up quoting their own defect in
 * a test failure. A parser sees code, not commentary.
 *
 * Rules
 * -----
 *   no-fixed-sleep        `page.waitForTimeout(...)` used as readiness.
 *   no-positional-selector  `.nth(n)` / `:nth-match` -- addresses by position.
 *   no-swallowed-error    a `catch` that contains an awaited interaction.
 *   no-unasserted-shot    `shot()` with no assertion before it in the function.
 *   no-text-selector      an interaction located by visible text (`has-text`).
 *   no-bare-element-query an interaction located by a tag name with no class
 *                         or test id, e.g. `input`, `button`, `select`.
 *   browser-closed        no `close()` inside a `finally` anywhere.
 *   fails-on-assertion    no non-zero `exitCode` assignment guarded by failure.
 *   diagnostics-attached  console / pageerror / requestfailed / response not
 *                         all subscribed.
 *
 * Usage:
 *   node scripts/capture_lint.cjs <file.cjs> [--pretty]
 */
const fs = require("fs");
const path = require("path");

const parser = require(path.join(
  __dirname, "..", "frontend", "node_modules", "@babel", "parser",
));

// ── Node helpers ────────────────────────────────────────────────────────

const CALLEE = (node) => {
  const c = node.callee;
  if (!c) return "";
  if (c.type === "Identifier") return c.name;
  if (c.type === "MemberExpression") {
    const object = c.object.type === "MemberExpression"
      ? `${CALLEE(c.object)}.${c.object.property.name || c.object.property.value}`
      : (c.object.name || c.object.value || "");
    return `${object}.${c.property.name || c.property.value}`;
  }
  if (c.type === "CallExpression") return `${CALLEE(c.callee)}()`;
  return "";
};

const memberProperty = (node) =>
  node && node.type === "MemberExpression" && !node.computed
    ? String(node.property.name || node.property.value)
    : "";

const contains = (node, predicate) => {
  let found = false;
  const walk = (n) => {
    if (found || !n || typeof n !== "object") return;
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (n.type && predicate(n)) { found = true; return; }
    for (const key of Object.keys(n)) {
      if (key === "loc" || key === "leadingComments" || key === "trailingComments") continue;
      walk(n[key]);
    }
  };
  walk(node);
  return found;
};

const lineOf = (node) => (node.loc && node.loc.start.line) || 0;

/**
 * The string value of a literal node, or null.
 *
 * Babel's AST uses `StringLiteral`; ESTree's uses `Literal`. This linter was
 * first written with `node.type === "Literal"` only, which made three of the
 * rules below silently match nothing -- a checker that passes because it is
 * broken is worse than no checker, so both shapes are accepted here.
 */
const stringValue = (node) => {
  if (!node) return null;
  if (node.type === "StringLiteral") return node.value;
  if (node.type === "Literal" && typeof node.value === "string") return node.value;
  return null;
};

/** An assertion: one of the `require*` helpers, or a wrapped API call. */
const isAssertion = (node) => {
  const name = CALLEE(node);
  return /^require(Visible|Absent|Text)$/.test(name) || name === "withApiWait";
};

/** A deliberate interaction with the page. */
const isInteraction = (node) => {
  const name = CALLEE(node);
  return /\.(fill|click|selectOption|type|press|check|hover|tap)$/.test(name);
};

const isShot = (node) => CALLEE(node) === "shot";

// ── Rules ───────────────────────────────────────────────────────────────

function checkNoFixedSleep(ast, report) {
  find(ast, (n) => n.type === "CallExpression"
    && memberProperty(n.callee) === "waitForTimeout")
    .forEach((n) => report("no-fixed-sleep", n,
      "waitForTimeout guesses at readiness; wait for the state instead"));
}

function checkNoPositionalSelector(ast, report) {
  find(ast, (n) => n.type === "CallExpression"
    && memberProperty(n.callee) === "nth")
    .forEach((n) => report("no-positional-selector", n,
      ".nth(n) addresses by position; a reordered form silently changes target"));
  find(ast, (n) => {
    const value = stringValue(n);
    return value !== null && /:nth-(match|child)\(/.test(value);
  })
    .forEach((n) => report("no-positional-selector", n,
      "a positional CSS selector"));
}

function checkNoTextSelector(ast, report) {
  find(ast, (n) => n.type === "CallExpression"
    && n.arguments.some((a) => (stringValue(a) || "").includes("has-text")))
    .forEach((n) => report("no-text-selector", n,
      "an element located by its visible text; use data-testid"));
}

function checkNoBareElementQuery(ast, report) {
  find(ast, (n) => n.type === "CallExpression"
    && memberProperty(n.callee) === "locator"
    && n.arguments.some((a) => /^\s*(input|button|select|textarea|a|div|span|form)\s*$/
      .test(stringValue(a) || "")))
    .forEach((n) => report("no-bare-element-query", n,
      "a bare tag selector matches whichever one appears first"));
}

function checkNoSwallowedError(ast, report) {
  // Both halves matter: the failing call lives in the `try` block, the swallow
  // is the handler. Checking only the handler body finds nothing in the common
  // `try { await ...fill(...) } catch {}` shape, which is exactly the shape
  // this rule exists to catch.
  find(ast, (n) => n.type === "TryStatement" && n.handler
    && n.handler.type === "CatchClause" && !n.handler.param)
    .forEach((statement) => {
      const swallowed = find(statement, (n) =>
        n !== statement
        && (n.type === "AwaitExpression" || n.type === "CallExpression")
        && (isInteraction(n) || isAssertion(n)));
      if (swallowed.length) {
        report("no-swallowed-error", statement,
          "a bare catch swallows the failure of an interaction or an assertion, "
          + "so a failed step is screenshotted as a success");
      }
    });
  // A `catch (e) { ... }` that returns nothing and rethrows nothing is the same
  // defect with extra decoration.
  find(ast, (n) => n.type === "CatchClause" && n.param)
    .forEach((clause) => {
      const rethrows = contains(clause.body, (c) =>
        c.type === "ThrowStatement" || /^(rethrow|raise|fail)$/.test(String(c.name || "")));
      const handles = contains(clause.body, (c) => c.type === "CallExpression"
        && /\.(fail|rethrow|raise|rethrow)$/.test(CALLEE(c)));
      if (!rethrows && !handles && find(clause.body, isInteraction).length) {
        report("no-swallowed-error", clause,
          "an interaction is caught without being rethrown or reported");
      }
    });
}

function checkNoUnassertedShot(ast, report) {
  find(ast, (n) => (n.type === "FunctionDeclaration" || n.type === "FunctionExpression"
    || n.type === "ArrowFunctionExpression") && n.body && n.body.type === "BlockStatement")
    .forEach((fn) => {
      // Walk this function's own statements, not nested closures: an assertion
      // inside a helper the shot calls is legitimate, one inside a sibling
      // callback is not.
      fn.body.body.forEach((stmt, index) => {
        if (!contains(stmt, isShot)) return;
        const before = fn.body.body.slice(0, index);
        if (before.some((s) => contains(s, isAssertion))) return;
        report("no-unasserted-shot", stmt,
          `a screenshot is taken in this function with no assertion before it; `
          + `a screenshot of an error page is worse than no screenshot`);
      });
    });
}

/**
 * A readiness wait whose timeout is discarded.
 *
 * `page.waitForResponse(...).catch(() => null)` is the shape of the defect: the
 * script asked whether the UI had become ready, was told no, and continued as
 * though the answer were yes. Distinct from `no-swallowed-error`, which covers
 * try/catch around an interaction -- this one is about the *wait* itself.
 *
 * Only readiness waits are flagged, so the deliberate bare catches elsewhere are
 * not false positives: `resolveToken` probes for a file that may not exist, and
 * `requireAbsent` implements "absent" by catching a wait that times out and
 * returning true.
 */
const READINESS_WAITS = [
  "waitForResponse",
  "waitForSelector",
  "waitForLoadState",
  "waitForFunction",
  "waitForURL",
];

/**
 * What a name refers to at a given point, following `const`/`let` bindings.
 *
 * `const pending = page.waitForResponse(...); await pending.catch(...)` is the
 * same discard as `page.waitForResponse(...).catch(...)`. A rule that only
 * recognised the inline form would pass on the refactored version of the exact
 * defect it exists to catch, which is worse than not having the rule.
 */
function resolveBinding(ast, identifierNode) {
  if (!identifierNode || identifierNode.type !== "Identifier") return null;
  const name = identifierNode.name;
  const bindings = find(ast, (n) => n.type === "VariableDeclarator"
    && n.id.type === "Identifier" && n.id.name === name);
  // The last binding textually before the use wins, as in the source itself.
  const before = bindings
    .filter((b) => b.loc && b.loc.start.line <= (identifierNode.loc.start.line || 0))
    .sort((a, b) => a.loc.start.line - b.loc.start.line);
  return before.length ? before[before.length - 1].init : null;
}

function checkNoSilentReadinessTimeout(ast, report) {
  find(ast, (n) => n.type === "CallExpression"
    && memberProperty(n.callee) === "catch")
    .forEach((n) => {
      const inner = n.callee.object;
      const target = inner.type === "CallExpression"
        ? inner
        : resolveBinding(ast, inner);
      if (!target || target.type !== "CallExpression") return;
      const waited = CALLEE(target);
      const method = memberProperty(target.callee);
      if (READINESS_WAITS.indexOf(method) === -1
        && READINESS_WAITS.indexOf(waited) === -1) return;

      const reported = contains(n.arguments[0], (c) =>
        c.type === "ThrowStatement"
        || /^(AssertionFailure|throw|fail|rethrow|raise)$/.test(String(c.name || ""))
        || (c.type === "CallExpression"
          && /^(AssertionFailure|fail|rethrow|raise)$/.test(CALLEE(c))));
      if (!reported) {
        report("no-silent-readiness-timeout", n,
          `a readiness wait (${waited}) has its timeout caught and discarded, `
          + "so a request that never happened reads as success");
      }
    });
}

function checkBrowserClosed(ast, report) {
  const finallyCloses = find(ast, (n) => n.type === "TryStatement" && n.finalizer
    && contains(n.finalizer, (c) => c.type === "CallExpression"
      && /close/.test(CALLEE(c))));
  if (!finallyCloses.length) {
    const anyClose = find(ast, (n) => n.type === "CallExpression"
      && memberProperty(n.callee) === "close");
    report("browser-closed", anyClose[0] || ast.program,
      anyClose.length
        ? "close() exists but not in a finally; a failed assertion leaks the browser"
        : "the browser is never closed");
  }
}

function checkFailsOnAssertion(ast, report) {
  const exits = find(ast, (n) => n.type === "AssignmentExpression"
    && memberProperty(n.left) === "exitCode");
  if (!exits.length) {
    report("fails-on-assertion", ast.program,
      "the process exit code is never set, so a failed assertion cannot fail the run");
  }
}

function checkDiagnosticsAttached(ast, report) {
  const required = ["console", "pageerror", "requestfailed", "response"];
  const subscribed = new Set();
  find(ast, (n) => n.type === "CallExpression")
    .forEach((call) => {
      const event = stringValue(call.arguments[0]);
      if (event && required.includes(event)) subscribed.add(event);
    });
  for (const name of required) {
    if (!subscribed.has(name)) {
      report("diagnostics-attached", ast.program,
        `no listener for "${name}"; the failure would be invisible`);
    }
  }
}

function find(ast, predicate) {
  const out = [];
  const walk = (n) => {
    if (!n || typeof n !== "object") return;
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (n.type && predicate(n)) out.push(n);
    for (const key of Object.keys(n)) {
      if (key === "loc" || key === "leadingComments" || key === "trailingComments") continue;
      walk(n[key]);
    }
  };
  walk(ast);
  return out;
}

const RULES = [
  checkNoFixedSleep,
  checkNoPositionalSelector,
  checkNoTextSelector,
  checkNoBareElementQuery,
  checkNoSwallowedError,
  checkNoSilentReadinessTimeout,
  checkNoUnassertedShot,
  checkBrowserClosed,
  checkFailsOnAssertion,
  checkDiagnosticsAttached,
];

/**
 * The rule codes this file can emit.
 *
 * Exported so the test suite can assert it covers exactly these -- a rule added
 * here without a fixture is a rule nobody has proved fires, and a fixture
 * without a rule is a test of nothing.
 */
const RULE_CODES = [
  "no-fixed-sleep",
  "no-positional-selector",
  "no-text-selector",
  "no-bare-element-query",
  "no-swallowed-error",
  "no-silent-readiness-timeout",
  "no-unasserted-shot",
  "browser-closed",
  "fails-on-assertion",
  "diagnostics-attached",
];

/** Names of the routes the capture script covers, in order. */
function routeNames(ast) {
  for (const node of ast.program.body) {
    if (node.type !== "VariableDeclaration") continue;
    for (const decl of node.declarations) {
      if (decl.id.name !== "ROUTES" || decl.init.type !== "ObjectExpression") continue;
      // Both node kinds matter: `explorer: async () => {}` is an ObjectProperty
      // and `async explorer() {}` is an ObjectMethod. Filtering on only the
      // former reports an empty route list, which reads like "no routes".
      return decl.init.properties
        .filter((p) => (p.type === "ObjectProperty" || p.type === "ObjectMethod")
          && p.key.type === "Identifier")
        .map((p) => p.key.name);
    }
  }
  return [];
}

/**
 * Every API path the script waits on, sorted.
 *
 * Exists so the test suite can check each one against the frontend's own API
 * client. A typo here is invisible and expensive: `withApiWait` used to catch
 * its own timeout, so a path no endpoint matched did not fail -- it burned the
 * full 120-second ceiling and then left the route to a DOM assertion that was
 * never meant to be its only check. Two paths had exactly that defect (a
 * benchmark route waiting on `/run_benchmark` when the endpoint is
 * `/api/benchmarks/run`, and a network route waiting on `/gpt2/run_prompt` when
 * it calls `/api/infer`), costing 241 seconds of one run.
 *
 * A flat list, not keyed by route: attributing a call to its enclosing route
 * needs position tracking that is easy to get subtly wrong, and the property
 * worth checking does not depend on which route asked.
 */
function waitedUrls(ast) {
  const urls = find(ast, (n) => n.type === "CallExpression"
    && CALLEE(n) === "withApiWait")
    .map((n) => stringValue(n.arguments[2]))
    .filter((u) => u !== null);
  return [...new Set(urls)].sort();
}

function lint(file) {
  const source = fs.readFileSync(file, "utf8");
  const ast = parser.parse(source, { sourceType: "script" });
  const violations = [];
  for (const rule of RULES) {
    rule(ast, (code, node, message) => violations.push({
      rule: code,
      line: lineOf(node),
      message,
    }));
  }
  violations.sort((a, b) => a.line - b.line || a.rule.localeCompare(b.rule));
  return {
    file,
    lines: source.split("\n").length,
    ast,
    routes: routeNames(ast),
    violations,
  };
}

/**
 * The `data-testid` values this script depends on.
 *
 * Extracted from the AST rather than by grep because they appear in two
 * different shapes -- `[data-testid="x"]` inside a locator string, and a bare
 * `"x"` as the third argument to a `require*` helper -- and a regex that knows
 * only one of them silently reports a partial list. A partial list reads exactly
 * like a complete one, which is how a selector renamed in a Vue template would
 * slip through and fail at runtime instead of here.
 */
function testIdDependencies(ast) {
  const ids = new Map();
  const note = (id, line) => {
    if (!ids) return;
    if (!ids.has(id)) ids.set(id, []);
    ids.get(id).push(line);
  };

  find(ast, (n) => n.type === "CallExpression"
    && /^require(Visible|Absent|Text)$/.test(CALLEE(n)))
    .forEach((call) => {
      const id = stringValue(call.arguments[2]);
      if (id) note(id, lineOf(call));
    });

  find(ast, (n) => stringValue(n) !== null
    && stringValue(n).indexOf("data-testid") !== -1)
    .forEach((n) => {
      const match = /data-testid=\\?["']([^"']+)\\?["']/.exec(stringValue(n));
      if (match) note(match[1], lineOf(n));
    });

  return [...ids.entries()]
    .map(([id, lines]) => ({ id, lines: [...new Set(lines)].sort((a, b) => a - b) }))
    .sort((a, b) => a.id.localeCompare(b.id));
}

function main(argv) {
  const args = argv.filter((a) => a !== "--pretty" && a !== "--ids");
  if (!args.length) {
    console.error("usage: node scripts/capture_lint.cjs <file.cjs> [--pretty] [--ids]");
    return 2;
  }
  let report;
  try {
    report = lint(args[0]);
  } catch (err) {
    console.error(`could not analyse ${args[0]}: ${err.message}`);
    return 2;
  }
  if (argv.includes("--ids")) {
    console.log(JSON.stringify(testIdDependencies(report.ast)));
    return 0;
  }
  if (argv.includes("--urls")) {
    console.log(JSON.stringify(waitedUrls(report.ast)));
    return 0;
  }
  const pretty = argv.includes("--pretty");
  console.log(JSON.stringify(
    {
      file: report.file,
      lines: report.lines,
      routes: report.routes,
      violations: report.violations,
    },
    null, pretty ? 2 : 0,
  ));
  return report.violations.length ? 1 : 0;
}

if (require.main === module) {
  process.exitCode = main(process.argv.slice(2));
}

module.exports = { lint, RULES, RULE_CODES, testIdDependencies, routeNames, waitedUrls };