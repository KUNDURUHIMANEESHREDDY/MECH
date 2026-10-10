"""A screenshot is not a test, so the screenshot scripts must assert.

The defect
----------
There were three capture scripts. All three documented the UI rather than
checking it::

    goto -> wait N seconds -> screenshot -> report success

which produces a green result and a PNG of an error page. The committed
`03-network.png` was a picture of the parameter ledger, with the attention wires
the route exists for never rendered -- the wiring panels sit behind
``v-if="wireTokens.length"`` and nothing in the script ran a prompt. And the
committed ``06-research-society.png`` and ``06b-research-society-progress.png``
are **byte-identical**: the "progress" capture was the same failed page twice,
which nothing noticed because a screenshot cannot fail.

The interactive version also addressed controls positionally --
``textInputs.nth(0..2)``, ``input[type="number"].first()``, ``select.first()`` --
so a reordered form silently changed which field got filled, and swallowed the
errors it hit in three bare ``catch {}`` blocks, each leaving a default value in
place.

How these tests are built
-------------------------
`scripts/capture_lint.cjs` parses the capture script and reports structural
violations; this file drives it. The important half is the negative controls:
every rule is proved to fire on a deliberately-broken fixture before it is
trusted to pass on the real script. A checker that reports nothing because it is
broken looks identical to a checker reporting a clean file, and during this
work the linter did exactly that twice -- it used ESTree's ``Literal`` where
Babel emits ``StringLiteral`` (three rules silently matched nothing), and it
inspected a ``catch`` handler while the swallowed call was in the ``try``. Both
were found by the fixtures below, not by reading the linter.

The remaining tests pin the properties that are not about the script's syntax:
that every selector it waits on still exists in a Vue template, that the
Python entry point cannot swallow a failure, and that there is only one
implementation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
CAPTURE = SCRIPTS / "capture_screenshots.cjs"
LINTER = SCRIPTS / "capture_lint.cjs"
WRAPPER = SCRIPTS / "capture_screenshots.py"
VUE_DIR = ROOT / "frontend" / "src"

sys.path.insert(0, str(ROOT))
from scripts import capture_screenshots as wrapper  # noqa: E402

#: Every rule the linter implements. Each is checked against a fixture below,
#: so this list and the fixtures must stay in step -- a rule added without one
#: is a rule nobody has proved works.
RULES = [
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
]

#: The routes the previous scripts captured. Collapsing three implementations
#: into one must not quietly drop a view.
LEGACY_ROUTES = [
    "explorer", "transformer", "network", "steering", "benchmark",
    "society", "workspace", "models", "settings", "plugins",
]


# ── Driving the linter ──────────────────────────────────────────────────

def run_linter(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", str(LINTER), *args],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)


def lint(file: Path) -> Dict:
    result = run_linter(str(file))
    assert result.returncode in (0, 1), (
        f"the linter itself failed ({result.returncode}): {result.stderr}")
    return json.loads(result.stdout)


def lint_source(tmp_path: Path, source: str) -> Dict:
    fixture = tmp_path / "fixture.cjs"
    fixture.write_text(source, encoding="utf-8")
    return lint(fixture)


def strip_js_comments(source: str) -> str:
    """Comments removed; string and template literals left intact.

    Needed because the capture script explains the patterns it removed *in the
    prose of the very function that removed them*, so a plain substring search
    finds `.catch(() => null)` in a comment describing why it is gone.

    Order matters. String and template literals are lifted out first, because a
    `//` inside one -- and this script has URLs like `http://127.0.0.1` in its
    message strings -- would otherwise be deleted as if it started a comment.
    Each literal is replaced by a same-length run of `x` so offsets stay aligned
    and the result remains readable.
    """
    lifted = re.sub(r"`(?:\\.|[^`\\])*`",
                    lambda m: "x" * len(m.group(0)), source, flags=re.S)
    lifted = re.sub(r'"(?:\\.|[^"\\\n])*"',
                    lambda m: "x" * len(m.group(0)), lifted)
    lifted = re.sub(r"'(?:\\.|[^'\\\n])*'",
                    lambda m: "x" * len(m.group(0)), lifted)
    lifted = re.sub(r"/\*.*?\*/", "", lifted, flags=re.S)
    return re.sub(r"(?m)//.*$", "", lifted)


def rules_fired(report: Dict) -> set:
    return {v["rule"] for v in report["violations"]}


# ── The real script must be clean ───────────────────────────────────────

@pytest.mark.parametrize("rule", RULES)
def test_the_capture_script_has_no_violations_of_that_rule(rule):
    """Each rule asserted separately.

    Parametrised rather than one "no violations" assertion because a single
    assertion over an empty list passes for every reason at once -- a typo in
    the rule name, a rule that finds nothing, a genuinely clean file -- and
    names none of them.
    """
    offenders = [v for v in lint(CAPTURE)["violations"] if v["rule"] == rule]
    assert not offenders, (
        f"{rule}:\n  " + "\n  ".join(
            f"line {v['line']}: {v['message']}" for v in offenders))


def test_the_linter_reports_no_violations_at_all_for_the_real_script():
    report = lint(CAPTURE)
    assert report["violations"] == [], report["violations"]


def test_no_silent_readiness_timeout_can_be_reintroduced(tmp_path):
    """The `.catch(() => null)` that `withApiWait` used to carry.

    A typo in a waited API path did not fail the route. The rejection became a
    null, the null read as "no error", the route carried on, and the whole
    120-second ceiling elapsed first. Two paths had exactly that defect, costing
    241 seconds of a single run -- and passing, on the DOM assertion alone.
    """
    fixture = tmp_path / "swallowed.cjs"
    fixture.write_text("""
async function withApiWait(page, route, url, action, ready) {
  const responsePromise = page
    .waitForResponse((r) => r.url().includes(url), { timeout: 120000 })
    .catch(() => null);
  await action();
  const response = await responsePromise;
  if (response && response.status() >= 400) { return; }
  await ready();
}
""", encoding="utf-8")

    violations = rules_fired(lint(fixture))

    assert "no-silent-readiness-timeout" in violations, (
        "a readiness wait whose timeout is caught and discarded is not "
        "checked")


def test_a_readiness_wait_that_reports_its_timeout_is_accepted(tmp_path):
    """The negative control for the rule above.

    Consuming a pending wait and turning its rejection into a reported failure
    is the correct shape, and must not be flagged -- otherwise the fix would be
    unrepresentable.
    """
    fixture = tmp_path / "reported.cjs"
    fixture.write_text("""
class AssertionFailure extends Error {}
async function withApiWait(page, route, url, action, ready) {
  const pending = page
    .waitForResponse((r) => r.url().includes(url), { timeout: 120000 })
    .then((response) => ({ response }), (cause) => ({ cause }));
  const outcome = await pending;
  if (!outcome.response) {
    throw new AssertionFailure(route, `no request to ${url}`, {});
  }
  await ready();
}
""", encoding="utf-8")

    assert "no-silent-readiness-timeout" not in rules_fired(lint(fixture))


def test_with_api_wait_really_does_report_a_missing_request():
    """The behaviour, checked against the real source.

    A structural check cannot see that `outcome.response` is tested; this reads
    the function and confirms both halves of the contract are present.
    """
    source = CAPTURE.read_text(encoding="utf-8")
    body = source[source.index("async function withApiWait"):]
    body = body[:body.index("\n}\n")]
    code = strip_js_comments(body)

    assert ".catch(() => null)" not in code, (
        "the null-on-timeout pattern is back in withApiWait")
    assert "if (!outcome.response)" in code, (
        "a missing response is not turned into a failure")
    # The pending wait is consumed on both paths, or Node may exit on an
    # unhandled rejection before the route's own error handler runs.
    assert code.count("await pending") == 1

    # The failure is *reported*, not merely constructed: the message has to
    # exist, because it is what tells a reader which path went unmatched. Checked
    # against the raw source, since `code` has literals masked out.
    assert "the action produced no request" in source, (
        "the missing-response branch throws without saying what went wrong")


# ── ...but only if the linter can still fail ────────────────────────────

#: One minimal fixture per rule. Each contains the smallest thing that should
#: trip that rule and nothing else that might, so a failure points at one cause.
FIXTURES: Dict[str, str] = {
    "no-fixed-sleep": """
async function route(page) {
  await requireVisible(page, "r", "x");
  await page.waitForTimeout(4500);
  await shot(page, "r", "n");
}
""",
    "no-positional-selector": """
async function route(page) {
  const inputs = page.locator('[data-testid="a"]');
  await requireVisible(page, "r", "x");
  await inputs.nth(2).fill("v");
  await shot(page, "r", "n");
}
""",
    "no-text-selector": """
async function route(page) {
  await requireVisible(page, "r", "x");
  await page.locator('button:has-text("Steer")').click();
  await shot(page, "r", "n");
}
""",
    "no-bare-element-query": """
async function route(page) {
  await requireVisible(page, "r", "x");
  await page.locator("select").selectOption("3");
  await shot(page, "r", "n");
}
""",
    "no-swallowed-error": """
async function route(page) {
  await requireVisible(page, "r", "x");
  try {
    await page.locator('[data-testid="a"]').fill("v");
  } catch {}
  await shot(page, "r", "n");
}
""",
    "no-unasserted-shot": """
async function route(page) {
  await page.goto("http://localhost:5173/#x");
  await shot(page, "r", "n");
}
""",
    "no-silent-readiness-timeout": """
async function withApiWait(page, route, url, action, ready) {
  const p = page.waitForResponse((r) => r.url().includes(url), { timeout: 60000 });
  await action();
  const response = await p.catch(() => null);
  if (response && response.status() >= 400) { return; }
  await ready();
}
""",
    "browser-closed": """
async function route(page) {
  await requireVisible(page, "r", "x");
  await shot(page, "r", "n");
}
async function main() {
  const b = await chromium.launch();
  await route();
  await b.close();
}
""",
    "fails-on-assertion": """
async function route(page) {
  await requireVisible(page, "r", "x");
  await shot(page, "r", "n");
}
async function main() {
  const b = await chromium.launch();
  try { await route(); } finally { await b.close(); }
}
""",
    "diagnostics-attached": """
async function route(page) {
  await requireVisible(page, "r", "x");
  await shot(page, "r", "n");
}
async function main() {
  const b = await chromium.launch();
  page.on("console", () => {});
  page.on("pageerror", () => {});
  page.on("requestfailed", () => {});
  try { await route(); } finally { await b.close(); }
  process.exitCode = 0;
}
""",
}


@pytest.mark.parametrize("rule", RULES)
def test_each_rule_fires_on_a_fixture_built_to_break_it(rule, tmp_path):
    """The negative control.

    This is what makes the clean result above mean something. Each fixture
    exercises one rule; the others may fire too (a fixture has to contain enough
    code to be a route), but *this* rule must be among them.
    """
    assert rule in FIXTURES, f"{rule} has no fixture; add one before trusting it"
    report = lint_source(tmp_path, FIXTURES[rule])
    assert rule in rules_fired(report), (
        f"{rule} did not fire on its own fixture. The fixture was:\n"
        f"{FIXTURES[rule]}\nreported: {sorted(rules_fired(report))}")


def linter_rule_codes() -> List[str]:
    """The rule codes the linter declares it can emit."""
    result = subprocess.run(
        ["node", "-e",
         f"console.log(JSON.stringify(require({str(LINTER)!r}).RULE_CODES))"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_test_suite_covers_exactly_the_rules_the_linter_implements():
    """A rule with no fixture is a rule nobody has proved works.

    The linter declares its own rule codes so this can be checked from both
    sides. Comparing against a count would pass as long as the two happened to
    be the same number.
    """
    assert set(linter_rule_codes()) == set(RULES), (
        "the linter and its tests disagree about which rules exist; a rule "
        "added to one side only is unchecked")
    assert set(FIXTURES) == set(RULES)


def test_the_fixtures_still_cover_every_rule_the_linter_names():
    """Belt and braces: no fixture may name a rule the linter never emits."""
    for rule, source in FIXTURES.items():
        assert rule in RULES, f"fixture for an unknown rule {rule}"
        assert source.strip(), f"the {rule} fixture is empty"


# ── Comments are not code ───────────────────────────────────────────────

def test_the_scripts_own_comments_do_not_trip_its_own_rules():
    """The trap this repository has already fallen into twice.

    The capture script *quotes* `textInputs.nth(0..2)` and `catch {}` in its
    header in order to explain what was wrong. A grep-based checker flags those
    explanations as if they were the defect, which is how
    `test_no_hash_derived_identifiers` and the boolean-marker check in
    `test_standalone_launcher.py` both failed on their own documentation. The
    AST approach is what avoids it, so this asserts the prose is still there --
    if it is deleted, the AST reasoning behind the checker needs revisiting.
    """
    source = CAPTURE.read_text(encoding="utf-8")
    assert ".nth(0..2)" in source
    assert "catch {}" in source
    assert "waitForTimeout" in source, (
        "the script no longer documents the fixed sleep it replaced; revisit "
        "why an AST is used instead of a grep")

    offenders = [v for v in lint(CAPTURE)["violations"]]
    assert not any(v["line"] <= 60 for v in offenders), (
        "a violation is reported inside the file's explanatory header, which "
        "means a comment is being read as code")


# ── Every selector the script waits on must still exist ─────────────────

def capture_test_ids() -> List[Dict]:
    """The `data-testid` values the capture script waits on.

    Deliberately not named `test_*`: pytest collects that prefix, and this is a
    helper, not a test.
    """
    result = run_linter(str(CAPTURE), "--ids")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_script_depends_on_more_than_a_handful_of_selectors():
    """Guards the extraction itself.

    An extractor that returns an empty or near-empty list is indistinguishable
    from one that found nothing wrong, and would make the test below vacuous.
    """
    ids = capture_test_ids()
    assert len(ids) >= 25, [i["id"] for i in ids]
    # Both shapes must be collected: `[data-testid="x"]` in a locator string,
    # and a bare `"x"` passed to a require* helper.
    assert any("steer-prompt" == i["id"] for i in ids)      # locator form
    assert any("steer-result" == i["id"] for i in ids)      # require* form


@pytest.mark.parametrize("entry", capture_test_ids(), ids=lambda e: e["id"])
def test_every_data_testid_the_script_waits_on_exists_in_a_template(entry):
    """The coupling that would otherwise fail only at runtime.

    The script waits on `[data-testid="steer-prompt"]`. If a Vue template drops
    or renames that attribute, every run of the script starts failing its
    assertion -- which is correct behaviour, but it fails as a mysterious
    30-second timeout rather than as "the selector moved".
    """
    needle = f'data-testid="{entry["id"]}"'
    holders = [p for p in VUE_DIR.rglob("*.vue") if needle in p.read_text(encoding="utf-8")]
    assert holders, (
        f'no Vue template defines {needle}, which scripts/capture_screenshots.cjs '
        f'waits on at line(s) {entry["lines"]}')


def waited_paths() -> List[str]:
    """The API paths the capture script waits on, via the linter's `--urls`.

    Deliberately not named `test_*`; pytest collects that prefix.
    """
    result = run_linter(str(CAPTURE), "--urls")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def frontend_endpoints() -> set:
    """Every `/api/...` path the frontend actually calls."""
    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for pattern in ("*.ts", "*.vue")
        for p in (VUE_DIR).rglob(pattern))
    return set(re.findall(r"(/api/[A-Za-z0-9_/]+)", sources))


def test_the_script_waits_on_some_paths():
    """Guards the extraction, for the same reason the test-id one exists."""
    assert len(waited_paths()) >= 5, waited_paths()


@pytest.mark.parametrize("path", waited_paths())
def test_every_api_path_the_script_waits_on_is_one_the_frontend_calls(path):
    """The check that catches a wrong path, which is otherwise invisible.

    A mistyped path did not fail the route. `withApiWait` discarded its own
    timeout, the null read as "no error", and the route continued -- after the
    full 120-second ceiling. Two paths were wrong (`/run_benchmark` for
    `/api/benchmarks/run`, and `/gpt2/run_prompt` on the network route, which
    calls `/api/infer`), costing 241 seconds of one run and leaving those routes
    checked by their DOM assertion alone.
    """
    endpoints = frontend_endpoints()
    assert any(path == e or path in e or e.endswith(path) for e in endpoints), (
        f"the capture script waits on {path}, which no frontend API call "
        f"matches. Checked against {len(endpoints)} endpoints.")


def test_the_script_has_no_route_that_passes_on_a_dom_assertion_alone():
    """Every route that drives a request must wait for that request.

    The routes that only navigated and screenshotted were exactly the ones that
    produced pictures of unrendered panels, so this pins that any route which
    clicks something also asserts the response -- and now that a *missing*
    response is a failure rather than a discarded timeout.
    """
    source = CAPTURE.read_text(encoding="utf-8")
    # Split on route method boundaries so each route can be considered alone.
    starts = [m.start() for m in re.finditer(r"\n  async \w+\(page, args\) \{", source)]
    assert len(starts) >= 10, "could not locate the route functions"
    ends = starts[1:] + [source.index("\n};", starts[-1])]

    for start, end in zip(starts, ends):
        body = source[start:end]
        name = re.search(r"async (\w+)\(page, args\)", body).group(1)
        has_interaction = re.search(r"\.(click|fill|selectOption)\(", body)
        has_api_wait = "withApiWait(" in body
        if has_interaction:
            assert has_api_wait, (
                f"the {name} route clicks something but never waits for a "
                f"response; it can only be checked by its DOM assertion, which "
                f"is how a screenshot of an error page passes")


def test_the_error_test_ids_the_script_asserts_absent_also_exist():
    """`requireAbsent` is only an assertion if the element can appear.

    A typo in an error selector makes the check permanently pass, which is the
    silent half of the original defect.
    """
    source = CAPTURE.read_text(encoding="utf-8")
    absent = re.findall(r'requireAbsent\([^,]+,[^,]+,\s*"([a-z0-9-]+)"', source)
    assert absent, "no requireAbsent call found; the check may have been removed"
    for test_id in absent:
        assert test_id.endswith("-error"), (
            f"{test_id} is asserted absent but is not named like an error")
        assert any(f'data-testid="{test_id}"' in p.read_text(encoding="utf-8")
                   for p in VUE_DIR.rglob("*.vue")), (
            f"{test_id} is asserted absent but no template can render it")


# ── No route was dropped in the collapse ────────────────────────────────

def test_no_route_was_dropped_when_the_three_scripts_became_one():
    """Three implementations became one, so one list has to be the survivor."""
    assert lint(CAPTURE)["routes"] == LEGACY_ROUTES


def test_the_route_order_is_stable():
    """Screenshot filenames are numbered by position in this list."""
    assert lint(CAPTURE)["routes"][:6] == LEGACY_ROUTES[:6]


# ── The Python entry point cannot swallow a failure ─────────────────────

def test_the_wrapper_returns_the_node_scripts_own_exit_code(monkeypatch):
    """The wrapper must not manufacture success.

    It exists so `python scripts/capture_screenshots.py` still works. If it
    ignored the child's status, a failed capture would exit 0 and any caller --
    a CI step, a shell chain -- would read it as a pass.
    """
    seen = {}

    class Result:
        returncode = 1

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return Result()

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    assert wrapper.main([]) == 1
    assert "capture_screenshots.cjs" in seen["cmd"][1]


@pytest.mark.parametrize("code", [0, 1, 2, 7])
def test_any_nonzero_code_survives_the_wrapper(monkeypatch, code):
    class Result:
        returncode = code

    monkeypatch.setattr(wrapper.subprocess, "run", lambda cmd, **kw: Result())
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: "node")
    assert wrapper.main([]) == code


def test_arguments_are_forwarded_to_the_node_script(monkeypatch):
    """A flag the wrapper silently drops is a flag the user thinks is on."""
    seen = {}

    class Result:
        returncode = 0

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return Result()

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    wrapper.main(["--only=steering", "--no-load-model"])

    assert "--only=steering" in seen["cmd"]
    assert "--no-load-model" in seen["cmd"]


def test_the_separator_is_accepted_and_stripped(monkeypatch):
    """`python scripts/capture_screenshots.py -- --only=x` is documented, so
    forwarding the `--` to node would make node reject the flag."""
    seen = {}

    class Result:
        returncode = 0

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return Result()

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    wrapper.main(["--", "--only=steering"])

    assert "--" not in seen["cmd"]
    assert "--only=steering" in seen["cmd"]


def test_a_missing_node_is_reported_rather_than_raising(monkeypatch):
    """Playwright lives in frontend/node_modules, so node is a hard dependency
    of this entry point. Saying so beats a FileNotFoundError traceback."""
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: None)
    assert wrapper.main([]) == 2


def test_the_wrapper_holds_no_capture_logic():
    """It delegates. Logic that crept back in would be a second implementation,
    which is precisely the divergence this removed."""
    source = WRAPPER.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#"))
    for forbidden in ("playwright", "chromium", "waitForTimeout", "PAGES", "screenshot("):
        assert forbidden not in code, (
            f"capture_screenshots.py contains {forbidden!r}; the capture logic "
            f"belongs in capture_screenshots.cjs")


# ── One implementation ──────────────────────────────────────────────────

def test_the_divergent_interactive_script_is_gone():
    """`capture_screenshots_interactive.cjs` had diverged from its sibling --
    only it loaded the model, only it visited the Society view -- so which
    screenshots were current depended on which file someone remembered to run."""
    assert not (SCRIPTS / "capture_screenshots_interactive.cjs").exists()
    assert not (SCRIPTS / "capture_screenshots_interactive.py").exists()


def test_there_is_exactly_one_place_that_screenshots():
    """One implementation, so there is one place to add an assertion."""
    offenders = [
        p.relative_to(ROOT).as_posix() for p in SCRIPTS.rglob("*")
        if p.is_file()
        and p.suffix in (".cjs", ".py")
        and p.name not in ("capture_screenshots.cjs", "capture_lint.cjs",
                           "capture_screenshots.py")
        and "page.screenshot" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, f"other scripts still screenshot: {offenders}"


# ── Authentication matches the backend ──────────────────────────────────

def test_the_token_source_order_matches_the_backend():
    """`backend/core/auth.py` checks the environment variable first, then the
    persisted file. If the capture script resolved them the other way round it
    would send the wrong token and fail on 401s that look like a UI bug."""
    backend = (ROOT / "backend" / "core" / "auth.py").read_text(encoding="utf-8")
    env_name = re.search(r'_TOKEN_ENV\s*=\s*"([A-Z_]+)"', backend)
    file_name = re.search(r'_TOKEN_FILENAME\s*=\s*"([^"]+)"', backend)
    assert env_name and file_name, "could not read the backend's token sources"

    script = CAPTURE.read_text(encoding="utf-8")
    assert env_name.group(1) in script, (
        f"the backend reads {env_name.group(1)} but the capture script does not")
    assert file_name.group(1) in script

    env_at = script.index(env_name.group(1))
    file_at = script.index(file_name.group(1))
    assert env_at < file_at, (
        "the capture script tries the file before the environment variable, "
        "which is the reverse of the backend's precedence")


def test_the_capture_announces_when_it_has_no_token():
    """A silent 401 storm is worse than a warning: the first run of this script
    produced three 401s and no mention of a missing token."""
    source = CAPTURE.read_text(encoding="utf-8")
    assert "No backend token found" in source
    assert "will fail on any authenticated route" in source


# ── The files parse ─────────────────────────────────────────────────────

def test_the_linter_rejects_a_missing_argument_with_a_usage_error():
    """Not a substitute for a syntax check -- a check that a missing argument is
    diagnosed rather than analysed as a filename."""
    assert run_linter().returncode == 2


def test_the_linter_reports_an_unparseable_file_rather_than_raising(tmp_path):
    """A syntax error in the script under analysis must be a clear refusal.

    Silently returning "no violations" for a file it could not parse would be
    the worst possible failure: a clean report for something never read.
    """
    broken = tmp_path / "broken.cjs"
    broken.write_text("async function ( { oops", encoding="utf-8")

    result = run_linter(str(broken))

    assert result.returncode == 2
    assert "could not analyse" in result.stderr


@pytest.mark.parametrize("script", [CAPTURE, LINTER])
def test_the_node_scripts_are_syntactically_valid(script):
    result = subprocess.run(
        ["node", "--check", str(script)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr