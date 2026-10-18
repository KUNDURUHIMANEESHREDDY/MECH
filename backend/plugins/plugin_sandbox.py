import ast
import sys
import types
from typing import Dict, Any

class SecurityViolation(Exception):
    pass


def _plugin_print(*args: Any, **kwargs: Any) -> None:
    """`print` for plugin code, redirected to stderr.

    The worker's stdout *is* the JSON protocol channel: the parent reads frames
    from it and a single frame that is not JSON deserialises as a protocol error
    and can desynchronise the stream for every later message. Exposing the real
    `print` meant a plugin debugging itself with one line of output could break
    the channel it is answering on.

    stderr is not protocol, and the runner already drains it for diagnostics,
    so plugin logging stays visible instead of being taken away. `print` stays
    available deliberately -- removing it would only teach plugin authors to
    open ``sys.stdout`` directly, which the AST gate would then have to police.
    """
    try:
        print(*args, file=sys.stderr, **kwargs)
    except Exception:
        # A logging call must never be the thing that fails a hook.
        pass


class PluginSandbox:
    """
    Provides a restrictive environment for executing plugin code.
    Prevents access to sensitive built-ins (like file IO, os, sys).
    """
    
    ALLOWED_BUILTINS = {
        'print': _plugin_print,
        'len': len,
        'range': range,
        'int': int,
        'float': float,
        'str': str,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'bool': bool,
        'isinstance': isinstance,
        'sum': sum,
        # Class construction. MechPlugin.manifest is an @property, so any
        # conforming plugin needs these to declare its subclass at all.
        # None of them exposes the interpreter: the class-object escapes
        # (type, __subclasses__, __mro__) stay out of this dict and are
        # rejected by analyze_ast.
        'property': property,
        'staticmethod': staticmethod,
        'classmethod': classmethod,
        'object': object,
        'super': super,
        'min': min,
        'max': max,
        'abs': abs,
        'enumerate': enumerate,
        'zip': zip,
        # Exception types. A plugin's own error handling needs these, and a
        # plugin cannot reach a dangerous builtin through an exception class.
        'Exception': Exception,
        'BaseException': BaseException,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'KeyError': KeyError,
        'IndexError': IndexError,
        'AttributeError': AttributeError,
        'NotImplementedError': NotImplementedError,
        'RuntimeError': RuntimeError,
        'StopIteration': StopIteration,
        'ZeroDivisionError': ZeroDivisionError,
    }

    # Importable modules. The MECH plugin SDK lives under
    # ``backend.plugins`` — without it no MechPlugin subclass can even be
    # declared, so the SDK packages are allowlisted alongside a small set of
    # safe standard-library modules. Everything else (os, sys, subprocess,
    # socket, the rest of backend, …) stays forbidden at install scan time.
    #
    # Note the boundary honestly: this scan gates installation only. Enabled
    # plugin code executes in the backend process with that process's
    # privileges, so install only directories you trust.
    ALLOWED_IMPORT_ROOTS = frozenset({
        '__future__',
        'math', 'typing', 'collections', 'dataclasses', 'abc', 'enum',
        'functools', 'itertools', 're', 'json', 'logging', 'datetime',
        'pathlib', 'hashlib', 'time', 'statistics', 'decimal', 'fractions',
    })
    ALLOWED_IMPORT_PREFIXES = ('backend.plugins.',)

    # Bare names that hand back unrestricted access to the interpreter.
    # ``__import__`` is the important one: it reaches every module the
    # import allowlist above is meant to exclude (os, subprocess, socket, …).
    FORBIDDEN_NAMES = frozenset({
        'eval', 'exec', 'open', 'compile', 'input', 'breakpoint',
        '__import__', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        '__builtins__', '__loader__', '__spec__',
    })

    # Attribute names that walk out of the sandbox. Together with a bare
    # ``type`` these reach every class in the interpreter:
    # ``().__class__.__bases__[0].__subclasses__()`` includes
    # ``subprocess.Popen``, so blocking the name is not sufficient.
    FORBIDDEN_ATTRS = frozenset({
        '__class__', '__bases__', '__mro__', '__subclasses__',
        '__globals__', '__builtins__', '__code__', '__closure__',
        '__func__', '__self__', '__dict__', '__reduce__', '__reduce_ex__',
        '__getattribute__', '__setattr__', '__delattr__', '__init_subclass__',
        '__loader__', '__spec__', '__import__', '__get__',
    })

    #: Attributes that are a *live module object* held on an allowlisted
    #: stdlib module. Reading one hands the plugin back the interpreter.
    #: Measured on this host across the modules actually imported:
    #:
    #:   collections._sys   logging.sys   logging.os   enum.sys   enum.bltns
    #:
    #: `logging.os` is the whole of `os` with no underscore to hide behind.
    #: These are refused only in a *load* context (see ``analyze_ast``):
    #: `self.os = 'win32'` and `self.builtins = []` are ordinary field writes,
    #: and the earlier attempt at a name-level rule failed precisely because it
    #: could not tell a module from a field. A store binds a field; only a load
    #: can be a pivot.
    ESCAPE_MODULE_ATTRS = frozenset({
        '_sys', 'sys', 'os', 'builtins', 'bltns', '_os',
    })

    # Deliberately NOT listed above: the module names an allowlisted module
    # holds as plain attributes (`pathlib.sys`, `pathlib.os`, `logging.io`, …).
    # They were added once, to close the `pathlib.sys.modules['_winapi']` pivot,
    # and removed because a name-level rule does neither job. Measured on this
    # host:
    #
    #   * it refused legitimate plugin code -- `self.io = None`,
    #     `self.os = 'win32'` and `self.builtins = []` are ordinary and were all
    #     rejected, because the rule cannot tell a module from a field;
    #   * and it did not block the pivot, because the same modules are reachable
    #     under other names -- `collections._sys`, `hashlib._sys`,
    #     `logging._sys` and `re._sys` are the real `sys` module and this scan
    #     accepted every one of them, as it did `enum.bltns` for `builtins`.
    #
    # A control that refuses working code while missing the payload it was
    # written for is worse than no control, because it implies a guarantee it
    # does not provide. What holds instead is the dynamic layer, which refuses
    # the *operation* however the module was reached: process creation by
    # namespace (`capabilities._PROCESS_EVENT_PREFIXES`) and FFI by namespace
    # (`capabilities._FFI_EVENT_PREFIXES`). Both are asserted in
    # `tests/pytest/test_plugin_capabilities.py`.
    #
    # Closed path, recorded so it is not reopened. An alias such as
    # `collections._sys` reached `sys.modules`, so a plugin could read
    # `os.environ` -- which raises no audit event -- and call into any module
    # the host had already imported, including `backend.core.auth`, whose
    # `get_bearer_token()` returned the live control-plane token. Measured
    # byte-for-byte against `backend/storage/.mech_api_token`. No purely static
    # rule can close that, because the pivot is a normal attribute read on an
    # allowlisted module under a name that is not a dunder.
    #
    # What closes it is `_guarded_module` below: the module the plugin receives
    # is a `types.ModuleType` proxy whose `__getattr__` refuses
    # `ESCAPE_MODULE_ATTRS`. The proxy wraps the real module in a closure rather
    # than an attribute, so the reference is not recoverable from the proxy's
    # own `__dict__`, and legitimate uses keep working because forwarding a
    # *function* (`collections.namedtuple`) leaves that function bound to the
    # real module globals, where its internal `_sys._getframe(1)` still
    # resolves. `analyze_ast` refuses the literal form first, so the install-time
    # gate and the runtime guard fail independently.

    @staticmethod
    def _is_dunder_literal(node: ast.AST) -> bool:
        """True for a bare ``__x__`` string, the getattr-escape payload."""
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 4
            and node.value.startswith('__')
            and node.value.endswith('__')
            and node.value.strip('_') != ''
        )

    @classmethod
    def _check_import(cls, node):
        """Allowlist every imported module, not just the first alias."""
        if isinstance(node, ast.ImportFrom):
            # A relative import (``from . import x``) has level >= 1 and its
            # module may be None, so the old root check skipped it entirely.
            if node.level:
                raise SecurityViolation(
                    "Relative imports are forbidden in sandbox.")
            if not node.module:
                raise SecurityViolation("Malformed relative import.")
            modules = [node.module]
        else:
            # ``import json, os`` must validate *every* alias, not names[0].
            modules = [alias.name for alias in node.names]

        for module in modules:
            root = module.split('.')[0]
            if root in cls.ALLOWED_IMPORT_ROOTS:
                continue
            if any(module == prefix[:-1] or module.startswith(prefix)
                   for prefix in cls.ALLOWED_IMPORT_PREFIXES):
                continue
            raise SecurityViolation(
                f"Importing module '{module}' is forbidden in sandbox.")

        # `from collections import _sys` is the same pivot as the attribute
        # form, spelled so that no `ast.Attribute` node is produced at all.
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name in cls.ESCAPE_MODULE_ATTRS:
                    raise SecurityViolation(
                        f"Importing '{node.module}.{alias.name}' is forbidden "
                        "in sandbox.")

    @classmethod
    def analyze_ast(cls, source_code: str):
        """
        Static analysis to block forbidden imports and interpreter escapes.

        This runs at install time *and* immediately before execution. It is
        the only automated gate on plugin code, so it fails closed.
        """
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                cls._check_import(node)

            elif isinstance(node, ast.Attribute):
                if node.attr in cls.FORBIDDEN_ATTRS:
                    raise SecurityViolation(
                        f"Accessing attribute '{node.attr}' is forbidden in sandbox.")
                # `collections._sys`, `logging.os`, `enum.bltns`: a read of a
                # module-valued attribute on a name imported from the
                # allowlist. Narrowed to a Name base so `self.os` stays legal,
                # and to a Load so `self.os = 'win32'` stays legal.
                if (node.attr in cls.ESCAPE_MODULE_ATTRS
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in cls.ALLOWED_IMPORT_ROOTS):
                    raise SecurityViolation(
                        f"Accessing '{node.value.id}.{node.attr}' is forbidden "
                        "in sandbox.")

            elif isinstance(node, ast.Name):
                if node.id in cls.FORBIDDEN_NAMES:
                    raise SecurityViolation(
                        f"Referencing '{node.id}' is forbidden in sandbox.")

            elif cls._is_dunder_literal(node):
                raise SecurityViolation(
                    "Dunder string literals are forbidden in sandbox.")

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ('eval', 'exec', 'open', '__import__'):
                    raise SecurityViolation(
                        f"Calling function '{node.func.id}' is forbidden.")
                         
    @classmethod
    def _module_permitted(cls, module: str) -> bool:
        """True when ``module`` is inside the import allowlist."""
        root = module.split('.')[0]
        if root in cls.ALLOWED_IMPORT_ROOTS:
            return True
        return any(module == prefix[:-1] or module.startswith(prefix)
                   for prefix in cls.ALLOWED_IMPORT_PREFIXES)

    @classmethod
    def _guarded_module(cls, module):
        """A view of ``module`` that refuses reads of ``ESCAPE_MODULE_ATTRS``.

        The real module is captured in a closure rather than stored on the
        proxy, so it is not recoverable from the proxy's own ``__dict__`` or
        through ``type()``. The plugin cannot reach the same object under a
        different name because computed access needs ``getattr``/``vars``/
        ``dir`` (absent from ``ALLOWED_BUILTINS``) and dunder reads are refused
        by ``FORBIDDEN_ATTRS``.

        Two checks, and the second is the one that generalises. ``name`` against
        ``ESCAPE_MODULE_ATTRS`` refuses the known aliases directly. Beyond that,
        a forwarded *module* must itself pass ``_module_permitted``: the
        allowlist is a rule about which modules may be reached, so it has to
        hold for an attribute as much as for an import statement. Without that
        rule the guard would be a list of spellings rather than a boundary.

        Forwarding a *function* returns that function unchanged, still bound to
        the real module's globals, so internal stdlib uses keep working
        (`collections.namedtuple` calls `_sys._getframe(1)` internally) while
        the plugin's own read of `collections._sys` is refused. A forwarded
        *module* is itself wrapped, so `logging.handlers` cannot be used to
        step around the guard and read `logging.sys` on the real object.
        """
        escape = cls.ESCAPE_MODULE_ATTRS
        module_name = getattr(module, "__name__", "<module>")

        class _GuardedModule(types.ModuleType):
            def __getattr__(self, name):
                if name in escape:
                    raise SecurityViolation(
                        f"Accessing '{module_name}.{name}' is forbidden in "
                        "sandbox.")
                value = getattr(module, name)
                if isinstance(value, types.ModuleType):
                    # The rule, not a list of names. A module the import
                    # allowlist refuses must not be reachable as an
                    # *attribute* of a module it permits, or the allowlist only
                    # governs the spelling of an import. Measured before this
                    # rule existed: 27 (root, attribute) forwarding reads across
                    # 21 distinct non-allowlisted modules were reachable from
                    # the allowlisted roots -- `logging.traceback`,
                    # `logging.threading`, `json.codecs`, `dataclasses.inspect`,
                    # `pathlib.ntpath`, `typing.contextlib`, `statistics.random`
                    # and others -- several of which hold `sys`, `os` or
                    # `builtins`. The reachability rule closes this forwarding
                    # boundary generally across all non-allowlisted modules
                    # rather than relying on a fixed list of attribute spellings.
                    child_name = getattr(value, "__name__", "")
                    if not cls._module_permitted(child_name):
                        raise SecurityViolation(
                            f"Accessing '{module_name}.{name}' is forbidden in "
                            f"sandbox: module '{child_name}' is outside the "
                            "import allowlist.")
                    if not isinstance(value, _GuardedModule):
                        return cls._guarded_module(value)
                return value

        proxy = _GuardedModule(module_name)
        # A module's expected public surface, copied so ordinary introspection
        # does not fail. `__loader__`/`__spec__` are deliberately not copied:
        # the AST gate refuses them anyway, and they carry import machinery.
        for attr in ("__doc__", "__file__", "__package__"):
            try:
                object.__setattr__(proxy, attr, getattr(module, attr))
            except AttributeError:
                pass
        return proxy

    @classmethod
    def _guarded_import(cls, name, globals=None, locals=None, fromlist=(), level=0):
        """``__import__`` restricted to the allowlist.

        The IMPORT_NAME opcode resolves imports through the ``__import__``
        entry in builtins, so a sandbox that simply omits it breaks every
        legitimate ``import json`` in a plugin. A guarded implementation keeps
        allowed imports working while still refusing os/subprocess/socket even
        if the AST gate above is bypassed.

        What it returns is a :meth:`_guarded_module` view, not the real module.
        Allowlisting a module was never the same as trusting every attribute on
        it: `collections._sys`, `logging.os` and `enum.bltns` are live module
        objects, and handing them back was the whole of the escape.
        """
        if level:
            raise SecurityViolation("Relative imports are forbidden in sandbox.")
        if not cls._module_permitted(name):
            raise SecurityViolation(
                f"Importing module '{name}' is forbidden in sandbox.")
        return cls._guarded_module(
            __import__(name, globals, locals, fromlist, level))

    @classmethod
    def execute(cls, source_code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute plugin code under a restricted global namespace.

        The AST gate above is the primary boundary; the restricted builtins and
        guarded importer are defence in depth behind it. Neither substitutes for
        running untrusted code in a separate, capability-restricted process.
        """
        cls.analyze_ast(source_code)

        restricted_builtins = dict(cls.ALLOWED_BUILTINS)
        restricted_builtins["__import__"] = cls._guarded_import
        # `class` statements compile to a call to __build_class__, so every
        # plugin that subclasses MechPlugin needs it. It grants no escape on
        # its own — the class-object escapes (__subclasses__, the type
        # builtin) stay blocked above.
        restricted_builtins["__build_class__"] = __builtins__["__build_class__"] \
            if isinstance(__builtins__, dict) else __build_class__

        sandbox_globals: Dict[str, Any] = {
            "__builtins__": restricted_builtins,
            "__name__": "_mech_sandbox",
        }

        if context:
            sandbox_globals.update(context)

        # One namespace for both globals and locals. Passing two separate
        # dicts binds `import`/`from ... import` names into locals while class
        # bodies and functions resolve names against globals only, so any
        # plugin declaring a class over an imported base raises NameError.
        namespace: Dict[str, Any] = dict(sandbox_globals)

        try:
            exec(compile(source_code, "<mech-plugin>", "exec"), namespace)
            return namespace
        except SecurityViolation:
            raise
        except Exception as e:
            raise SecurityViolation(f"Plugin execution failed: {str(e)}")
