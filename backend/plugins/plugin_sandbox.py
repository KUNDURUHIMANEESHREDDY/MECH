import ast
import sys
import time
from typing import Any, Dict, Optional

class SecurityViolation(Exception):
    """Raised when plugin code attempts a forbidden operation or exceeds its budget."""
    pass


class _RestrictedProxy:
    """Wraps an arbitrary object so plugins cannot reach dunder / introspection
    attributes at runtime (defense-in-depth alongside the static AST checks).

    Normal operations (attribute access, item access, iteration, calling) are
    delegated to the wrapped object. Dunder access that is not required for
    ordinary use (e.g. ``__class__``, ``__subclasses__``, ``__globals__``) is
    blocked. Primitive values are never wrapped so arithmetic / string ops keep
    working transparently.
    """

    __slots__ = ("_wrapped",)

    # Dunders required for ordinary object use. Everything else dunder is blocked.
    _ALLOWED = {
        "__getitem__", "__setitem__", "__delitem__",
        "__iter__", "__len__", "__contains__", "__call__",
        "__next__", "__enter__", "__exit__", "__reversed__",
    }

    def __init__(self, obj):
        object.__setattr__(self, "_wrapped", obj)

    def __getattribute__(self, name):
        if name in ("_wrapped", "_ALLOWED"):
            return object.__getattribute__(self, name)
        allowed = object.__getattribute__(self, "_ALLOWED")
        if name.startswith("__") and name not in allowed:
            raise SecurityViolation(
                f"Access to '{name}' is forbidden in the plugin sandbox."
            )
        wrapped = object.__getattribute__(self, "_wrapped")
        return _wrap(getattr(wrapped, name))

    def __setattr__(self, name, value):
        allowed = object.__getattribute__(self, "_ALLOWED")
        if name.startswith("__") and name not in allowed:
            raise SecurityViolation(
                f"Setting '{name}' is forbidden in the plugin sandbox."
            )
        wrapped = object.__getattribute__(self, "_wrapped")
        setattr(wrapped, name, _wrap(value))

    def __getitem__(self, key):
        return _wrap(self._wrapped[key])

    def __setitem__(self, key, value):
        self._wrapped[key] = _wrap(value)

    def __delitem__(self, key):
        del self._wrapped[key]

    def __iter__(self):
        return iter(self._wrapped)

    def __len__(self):
        return len(self._wrapped)

    def __contains__(self, item):
        return item in self._wrapped

    def __call__(self, *args, **kwargs):
        return _wrap(self._wrapped(*args, **kwargs))

    def __next__(self):
        return _wrap(next(self._wrapped))

    def __enter__(self):
        return _wrap(self._wrapped.__enter__())

    def __exit__(self, *exc):
        return self._wrapped.__exit__(*exc)


_PRIMITIVES = (type(None), bool, int, float, str, bytes, complex)


def _wrap(obj):
    """Wrap non-primitive objects in a _RestrictedProxy; primitives pass through."""
    if obj is None or isinstance(obj, _PRIMITIVES) or isinstance(obj, type):
        return obj
    if isinstance(obj, _RestrictedProxy):
        return obj
    return _RestrictedProxy(obj)


class PluginSandbox:
    """
    Provides a restrictive environment for executing plugin code.

    Defense is layered:
      1. Static AST whitelisting blocks forbidden imports, calls and dunder
         attribute access before any code runs.
      2. A restricted builtins table removes dangerous callables.
      3. Context objects are wrapped in a runtime proxy that blocks dunder /
         introspection access even if the static check is bypassed.
      4. A wall-clock timeout terminates runaway plugin loops.

    NOTE: this is defense-in-depth, not a hard isolation boundary. A determined
    attacker with arbitrary Python can still find escapes (e.g. via C extension
    objects passed in through the context). It is suitable for restricting
    first-party / vetted plugins, not for executing fully untrusted code.
    """

    # Default wall-clock budget for a single plugin execution (seconds).
    # Pure-Python loops are interrupted; C-extension busy loops are not.
    DEFAULT_TIMEOUT = 15.0

    SAFE_BUILTINS = {
        'print': print,
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
        'type': type,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'enumerate': enumerate,
        'zip': zip,
        'sorted': sorted,
        'reversed': reversed,
        'map': map,
        'filter': filter,
        'Exception': Exception,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'KeyError': KeyError,
        'IndexError': IndexError,
        'AttributeError': AttributeError,
        'None': None,
        'True': True,
        'False': False,
        '__build_class__': __build_class__,
    }

    DANGEROUS_NAMES = {
        'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
        'delattr', 'hasattr', 'open', 'input', '__builtins__', 'breakpoint',
        'globals', 'locals', 'vars', 'dir', 'help', 'exit', 'quit',
        'copyright', 'credits', 'license',
    }

    DANGEROUS_ATTRS = {
        '__class__', '__bases__', '__subclasses__', '__import__',
        '__builtins__', '__globals__', '__code__', '__closure__',
        '__func__', '__self__', '__module__', '__dict__', '__slots__',
        '__getattribute__', '__setattr__', '__delattr__', '__init__',
        '__new__', '__call__', '__iter__', '__next__', '__enter__',
        '__exit__', '__repr__', '__str__', '__bytes__', '__format__',
        '__hash__', '__bool__', '__len__', '__getitem__', '__setitem__',
        '__delitem__', '__contains__', '__add__', '__sub__', '__mul__',
        '__truediv__', '__floordiv__', '__mod__', '__pow__', '__lt__',
        '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__and__',
        '__or__', '__xor__', '__lshift__', '__rshift__', '__iadd__',
        '__isub__', '__imul__', '__itruediv__', '__ifloordiv__',
        '__imod__', '__ipow__', '__ilshift__', '__irshift__',
        '__iand__', '__ior__', '__ixor__', '__neg__', '__pos__',
        '__abs__', '__invert__', '__complex__', '__int__', '__float__',
        '__round__', '__trunc__', '__floor__', '__ceil__', '__index__',
        '__sizeof__', '__reduce__', '__reduce_ex__', '__getstate__',
        '__setstate__', '__del__', '__copy__', '__deepcopy__',
        '__getnewargs__', '__getnewargs_ex__', '__format__',
        '__match_args__', '__init_subclass__', '__set_name__',
        '__class_getitem__', '__mro_entries__', '__prepare__',
    }

    ALLOWED_NODE_TYPES = {
        ast.Module,
        ast.Expr,
        # Operator / expression-context nodes (no children, safe to permit).
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.MatMult, ast.UAdd, ast.USub, ast.Not, ast.Invert,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.While,
        ast.If,
        ast.Raise,
        ast.Assert,
        ast.Pass,
        ast.Break,
        ast.Continue,
        ast.Return,
        ast.Yield,
        ast.YieldFrom,
        ast.Compare,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.IfExp,
        ast.Dict,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Name,
        ast.Constant,
        ast.Subscript,
        ast.Slice,
        ast.Attribute,
        ast.Call,
        ast.Starred,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Lambda,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.NamedExpr,
        ast.keyword,
        ast.arguments,
        ast.arg,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.With,
        ast.AsyncWith,
        ast.AsyncFor,
        ast.Await,
        ast.FormattedValue,
        ast.JoinedStr,
    }

    ALLOWED_BIN_OPS = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.MatMult,
    }

    ALLOWED_UNARY_OPS = {
        ast.UAdd, ast.USub, ast.Not, ast.Invert,
    }

    ALLOWED_CMP_OPS = {
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
    }

    @classmethod
    def _check_node(cls, node):
        """Recursively check AST node for dangerous patterns."""
        if type(node) not in cls.ALLOWED_NODE_TYPES:
            raise SecurityViolation(f"Node type '{type(node).__name__}' is forbidden in sandbox.")

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SecurityViolation("Import statements are forbidden in sandbox.")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in cls.DANGEROUS_NAMES:
                    raise SecurityViolation(f"Calling '{node.func.id}' is forbidden.")
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in cls.DANGEROUS_ATTRS or attr_name.startswith('__'):
                    raise SecurityViolation(f"Calling attribute '{attr_name}' is forbidden.")

        if isinstance(node, ast.Attribute):
            if node.attr in cls.DANGEROUS_ATTRS or node.attr.startswith('__'):
                raise SecurityViolation(f"Accessing attribute '{node.attr}' is forbidden.")

        if isinstance(node, ast.Name):
            if node.id in cls.DANGEROUS_NAMES or node.id.startswith('__'):
                raise SecurityViolation(f"Using name '{node.id}' is forbidden.")

        if isinstance(node, ast.BinOp):
            if type(node.op) not in cls.ALLOWED_BIN_OPS:
                raise SecurityViolation(f"Binary operator '{type(node.op).__name__}' is forbidden.")

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in cls.ALLOWED_UNARY_OPS:
                raise SecurityViolation(f"Unary operator '{type(node.op).__name__}' is forbidden.")

        if isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in cls.ALLOWED_CMP_OPS:
                    raise SecurityViolation(f"Comparison operator '{type(op).__name__}' is forbidden.")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                cls._check_node(decorator)
            for arg in node.args.args:
                cls._check_node(arg)
            if node.args.vararg:
                cls._check_node(node.args.vararg)
            if node.args.kwarg:
                cls._check_node(node.args.kwarg)
            for default in node.args.defaults:
                cls._check_node(default)
            for default in node.args.kw_defaults:
                if default:
                    cls._check_node(default)

        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                cls._check_node(base)
            for decorator in node.decorator_list:
                cls._check_node(decorator)
            for keyword in node.keywords:
                cls._check_node(keyword)
            for stmt in node.body:
                cls._check_node(stmt)

        for child in ast.iter_child_nodes(node):
            cls._check_node(child)

    @classmethod
    def analyze_ast(cls, source_code: str):
        """
        Static analysis to block forbidden imports, calls, and attribute access.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise SecurityViolation(f"Syntax error: {e}")

        cls._check_node(tree)

    @classmethod
    def execute(
        cls,
        source_code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes code safely within a restricted global namespace.

        ``timeout`` is the wall-clock budget in seconds. ``None`` (the default)
        uses :data:`DEFAULT_TIMEOUT`; pass ``0`` or a negative value to disable
        the timeout entirely.
        """
        cls.analyze_ast(source_code)

        sandbox_globals: Dict[str, Any] = {
            "__builtins__": cls.SAFE_BUILTINS,
            "__name__": "__plugin__",
        }

        if context:
            for key, value in context.items():
                if key in cls.DANGEROUS_NAMES or key.startswith("__"):
                    raise SecurityViolation(f"Context key '{key}' is forbidden.")
                sandbox_globals[key] = _wrap(value)

        sandbox_locals: Dict[str, Any] = {}

        budget = cls.DEFAULT_TIMEOUT if timeout is None else timeout
        use_timeout = budget is not None and budget > 0

        try:
            code = compile(source_code, '<plugin>', 'exec')
            if use_timeout:
                deadline = time.monotonic() + budget

                def _tracer(frame, event, arg):  # noqa: ANN001, ARG001
                    if time.monotonic() > deadline:
                        raise SecurityViolation(
                            f"Plugin execution exceeded the {budget:g}s time limit."
                        )
                    return _tracer

                previous = sys.gettrace()
                sys.settrace(_tracer)
                try:
                    exec(code, sandbox_globals, sandbox_locals)
                finally:
                    sys.settrace(previous)
            else:
                exec(code, sandbox_globals, sandbox_locals)
            return sandbox_locals
        except SecurityViolation:
            raise
        except Exception as e:
            raise SecurityViolation(f"Plugin execution failed: {str(e)}")
