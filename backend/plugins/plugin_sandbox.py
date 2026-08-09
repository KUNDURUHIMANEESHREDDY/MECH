import ast
import types
from typing import Dict, Any

class SecurityViolation(Exception):
    pass

class PluginSandbox:
    """
    Provides a restrictive environment for executing plugin code.
    Uses strict AST whitelisting to prevent dangerous operations.
    """
    
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
        ast.NameConstant,
        ast.Subscript,
        ast.Slice,
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
        
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            raise SecurityViolation("Import statements are forbidden in sandbox.")
        
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in cls.DANGEROUS_NAMES:
                    raise SecurityViolation(f"Calling '{node.func.id}' is forbidden.")
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in cls.DANGEROUS_ATTRS or attr_name.startswith('__'):
                    raise SecurityViolation(f"Accessing attribute '{attr_name}' is forbidden.")
        
        if isinstance(node, ast.Attribute):
            if node.attr in cls.DANGEROUS_ATTRS or node.attr.startswith('__'):
                raise SecurityViolation(f"Accessing attribute '{node.attr}' is forbidden.")
        
        if isinstance(node, ast.Name):
            if node.id in cls.DANGEROUS_NAMES:
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
        
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
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
    def execute(cls, source_code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes code safely within a restricted global namespace.
        """
        cls.analyze_ast(source_code)
        
        sandbox_globals = {
            "__builtins__": cls.SAFE_BUILTINS,
        }
        
        if context:
            for key, value in context.items():
                if key in cls.DANGEROUS_NAMES:
                    raise SecurityViolation(f"Context key '{key}' is forbidden.")
                sandbox_globals[key] = value
        
        sandbox_locals = {}
        
        try:
            code = compile(source_code, '<plugin>', 'exec')
            exec(code, sandbox_globals, sandbox_locals)
            return sandbox_locals
        except SecurityViolation:
            raise
        except Exception as e:
            raise SecurityViolation(f"Plugin execution failed: {str(e)}")
