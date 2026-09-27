import ast
import types
from typing import Dict, Any

class SecurityViolation(Exception):
    pass

class PluginSandbox:
    """
    Provides a restrictive environment for executing plugin code.
    Prevents access to sensitive built-ins (like file IO, os, sys).
    """
    
    ALLOWED_BUILTINS = {
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
        'Exception': Exception,
        'ValueError': ValueError,
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

    @classmethod
    def analyze_ast(cls, source_code: str):
        """
        Static analysis to block forbidden imports and calls.
        """
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                module = node.names[0].name if isinstance(node, ast.Import) else node.module
                if module is None:
                    continue
                root = module.split('.')[0]
                if root in cls.ALLOWED_IMPORT_ROOTS:
                    continue
                if any(module == prefix[:-1] or module.startswith(prefix)
                       for prefix in cls.ALLOWED_IMPORT_PREFIXES):
                    continue
                raise SecurityViolation(f"Importing module '{module}' is forbidden in sandbox.")
                    
            if isinstance(node, ast.Call):
                 if isinstance(node.func, ast.Name):
                     if node.func.id in ['eval', 'exec', 'open']:
                         raise SecurityViolation(f"Calling function '{node.func.id}' is forbidden.")
                         
    @classmethod
    def execute(cls, source_code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes code safely within a restricted global namespace.
        """
        cls.analyze_ast(source_code)
        
        sandbox_globals = {
            "__builtins__": cls.ALLOWED_BUILTINS.copy()
        }
        
        if context:
            sandbox_globals.update(context)
            
        sandbox_locals = {}
        
        try:
            # Compiled with restrictive globals
            exec(source_code, sandbox_globals, sandbox_locals)
            return sandbox_locals
        except Exception as e:
            raise SecurityViolation(f"Plugin execution failed: {str(e)}")
