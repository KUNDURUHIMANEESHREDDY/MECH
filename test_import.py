import sys
import os
backend_dir = os.path.abspath('backend')
sys.path.insert(0, backend_dir)
print('path set')

# Try importing knowledge_graph package
import importlib.util
spec = importlib.util.spec_from_file_location("knowledge_graph", os.path.join(backend_dir, "knowledge_graph", "__init__.py"))
kg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kg)
print('knowledge_graph OK')
