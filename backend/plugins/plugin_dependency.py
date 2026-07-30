from typing import Dict, List, Set

class PluginDependencyResolver:
    """
    Resolves plugin dependencies and determines load order using topological sort.
    """
    
    @staticmethod
    def resolve(plugin_deps: Dict[str, List[str]]) -> List[str]:
        """
        Takes a dict of plugin_name -> [dependency_names].
        Returns a valid load order, or raises Exception if cycle detected.
        """
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []

        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving plugin '{node}'")
            if node not in visited:
                temp_visited.add(node)
                for dep in plugin_deps.get(node, []):
                    visit(dep)
                temp_visited.remove(node)
                visited.add(node)
                order.append(node)

        for plugin in plugin_deps:
            if plugin not in visited:
                visit(plugin)

        return order
