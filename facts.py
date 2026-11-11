"""Conservative Python facts; no imports or execution of workspace modules."""
import ast
from pathlib import PurePosixPath

def extract(path, source):
    facts = {'definitions': [], 'imports': [], 'calls': [], 'limitations': []}
    if not path.endswith('.py'):
        facts['limitations'].append('Text description only; structural extraction supports Python.')
        return facts
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        facts['limitations'].append(f'Python parse error at line {exc.lineno}: {exc.msg}')
        return facts
    class Visitor(ast.NodeVisitor):
        stack = []
        def definition(self, node):
            name = '.'.join(self.stack + [node.name])
            signature = node.name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signature += '(' + ast.unparse(node.args) + ')'
                if node.returns:
                    signature += ' -> ' + ast.unparse(node.returns)
            else:
                signature += '(' + ', '.join(ast.unparse(b) for b in node.bases) + ')'
            facts['definitions'].append({'name': name, 'signature': signature,
                'line': node.lineno, 'end': node.end_lineno, 'docstring': ast.get_docstring(node) or ''})
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()
        visit_FunctionDef = definition
        visit_AsyncFunctionDef = definition
        visit_ClassDef = definition
        def visit_Import(self, node):
            for alias in node.names:
                facts['imports'].append({'module': alias.name, 'name': '',
                    'alias': alias.asname or alias.name.split('.')[0], 'level': 0})
        def visit_ImportFrom(self, node):
            for alias in node.names:
                facts['imports'].append({'module': node.module or '', 'name': alias.name,
                    'alias': alias.asname or alias.name, 'level': node.level})
        def visit_Call(self, node):
            facts['calls'].append({'expression': ast.unparse(node.func), 'line': node.lineno,
                                  'scope': '.'.join(self.stack), 'target': 'unresolved/external/dynamic'})
            self.generic_visit(node)
    Visitor().visit(tree)
    facts['limitations'].append('Call resolution covers unambiguous imports only; dynamic dispatch and shadowing remain unresolved.')
    return facts

def resolve(all_facts):
    modules = {}
    for path in all_facts:
        if path.endswith('.py'):
            module = path[:-3].replace('/', '.')
            if module.endswith('.__init__'):
                module = module[:-9]
            modules.setdefault(module, []).append(path)
    dependencies = {}
    for path, facts in all_facts.items():
        deps = set()
        aliases = {}
        for entry in facts['imports']:
            module = entry['module']
            if entry['level']:
                parents = list(PurePosixPath(path).parent.parts)
                keep = len(parents) - entry['level'] + 1
                module = '.'.join(parents[:max(keep, 0)] + ([module] if module else []))
            targets = modules.get(module, [])
            # from package import submodule
            submodule = modules.get('.'.join(filter(None, [module, entry['name']])), [])
            if submodule:
                targets = submodule
            if len(targets) == 1:
                target = targets[0]
                deps.add(target)
                entry['target'] = target
                aliases[entry['alias']] = (target, entry['name'] if not submodule else '')
        # Record possible imported definitions, rather than claiming lexical binding analysis.
        for call in facts['calls']:
            head, _, tail = call['expression'].partition('.')
            if head in aliases:
                target, imported = aliases[head]
                symbol = '.'.join(filter(None, [imported, tail]))
                matches = [d for d in all_facts[target]['definitions'] if d['name'] == symbol]
                if len(matches) == 1:
                    call['candidate'] = f"{target}:{matches[0]['line']} ({symbol}; binding not verified)"
        dependencies[path] = sorted(deps)
    return dependencies
