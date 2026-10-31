"""Load named, unmodified definitions from the pinned original source for audits.

Avoids executing unrelated top-level CLIs/imports (some launch GPUs). Algorithms
are compiled from original AST nodes without transformations. This is not a
replacement implementation or authorization to resolve upstream ambiguities.
"""
import ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'references/eval_political_worldviews'

def load_definitions(relative,namespace,names=None,include_assignments=False):
    path=SOURCE/relative
    tree=ast.parse(path.read_text(),filename=str(path));nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names):nodes.append(node)
        elif include_assignments and isinstance(node,ast.Assign):nodes.append(node)
    module=ast.Module(body=nodes,type_ignores=[])
    exec(compile(module,str(path),'exec'),namespace)
    return {'source':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'loaded_AST_sha256':hashlib.sha256(ast.dump(module).encode()).hexdigest(),'definitions':[n.name for n in nodes if hasattr(n,'name')]}
