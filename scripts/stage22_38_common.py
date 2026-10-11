"""Small CPU-only artifact helpers for the frozen imputation diagnosis."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / 'runtime/stage22_38_diagnostic'
STATE = RT / 'state.json'
CONFIG = ROOT / 'configs/stage22_38_diagnostic'
MANIFEST = RT / 'config_manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text); os.replace(tmp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def namespace_values(log):
    line = next(x for x in log.splitlines() if x.startswith('Namespace('))
    node = ast.parse(line, mode='eval').body
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Namespace'
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg != 'device'}
