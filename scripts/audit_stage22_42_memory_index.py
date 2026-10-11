#!/usr/bin/env python3
"""CPU synthetic audit of original memory-update methods; never patch them."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from stage22_42_diagnostic import ROOT, digest, write_json


def reproduce(path, capacity):
    source = path.read_text()
    parsed = ast.parse(source)
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'FeatureMemory')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'add_features_from_sample_learned')
    module = ast.Module(body=[method], type_ignores=[])
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    old = [torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])]
    state = SimpleNamespace(memory=old, repre=torch.cat(old), memory_size=capacity,
                            num_dic=[1, 1]+[0]*(capacity-2), M=1, count=0,
                            thres1=.95, thres2=.6, per_mem_size=5)
    with torch.no_grad():
        scope[method.name](state, torch.tensor([[-1., 0.]]))
    resident = state.memory[-1].mean(dim=0)
    representative = state.repre[-1]
    assert torch.equal(resident, torch.tensor([-1., 0.]))
    assert torch.equal(representative, torch.tensor([0., 1.]))
    assert not torch.equal(resident, representative)
    return dict(source=str(path), source_sha256=digest(path), method_start_line=method.lineno,
        capacity=capacity, source_method_used_without_modification=True,
        actual_latest_resident_mean=resident.tolist(), actual_latest_representative=representative.tolist(),
        mismatch_l2=float(torch.linalg.vector_norm(resident-representative)), confirmed_index_error=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    paths = [ROOT / 'baselines/S4M/s4m/model/Memory.py', ROOT / 'upstream/S4M/s4m/model/Memory.py']
    hashes = {str(p): digest(p) for p in paths}
    cases = [reproduce(path, capacity) for path in paths for capacity in (4, 2)]
    assert hashes == {str(p): digest(p) for p in paths}
    assert not torch.cuda.is_initialized()
    payload = dict(stage='22.42', status='SUCCESS', passed_cases=4, records=cases,
        device='cpu', no_model_instantiated=True, no_model_forward=True, no_new_training=True,
        uses_test_split=False, source_unchanged=True, baseline_or_upstream_modified=False,
        decision='UPSTREAM_NEW_CLUSTER_CENTROID_INDEX_ERROR_CONFIRMED_NOT_SEED2028_CAUSAL_PROOF',
        causal_limit='Both original and distance-fixed models inherit the bug. No effect on forecasting or seed2028 attribution measured.',
        script_sha256=digest(Path(__file__)), source_hashes=hashes)
    write_json(ROOT / 'reports/stage22_42_memory_index_audit.json', payload)
    print('PASS 4 original-method CPU fixtures; inherited index error confirmed; no causal inference')
