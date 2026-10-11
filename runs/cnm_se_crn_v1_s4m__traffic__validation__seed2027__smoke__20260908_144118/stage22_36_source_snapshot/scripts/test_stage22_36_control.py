#!/usr/bin/env python3
import copy
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('baselines/S4M/s4m','baselines/S4M/s4m/model',
    'baselines/S4M/s4m/model/S4/s4','baselines/S4M/s4m/model/S4/s4/extensions/kernels')]
from utils.paired_training_rng import stream,AuditedRandomSampler,common_state_hash,training_stream
from stage22_35_diagnose import parsed_namespace
from run_s4m_experiment import validate_experiment_config as validate_config
import run_stage22_36_queue as queue
EVIDENCE={}


class Tests(unittest.TestCase):
    def test_decorator_keyword_forward_and_backward(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.config=SimpleNamespace(seed=2027,stage22_36_paired_training_rng=True)
            @training_stream('toy')
            def forward(self,Q): return Q*torch.rand_like(Q)
        model=Toy();x=torch.ones(8,requires_grad=True);before=torch.get_rng_state().clone()
        y=model(Q=x);y.sum().backward()
        self.assertTrue(torch.equal(y.detach(),model(x).detach()))
        self.assertIsNotNone(x.grad);self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_aux_stream_restores_cpu_and_primary_draws(self):
        args=SimpleNamespace(seed=2027,stage22_36_paired_training_rng=True,_paired_epoch=2,_paired_batch=7)
        with stream(args,'primary'):
            a=torch.rand(9); expected=torch.rand(9)
        before=torch.get_rng_state().clone()
        with stream(args,'primary'):
            self.assertTrue(torch.equal(a,torch.rand(9)))
            with stream(args,'aux'): torch.rand(700)
            self.assertTrue(torch.equal(expected,torch.rand(9)))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_disabled_preserves_legacy_rng_and_exception_restores(self):
        args=SimpleNamespace(seed=2027,stage22_36_paired_training_rng=False)
        torch.manual_seed(1); expected=torch.rand(3);torch.manual_seed(1)
        with stream(args,'ignored'): actual=torch.rand(3)
        self.assertTrue(torch.equal(expected,actual))
        args.stage22_36_paired_training_rng=True;before=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with stream(args,'aux'): torch.rand(5);raise RuntimeError('test')
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_sampler_order_is_isolated_across_epochs_and_streams(self):
        a=AuditedRandomSampler(range(77),2027,'train');b=AuditedRandomSampler(range(77),2027,'train')
        orders=[]
        for _ in range(3):
            first=list(a);torch.rand(1234);self.assertEqual(first,list(b));orders.append(first)
        self.assertEqual(a.history,b.history);self.assertNotEqual(orders[0],orders[1])
        self.assertEqual(sorted(orders[0]),list(range(77)))

    def test_formal_budget_and_dependency_isolation(self):
        configs=list((ROOT/'configs/stage22_36_control').glob('*_train_*.yaml'))
        self.assertEqual(len(configs),4)
        for path in configs:
            cfg=yaml.safe_load(path.read_text());validate_config(cfg)
            source=yaml.safe_load(Path(cfg['provenance']['source_config']).read_text())
            expected=copy.deepcopy(source['official_args']);expected['stage22_36_paired_training_rng']=True
            self.assertEqual(cfg['official_args'],expected)
            self.assertEqual(cfg['data'],source['data'])
        jobs={j['id']:j for j in queue.scheduler.JOB_DEFINITIONS}
        self.assertEqual(len(jobs),9)
        self.assertEqual(jobs['s4m_train_formal']['depends_on'],['s4m_eval_smoke'])
        job=copy.deepcopy(jobs['pg95_train_formal']);job['required_memory_estimate_mib']=22000
        queue.refresh(job,{})
        self.assertGreaterEqual(job['required_memory_estimate_mib'],22000)

    def test_real_models_common_initial_state_identical(self):
        from model.S4M import Model
        report=json.loads((ROOT/'reports/stage22_35_diagnosis.json').read_text())
        source=next(r for r in report['records'] if r['seed']==2027)['sources']
        states=[]
        for family in ('s4m','pg95'):
            args=parsed_namespace(Path(source[family]['source'],'train.log').read_text())
            args.stage22_36_paired_training_rng=True
            torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
            model=Model(args).float()
            h,n=common_state_hash(model)
            states.append((h,n))
            del model
        self.assertEqual(states[0],states[1]);self.assertEqual(states[0][1],314)
        EVIDENCE.update(expected_common_sha256=states[0][0],common_entries=314,changed_common_entries=0)


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (ROOT/'reports/stage22_36_tests.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,**EVIDENCE),indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
