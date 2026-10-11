#!/usr/bin/env python3
"""CPU contracts for Step-22.30 PBA-CMVD."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
]
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
from utils.tools import EarlyStopping  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from run_s4m_experiment import verify_checkpoint  # noqa: E402


class MemoryBox(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory = torch.tensor([1.0])
        self.repre = torch.tensor([2.0])
        self.num_dic = torch.tensor([3.0])


class TinyPBA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3)
        self.s4_layers = torch.nn.ModuleList([
            torch.nn.Linear(3, 3) for _ in range(8)
        ])
        self.norm = torch.nn.LayerNorm(3)
        self.out_proj = torch.nn.Linear(3, 2)
        self.mem_net = torch.nn.Module()
        self.mem_net.mipp = torch.nn.Linear(1, 1, bias=False)
        self.mem_net.memory = MemoryBox()


def main() -> int:
    torch.manual_seed(2027)
    selection = {
        key: Exp._pba_state_key_selected(key, 8, 2)
        for key in (
            "frozen.weight", "s4_layers.5.weight", "s4_layers.6.weight",
            "s4_layers.7.bias", "norm.weight", "out_proj.bias",
            "module.s4_layers.6.bias", "module.out_proj.weight",
        )
    }
    student = TinyPBA()
    teacher = copy.deepcopy(student)
    storage_disjoint = all(
        dict(student.named_parameters())[name].data_ptr()
        != dict(teacher.named_parameters())[name].data_ptr()
        for name, _ in student.named_parameters()
    )

    target = torch.zeros(2, 2, 3)
    source = torch.tensor([
        [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]],
    ])
    frozen_teacher = torch.tensor([
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
    ], requires_grad=True)
    student_bias = torch.nn.Parameter(torch.tensor(0.25))
    prediction = source + student_bias
    observed = torch.ones_like(target, dtype=torch.bool)
    diagnostics = Exp._bf_cmvd_objective(
        student_prediction=prediction,
        backbone_prediction=source,
        teacher_prediction=frozen_teacher,
        target=target,
        target_mask=observed,
        filter_granularity="token",
    )
    diagnostics["total_loss"].backward()

    delta_checks = {}
    with tempfile.TemporaryDirectory(prefix="pba_delta_") as temporary:
        root = Path(temporary)
        source_path = root / "source.pth"
        trained = TinyPBA()
        source_state = {
            key: value.detach().clone()
            for key, value in trained.state_dict().items()
        }
        torch.save({
            "model": source_state,
            "memory": torch.tensor([11.0]),
            "repre": torch.tensor([12.0]),
            "num_dic": torch.tensor([13.0]),
        }, source_path)
        with torch.no_grad():
            trained.frozen.weight.add_(5.0)
            trained.s4_layers[6].weight.add_(1.0)
            trained.s4_layers[7].bias.sub_(2.0)
            trained.norm.bias.add_(3.0)
            trained.out_proj.weight.mul_(0.5)
        args = SimpleNamespace(
            model="S4M",
            pba_delta_checkpoint=True,
            pba_backbone_checkpoint=str(source_path),
            pba_tail_layers=2,
            e_layers=8,
            mcpr_delta_checkpoint=False,
        )
        saver = EarlyStopping(args, patience=1, more_save=True)
        saver.save_checkpoint(1.0, trained, str(root))
        delta_path = root / "checkpoint.pth"
        verification = verify_checkpoint(
            delta_path, require_mipp=True, require_pba=True
        )
        reconstructed = TinyPBA()
        fake_experiment = SimpleNamespace(
            model=reconstructed,
            device=torch.device("cpu"),
            _file_sha256=Exp._file_sha256,
        )
        Exp.load_model(fake_experiment, str(delta_path))
        reconstructed_state = reconstructed.state_dict()
        expected_selected = {
            key: value for key, value in trained.state_dict().items()
            if Exp._pba_state_key_selected(key, 8, 2)
        }
        expected_frozen = {
            key: value for key, value in source_state.items()
            if not Exp._pba_state_key_selected(key, 8, 2)
        }
        bad_source_rejected = False
        tampered = torch.load(source_path, map_location="cpu")
        tampered["memory"] = torch.tensor([99.0])
        torch.save(tampered, source_path)
        try:
            verify_checkpoint(delta_path, require_pba=True)
        except RuntimeError:
            bad_source_rejected = True
        delta_checks = {
            "format": verification["checkpoint_format"] == "pba_delta_v1",
            "tail_depth": verification["checkpoint_pba_tail_layers"] == 2,
            "manifest_nonempty": bool(verification["checkpoint_pba_state_keys"]),
            "selected_reconstructed": all(
                torch.equal(reconstructed_state[key], value)
                for key, value in expected_selected.items()
            ),
            "frozen_reconstructed_from_source": all(
                torch.equal(reconstructed_state[key], value)
                for key, value in expected_frozen.items()
            ),
            "memory_reconstructed": (
                torch.equal(reconstructed.mem_net.memory.memory, torch.tensor([11.0]))
                and torch.equal(reconstructed.mem_net.memory.repre, torch.tensor([12.0]))
                and torch.equal(reconstructed.mem_net.memory.num_dic, torch.tensor([13.0]))
            ),
            "tampered_source_rejected": bad_source_rejected,
        }

    checks = {
        "only_frozen_prefix_rejected": selection["frozen.weight"] is False,
        "layer5_rejected": selection["s4_layers.5.weight"] is False,
        "layer6_selected": selection["s4_layers.6.weight"] is True,
        "layer7_selected": selection["s4_layers.7.bias"] is True,
        "norm_selected": selection["norm.weight"] is True,
        "out_proj_selected": selection["out_proj.bias"] is True,
        "dataparallel_prefix_supported": (
            selection["module.s4_layers.6.bias"] is True
            and selection["module.out_proj.weight"] is True
        ),
        "teacher_student_storage_disjoint": storage_disjoint,
        "token_filter_has_wins_and_rejections": (
            int(diagnostics["beneficial_mask"].sum()) == 2
            and int(diagnostics["eligible_mask"].sum()) == 4
        ),
        "student_receives_gradient": (
            student_bias.grad is not None
            and torch.isfinite(student_bias.grad)
            and float(student_bias.grad.abs()) > 0.0
        ),
        "teacher_is_detached": frozen_teacher.grad is None,
        "distillation_is_positive": float(diagnostics["distill_l1"]) > 0.0,
        **{f"delta_{key}": value for key, value in delta_checks.items()},
    }
    report = {
        "step": "22.30",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports/stage22_30_pba_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
