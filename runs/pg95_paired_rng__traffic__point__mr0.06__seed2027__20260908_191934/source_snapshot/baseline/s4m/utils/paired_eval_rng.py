"""Opt-in common random numbers for frozen-checkpoint evaluation only."""
from contextlib import contextmanager
import hashlib
import torch


@contextmanager
def paired_eval_rng(enabled, seed, batch_index, device):
    if not enabled:
        yield
        return
    device = torch.device(device)
    devices = [] if device.type != 'cuda' else [
        device.index if device.index is not None else torch.cuda.current_device()]
    key = f'stage22_34_crn_v1:{int(seed)}:{int(batch_index)}'.encode()
    derived = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(derived)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(derived)
        yield
