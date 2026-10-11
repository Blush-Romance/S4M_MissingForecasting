"""Opt-in Step22.36 random-stream coupling. No model/loss/budget changes."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import torch
from torch.utils.data import Sampler


def seed_for(seed, name, epoch=0, batch=0):
    key=f'stage22_36_v1:{seed}:{name}:{epoch}:{batch}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % (2**63-1)


@contextmanager
def stream(args, name, device='cpu'):
    if not getattr(args,'stage22_36_paired_training_rng',False):
        yield
        return
    device=torch.device(device)
    devices=[] if device.type!='cuda' else [device.index if device.index is not None else torch.cuda.current_device()]
    seed=seed_for(args.seed,name,getattr(args,'_paired_epoch',0),getattr(args,'_paired_batch',0))
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for d in devices: torch.cuda.default_generators[d].manual_seed(seed)
        yield


def training_stream(name):
    def decorate(function):
        @wraps(function)
        def call(self,*args,**kwargs):
            x=args[0] if args else kwargs[function.__code__.co_varnames[1]]
            config=getattr(self,'configs',None) or self.config
            if self.training:
                with stream(config,name,x.device):
                    return function(self,*args,**kwargs)
            return function(self,*args,**kwargs)
        return call
    return decorate


def common_state_hash(model):
    h=hashlib.sha256(); count=0
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith('mem_net.mipp.'):
            continue
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes()); count+=1
    return h.hexdigest(),count


class AuditedRandomSampler(Sampler):
    def __init__(self,data,seed,split):
        self.data=data; self.seed=seed; self.split=split; self.history=[]
    def __len__(self): return len(self.data)
    def __iter__(self):
        epoch=len(self.history)
        generator=torch.Generator().manual_seed(seed_for(self.seed,'sampler/'+self.split,epoch))
        order=torch.randperm(len(self.data),generator=generator)
        self.history.append(dict(iteration=epoch,samples=len(order),sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest()))
        return iter(order.tolist())
