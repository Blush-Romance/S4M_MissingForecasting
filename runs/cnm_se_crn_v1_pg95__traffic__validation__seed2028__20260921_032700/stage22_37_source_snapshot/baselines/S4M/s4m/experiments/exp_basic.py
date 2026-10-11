import os
import torch
from model import S4M


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S4M':S4M
        }
        self.device = self._acquire_device()
        self.model = self._build_model()
        if getattr(self.args,'stage22_36_paired_training_rng',False):
            from utils.paired_training_rng import common_state_hash
            self.paired_initial_sha256,self.paired_initial_entries=common_state_hash(self.model)
            print(f'PAIRED_INITIAL entries={self.paired_initial_entries} sha256={self.paired_initial_sha256}',flush=True)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())
        buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in self.model.buffers())
        self.parameter_count = parameter_count
        self.parameter_bytes = parameter_bytes
        self.buffer_bytes = buffer_bytes
        print(
            "MODEL_FOOTPRINT "
            f"parameters={parameter_count} "
            f"parameter_mib={parameter_bytes / 1024**2:.2f} "
            f"buffer_mib={buffer_bytes / 1024**2:.2f}"
        )
        self.model = self.model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            external_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if external_visibility:
                # The project scheduler selects a physical GPU by UUID before
                # this process starts. Replacing that UUID with the server's
                # logical index 0 silently routes work to physical GPU 1.
                # Inside an externally restricted process the selected device
                # is always local cuda:0.
                self.args.gpu = 0
                device = torch.device('cuda:0')
                print(
                    'Use GPU: cuda:0 '
                    f'(preserved external CUDA_VISIBLE_DEVICES={external_visibility})'
                )
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
