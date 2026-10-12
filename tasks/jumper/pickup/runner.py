"""Keep same-width left/right checkpoints from being silently interchanged."""
import io
import torch
from mjlab.rl import MjlabOnPolicyRunner


class PickupRunner(MjlabOnPolicyRunner):
    def contract(self):
        term = self.env.unwrapped.action_manager.get_term('claw')
        return {'schema': 'jumper.pickup/1', 'side': term.cfg.side,
                'joint_names': list(term.cfg.joint_names),
                'object_scales': list(term.cfg.object_scales),
                'physics': 'full-cad-chassis+distal-front-arm-contact/1',
                'actor_inputs': 'proprioception+delayed-noisy-position-size/1',
                'rewards': 'discounted-potentials+unsupported-pickup-settled-place/1'}

    def save(self, path, infos=None):
        return super().save(path, {**(infos or {}), 'pickup_contract': self.contract()})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Restricted tensor/primitive decoding before model/optimiser mutation.
        # The common loader owns actual loading; this task only guards semantics
        # that network shape checks cannot distinguish (both sides have 5 actions).
        # Snapshot the file once. Validating one pathname read then loading a
        # second lets replacement between reads bypass the side/version guard.
        with open(path, 'rb') as source:
            contents = source.read(32 * 1024 * 1024 + 1)
        if len(contents) > 32 * 1024 * 1024:
            raise RuntimeError('pickup checkpoint exceeds the 32 MiB limit')
        snapshot = io.BytesIO(contents)
        loaded = torch.load(snapshot, map_location='cpu', weights_only=True)
        if not isinstance(loaded, dict) or not isinstance(loaded.get('infos'), dict):
            raise RuntimeError('pickup checkpoint has no validated task contract')
        if loaded['infos'].get('pickup_contract') != self.contract():
            raise RuntimeError('pickup checkpoint contract differs: use the same claw side and task version')
        del loaded
        snapshot.seek(0)
        infos = super().load(snapshot, load_cfg=load_cfg, strict=strict, map_location=map_location)
        # Checkpoints restore the global training clock, not simulator state.
        # Start fresh episodes at that restored clock so stale captures and the
        # one-second settle gate cannot be skipped after a resume.
        self.env.unwrapped.reset()
        return infos
