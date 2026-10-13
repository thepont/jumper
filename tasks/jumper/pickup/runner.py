"""Keep same-width left/right checkpoints from being silently interchanged."""
import io
import hashlib
from copy import deepcopy
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO


class FrozenActorNormalizationPPO(PPO):
    """Specialise mode transitions, not the upstream PPO loss or optimiser."""

    def train_mode(self):
        normalizer = self._raw_actor.obs_normalizer
        if not hasattr(normalizer, 'count') or int(normalizer.count) <= 0:
            raise RuntimeError('retained PPO needs a checkpoint with fitted actor normalization')
        super().train_mode()
        # Freezing only these buffers leaves actor gradients, exploration and
        # critic learning active. Reapply after EVERY recursive train transition.
        normalizer.eval()


class PickupRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        term = env.unwrapped.action_manager.get_term('claw')
        self._freeze_actor_normalization = getattr(term.cfg, 'freeze_actor_normalization', False)
        self._loaded_training_policy = None
        self._imitation_provenance = None
        self._actor_transfer_provenance = None
        if self._freeze_actor_normalization:
            from .rl_cfg import retained_agent_cfg
            profile = retained_agent_cfg().algorithm
            train_cfg = deepcopy(train_cfg)
            train_cfg['algorithm'].update(class_name=profile.class_name,
                                          learning_rate=profile.learning_rate, schedule=profile.schedule)
            self._retained_learning_rate = profile.learning_rate
        super().__init__(env, train_cfg, log_dir, device)

    def contract(self, *, training_stage=None):
        term = self.env.unwrapped.action_manager.get_term('claw')
        stage = training_stage or getattr(term.cfg, 'training_stage', 'place')
        if stage not in ('place', 'grasp', 'place-near'):
            raise ValueError('unsupported checkpoint objective')
        contract = {'schema': 'jumper.pickup/1', 'side': term.cfg.side,
                'joint_names': list(term.cfg.joint_names),
                'object_scales': list(term.cfg.object_scales),
                'physics': 'full-cad-chassis+distal-front-arm-contact+substep-intrusion/2',
                'actor_inputs': 'proprioception+delayed-noisy-position-size/1',
                'rewards': 'discounted-potentials+unsupported-pickup-settled-place/1'}
        if stage == 'grasp':
            contract.update(objective='grasp-first/1', start=term.cfg.grasp_start,
                            reset_profile='checked-left-pregrasp/1' if term.cfg.grasp_start == 'near'
                                          else 'original-left-stow/1',
                            qualification_samples=term.cfg.grip_min_samples,
                            action_gate='object-only/1',
                            rewards='discounted-reach-grip-lift+qualified-pickup-bonus/1')
        if stage == 'place-near':
            contract.update(objective='carry-place/1', start='near',
                            reset_profile='checked-left-pregrasp/1', qualification_samples=11,
                            action_gate='both-detections/1',
                            rewards='discounted-potentials+failure-aware-settled-placement/1')
            if getattr(term.cfg, 'bounded_action_history', False):
                contract['action_feedback'] = 'bounded-command-history/1'
        return contract

    def transfer_grasp_actor(self, path):
        """Explicit fresh-learner transfer, never a cross-objective resume."""
        term = self.env.unwrapped.action_manager.get_term('claw')
        if (term.cfg.training_stage != 'place-near' or not self._freeze_actor_normalization or
                self.current_learning_iteration != 0 or self.alg.optimizer.state or
                int(self.alg._raw_actor.obs_normalizer.count) != 0):
            raise RuntimeError('grasp transfer requires a fresh retained place-near learner')
        with open(path, 'rb') as source:
            contents = source.read(32 * 1024 * 1024 + 1)
        if len(contents) > 32 * 1024 * 1024:
            raise RuntimeError('transfer checkpoint exceeds 32 MiB')
        payload = torch.load(io.BytesIO(contents), weights_only=True, map_location='cpu')
        # Explicit actor-only transfer from the unchanged grasp contract. Unlike
        # resume, this starts a NEW observation-feedback scope and fresh learner.
        source_contract = self.contract(training_stage='grasp')
        # Actor-only transfer from the original qualified grasp distribution
        # may initialise a narrower size curriculum. Resume remains exact-match;
        # this does not adopt the source critic, optimizer or episode clock.
        from .env_cfg import DEFAULT_OBJECT_SCALES
        source_contract['object_scales'] = list(DEFAULT_OBJECT_SCALES)
        if (not isinstance(payload, dict) or not isinstance(payload.get('infos'), dict) or
                payload['infos'].get('pickup_contract') != source_contract or
                payload['infos'].get('pickup_training_policy') != 'frozen-actor-normalization-small-step/1'):
            raise RuntimeError('source must be a compatible retained left near-grasp checkpoint')
        state = payload.get('actor_state_dict')
        if not isinstance(state, dict) or not all(
                isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values()):
            raise RuntimeError('transfer actor state must be finite tensors')
        candidate = deepcopy(self.alg._raw_actor)
        expected = candidate.state_dict()
        if state.keys() != expected.keys() or any(
                value.dtype != expected[key].dtype or value.shape != expected[key].shape
                for key, value in state.items()):
            raise RuntimeError('transfer actor keys, shapes or dtypes differ')
        candidate.load_state_dict(state, strict=True)
        if int(candidate.obs_normalizer.count) <= 0:
            raise RuntimeError('source actor normalization was not fitted')
        self.alg._raw_actor.load_state_dict(candidate.state_dict(), strict=True)
        self._imitation_provenance = payload['infos'].get('imitation')
        self._actor_transfer_provenance = {
            'schema': 'grasp-to-near-placement-actor-only/1',
            'source_sha256': hashlib.sha256(contents).hexdigest(),
            'source_contract': payload['infos']['pickup_contract'],
            'destination_contract': self.contract(), 'critic_optimizer_clock': 'fresh'}
        self.env.unwrapped.reset()

    def save(self, path, infos=None):
        infos = {**(infos or {}), 'pickup_contract': self.contract()}
        if self._actor_transfer_provenance is not None:
            infos['actor_transfer'] = self._actor_transfer_provenance
        if self._freeze_actor_normalization or self._loaded_training_policy is not None:
            infos['pickup_training_policy'] = 'frozen-actor-normalization-small-step/1'
            if self._imitation_provenance is not None:
                infos['imitation'] = self._imitation_provenance
        return super().save(path, infos)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self._loaded_training_policy is not None and not self._freeze_actor_normalization:
            raise RuntimeError('retained checkpoint needs --freeze-actor-normalization to continue PPO')
        if self._freeze_actor_normalization:
            # Validate the fitted transform before logging/rollout side effects.
            # The algorithm reasserts the same policy at every mode transition.
            self.alg.train_mode()
        return super().learn(num_learning_iterations, init_at_random_ep_len)

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
        training_policy = loaded['infos'].get('pickup_training_policy')
        if training_policy is not None and (
                type(training_policy) is not str or
                training_policy != 'frozen-actor-normalization-small-step/1'):
            raise RuntimeError('pickup checkpoint has an unsupported training-policy version')
        del loaded
        snapshot.seek(0)
        infos = super().load(snapshot, load_cfg=load_cfg, strict=strict, map_location=map_location)
        self._loaded_training_policy = training_policy
        self._imitation_provenance = infos.get('imitation')
        self._actor_transfer_provenance = infos.get('actor_transfer')
        if self._freeze_actor_normalization:
            # The common loader restores optimizer groups, including the OLD
            # learning rate. Explicit adoption must reapply this profile AFTER it.
            self.alg.learning_rate = self._retained_learning_rate
            for group in self.alg.optimizer.param_groups:
                group['lr'] = self._retained_learning_rate
        # Checkpoints restore the global training clock, not simulator state.
        # Start fresh episodes at that restored clock so stale captures and the
        # one-second settle gate cannot be skipped after a resume.
        self.env.unwrapped.reset()
        return infos
