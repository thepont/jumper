"""Stationary physical pickup/place learning, separate from walking policies."""
from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(id="jumper.pickup", assets=JUMPER_ASSETS,
         description="Learn one-claw pickup and settled placement from varied nearby positions",
         tags=("jumper", "manipulation", "pickup", "training"))


def cli_args(group):
    group.add_argument("--claw-side", choices=("left", "right"), default="left")
    group.add_argument("--position-jitter-mm", type=float, default=10.,
                       help="per-reset object XY variation, bounded to 0–30 mm")
    group.add_argument("--detection-dropout", type=float, default=.05,
                       help="synthetic detection dropout probability, in [0, 1]")
    group.add_argument("--object-size", choices=("mixed", "small", "standard", "large"),
                       default="mixed", help="physical size variants, sampled anew on reset")
    group.add_argument("--training-stage", choices=("place", "grasp", "place-near"), default="place",
                       help="place preserves the original task; grasp ends on lift; place-near starts at pregrasp")
    group.add_argument("--grasp-start", choices=("near", "reach"), default="near",
                       help="grasp only: checked pregrasp reset, or the original stowed-arm reset")
    group.add_argument("--freeze-actor-normalization", action="store_true",
                       help="near curriculum PPO: retain loaded actor scaling and use fixed 3e-5 learning rate")
    group.add_argument("--bounded-action-history", action="store_true",
                       help="place-near only: feed bounded commands back to the actor; distinct checkpoint contract")
    group.add_argument("--finger-close-speed", type=float, default=None,
                       help="closing-target slew in rad/s; opening unchanged; distinct contract")
    return ("claw_side", "position_jitter_mm", "detection_dropout", "object_size",
            "training_stage", "grasp_start", "freeze_actor_normalization", "bounded_action_history",
            "finger_close_speed")
