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
    return ("claw_side", "position_jitter_mm", "detection_dropout", "object_size")
