# Pickup learning

`jumper.pickup` is a separate stationary manipulation task. It uses the checked
five-foot bench physics, but PPO controls the four arm joints **and the finger**.
The other sixteen driven stance joints are held at their calibrated reset pose.
It does not run the scripted IK controller, remove old training code, overwrite
walking checkpoints, or teach walking while carrying.

## Train and replay

From the root repository's editable environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MJRL_CPU_THREADS=4 \
python scripts/train.py --task jumper.pickup --backend native --device cpu \
  --num_envs 16 --headless --no-tensorboard --max-iterations 3000 \
  --claw-side left --position-jitter-mm 10 --object-size mixed
```

Use `--claw-side right` for a separate right-claw policy. Both sides have five
actions, but their joint names and directions differ: checkpoints carry a side,
input/reward/physics version and action-order contract. A mismatch is rejected before
loading weights. Early experimental checkpoints without that contract are also
rejected; none is deleted. Resume uses a new timestamped directory:

```bash
python scripts/train.py --task jumper.pickup --backend native --device cpu \
  --num_envs 16 --headless --no-tensorboard --resume --checkpoint <checkpoint.pt> \
  --claw-side left --max-iterations 100

MUJOCO_GL=egl python scripts/play.py --task jumper.pickup --backend native \
  --device cpu --num_envs 1 --checkpoint <checkpoint.pt> --claw-side left \
  --object-size standard --headless --steps 1400 --video demos/pickup-nn.mp4
```

These are simulated training/replay commands, not a deployment recipe. There
are no operator command axes or `controls.yaml`: the task samples an object and
a goal, rather than taking a walking command from a stick or key. The standard
entry points and registry are unchanged apart from registering this new task.
The registered Jumper asset is required; other models are rejected rather than
silently reusing measured claw geometry.

## What varies

Every reset selects a real collision/mass variant of the rounded bar:

| variant | dimensions (mm) | mass (g) |
|---|---|---|
| small, scale 0.8 | 52.8 × 28.8 × 78.4 | 123.9 |
| standard, scale 1.0 | 66 × 36 × 98 | 242 |
| large, scale 1.2 | 79.2 × 43.2 × 117.6 | 418.2 |

Mass scales with volume, preserving density. `--object-size mixed` samples all
three; `small`, `standard` and `large` select one for controlled evaluation.
Inactive variants are parked beyond the arm's reach, never preloaded into the
tray or attached to the claw. Different shapes, materials, arbitrary orientation
and a scene containing several reachable objects are **not** covered yet.

`--position-jitter-mm` is finite and in [0, 30], default 10. Object X varies
outwards from the bench centre, with a 10 mm forward clearance margin and a
physical-width adjustment for each size; Y varies both ways. The tray varies by
one quarter of that width, and is pushed out if necessary to keep the whole
upright object outside its outer footprint. The complete layout reflects across
the centreline for RF. Only reset writes these poses. The first second holds
servo targets while the free-standing robot settles, and rejects any startup
robot/object contact. The original negative-X sampler could hit the stowed pad;
the revised four-world seed-7 native probe had no such contact (2026-10-13).

## Inputs and controls

The actor has three frames of joint positions/velocities, its own targets and
actions, gravity and angular velocity, plus **explicitly synthetic** position
and size estimates. Its 117 inputs contain no simulator contact, pickup,
containment or success labels. The 59-input critic may use privileged object
pose/velocity and evaluator state.

The synthetic producer samples at 10 Hz, delivers after 60 ms, adds 2 mm Gaussian
noise to positions and dimensions, and expires estimates after 250 ms. Detection
dropout defaults to 5%; `--detection-dropout` accepts [0, 1]. It approximates
horizontal field of view, retains capture pose, and masks unavailable positions
and dimensions. It is **not a camera**: no occlusion, detector, image calibration
or measured hardware accuracy is claimed. Robot encoders, IMU and base-motion
compensation remain ideal simulator readings.

The five outputs are bounded to [-1, 1]. Four advance joint targets at no more
than 0.45 rad/s (0.009 rad per 20 ms control), inside measured limits. The fifth
requests aperture from bench-open to closed, using the existing direction-aware
squeeze limiter. Targets hold when required detections are unavailable, retaining
the previous grip request. Non-finite actions fail before a physics step; the
failed run must be reset or closed, not continued with contaminated action history.

## Rewards and physical success

Dense shaping is **progress**, not income for hovering or holding forever:
`(gamma × potential_now − potential_previous) / dt`, with the same gamma as PPO.
True terminal potential is zero; timeout potential remains for bootstrapping.
Reach/grip/lift/carry/release potentials are gated by a valid settled fixture.
A separate completion bonus is paid once, when success terminates the episode.
Weights and the initial PPO settings are starting choices, not tuned convergence
claims. A controlled five-second holding probe changed from +53.11 reward to
−2.66 after this correction (native task probe, 2026-10-13).

Pickup requires both actual grip inserts, the whole physical hull more than
15 mm above the floor, and 200 ms continuous qualification. **Every contact must
belong to the carried palm subtree**: tray-rim, floor and other-leg support do
not count. Placement requires prior pickup, the whole hull inside the tray at
resting height, an open finger, no robot contact and a half-second settling
window with less than 3 mm vertex drift. Merely pushing an object into the tray
is not success. These checks are privileged simulation evaluation, not a real
robot's deployed completion detector.

Metrics distinguish `picked_up`, `placed`, `invalid_fixture`, grip contact,
object floor clearance, size scale and grip-point/object-centre distance.
Partial resets clear only that world's evidence and sensor buffers. Resume
restores the learner/global clock, then starts fresh episodes at that clock;
it does not restore a mid-grasp simulator state or bypass initial settling.

## Initial evidence and next stage

On the local native CPU backend, MuJoCo 3.11.0 (2026-10-13), a 16-world pilot
completed 100 PPO iterations and 50 more via guarded resume: 153,600 control
world-steps in total, with real optimiser updates and saved checkpoints. A
separate right-side four-world PPO smoke also completed. Training recorded no
successful pickup or placement. This proves a runnable learning path, **not a
trained cleanup policy**. After the safe-spawn correction, a first-episode
evaluation of checkpoint 149 completed 12 valid attempts (four per size), with
no startup-contact rejections and **zero pickups or placements**. It moved the
arm by up to 0.96 rad and reduced grip/object-centre distance, but did not grasp.
That pilot inherited the walking task's disabled arm/chassis contacts, so the
movement is **not evidence of physically feasible manipulation**. Its checkpoints
are retained, but the corrected pickup physics contract rejects loading them.
A longer run with the old physics would not resolve this defect.

## Chassis collision boundary

Pickup decorates the existing bench spec factory rather than modifying the
walking model or duplicating jaw/side configuration. All five chassis components
(base, upper shell, display, camera and ToF housing) use their full CAD meshes as
**conservative convex envelopes**. They protect the outer surfaces without the
inward gaps left by reduced walking hulls; they do not represent internal shell
concavities or prove hardware fit.

A separate mask channel enables both front arms' upper arm, forearm, palm,
finger, pads and grip inserts against these envelopes. Adjacent shoulder mounts
and stance hips keep their original exclusions. Parent filtering remains enabled:
globally disabling it exposes intentional mounting/hull overlap. Ground,
cross-leg contact, jaw decomposition, mass and inertia remain unchanged.

Failing-first task probes ran outside the repository's framework suite, following
`CLAUDE.md`. On the local native CPU/MuJoCo 3.11.0 backend (2026-10-13), four
collision/geometry checks failed before the change; eight checks then passed,
covering both configured claws, all five chassis masks, full CAD selection,
calibrated starts and actual negative-distance contacts in obstructed joint poses.
Mask checks alone are insufficient: physical contacts must also be generated.
The old scripted pickup benchmark retains its original physics; its success
rates must not be presented as validation of the corrected pickup scene.

The next learning work is a grasp-first curriculum or verified demonstration
warm-start, followed by held-out position/size checks. Then combine manipulation
with locomotion actions and rewards for retaining a grip while travelling, and
finally sequence objects. Replace the synthetic frontend with measured camera
detections before any real-robot claim. GPU throughput and hardware deployment
have not been validated for this new task.
