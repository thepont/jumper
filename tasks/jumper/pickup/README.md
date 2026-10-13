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

## Grasp-first stage (opt-in)

`--training-stage grasp --grasp-start near` starts the left arm at the checked
open pregrasp pose, with the object independently on the floor. Only reset writes
poses and servo targets. The actor still controls all five actions; there is no
scripted IK controller, attachment, post-reset object teleport or extra force in
training. The one-second settling/contact-free-start gate, unsupported pickup
witness and full chassis collision checks remain active. No privileged grip or
completion label is added to the 117 actor inputs.

This stage has a 10-second deadline and rewards reach, grip and lift progress,
then a one-shot completion bonus. It does **not** reward or terminate on bin
placement. Completion needs both grip inserts, the whole hull above 15 mm, no
external support, valid fixture/trajectory and no simultaneous physical failure.
Eleven successive 50 Hz qualifying samples span at least 200 ms between first
and last; this is sampled contact evidence, not proof of uninterrupted contact
between physics substeps. The legacy place mode retains its original qualifier.
Bonus weight 1000 delivers 20 reward at the 20 ms control period; hovering or
holding indefinitely cannot farm potential-shaping income.

Grasp motion requires a usable **object** detection, not a tray detection. The
tray estimate remains in the observation layout, and the placement mode retains
its original both-detection gate. Lost required detections stop new target
updates, not physical servo motion. The new objective, reset profile, eleven-
sample qualification, action gate and reward version are recorded in the grasp
checkpoint contract; placement checkpoints remain unchanged and cannot be
silently resumed as grasp checkpoints.

Start a fresh standard-object/fixed-layout pilot, not a resume of model_304:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/train.py --task jumper.pickup --backend native --device cpu \
  --num_envs 128 --cpu_threads 12 --headless --no-tensorboard \
  --max-iterations 100 --training-stage grasp --grasp-start near \
  --claw-side left --object-size standard --position-jitter-mm 0

MUJOCO_GL=egl python scripts/play.py --task jumper.pickup --backend native \
  --device cpu --num_envs 1 --training-stage grasp --grasp-start near \
  --claw-side left --object-size standard --position-jitter-mm 0 \
  --checkpoint <grasp-checkpoint.pt> --physics-hz 200 --headless \
  --steps 500 --stop-on-done --video demos/grasp-first-nn.mp4
```

Native CPU feasibility probes (MuJoCo 3.11.0, 2026-10-13) used scripted commands
through the same bounded action term, **not a learned policy**. Left small,
standard and large variants qualified at 4.12/4.78/5.58 seconds with 25.10/27.61/
22.02 mm floor clearance and zero chassis penetration. The stricter new grasp
stage qualified the standard object at 4.92 seconds with 32.98 mm clearance.
Open-finger and no-lift controls timed out with no pickup. Right-claw close/lift
attempts did not qualify; this opt-in curriculum is deliberately left-only until
a right-specific sequence is verified. Existing right placement remains available.

Progress manually, changing one difficulty at a time: fixed standard object,
small held-out position variation, size variants, then the original stowed-arm
start (`--grasp-start reach`) and finally placement. There is no automatic
promotion or unchecked interpolation between arm poses. Near/reach checkpoint
contracts differ; explicit actor-only transfer is not yet implemented. Before
promotion, use two held-out sets of 20 first episodes with at least 18 successes
each, no invalid starts or intrusion failures, and retained previous-stage
performance. These are proposed promotion gates, not measured learning results.

## Demonstration warm start (CPU, opt-in)

`python -m tasks.jumper.pickup.bootstrap` collects the checked left-claw
near-reset sequence through the normal five-action boundary, fits the existing
actor, saves a normal `PickupRunner` checkpoint, then evaluates a **fresh loaded
NN without the teacher**. It does not change the standard PPO entry point or
defaults, and does not convert a placement checkpoint into a grasp checkpoint.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tasks.jumper.pickup.bootstrap \
  --output logs/jumper/jumper.pickup/bootstrap-example \
  --episodes 24 --holdout-episodes 4 --eval-episodes 20 \
  --epochs 400 --initial-std 0.1
```

The output directory must be new. It contains `demonstrations.pt`,
`warm_start.pt` and `report.json`; previous code, datasets and checkpoints are
never overwritten. Only complete qualifying episodes are kept. Timeouts,
falls, invalid starts and chassis-intrusion failures are discarded, including
simultaneous success/failure. Teacher/IK outputs must be finite **before**
saturation. Commands are bounded **before stepping**, because previous raw
commands are themselves actor inputs. Each record pairs a cloned pre-step
117-component actor observation with the next bounded command; critic state,
teacher clocks, contact labels and true object positions are not student inputs.

Whole episodes, not neighbouring frames, form the training/validation split.
Only training observations set normalisation statistics. Behaviour cloning
updates the actor MLP with a separate Adam optimiser, selects the epoch with
lowest whole-episode validation MSE, and retains the fresh critic and fresh PPO
optimiser. The checkpoint includes the existing task contract and imitation
provenance; current runs also record startup source hashes and resolved jitter,
detection, physics and exploration settings. A fresh strict load must reproduce
the fitted actor's deterministic outputs exactly before autonomous evaluation.
Low validation MSE is not itself a pickup-success gate.

Evaluation and normal replay/PPO retain the existing raw NN action convention:
servo commands are bounded by the action term, but raw predictions remain in
proprioception. The evaluator reports prediction overshoot rather than silently
clipping only evaluation and making it differ from training. The scripted teacher
is a collection-only Strategy; the evaluation port accepts observations only.

Native CPU measurements, Ryzen 9 3900X/MuJoCo 3.11.0, 2026-10-13, dirty working
checkout (not a clean-fork reproducibility claim):

- The first 16-attempt collection kept 15 pickups. Its final-epoch actor achieved
  9/12 unassisted pickups, with one intrusion rejection; this is not the stronger
  checkpoint below.
- `bootstrap-verified-2026-10-13/warm_start.pt`: 21/24 demonstrations qualified;
  17 episodes trained the actor and four validated it. Epoch 30 was selected from
  400 by validation MSE. Two disjoint first-episode seed sets (1017–1036 and
  2017–2036) each achieved **20/20** deterministic standard-object pickups, with
  no invalid starts or chassis-intrusion failures. The first set's maximum
  chassis penetration was 1.659 mm, below the active 2 mm rejection threshold;
  this is not a claim of zero contact.
- Held-out ±2 mm position variation (3017–3036): **19/20**, with one intrusion
  rejection at 2.032 mm. It fails the no-intrusion promotion gate.
- Gaussian exploration std 0.6 was **not** learned by imitation: stochastic
  evaluation (4017–4028) achieved **0/12**, with six intrusion rejections. A
  five-update PPO continuation loaded and saved correctly with finite losses,
  but that does not establish a reliable stochastic grasp policy.
- A separate checkpoint changed only exploration std from 0.6 to 0.1, retaining
  exactly the same deterministic mean. The same stochastic seed set achieved
  **11/12**, with no invalid starts or intrusion failures. The opt-in bootstrap
  now defaults to 0.1 and records it explicitly; ordinary task PPO still defaults
  to 0.6. Twelve trials are not the two-set, 20-episode promotion gate.

These results concern a standard object, the near reset and the left claw only.
They do not establish arbitrary-size pickup, a stowed-arm approach, locomotion,
right-claw pickup, bin placement or real-hardware performance. The teacher has
timed phases; the actor has only short sensor history, not that clock. Phase
ambiguity and off-demonstration states remain reasons to test held-out episodes.

## Measured local CPU profile

On the local AMD Ryzen 9 3900X (12 physical cores, 24 logical), native MuJoCo
3.11.0, 2026-10-13, sequential five-update PPO benchmarks kept the corrected
collision geometry, four physics substeps, 64 rollout steps, mixed sizes and
10 mm position variation unchanged. The first two updates were excluded as
warm-up; rates below include collection and learning, but exclude startup,
checkpoint writing and logging overhead. These are short measurements, not
steady-state guarantees.

| Worlds | CPU workers | Control world-steps/s | Seconds/PPO update |
|---|---|---|---|
| 4 | 2 | 238 | 1.08 |
| 16 | 4 | 563 | 1.82 |
| 32 | 4 | 892 | 2.30 |
| 32 | 8 | 945 | 2.17 |
| 64 | 8 | 1,239 | 3.31 |
| 64 | 12 | 1,298 | 3.16 |
| 128 | 8 | 1,486 | 5.51 |
| 128 | 12 | 1,587 | 5.16 |

The selected 128-world/12-worker launch is 6.7 times the sequential four-world
baseline in **experience throughput**, approximately 6,346 physics world-steps/s.
No backend code, reward, servo, sensor frequency or collision tolerance was
changed for this improvement. Limit BLAS/PyTorch to one thread so the native
physics pools can use the CPU without competing with nested matrix thread pools:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/train.py --task jumper.pickup --backend native --device cpu \
  --num_envs 128 --cpu_threads 12 --headless --no-tensorboard \
  --max-iterations 300 --claw-side left --position-jitter-mm 10 --object-size mixed
```

The longer local run resumes the selected benchmark's collision-enabled
`2026-10-13_10-58-28/model_4.pt` via `--checkpoint`, in a new log directory.
It runs 300 additional updates (indices 5 through 304): 2,457,600 new control
world-steps. The short benchmark projects about 26 minutes of learner time;
actual episode dynamics and other CPU users can change that estimate.

At 128 worlds, native automatically strips visual-only meshes while retaining
full CAD chassis colliders: model copies measured 76.4 -> 22.4 MiB each,
approximately 2.80 GiB for 128 copies, excluding other runtime memory. A
128-world/12-worker preflight on **both** claws verified full CAD collider names,
masks, 5 ms physics / 20 ms controls, four-substep contact history, finite
observations/rewards, intrusion failures, partial resets and checkpoint guards.
Twenty-six native determinism/randomisation/backend-resolution checks passed.

More worlds change the PPO batch: 8,192 samples per update here versus 256 at
four worlds, still four epochs and four minibatches. Faster collection is **not**
a measured improvement in convergence, collision avoidance or grasp success.
Compare held-out pickup/placement and body contact at equal collected samples;
do not compare reward or update counts alone across different world counts.

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

Both arms have chassis contact sensors with deepest-contact reduction and a
history covering every physics substep of the 20 ms control frame. A penetration
greater than the task-owned 2 mm solver tolerance latches an invalid trajectory,
terminates it as a failure (not a timeout), and prevents pickup/placement credit.
It is not merely a reward penalty. Partial resets clear only the selected world's
latch and sensor history. Contact count, maximum penetration and intrusion are
reported as privileged metrics; none is added to the actor's observations.

Native two-world integration probes verified 80 normal settling frames per claw
with zero chassis penetration. Deliberately injected invalid poses generated
resisting contacts (LF forearm/base: 15.25 mm, 260.44 N; RF palm-piece/base:
7.22 mm, 122.88 N), were detected by the sensors, and failed. Separate
completion-frame probes supplied an otherwise-successful placement as a positive
control: 0 and 2 mm penetration retained success, while 2.001 mm rejected it and
paid zero placement bonus. Clearing contact did not clear the failure latch.
Those large forces are **solver responses to invalid diagnostic poses**,
not safe hardware forces or normal operating measurements. Failure latches and
sensor histories passed partial-reset isolation checks on both sides; the old
pilot checkpoint was rejected before weight loading. New checkpoints passed a
save/load round trip and preserved the fresh-episode settling gate.

Failing-first task probes ran outside the repository's framework suite, following
`CLAUDE.md`. On the local native CPU/MuJoCo 3.11.0 backend (2026-10-13), four
collision/geometry checks failed before the change; eight checks then passed,
covering both configured claws, all five chassis masks, full CAD selection,
calibrated starts and actual negative-distance contacts in obstructed joint poses.
Mask checks alone are insufficient: physical contacts must also be generated.
The old scripted pickup benchmark retains its original physics; its success
rates must not be presented as validation of the corrected pickup scene.

Three-update, four-world native CPU PPO smoke runs completed for each claw with
mixed sizes and 30 mm position variation. These validate training and saving
under the corrected contract, **not grasp learning or hardware safety**. Fourteen
external task checks and 61 focused framework checks passed; the full suite had
1,277 passes, 28 skips and the same 11 pre-existing failures as the prior run.

Grasp-first and verified demonstration warm-start are now opt-in paths above.
The next learning work is held-out position/size robustness and safe stochastic
PPO continuation. Then combine manipulation
with locomotion actions and rewards for retaining a grip while travelling, and
finally sequence objects. Replace the synthetic frontend with measured camera
detections before any real-robot claim. GPU throughput and hardware deployment
have not been validated for this new task.
