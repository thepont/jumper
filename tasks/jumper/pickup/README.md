# Pickup learning

`jumper.pickup` is a separate stationary manipulation task. It uses the checked
five-foot bench physics, but PPO controls the four arm joints **and the finger**.
The other sixteen driven stance joints are held at their calibrated reset pose.
It does not run the scripted IK controller, remove old training code, overwrite
walking checkpoints, or teach walking while carrying.

## Train and replay

### Bounded alternate IK starts — 2026-10-15

The next maths-advisor slice uses the existing `PointIK` authority with ten
deterministic starts per waypoint: measured/previous joints, joint-range centre,
and eight seeded interior samples. Only private arm seed joints change. Each
attempt records separate position/alignment residuals, admission and selection.
Admitted fits are preferred, then smaller joint travel; rejected fits are ranked
by their worst normalised residual. Collision/path/closure admission remains
mandatory. This is a sampled local search, not exhaustive branch enumeration or
joint optimisation across all waypoint combinations. Existing 4 mm/10-degree
thresholds and the 1 mm clearance margin are unchanged.

On the same local native CPU fixture and identical sensor packet/reset fingerprint
as the prototype below, all **540 fits** (18 proposals, three waypoints, ten
starts) failed joint position/alignment admission. The best complete proposal's
maximum position residual remains **14.281 mm**. No approach, closure or lift
path was admitted or executed. The one clearance query was the measured-start
positive control. Live qpos and collision meshes remained unchanged by planning.

A separate diagnostic removed the orientation objective, retaining the same ten
starts per waypoint. Position-only admission counts were 1/18 approach, 1/18
grasp and 2/18 lift; no complete proposal passed even position-only admission
(best maximum residual 4.987 mm). The four passing individual point solutions
had signed pad-axis errors 82.865, 80.843, 103.242 and 103.637 degrees. They are
diagnostic solutions, never executable grasp admissions. An earlier single-start
version is preserved separately and is not treated as the matched comparison.
These observations support investigating reachable relative positioning and
support-leg motion; they do not prove that every fixed-stance grasp is impossible.
Any repositioning must be reached through checked servo motion, with support,
tracking and collision measurements, before claiming physical pickup.

Seven new BDD controls were observed red at the missing API, then green. They
cover alternate-solution recovery, unreachable and misaligned rejection,
determinism, bounded work, unchanged non-arm state and the single-start control.
Together with 35 retained advisor checks, 16 slow-claw controls and 42 focused
framework checks, **100 checks passed**. Ruff and whitespace checks passed.
No full-suite, physical pickup, training improvement or hardware claim is made.
Evidence and external controls are under
`logs/jumper/jumper.pickup/math-grasp-branches-2026-10-15/`.

### Estimated-pose maths grasp advisor prototype — 2026-10-15

The user authorised a maths-based grasp assistant with TDD/BDD. This first slice
is an **advisory library and progression gate**, not a working pickup controller:
it is not wired into PPO, camera inference, autonomous servo execution or hardware.
No new NN training, demonstrations or deployment were started. The original
teachers, policies, sensor observations, physics and reward functions are unchanged.

The performance contract remains an independently verified physical pickup:
11 consecutive pinched, unsupported samples with whole-hull floor clearance
above 15 mm and no concurrent failure. Central reference points, IK fits,
clipped geometry and nearby pads do not certify that outcome. The environment
is the known static bench and one known convex SOAP shape, left claw and fixed
stance; this does not solve arbitrary clutter, other shapes, either arm or
robot repositioning. Prospective actuation uses the existing rate-bounded
`ClawActions`, including its opt-in slower closing target, not pose writes.

Three task-owned modules separate responsibilities:

- `grasp_advisor.py`: validated `ObjectPose`/candidate boundaries, 18 bounded
  proposals on the known convex hull's +/-Y sides, explicit admission evidence
  and the `GripGate` State machine. Sampling uses hull half-spaces, not the
  rectangular bounding-box face. Candidates rotate with measured object pose.
- `grasp_sensing.py`: opt-in `PoseDetections` Adapter, reusing the existing
  position/size capture schedule. Orientation is captured at 10 Hz, delivered
  after 60 ms and expires with the same 250 ms/dropout validity. Its independent
  RNG leaves the existing 13-field detection observation and position random
  stream byte-identical in a 20-step control. Real orientation estimation and
  calibration are **not implemented**; this is explicitly synthetic sensing.
- `grasp_planning.py`: native single-world checking Strategy, reusing `PointIK`,
  `plan_joint_path` and `GeometryClearance`. It derives the fixed insert normal
  from compiled collision geometry at the existing on-facet reference. Object
  pose and dimensions come from the estimate; a private recompiled object mesh
  avoids retaining the true simulator size. Static bench CAD and ideal robot
  proprioception remain assumptions, not a general sensor-derived scene map.

`grasp_candidates(estimate, reference_hull, now_s=...)` returns no proposals for
missing, stale, future-dated or orientation-free detections. It requires a
usable estimate with synthetic availability confidence at least 0.6. Confidence
1 in the synthetic Adapter means valid delivery, not calibrated camera accuracy.
`choose_grasp(candidates, checker.assess)` ranks admitted proposals centrally
without turning admission into physical success. Never replace an absent
orientation with the identity quaternion merely to obtain a plan.

The checking Strategy requires signed insert alignment within 10 degrees and
position error at most 4 mm for a 17 mm stand-off, 2 mm surface stand-off and
50 mm lift proposal. It bounds IK/path/clearance work; at most 512 combined
clearance queries, one bounded path attempt per segment, and 33 sampled jaw
apertures. Nonadjacent front-arm/self pairs, arm/floor and non-insert/object
clearance retain a 1 mm planning margin. Intentional rigid/adjacent interfaces
are still excluded by the existing pair authority: this is not complete
no-scrape or hardware-safety certification. The margin is not a calibrated
bound on 2 mm sensing error, tracking error or model uncertainty.

Closure screening uses the actual curved insert model, stops on the first
obstruction or excessive insert penetration, and does not require closing
through an already near-seated object. Frozen-object closure and rigid-carriage
lift remain limited geometric screening models, not dynamic seating proof or
proof of global impossibility. Not-run checks stay explicit in detailed reports;
their admission booleans are false, not measured physical failures. Concurrent
robot/object rejections and query-budget exhaustion remain visible.

The separate `GripGate` requires ten aligned samples over at least 180 ms before
closing and eleven loaded-central samples over at least 200 ms before lift.
Missing pose, lost alignment, safety failure and bounded phase timeouts block
progress. Only independent canonical pickup feedback can mark completion.
The caller still must supply these measurements and obey the phase; the gate
itself does not issue servo commands. Native contact loads are not an available
hardware tactile sensor. Loaded-central quality is not the historical unweighted
contact-centroid <=0.35 criterion and is not inferred from candidate centrality.

**Measured result, local native CPU/MuJoCo 3.11.0:** one seed-970001 standard
SOAP fixture, +/-2.5% size configuration and 2 mm XY reset jitter. After 65
open-jaw settling steps, a synthetic packet with 2 mm position/size noise and
2-degree per-axis rotation-vector noise generated 18 proposals. **0/18 passed
all-stage IK** at the existing thresholds. Approach, closure and lift paths
were consequently not evaluated for these proposals; no close or lift was
executed. The gate reported `blocked: no admitted grasp plan`, not pickup success.
The best candidate's maximum three-stage position residual was 14.281 mm,
versus the 4 mm limit. This is one starting branch, stance and bounded proposal
family, not proof that no feasible central grasp exists.

Independent controls gave zero position error for a known FK-derived reachable
target and 4.966 m error for an intentionally unreachable target. The measured
start passed the explicit clearance check. Private planning dimensions matched
the packet (63.583 / 36.030 / 100.616 mm), and live qpos and collision-mesh hashes
were unchanged. Only that positive-control clearance query ran in the final
check; it does not exercise every planning/execution branch.

TDD evidence: absent-module/API controls failed first; four malformed-candidate
controls and two concurrent-rejection/budget controls were also observed red
before their repairs. A native assertion caught the initial true-size geometry
leak, then passed with private estimated geometry. Earlier diagnostic reports
are preserved, not counted as independent trials or improvement. Final checks:
35 advisor/sensor/clearance task probes, 16 retained slow-claw probes and 86
focused framework checks passed (137 total); Ruff passed on the new modules
and E4/E7/E9/F on external probes. No full-suite claim is made.

Evidence and external probes are preserved in
`logs/jumper/jumper.pickup/math-grasp-advisor-2026-10-15/` rather than adding
task tests to the framework suite. No checkpoint was selected or modified.
The next issue is reachable approach/grasp/lift geometry under the unchanged
guards. Compare bounded alternative IK branches or a separately validated
repositioning proposal before enabling closure; do not weaken tolerances or
start another NN run to conceal this rejection. Physical closure/retention,
fresh pose tracking, real perception, tactile feedback and an obeying servo
executor remain unverified/unimplemented as described above.

### Opt-in slower closing target — 2026-10-15

After the user observed rapid premature closure and tipping, pickup now accepts
`--finger-close-speed 0.5` (radians/second). The default is absent and retains
historical behaviour. Supply this flag explicitly in a **fresh** run; no new
training was launched for this change. Finite positive rates are accepted;
booleans, zero, negative and nonfinite rates are rejected at configuration
construction. This is a target-slew experiment, not a calibrated hardware limit.

The existing `ClawActions` authority limits the emitted closing target after
the existing squeeze limiter, every 5 ms physics substep, rather than every
20 ms policy step. Mirrored claw signs are handled canonically. Opening is
not slowed; partial resets restart only the selected world's target history
from its measured joint angle. History stores the emitted squeezed target so
stalled contact cannot accumulate hidden lead. No rewards, reversal penalties,
arm rates, gains, collision guards, geometry or perception were changed.
This does not gate premature closure or enforce a physical velocity ceiling:
servo lag and momentum can make actual velocity exceed target slew.

Checkpoint contracts include `finger_closing_target_slew_rad_s` only when
enabled. Legacy mode retains its old contract. Changing the configured rate
rejects an incompatible checkpoint **before loading model/optimizer weights**;
do not silently reinterpret an old policy under slower action dynamics.

A bounded native CPU comparison used one matched reset at seed 970001,
standard SOAP, +/-2.5% size configuration and 2 mm position jitter. Both
325-step (6.5-second) fixed-arm command runs completed without termination or
truncation. They held open for 1.2 seconds, requested close for 4.5 seconds,
then requested open for 0.8 seconds. Starting qpos fingerprints matched
(`d634fc0f99be5f0380e9981048133431350135c2b3c8811c1b1c0987378ff031`).
The original guards plus explicit mask-independent 1 mm self-clearance rejection
remained enabled. This is not replay of a trained actor or a hardware safety test.

| Measurement | Original | Closing-target limit 0.5 rad/s |
|---|---:|---:|
| Maximum emitted closing-target rate, 200 Hz | 89.8677 rad/s | 0.500011 rad/s |
| Maximum observed closing velocity, sampled at 50 Hz | 17.3384 rad/s | 0.6595 rad/s |
| Pinched control samples | 0 | 0 |

The target-rate excess over 0.5 is floating-point accumulation error. The
observed jaw speed is lower, but **neither run established a grip or pickup**.
This isolates the speed-control change; it does not establish less tapping,
better timing, central contact or a better learned policy. Timing and repeated
opening/closing remain separate, untested interventions. The 0.5 value was
chosen before this comparison; no parameter sweep or snapshot selection ran.

Fourteen external behavioural controls failed first on the absent option;
after implementation, sixteen passed, including a no-limit negative control
and incompatible-checkpoint rejection. Eighty-six focused registry, layout,
checkpoint, seam and native-determinism framework checks passed. Ruff's
E4/E7/E9/F checks and whitespace checks passed; full Ruff still reports existing
import ordering, line-length and style findings, and no full-suite pass is
claimed. The first native probe stopped on an incorrect diagnostic field name
before comparison; that probe was corrected to the existing `root_link_pos_w`
authority, then the single complete matched comparison above was recorded.

Durable working evidence is under
`logs/jumper/jumper.pickup/slow-claw-2026-10-15/`: exact traces, logs,
external controls and driver. These task-specific probes stay outside the
framework test suite per `CLAUDE.md`. Existing policies, code and experiment
artifacts remain preserved. The next learning experiment must explicitly opt
into the new action contract and retain independent physical pickup metrics;
that experiment is not started or selected by this change.

### Fresh 10,000-update Gemini run — 2026-10-15

The user authorised a longer run on the faster machine after the 1,000-update
policies achieved no qualified pickups. This is **fresh** PPO at seed 970001,
not a resume: 10,000 updates, eight worlds and 64 rollout steps give exactly
5,120,000 environment steps. It retains the Gemini native CPU runtime, two
native threads, one PyTorch thread, 60-second training episodes, original
rewards/physics/synthetic sensors and the explicit 1 mm self-clearance guard.
The central potential/qualified-pickup bonus weights remain 1/250, and SOAP
scales/jitter remain 0.975/1/1.025 and 2 mm. The sole experimental learning
change is the total training budget; the prior policies remain untouched.

Training has a six-hour inclusive wall cap and stops on source-hash drift,
errors or nonfinite actor checkpoint state. Successful training automatically
permits one separate thirty-minute evaluation of final `model_9999.pt` only.
The same twelve sixty-second diagnostic scenarios from the short Gemini run
are reused (eight standard, two small, two large), including its first-scenario
video. This is exposed diagnostic evidence, not independent confirmation.
No extra snapshot selection, reward tuning, retries, promotion or deployment
is authorised. It does not resolve the earlier 30-versus-60-second
matched-attempt question or establish algorithm improvement from one seed.

An external Observer records exact completed attempts per world and separates
true terminations from timeout-only endings. Simultaneous termination/truncation
is counted once; started attempts include each autoreset. It preserves original
actions and step return values and does not query/change rewards. Counts are
saved every ten updates alongside steps/time, fixing the missing-count evidence
for this new run without inventing historical counts. Seven external controls
passed after failing budget/contract/counter controls; Ruff passed. A fresh
remote one-update/reload/video smoke passed before launch; initial weights and
reset hashes match the earlier Gemini run, with a measured 3,000-control-step
episode limit. No production
source changes or new full-suite result are claimed.

Protocol, frozen external driver/control copies and local review evidence are
under `logs/jumper/jumper.pickup/long10k-gemini-2026-10-15/`; active remote files
are `/home/pont/jumper-horizon-20261015.WYApvB/long10k-20261015/`. All previous
code, artifacts and checkpoints are preserved. At the preceding measured rate
of 467 environment steps/second, 5.12 million steps project to about three hours
of training; this is an estimate, not a completion promise or learning claim.

#### Completed 10k result — 2026-10-15

The Gemini run completed all 10,000 updates / 5,120,000 environment steps in
10,405.317 seconds (2 h 53 min; 492.06 environment steps/s, training only).
Training and evaluation exited zero without wall timeouts; final reports are
complete. Exact completed training attempts total 1,816: 162 true terminations
and 1,654 timeout-only endings. True terminations include all non-time-limit
termination types; this counter alone does not identify failure/success reasons.
Per-world completions are 227/229/224/225/236/225/223/227. Including each initial
world and autoreset, 1,824 attempts were started; eight remained unfinished at
the fixed step budget. These counts are not inferred from rolling averages.

| Diagnostic size | Evaluated | Qualified pickups | Auxiliary central-loaded pickups |
| --- | ---: | ---: | ---: |
| Standard | 8 | 0 | 0 |
| Small | 2 | 0 | 0 |
| Large | 2 | 0 | 0 |

All twelve evaluation episodes ran the full sixty seconds and timed out; none
ended for explicit clearance failure, detected self-contact, chassis intrusion,
fall, too-low state or invalid fixture. No observed guard stop is not hardware
safety proof. Maximum auxiliary loaded score was exactly zero. Maximum whole
object floor clearance was 0.151657 mm, below the unchanged >15 mm / eleven-frame
physical pickup requirement. No pickup-time estimate exists because no qualified
pickup occurred. The final checkpoint is preserved with SHA-256
`0f91a6ff1bff1b5d0d43705affc8d465ad1b250663a258e32bb81df42381a585`.

Final audit matched all twelve starting-state fingerprints against the earlier
short Gemini diagnostic evaluation (also 0/12 pickups). All 634 frozen source/
asset hashes and three control hashes matched on Gemini and in the local review
copy. There are no incomplete reports or excluded diagnostic episodes. The first
scenario MP4 is 1,501 frames / 60.04 seconds at 25 fps; its late frame was inspected
alongside the reports. Software rendering retains background line artefacts.

The full training log provides a correction to reading only the latest summaries:
the `grip_contact` metric occasionally reached 1.0, although its final printed
value was zero. Across 1,667 printed episode-summary samples, pickup, lift reward
and central-pickup reward remained 0.0000. Central-load reward summaries ranged
from -0.0001 to 0.0000; printed zeros are rounded, not proof of exactly zero raw
training reward. Reported reach distance ranged 0.0585–0.3417 m and ended at
0.1835 m. Mean action standard deviation decreased from 0.60 to 0.10 over 10,000
updates. These are training diagnostics, not independent evaluation, exact contact
counts or an isolated causal explanation. Occasional contact did not establish
stable grip or qualified lift, and decreasing exploration alone does not prove
the cause of failure.

Decision: preserve the candidate and every earlier checkpoint, promote neither
short nor long policy, and close this bounded run without automatic replacement.
Tenfold training steps did not produce a successful final policy in these trials;
the claim that this particular larger budget would suffice is unsupported. This
does not prove further training can never work, identify a reward/control defect,
or establish algorithm-level equality from one seed and a reused diagnostic set.
It does not resolve the earlier 30-versus-60-second matched-attempt question.
The next proposed investigation is to trace approach/contact reward signals and
exploration against actual motion before another unchanged training extension;
no such investigation or reward tuning was launched by this review.

Seven external orchestration/counter controls and final artifact checks passed;
Ruff passed for the final audit. No new full-suite or hardware check is claimed.
The machine-readable review, reports, logs, frozen controls and final checkpoint
are retained under `long10k-gemini-2026-10-15/`, with scoped evidence ZIP and first
evaluation video published separately. Hosted copies expire after fourteen days;
the original remote checkpoint collection and local evidence remain preserved.

### Matched episode-horizon pilot — 2026-10-15

The user authorised one run on each machine to investigate whether the prior
30-second training episodes curtailed pickup learning. The frozen comparison
uses 30 seconds on music-room and 60 seconds on arch-gemini, both native CPU:
eight worlds, two native threads, one PyTorch thread, seed 970001, 1,000 PPO
updates of 64 vector steps (512,000 environment steps per run). Training is
bounded at two hours per host including startup; only successful completion
automatically starts a separate 30-minute evaluation. No replacements,
promotion, reward/physics/perception changes or hardware deployment are allowed.

Both conditions retain the experimental central-load potential weight 1,
canonical-pickup-gated central bonus weight 250, synthetic delayed/noisy
detections, SOAP scales 0.975/1/1.025, 2 mm position jitter and the 1 mm explicit
nonadjacent self-clearance failure latch. Only the final update-999 checkpoint
is evaluated. Both evaluators allow 60 seconds on the same twelve fresh scenarios
(seeds 980001–980012, eight standard and two each small/large). Success means the
unchanged physical pickup witness, not reward or contact alone. Loaded-contact
score >=0.9 for its first eleven witness frames is an auxiliary outcome, not
equivalent to the historical centroid-fraction <=0.35 measure. This is grasp-first
evaluation; it establishes neither placement nor general household grasping.

Report denominators, timeouts versus guard stops, per-size pickups, auxiliary
central pickups, actual first-pickup time and safety regressions. Check matched
starting-state fingerprints before paired interpretation. These are two fixed
policies from one seed, with condition confounded by host: hardware, Python and
CPU wheel build differ even after matching core package release versions. This
pilot can supply tentative evidence, not prove an algorithm improvement or
equality. No automatic policy selection or safety relaxation is authorised.

The local training environment uses Python 3.13; Gemini's dedicated Docker
runtime uses Python 3.12, PyTorch 2.11.0 CPU rather than the untouched container's
ROCm PyTorch 2.14.0. The frozen source is equivalent to root commit afc7e9e;
634 source/asset files plus external controls are hash-checked before each
training chunk and evaluation episode. The old remote dirty checkout, old
container and all policies remain preserved. The user explicitly allowed
training-only readiness checks instead of host controller-build gates E/F.
Inherited AMD wheel metadata in the new runtime is incompatible with its CPU
torchvision version but unused by this CPU-only path; import paths and actual
CPU execution must pass the bounded preflight rather than treating metadata as
runtime proof.

Five external orchestration controls passed after a failing training-horizon
identity control; fourteen existing external reward/guard checks and sixteen
framework checkpoint/native checks passed (44 framework warnings). Ruff passed
for the new driver and controls. Each host passed a one-update training smoke,
fresh-environment checkpoint reload and short MP4 recording; those checks are
not evidence of learned pickup. The actor/critic initial-state hash agreed
(`4040a886...afaf9b8`), as did the eight-world reset hash
(`0571308b...190f1`), control dt 0.02 seconds and core release versions. Measured
training limits are 1,500 and 3,000 control steps. No full-suite rerun is claimed.

Evidence is preserved under
`logs/jumper/jumper.pickup/horizon-pair-2026-10-15/` locally and
`/home/pont/jumper-horizon-20261015.WYApvB/` on Gemini. External driver/control
copies, manifests, preflight records, reports, logs, final checkpoints and the
first non-selected evaluation clip are retained. Incomplete runs are reported
as incomplete, never replaced automatically.

#### Completed result and interpretation limit

Both training stages completed exactly 1,000 updates / 512,000 environment
steps, and both separate evaluations completed all twelve scenarios. All four
stage exits were zero with no wall timeout. The twelve paired seed/size/reset
fingerprints agree. Final audit rechecked all 634 frozen source/asset hashes and
three control hashes on each host, plus both final checkpoint hashes. There
were no incomplete reports, excluded evaluation episodes or replacement runs.

| Training episode limit / host | Training seconds | Env steps/s, training only | Pickups standard / small / large | Auxiliary central pickups | 60-second evaluation timeouts | Guard stops |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 30 s / music-room | 2,517.815 | 203.35 | 0/8 / 0/2 / 0/2 | 0/12 | 12/12 | 0/12 |
| 60 s / arch-gemini Docker | 1,095.816 | 467.23 | 0/8 / 0/2 / 0/2 | 0/12 | 12/12 | 0/12 |

Both evaluators allowed the entire sixty seconds on every episode; none ended
early for detected self-contact, explicit clearance failure, chassis intrusion,
fall, too-low state or invalid fixture. This is absence of observed guard
failures in these trials, not hardware safety proof. Maximum auxiliary loaded
central score was exactly zero for both policies. Maximum whole-hull floor
clearance was 1.231972 mm for 30-second training and 0.033857 mm for 60-second
training, below the canonical >15 mm / eleven-frame pickup requirement.
No qualified pickup occurred, so there is no observed pickup-time estimate.
Zero auxiliary score does not distinguish every possible single-pad contact,
grip geometry, load or exploration failure; it does not isolate a cause.

The user's subsequent clarification is binding: equal environment steps are
**not equal training attempts**. Completed training-attempt totals were not
persisted as an exact count; rolling mean episode lengths cannot reconstruct
them reliably. The proposed 30-second half-budget checkpoint comparison has
**not been evaluated**. No extra snapshot was selected or new work launched to
fill that gap. Therefore these results establish only that these two fixed
final policies failed these trials. They do not establish that longer training
episodes are worse, equivalent or sufficiently trained, nor answer performance
at equal attempts. Comparing attempt-matched learning remains unresolved, along
with single-seed and host/software confounds. No significance or equality claim
is made, and the 60-second policy is not rejected because it had fewer attempts.

Decision: preserve both candidates and all prior policies; promote neither.
The fixed-compute protocol is closed without claiming the user's matched-attempt
question is resolved. Both first-standard-scenario videos are retained, each
1,501 frames / 60.04 seconds at 25 fps, rather than selected for a favourable
outcome. Their late frames were inspected against the reports; Gemini's
software-rendered clip has background line artefacts, not evidence of contact.
Machine-readable audit is `horizon-pair-2026-10-15/final_review.json`; scoped
reports, logs, controls, final checkpoint identities and checkpoints are archived
beside the original evidence. The chat-hosted copies expire after fourteen days;
the local/remote run directories remain the durable working copies.

Final artifact checks and the five external orchestration controls passed. This
documentation-only result adds no production change and makes no new full-suite
claim. Final checkpoint SHA-256 values are
`5630c053981bb418b902aaaa3799c4a2eca2ecc98bab211a7fc528c444ad3362`
(30 seconds) and
`18f8c7c82b3405e125c6881aee604399ab421b7e31668751392b6480a0cbbdc4`
(60 seconds).

### Teacher clearance repair — 2026-10-14

The default left pickup teacher now uses measured arrival (10 samples within
4 mm), checks IK residuals, holds its approach pose during closure, and requires
three dual-contact samples before attempting lift. Phase timeouts reject the
attempt. `TimedLeftPickupTeacher` preserves the historical controller explicitly;
old policies and datasets have not been removed. This is a teacher-control repair,
not an NN improvement or central-grasp qualification.

Before issuing a pickup command, a private-state geometry adapter samples its
arm/aperture segment at no more than 0.009 rad intervals and checks 779 explicit
nonadjacent front-arm/robot collision pairs, independently of contact masks.
The provisional 1 mm collision-hull margin is an experimental planning choice,
not validated hardware clearance. Same-rigid and adjacent-body interfaces remain
excluded and require a separate audit. Sampling is not continuous collision
certification; collision hulls may conservatively reject real free space.
Physics, collision masks and the existing task success criteria are unchanged.
This guard covers pickup teacher commands, not learned actors or the subsequent
placement carry controller. Collection propagates pickup rejection through the
placement wrapper and stops before executing or accepting the rejected command.

On the local Ryzen 3900X/native CPU fixture, the single seed-610018 integration
probe stopped after 236 executed steps (4.72 simulated seconds): left finger
grip insert versus right finger hull measured 0.531 mm in the proposed closure
segment. The command was rejected, with zero accepted episodes. Evidence and
pre-run source hashes are in
`logs/jumper/jumper.pickup/guarded-teacher-integration-2026-10-14/`.
The earlier externally guarded probe reproduced the same boundary. Neither
probe achieved a qualified central grasp; no fitting or PPO was launched.

The separate saved-pose CAD witness audit
(`self-cad-witness-2026-10-14/report.json`) left all ten directional solid-side
queries ambiguous because of mesh topology. Unsigned surface distances do not
prove that the masked hull overlaps are real scrapes or false obstructions.
Do not remove colliders or relax margins on that evidence. The next manipulation
investigation must resolve the relative arm/support positioning at closure while
retaining support and checking the full trajectory, before collecting new data.

Validation: six framework clearance tests and six external teacher/collection
controls passed; new controls were observed failing before their repairs.
Together with adjacent framework checks, 60 checks passed (35 warnings, 5.90 s),
and Ruff passed. The prior self-contact full-suite run recorded 1,293 passes,
28 skips and the same 11 baseline failures; that is historical evidence, not a
full-suite run of this teacher change. Controls are preserved beside the new
integration report rather than added as task-specific framework tests.

### Supported teacher and central-contact investigation — 2026-10-14

On the same local Ryzen 3900X/native CPU fixture, the teacher now waits for
ten consecutive five-foot ground-contact samples after one second, times out
after four seconds, and rejects active manipulation with fewer than four support
contacts. Joint target increments preserve the checked straight joint segment;
independent joint clipping did not. Aperture prediction uses the existing
squeeze limiter and measured finger velocity, not an instantaneous full-close
target. No actuator, reward, perception, collision mask or physics was changed.

Approach rejects non-insert left-arm/object interference. During closure/lift,
the object's existing palm-subtree contact permission is retained, while contact
with other robot parts is rejected. This is separate from robot self-clearance:
permitted object/claw contact does not permit the arm to scrape the robot.
The release integration (`supported-teacher-release-integration-2026-10-14`)
rejected the first proposed forbidden palm-pad/object gap at 0.99445 mm after
3.30 simulated seconds, accepting zero episodes. It verifies fail-closed
collection, not a successful pickup. The historical teacher remains available.

An ambiguous native mesh-distance zero in a proposed lift was independently
checked in `mesh-gap-audit-2026-10-14`: a convex-hull separating plane certified
at least 1.95074 mm separation. The adapter now accepts a native zero only if
compiled mesh geometry independently certifies the requested margin, subtracting
10 nm for roundoff. Native negative distances are never overridden; insufficient
or unavailable certificates still reject. This is a conservative lower bound,
not an exact distance or a model/solver repair. Positive separation, sub-margin,
touching, overlapping and transformed-frame framework controls cover the change.

The separate servo-driven support prototype achieved a 20 mm forward preparation
in 329 physical steps with all five feet and at most 3.56 mm recorded foot drift
(`supported-body-prepare-2026-10-14`). It never writes the live base/object pose
after reset. A restricted 90-query central reach screen had zero admissions at
the original stance and 18 after that actual preparation: 13 legacy-reference
and five facet-centroid candidates. Kinematic admission is not grip evidence.
The first prepared-screen report had a NumPy-boolean serialization error; its
incomplete record is preserved, with corrected publication in the separate r2.

Physical legacy-reference runs obtained dual-pad edge pinches but no qualified
central pickup: contact fractions were approximately 0.8735 and 0.8779 against
the unchanged 0.35 limit. The first nominal inverted-branch run did not actually
honour its IK seed; only `central-inverted-contact-r2-physical-2026-10-14`
executed the inverted branch. Neither enters demonstration fitting.
The first centroid-target run stopped before approach because its planning guard
did not match the upcoming grasp contact contract. Two external controls observed
that defect red, then green. The repaired r2 physically approached but rejected
the subsequent seating IK (`central-facet-contact-physical-r2-2026-10-14`),
with zero qualified pickups. Reference-frame replay measured the tangential
reference-to-centroid displacement as approximately -16.3 mm along object X,
not Z (`central-reference-frame-audit-2026-10-14`). That can contribute to edge
contact, but is not an isolated causal proof. A separate bounded 30 mm requested
servo preparation followed by centroid approach stopped at measured arrival
timeout after 682 physical steps (`central-facet-30mm-physical-2026-10-14`),
with all five recorded supports, 6.77 mm final foot drift and no qualified pickup.
Its inherited intermediate reason text still says 20 mm; the frozen plan and
source specify 30 mm. The trace, not that stale string, identifies this trial.

#### Longer approach comparison — 2026-10-14

At the user's request, one same-seed diagnostic extended only the prototype's
approach deadline to at least 15 seconds (40-second episode cap, 240-second
wall limit, one native CPU world/thread). Collision, support, action and
centrality criteria were unchanged. Source copies and hashes were saved before
execution in `central-facet-long-approach-2026-10-14`; the added read-only trace
records actual position/axis errors and joint-target tracking.

The old run stopped at step 682 (13.64 seconds), with this matched replay showing
4.158 mm position error against the 4 mm arrival gate. The longer run began
closure at step 710 (14.20 seconds), just 0.56 seconds later. All 682 shared
base-pose records match exactly; this checks that recorded channel, not every
unrecorded state. The earlier arrival deadline was insufficient for this case.

The run ended at step 738 (14.76 seconds) with three dual-pad contact frames,
zero central frames and zero qualified pickups. The final contact fraction was
0.5026 against the unchanged 0.35 limit. Its generic stop label is `no checked
lift corridor`, but the detailed rejection is specifically fewer than three
central-contact samples, not a failed geometric lift search. All five recorded
supports remained; final recorded foot drift was 7.08 mm. Extra time enabled
arrival and pinch, but did not demonstrate a central grip or NN improvement.
No replacement attempt or training was launched. This is one diagnostic case,
not a robustness comparison or an independently confirmed learned improvement.

These support/central prototypes are not promoted into the actor's five-action
interface: they command support joints that the existing pickup actor cannot.
Some early combined prototype plan text incorrectly said support-only commands;
the preserved source/trace also contains left-claw commands. Those runs are
diagnostic evidence, not evidence of NN improvement. No NN fitting or PPO was
run, and no graded centrality reward was added. Training waits for actual central
pickup and whole-placement qualification; existing policies/data remain intact.
Adjacent robot interfaces, continuous swept clearance, mask-independent learned
actor evaluation and hardware safety remain unverified.

Release checks: 80 focused framework/external controls passed (35 warnings,
4.91 s), Ruff and whitespace checks passed. This is not a new full-suite result;
the historical baseline above remains separately labelled. Task controls stay
outside `tests/` and are preserved with the evidence archive.

### Overnight central-reward PPO pilot — 2026-10-15

The user authorised substantial overnight reward-based training. A perfect
scripted teacher is not a prerequisite for from-scratch PPO; it remains necessary
to qualify demonstrations before imitation. This pilot does not use the support
preparation prototype or change the five-action actor interface.

The opt-in native CPU module `central_reward.py` scores active, nonpositive-gap,
positively loaded insert/object contacts in the selected object's frame and scale.
Each point receives `exp(-0.5 * (max(abs(x)/hx, abs(z)/hz) / 0.35)^2)` **before**
normal-load averaging; the weaker pad determines the score. Opposite edge points
cannot cancel into a fabricated central bonus. Missing loaded pads score zero.
These are simulator post-forward constraint loads, not hardware pressure readings
or proof of sufficient grip force. Positive load, inactive contact, positive gap,
edge cancellation and scaling controls include real native sphere/box fixtures.

The six serial fresh learners use two training seeds (930001/930002) and central
profile weights 0/1/4. Each profile changes both the auxiliary discount-correct
potential and a canonical-pickup-gated bonus of weight `250 * profile_weight`;
it is a comparison of reward recipes, not an isolated shaping-component effect.
Existing reach/grip/lift shaping, actor inputs, perception noise, physics and
canonical physical pickup criteria remain unchanged. Episodes are 30 seconds
rather than the default 10-second grasp episode. Eight native worlds use two
native threads and one PyTorch thread; no other machine or hardware is involved.

All profiles add the same experiment-only 1 mm mask-independent self-clearance
rejection, latched into existing failure eligibility before pickup completion.
This checks nonadjacent front-arm/robot pairs at control boundaries; it is not
continuous swept clearance or hardware safety. Adjacent interfaces remain outside
that guard. The ordinary task, old policies and old checkpoint contracts remain
unchanged; experimental checkpoints have a distinct contract and are not deployed.

Queue limits: six hours overall, at most 2,000 PPO updates and 45 minutes per
learner, then at most ten minutes for 12 matched selection episodes (eight
standard, two small and two large at +/-2.5% scale and 2 mm position jitter).
Source hashes are checked between training chunks and before evaluation; a source
change, nonfinite state, failed process, incomplete report or paired-start mismatch
stops the queue without replacement. Every checkpoint and failed result is kept.
Complete paired selection compares central pickups and pickup retention by size,
with no aggregate safety regression. Retention is experimental, not promotion.
If any candidate meets the fixed selection gate, one is compared with its same-seed
zero-bonus baseline on 12 new reserved seeds, within the remaining overall budget.
Sparse pilot results must not be described as proven algorithm improvement.

Evaluation reports legacy physical pickup separately. The auxiliary score >=0.9
is **not** equivalent to the old reported-centroid fraction <=0.35. The central
pickup label requires both signals for the first 11-frame physical witness;
neither signal alone certifies a lift. A high score alone does not establish
benign penetration or hardware safety. Grasp-only episodes do not test placement.

Protocol/driver/checkpoints/reports live under
`logs/jumper/jumper.pickup/overnight-central-2026-10-14/`; that frozen directory
name retains the planning date. The bounded coordinator is
`/tmp/jumper_overnight_central.py`, copied into the evidence bundle before launch.
No automatic deletion, widening, reward mutation, policy promotion or hardware
deployment is authorised. Rejected candidates are preserved, not removed.

Preflight on the local Ryzen 3900X/native CPU: 14 external reward/guard controls
and adjacent framework checks totalled 73 passes (1.56 seconds); Ruff and
whitespace checks passed. Three native PPO updates completed in approximately
five seconds, saving finite checkpoints. Cold reload completed all 12 episodes
with zero pickups and explicit-clearance stops; this verifies the executable
path, not learned grasping. A stationary open-claw positive control ran 300
steps/six seconds without rejection. Earlier device-type and logger integration
failures are preserved in their separate preflight directories/logs. No new
full-suite result is claimed.

#### Final audit: stopped queue, no selected candidate — 2026-10-15

On music-room (Ryzen 3900X, native CPU), the coordinator stopped with
`stopped-training-error` / `wall-timeout` on run 05. Of six planned jobs,
four completed training plus all 12 selection episodes, five saved final
training reports/checkpoints, and run 06 never started. Run 05 reports
`complete: true` after 1,100 updates, but its process timed out before a
successful coordinator exit; it has no evaluation or selection decision.
That report does not override the stopped queue. No retry, replacement,
confirmation, promotion or deployment was performed.

| Run | Seed | Profile weight | Updates | Training seconds | Selection | Decision |
|---|---:|---:|---:|---:|---|---|
| 01 | 930001 | 0 | 1,100 | 2681.002 | 12/12 | baseline preserved |
| 02 | 930001 | 1 | 970 | 2671.630 | 12/12 | rejected, preserved |
| 03 | 930001 | 4 | 1,120 | 2671.685 | 12/12 | rejected, preserved |
| 04 | 930002 | 0 | 1,320 | 2687.710 | 12/12 | baseline preserved |
| 05 | 930002 | 1 | 1,100 | 2692.750 | absent | unevaluated, preserved |
| 06 | 930002 | 4 | — | — | absent | not started |

All four evaluated policies produced **zero qualified pickups and zero
combined central first-witness pickups**: each had 0/8 standard, 0/2 small
and 0/2 large pickups, totalling 0/48. The combined witness requires raw
current dual-pad centroid fraction <=0.35 and auxiliary loaded score >=0.9,
both sustained for the first 11-frame physical pickup witness. No physical
witness occurred: the saved witness values are null, and maximum auxiliary
scores are zero throughout. Separate raw per-frame centroid traces were not
persisted; these reports cannot establish raw central-contact occupancy.
Per-size retention against the seed-930001 baseline is equality at zero,
not evidence of retaining a working grip.

| Run | Explicit self-clearance episodes (standard/small/large) | Self-contact episodes | Chassis intrusion | Fall / too low | Invalid fixture | Timeouts |
|---|---|---:|---:|---:|---:|---:|
| 01 | 2/0/1 | 1 | 0 | 0/0 | 0 | 9 |
| 02 | 3/0/2 | 2 | 0 | 0/0 | 0 | 7 |
| 03 | 1/0/0 | 0 | 0 | 0/0 | 0 | 11 |
| 04 | 0/0/0 | 0 | 0 | 0/0 | 0 | 12 |

Self-contact overlaps explicit clearance, so unsafe episode totals are
3/5/1/0, not their summed termination counts. The weight-1 candidate has
more unsafe episodes than its baseline; weight 4 has fewer but no pickups.
Neither passes the fixed central-pickup improvement gate: zero candidates
were experimentally retained, and both rejected candidates remain intact.
There are no independent confirmation reports. The second training seed
has only an evaluated baseline, so the planned two-seed recipe comparison
is incomplete. These selection results do not prove algorithm improvement,
grip adequacy, placement, actual vision performance or hardware safety.

Audit checked **485/485 frozen source/asset SHA256 hashes**, including the
external coordinator, with no mismatch; all five final checkpoints match
their recorded hashes. All 12 seed/size/start-qpos fingerprints match across
all four evaluations, including each completed candidate/baseline pairing.
The source snapshot records shared commit `afc7e9e7d92771d8a3f193e31e7a289ea2cc940c`;
the opt-in module was published separately as `278238b`. Exact final
checkpoint paths and SHA256 identities are in each run's `training.json`
and the final evidence archive's audit/manifest. No intermediate checkpoints
are published; all local policies and artifacts are preserved. The archive
contains the frozen plan/queue, progress/training/evaluation reports, scoped
logs, verification copies and final checkpoints. First-standard recordings
are chronological protocol episodes, not favourable clip selection.

Documentation validation in the isolated publish worktree: `git diff --check`
passed. The full suite produced 1,307 passes, 24 skips and three failures in
143.76 seconds. All three reproduced against the unchanged HEAD README:
missing CycloneDDS `idlc` in deploy provenance, the posture footprint mirror
assertion, and TensorBoard's lingering-port assertion. They are pre-existing
failures, not fixed by this evidence-only change. Ruff could not run because
it is absent from the available environment; the earlier preflight Ruff pass
remains a separate result. No shared source was changed for these checks.

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
- The next independent stochastic set at std 0.1 (5017–5036) achieved **19/20**
  but had one intrusion rejection at 2.040 mm. Lower noise helps, but still fails
  the no-intrusion gate; do not treat it as certified safe PPO exploration.
- The committed workflow produced
  `bootstrap-committed-2026-10-13/warm_start.pt` from source `d955ac1`, with source
  hashes and resolved settings captured before collection. A 30-epoch run
  reproduced **20/20** on 1017–1036. Its actor mean and normalisation tensors are
  exactly equal to the stronger checkpoint above; only the Gaussian std differs.
  Use this explicit checkpoint for replay, not an automatic newest-PPO search.

Do not replace the verified warm start with a merely loadable PPO smoke
checkpoint. Both five-update continuations were bounded compatibility checks;
finite losses and successful saving are not evidence that grasp skill survived.
At std 0.1, `2026-10-13_13-52-11/model_5.pt` achieved **0/4** deterministic
pickups on 6017–6020, while the preserved warm start achieved **4/4** on those
same seeds. Keep the warm start; PPO skill retention needs separate work.

These results concern a standard object, the near reset and the left claw only.
They do not establish arbitrary-size pickup, a stowed-arm approach, locomotion,
right-claw pickup, bin placement or real-hardware performance. The teacher has
timed phases; the actor has only short sensor history, not that clock. Phase
ambiguity and off-demonstration states remain reasons to test held-out episodes.

## Retaining the grasp during PPO (opt-in)

`--freeze-actor-normalization` explicitly adopts the grasp-only continuation
profile: retain the loaded actor's observation transform and use a fixed
learning rate of `3e-5`. It requires fitted actor statistics from a checkpoint;
it refuses a fresh, unnormalised start. Actor weights, Gaussian exploration
parameters and the critic remain trainable. Critic normalisation still updates.
The small task-local PPO subclass specialises training-mode transitions only;
it reuses the upstream PPO rollout, loss, optimiser and checkpoint machinery.
No teacher, IK controller, demonstration replay loss or simulator assistance
runs during this continuation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/train.py --task jumper.pickup --backend native --device cpu \
  --num_envs 128 --cpu_threads 12 --headless --no-tensorboard \
  --max-iterations 20 --training-stage grasp --grasp-start near \
  --claw-side left --object-size standard --position-jitter-mm 0 \
  --freeze-actor-normalization \
  --checkpoint logs/jumper/jumper.pickup/bootstrap-committed-2026-10-13/warm_start.pt
```

The original physical/task checkpoint contract is unchanged. Retained
checkpoints additionally record
`pickup_training_policy: frozen-actor-normalization-small-step/1` and preserve
the source imitation provenance. Unknown policy versions are refused before
model/optimiser mutation. They can be replayed without the training flag, but
continuing PPO without it is refused. Full checkpoint loading restores the old
optimiser learning rate; explicit adoption reapplies `3e-5` afterwards. Ordinary
PPO and behaviour-cloning defaults retain their existing semantics.

Native CPU/3900X/MuJoCo 3.11.0, 2026-10-13, same dirty checkout as above:

- Failed five-update checkpoint, seeds 6017–6020: restoring only the original
  MLP retained **0/4** pickups; restoring only the original normaliser gave
  **1/4**, with three intrusion failures. Restoring scaling after damaging
  updates is not a substitute for keeping the transform stable during learning.
- A single-variable five-update probe froze actor statistics before PPO,
  leaving ordinary learning settings intact: **3/4**, no intrusions. A fixed
  `3e-5` follow-up retained **4/4** on the same seeds.
- `2026-10-13_14-27-44/model_20.pt`: 20 PPO updates/163,840 control-world steps.
  Two new deterministic sets (7017–7036 and 8017–8036) gave **40/40** pickups;
  stochastic actions (9017–9036) gave **20/20**; held-out ±2 mm XY positions
  (10017–10036) gave **20/20**. All had zero invalid starts, intrusion failures
  and measured chassis penetration. Actor normalisation buffers were bit-for-bit
  unchanged; actor weights changed and critic statistics incorporated 163,840
  observations. These are actual PPO updates, not a frozen-actor replay.
- Full-state continuation for five more updates produced
  `2026-10-13_14-33-58/model_25.pt`: **20/20** on 11017–11036, no intrusion
  failures or measured chassis penetration. Actor statistics remained identical
  to the warm start, critic count reached 204,800 and optimiser LR stayed `3e-5`.
- External failing-first task probes and focused framework checks: **93 passed**.
  Cold loads covered explicit adoption, replay without the flag, refusal of
  unflagged training and unknown versions before parameter mutation. The full
  suite retained 1,277 passes, 28 skips and 11 pre-existing failures.

This establishes bounded pickup-skill retention, not indefinitely stable
training or bin-placement skill. The 60-second NN replay consists of repeated
pickup episodes: the simulator resets the object after each terminal lift.
Those resets are not packing items away, and no teacher supplies replay actions.

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

## Near-start carry/place curriculum (experimental)

`--training-stage place-near` extends the checked left pregrasp reset to a
30-second pickup → carry → release → settle objective. It uses the same standard
object, physical jaws and collision checks, not a pre-attached object. The actor
still receives only the existing 117 noisy/delayed sensor-history inputs; both
object and bin detections gate actions. Completion requires qualified pickup,
whole-hull containment, open-finger release, no robot contact and settling.
Same-frame falls, invalid fixtures and chassis intrusion reject completion and
its bonus. Ordinary `place` and `grasp` defaults remain unchanged.

The placement checkpoint has a distinct objective/reset/qualification/action-gate
contract. Ordinary resume still rejects a grasp checkpoint. The explicit
`PickupRunner.transfer_grasp_actor()` operation validates compatible near-grasp
physics, actor inputs, shapes, dtypes and finite tensors before copying only the
actor (including fitted normalizer and exploration scale). It requires a fresh
retained placement learner; critic, optimizer and clock remain fresh. Recorded
source SHA-256 identifies the transfer, not proof of source performance: use an
independently qualified source checkpoint.

Collect complete qualified demonstrations, fit with source normalization frozen,
then cold-load and evaluate the saved NN and its original pickup skill:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tasks.jumper.pickup.place_train \
  --grasp-checkpoint logs/jumper/jumper.pickup/2026-10-13_14-33-58/model_25.pt \
  --output logs/jumper/jumper.pickup/place-near-next \
  --episodes 20 --holdout-episodes 4 --eval-episodes 20 --epochs 300
```

The collection-only teacher composes the checked pickup with continuous,
position-only IK carry, lowering and open-finger withdrawal. It commands bounded
servo actions, never attachments, object-state writes or extra actor inputs.
Only complete non-failing placements become labels; whole episodes are held out.
`transferred_actor.pt`, `demonstrations.pt`, `placement_warm_start.pt` and
`report.json` are saved in a new output directory, preserving old runs.

Local native CPU/MuJoCo 3.11.0 evidence, 2026-10-13:

- Position-only scripted feasibility completed seed 17 in 11.88 s with zero
  measured chassis penetration. Constraining the closing axis horizontal lost
  the object; a no-release control received no placement credit.
- `place-near-pilot-2026-10-13` collected **8/16** complete placements;
  four episodes trained and four were held out. The best holdout-fit epoch was
  replayed, not necessarily the final epoch.
- Saved-NN unseen placement seeds 1017–1028: **0/12 placements**, with seven
  chassis-intrusion failures. Raw outputs reached 5.73 despite bounded teacher
  labels; the existing action term clamps servo commands, not raw history.
- The same actor retained **9/12 pickups** on original grasp seeds 2017–2028,
  with zero measured chassis penetration. This is a regression from the
  separately verified source grasp policy, not a promotion candidate.

Improved held-out action MSE did **not** establish closed-loop placement.
The candidate is not ready for longer PPO, new objects, position expansion or
hardware. Investigate coverage of carry/release observations, accumulated action
drift and pickup rehearsal before expanding the curriculum. These are hypotheses,
not established causes. Keep using an explicit checkpoint: newest-file selection
is not qualification. Replay training-matched physics with `--physics-hz 200`:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/play.py --task jumper.pickup \
  --backend native --device cpu --num_envs 1 --headless --physics-hz 200 \
  --training-stage place-near --object-size standard \
  --checkpoint logs/jumper/jumper.pickup/place-near-pilot-2026-10-13/placement_warm_start.pt \
  --steps 1500 --stop-on-done --video /tmp/place-near-nn.mp4
```

External failing-first stage/transfer probes and 102 focused checks passed.
Eleven additional placement/frozen-fit checks passed after the final safeguards.
The framework suite recorded 1,277 passed, 28 skipped and the same 11 known
unrelated failures. Task-specific probes remain outside `tests/` under the
repository's testing policy. No old code or checkpoints were removed.

### Bounded action-history experiment

`--bounded-action-history` is an opt-in for `place-near` only. It feeds the
clamped nominal command into proprioceptive history while preserving the raw
network action for diagnostics, rewards and PPO likelihoods. Servo execution,
perception gating, dimensions and default observations are unchanged. The
checkpoint contract adds `action_feedback=bounded-command-history/1`; ordinary
resume rejects mixing the two observation scopes. Only explicit actor-only
transfer from the original qualified grasp scope starts this new curriculum.
This is not a collision-prevention guarantee and does not turn the NN into a
placement policy.

3900X/native CPU, MuJoCo 3.11.0, 2026-10-13: twelve paired near-layout trials
(seeds 1017–1028, original saved placement actor, **±10 mm XY variation**)
produced **0/12 placements** in both modes. The first batched comparison gave
6 versus 1 intrusion failures;
repeating with a fresh global clock for each episode gave **4/12 raw-history
versus 2/12 bounded-history** failures. Peak raw NN output in the fresh-clock
comparison fell from 6.12 to 2.97. Episode grouping/capture timing affects the
numbers: this is bounded diagnostic evidence, not a general safety guarantee.
The unchanged default and raw diagnostic values are covered by external
acceptance probes, observed failing before the feature was added. New-feedback
save/load and rejection of old-feedback checkpoints passed integration checks.
The first diagnostic inherited the environment's 10 mm default; it was not
the fixed layout used to train the source. Repeating all twelve fresh-clock
pairs with **explicitly zero position jitter** gave **7/12 raw-history versus
0/12 bounded-history** intrusion failures, again **0/12 placements** in both
modes. Do not confuse improvement in failure rate with learned placement.

### Optional pickup rehearsal

The placement trainer accepts `--grasp-demonstrations` for a compatible
`jumper.pickup.demonstrations/1` dataset. Restricted decoding is size-bounded;
the contract, finite/bounded labels, unique qualified episodes and original
whole-episode split are checked before fitting. Pickup training excludes every
placement holdout seed; pickup validation excludes every placement training
seed. The saved placement dataset contains the actual pickup train/holdout
tensors, and the checkpoint records their seeds. Source normalization stays
frozen. This is imitation rehearsal, not a second deployed policy or a simulator
success label added to actor inputs.

3900X/native CPU/MuJoCo 3.11.0, 2026-10-13: a separate 150-epoch comparison
using the existing four placement training episodes, with versus without
13 compatible pickup rehearsal episodes, retained **11/12 versus 8/12** pickups
on unseen seeds 4017–4028. Both achieved **0/6 placements** on seeds 3017–3022
and each had four placement intrusion failures; neither had a pickup intrusion.
Both evaluation environments used **±10 mm position variation** (their default);
these were not the original fixed-layout retention seeds.
Rehearsal improved pickup retention in this bounded check, not carry/release.
These runs used original raw feedback, deliberately isolating rehearsal from
the bounded-feedback experiment. They are not qualified cleanup policies.

Example of the opt-in combined workflow (new output directory required):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tasks.jumper.pickup.place_train \
  --grasp-checkpoint logs/jumper/jumper.pickup/2026-10-13_14-33-58/model_25.pt \
  --grasp-demonstrations logs/jumper/jumper.pickup/bootstrap-committed-2026-10-13/demonstrations.pt \
  --bounded-action-history --episodes 32 --holdout-episodes 4 \
  --output logs/jumper/jumper.pickup/place-near-rehearsal-next
```

Eighty focused external/framework checks passed after these additions;
ruff and whitespace checks passed. Existing full-suite baseline failures are
documented above; this turn did not rerun the full suite. New demonstrations
still require physical qualification, and phase-aware recovery or collision-free
path planning has not yet been implemented. No hidden simulator phase is fed
to the actor and no hardware deployment is authorised by these experiments.

### Larger fixed-layout placement pilot

`place-near-bounded-32-2026-10-13` used explicit **zero position jitter**, the
same standard object, source grasp actor and frozen input normalization. Of
32 new teacher attempts (seeds 33–64), **16 qualified placements** were saved;
eight episodes trained and eight were held out. Epoch 22 had the best holdout
action MSE (0.08337), and was saved rather than the final epoch 300.

Cold-loaded NN evaluation on seeds 1033–1040 achieved **3/8 placements**, with
zero chassis-intrusion failures. Original grasp-scope evaluation on seeds
2033–2040 retained **7/8 pickups**, also without intrusion. A separate
per-episode fresh-clock audit of the placement seeds again achieved **3/8**,
although the successful seeds changed (1034, 1037 and 1039): capture timing
remains a source of variation. These are overlapping seed sets, not sixteen
independent successful trials.

An independent fixed-layout MP4 replay qualified seed 1034 in **17.76 s** with
real pickup, release, containment, settling and zero measured penetration.
Only the saved NN controlled actions; no teacher or IK ran during replay.
This is partial learned placement, not reliable cleanup, different-object
generalisation, locomotion or hardware validation.

A further diagnostic fit combined those eight placement training episodes with
pickup rehearsal and bounded feedback. With ±10 mm evaluation variation it
retained **12/12 pickups**, but placed **0/6 objects** (seeds 5033–5038 and
6033–6044). That candidate is separate, not promoted over the 3/8 placement
candidate, and its different seeds/layout are not a controlled comparison.
Adding rehearsal can preserve grasping while still failing to learn release;
stage-specific recovery and realistic phase estimation remain unfinished.

### Inspecting imitation snapshots without changing the default fit

`place_train --save-fit-every N` optionally saves `fit_epoch_N.pt` checkpoints
during fitting, including the final epoch when it is not a multiple of N.
Zero (the default) disables extra snapshots. The best holdout-MSE state is still
restored and written to `placement_warm_start.pt`; this option does not change
the loss, input scaling, observations, physics or default checkpoint selection.
The shared fitting function exposes an optional read-only epoch observer so the
existing runner remains the single checkpoint writer. Snapshot provenance records
its epoch, dataset digest and `selection=not-qualified`.

These files are candidates, not automatically improved policies. Compare actual
placement, intrusion failures and pickup retention on a fixed selection set, then
check a selected candidate against the reference on separate unseen seeds. Use
explicit checkpoint paths, one evaluation worker and a bounded timeout; do not
run every candidate in its own simulator process at once. Preserve failures and
partial logs separately from complete evaluation reports.

The opt-in callback's failing-first external probe also verifies that an observer
leaves the default fitted actor unchanged. The 300-epoch snapshot refit reproduced
the original epoch-22 MSE-selected actor tensor-for-tensor on the fixed dataset.
This establishes comparable fitting, not better closed-loop performance.

### Completed snapshot comparison (2026-10-13)

Results are saved under
`logs/jumper/jumper.pickup/placement-fit-snapshot-sweep-2026-10-13/`, including
`fit_report.json`, `comparison_report.json`, individual evaluation reports and
all four snapshots. The reference remains the epoch-22 actor in
`place-near-bounded-32-2026-10-13/placement_warm_start.pt`; no checkpoint was
overwritten or promoted to hardware.

The fixed selection seeds were 7017–7028. All evaluations used native CPU,
one world and one worker, one native/PyTorch thread, deterministic inference,
fresh episode clocks, zero position jitter, the left claw and standard object.
Physics ran at 200 Hz, controls at 50 Hz, with a 30-second episode limit.
The existing delayed/noisy detections (2 mm noise, 5% dropout), bounded action
history, full-CAD collision checks and physical placement qualification stayed
unchanged. These are narrow-layout tests, not object-size or position robustness.

| Actor | Qualified placements / 12 | Chassis intrusion failures / 12 |
|---|---:|---:|
| Reference, epoch 22 | 6 | 1 |
| Snapshot, epoch 25 | 0 | 9 |
| Snapshot, epoch 50 | 0 | 0 |
| Snapshot, epoch 100 | 4 | 0 |
| Snapshot, epoch 300 | 1 | 2 |

Epoch 100 was the only placing snapshot without an intrusion failure, so it was
checked against the retained reference on unseen seeds 9017–9028. It placed
4/12 versus the reference's 3/12, with zero intrusion failures for both.
On separate original-grasp-scope seeds 10017–10028, the same saved actors
picked up 12/12 versus 9/12, again without intrusion failures. This grasp check
uses the original observations/history, not a cross-objective optimizer resume.

The one-success unseen placement difference is inconclusive: across selection
and holdout, epoch 100 placed 8/24 versus the reference's 9/24. There was only
one independent fitting run. Keep epoch 100 as an experimental release-focused
candidate, not a verified placement improvement or a production replacement.
Its snapshot metadata remains `selection=not-qualified`.

In every unseen epoch-100 placement trial the object was picked up and fully
inside the bin at episode end. All eight misses had `inside=True` but
`released=False`. The measured bottleneck in this set is completing release
and withdrawal; these terminal observations do not establish why the actor
chooses its finger commands. No release/reward/controller change was made as
part of this comparison.

An NN-only replay of successful holdout seed 9020 completed qualified placement
in 13.02 simulated seconds with zero measured chassis penetration. It used
the existing replay recorder and evaluator without a teacher, IK controls or
injected object states. This selected success illustrates behavior; it is not
an additional independent trial or a reliability estimate.

Verification: 85 focused framework and external pickup acceptance tests passed,
including snapshot cadence, forced final-epoch saving, disabled/default behavior
and invalid-option rejection. Pickup Ruff and `git diff --check` passed.
The full framework suite finished with 1,274 passed, 28 skipped and 14 failures:
ten matched earlier deployment provenance, five-foot objects, script vocabulary,
missing-JAX MJX, posture and seam failures; four local-socket tests were blocked
by the sandbox. Rerunning the TensorBoard file with local socket access produced
26 passed and the previously observed TIME_WAIT failure. The full suite is not
green; its unrelated failures were not changed to make this feature pass.

### Gradual object-size curriculum

`place_train --size-variation .05` opts into uniformly sampled small, reference
and large objects at scales `.95, 1, 1.05`. Without this option the trainer still
uses the original standard-only distribution. `env_cfg(size_variation=...)`
bounds the fraction to `[0, .2]`; geometry, collision hull, reset height and
mass (constant density, cubic scaling) change together at environment creation.
This is uniform scaling of one soap shape, not independent aspect ratios or
different object classes. Each rollout keeps its object's physical size fixed.

Decision specification: success means real pickup followed by complete release,
whole-hull containment and settling, with the existing failure-first collision
criteria. The environment is the near-start bench and fixed tray; actuators are
the same four incremental arm controls and one claw control. The actor receives
proprioception plus synthetic size/position estimates in metres in the current
base frame, sampled at 10 Hz with 60 ms delivery latency, 2 mm noise, 5% dropout
and 250 ms expiry. Invalid detections zero their position/size and carry validity
and age. The privileged teacher/critic may use contact and true state; the actor
does not. No camera pipeline or deployable perception adapter is claimed here.
Pickup has no deployment hook/control binding; this remains simulation training.

The original qualified near-grasp actor can initialise this narrower distribution
through the existing explicit actor-only transfer. Its critic, optimizer and clock
are not adopted. Ordinary checkpoint resume remains exact-contract-match,
including the size scales; widening must not masquerade as an unchanged resume.

The task's `size_curriculum.next_variation` defines the provisional ladder
`±5% -> ±10% -> ±20%`. A driver may advance only when **each** size's separate
12 unseen trials has at least eight qualified placements and ten pickups,
without intrusion/fall/invalid-layout failures or measured chassis penetration.
Malformed/incomplete reports fail closed. Drivers must reserve new evaluation
seeds at every gate; previously examined sets are not fresh validation.
These are initial experiment gates, not calibrated reliability guarantees.

First experiment budget: one CPU collection/fitting worker, at most 36 physical
teacher attempts, six whole successful episodes held out, 200 fitting epochs and
snapshots at 50/100/150/200. Compare the retained epoch-100 reference and the
MSE-selected new actor on 12 unseen trials per size, with fresh clocks, unchanged
noise/failure criteria and zero XY jitter. Run only this narrow rung unless its
gate passes; incomplete collection or unsafe outcomes are reported, not relabelled
as success. The NN learns the verified demonstrations; NN-only evaluation never
constructs the teacher. No new reward or scripted policy is inserted at replay.
Use a 15-minute collection/fitting deadline and preserve partial logs on timeout.

#### Serial experiment queue

`python -m tasks.jumper.pickup.size_queue --output <new-directory> --runs 5
--minutes 45` queues at most five serial jobs inside a 45-minute overall budget.
Each job has a 15-minute training-plus-evaluation budget. The queue starts from
the explicit retained epoch-100 checkpoint above, not whichever checkpoint is
newest. Its `queue.json` is atomically updated as jobs progress; every job has
a separate directory, training log, dataset, checkpoints and per-size reports.

Each job makes 12 physical demonstration attempts per size and holds out two
qualified whole episodes per size. Fewer than three successes in any size fails
collection; it must not train a supposedly mixed-size actor on reference-only
examples. All narrow collections start at a fresh clock. The existing teacher
owns labels only, and inference remains the noisy-observation NN.

`--placement-checkpoint` and `--source-size-variation` let the canonical trainer
continue from an explicitly retained placement actor with a fresh critic and
optimizer. The source's side, physics, observations, objective, history and
declared size range must match; ordinary resume is still exact-match. Transfer
reuses the same guarded writer/validation path as the original grasp transfer.

For every new job, the reference and candidate are evaluated on the same newly
reserved seeds (12 per size). A candidate is retained only with zero measured
penetration/failure, no per-size placement or pickup regression, and at least
one strict improvement. Otherwise it is marked `rejected-preserved`; nothing is
deleted. A retained candidate widens only if the absolute ladder gate also
passes. Unsafe candidates shrink one rung, as far as ±2.5%; safe release failures
keep the range and retry with new collection seeds. Only size distribution and
dataset samples mutate, not reward weights, robot mechanics or sensor truth.

The queue stops at its run/time limit and requests review. Timed-out children
are reaped by the subprocess deadline; evaluations check the deadline before
each bounded physical episode, so shutdown may exceed it by the final episode.
Rejected or failed jobs never replace the incumbent. All queued comparisons
are selection evidence: retained candidates still need a separate confirmation
set before any improvement claim, broader curriculum or hardware promotion.

2026-10-13, Ryzen 9 3900X/native CPU verification: the new size probes were
observed failing before implementation. The unchanged transfer extraction
passed 25 checks; the final focused matrix passed 110 checks, followed by the
two additional timeout/next-job tests in a seven-test orchestration pass.
Pickup Ruff and whitespace checks passed. The final framework run reported
1,277 passed, 28 skipped and the same 11 pre-existing failures listed above
(the local-socket sandbox failures were excluded by granting localhost access).
The first five-job queue was launched in
`logs/jumper/jumper.pickup/size-queue-five-runs-2026-10-13/`, with the original
epoch-100 actor retained until a new candidate earns selection. Launch and
collection progress are not evidence of learned size robustness; consult
`queue.json` and the per-job reports for completed measurements.

#### First queue outcome: budget exhausted, no candidate promotion

The 2026-10-13 Ryzen 9 3900X/native CPU queue stopped after 2,709.13 seconds
(45 minutes plus the final bounded episode). Four jobs were processed; the
fifth never started. The scoped coordinator and training children exited.
`review.json` reconstructs complete and partial reports from the preserved logs.
All jobs stayed at ±5%; neither widening nor safety-driven narrowing was
exercised because no complete candidate comparison reached the decision gate.

| Job | Qualified demonstrations: small / standard / large | Fitting | Final status |
|---|---|---|---|
| 1 | 9/12, 6/12, 5/12 | 200 epochs; MSE selected epoch 14 | Comparison timed out |
| 2 | 9/12, 5/12, 2/12 | Not started: insufficient large examples | Collection failed |
| 3 | 4/12, 2/12, not attempted | Not started: insufficient standard examples | Collection failed |
| 4 | 6/12, 6/12, 3/12 | 200 epochs; MSE selected epoch 23 | Comparison timed out |

Jobs 1 and 4 saved `fit/placement_warm_start.pt` and all four epoch snapshots.
Their single-episode placement smoke checks were respectively 1/1 and 0/1;
original-grasp-scope smoke checks were 1/1 for each. These mixed-layout smoke
checks are not per-size benchmarks and do not establish a comparative gain.
Jobs 2 and 3 preserved their logs and transferred actors, but created no fitted
candidate. The collection gate did not lower qualification to manufacture labels.

The comparison runs **all reference sizes before the candidate**, and the job
deadline includes collection, fitting and evaluation. This combination starved
candidate evaluation on this CPU: job 1 recorded 28 reference trials and job 4
recorded 21, with **zero candidate comparison trials**. Partial reference results
must not be scored as zero-success candidate trials:

| Job | Size | Reference placement / observed trials | Pickup / observed trials | Intrusion failures |
|---|---|---:|---:|---:|
| 1 | Small | 2/12 | 10/12 | 1 |
| 1 | Standard | 3/12 | 12/12 | 0 |
| 1 | Large (incomplete) | 1/4 | 4/4 | 0 |
| 4 | Small | 0/12 | 10/12 | 0 |
| 4 | Standard (incomplete) | 2/9 | 7/9 | 0 |
| 4 | Large | Not evaluated | Not evaluated | Not measured |

The deepest recorded reference penetration was 2.014 mm in job 1's small set;
job 4's incomplete standard set reached 0.635 mm without an intrusion failure.
Zero reported intrusion failures do not mean zero penetration. Across these
49 observed reference trials, 26 placement misses ended inside the bin without
complete release. This is a reference diagnostic, not evidence about either
newly fitted candidate. Per-demonstration logs do not expose contact depths or
individual failure causes, so do not infer collision-free teacher collection.

No candidate was promoted or qualified as improved, no policy was deleted,
and nothing was deployed. The monitoring heartbeat was removed after completion.
The experiment's size-robustness hypothesis remains **inconclusive**. The next
bounded evaluation should finish the two existing candidates with interleaved
matched reference/candidate trials, resumable partial evidence and an evaluation
budget measured separately from collection/fitting. The collector's inconsistent
qualification also needs investigation before more queued fits. These are next
steps, not changes applied to this completed queue; no replacement run was launched.

#### Completed fixed-policy confirmation (2026-10-14)

The saved candidates were evaluated without additional training under a frozen
protocol in `logs/jumper/jumper.pickup/size-confirmation-2026-10-14/`.
`plan.json` reserves 62 previously unused matched scenarios: 50 random
draws from the 50/50 smaller/larger size mixture (21 small, 29 large realised),
plus 12 standard-size retention checks. Noise seeds were sampled uniformly
without replacement from an unused range. All three saved policies ran every
scenario, with rotating execution order, fresh episode clocks and identical
initial observation/qpos/qvel fingerprints. Each finished trial was atomically
published without replacing prior evidence. No optional stopping, checkpoint
selection, teacher actions or post-hoc sample extension was used.

On the Ryzen 9 3900X, two native CPU workers (one world and one native/PyTorch
thread each) completed all **186 episodes** in **1,137.26 seconds** (18.95 minutes),
inside a separate 45-minute evaluation budget. Both exit codes were zero.
Physics/control remained 200/50 Hz, with the same full-CAD intrusion checks,
qualified pickup, whole-hull placement/release/settling predicate, zero XY jitter,
2 mm synthetic detection noise and 5% dropout. Actor inputs, weights and the
original reference checkpoint's checksum were unchanged throughout.

| Policy | Varied-size placements / 50 | Standard placements / 12 | Chassis intrusion failures / all 62 |
|---|---:|---:|---:|
| Retained epoch-100 reference | 4 | 4 | 0 |
| Queue run 1, fitted epoch 14 | 32 | 6 | 1 |
| Queue run 4, fitted epoch 23 | 8 | 2 | 8 |

The primary endpoint was placement on the 50 varied-size scenarios, compared
with the reference using an [exact paired McNemar test](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).
Bonferroni correction covered both candidates (two-sided alpha .025 each).
Conservative simultaneous 95% bounds on the placement probability difference
subtract [exact binomial confidence bounds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
for candidate-only/reference-only successes, using .9875 confidence per cell
and a union bound across the four cells. The sampling unit is a scenario,
not a correlated timestep; per-size counts below are descriptive guard checks.

Run 1 gained **56 percentage points** (64% versus 8%). It uniquely succeeded
on 29 matched scenarios and uniquely lost one; corrected p = **1.15e-7**.
The simultaneous gain interval was **+26.0 to +75.0 percentage points**, supporting
a material improvement larger than the predeclared ten-point threshold in this
fixed evaluation distribution. This is strong evidence of improved learned
placement for this saved actor, not merely lower imitation loss or a chosen video.

However, the **combined safety/retention improvement is rejected** by its
predeclared guard: small-object pickups fell to **14/21 versus 20/21**, and one
small-object episode latched chassis intrusion at **2.026 mm** penetration.
Its large-object placements were 22/29 versus 2/29, small-object placements
10/21 versus 2/21, and standard placements 6/12 versus 4/12. Neither the improved
placement aggregate nor the statistical test overrides those safety/retention
failures. `improvement_supported=False` in the report denotes this composite
guard; it must not be read as absence of a placement improvement.

Run 4 gained eight points, but corrected p = **.7754**, with a simultaneous
interval of **-17.0 to +31.3 points**. Placement improvement is **inconclusive**;
the data neither establish equality nor rule out a ten-point gain. It also fails
the combined guard: eight intrusion failures (one standard, seven large),
standard placements 2/12 versus 4/12, and lower pickups on both small and large
objects. Its maximum penetration was 2.049 mm. The reference had no intrusion
failures, but reached 0.577 mm penetration on a large-object episode; zero
failures do not prove zero contact or hardware safety.

Decision: preserve both candidates and the reference, with **no automatic
promotion, widening or hardware deployment**. Run 1 is the evidence-backed
placement candidate for future small-object grasp/safety work. The confirmation
does not cover independent aspect ratios, other object classes, random positions,
the right claw or real-camera perception, and it does not establish the training
algorithm's performance across independent retraining seeds.

The external acceptance probes were observed red before implementation and
passed all 11 controls, including sparse-result uncertainty, misleading one-win
gains, stale/corrupt resume data and mismatched starts. The nine-episode runtime
smoke passed all matching-state checks without an improvement decision. A final
audit rebuilt the report from all 186 raw trial files, checked frozen source and
checkpoint hashes, and independently reproduced the exact p-values by counting
binomial probabilities. The monitoring heartbeat was removed after completion;
no replacement experiment was started. All source/probe copies and raw evidence
are retained beside the report.

Final verification: 52 focused framework/probe checks passed, along with pickup
and diagnostic Ruff checks and `git diff --check`. The complete framework suite
again reported 1,277 passed, 28 skipped and the same 11 pre-existing failures.
No training/evaluation implementation was changed in this confirmation hand-off;
the only tracked change is this measured documentation.

## Narrower, longer continuation protocol (2026-10-14)

The completed +/-5% confirmation above supports run 1's placement gain, but
not its safety or small-object pickup retention. The user requested smaller size
variation and a longer run. The next bounded experiment is
`logs/jumper/jumper.pickup/size-narrow-long-2026-10-14/`, supervised by the
external `/tmp/jumper_narrow_long_run.py` coordinator. This is one experimental
continuation, not a restarted queue, an automatic promotion or a deployment.

Continue only the run 1 placement actor (SHA-256
`d255264e0f3c5e196cb8b1c75dc1c1e96c3e672c2c000814c9f8c1795763569c`),
declaring its source width .05. Narrow the destination to scales .975, 1 and
1.025. Use 96 balanced collection attempts, 32 per size, with three qualified
whole-episode holdouts per size, seed 410017, zero XY jitter and up to 600
imitation epochs. Collection must have more than three qualified demonstrations
in every size stratum; otherwise stop before fitting and preserve the evidence.
Only physically qualified placement episodes enter fitting. Physics, sensors,
actor inputs, frozen source normalisation, objective and failure criteria stay
unchanged. One native CPU world/thread collects demonstrations. This is
demonstration-based fitting, not PPO training.

Allow 60 minutes for collection, fitting and the trainer's small replay checks.
Save unqualified snapshots every 100 epochs, but preselect only the final
`placement_warm_start.pt`, restored from the best whole-episode holdout MSE.
Longer fitting does not itself demonstrate better closed-loop control. Old
policies, code and artifacts remain untouched, and no extra candidate snapshots
or replacement jobs are authorised by this protocol.

On successful training, reserve a separate 45-minute evaluation-only budget:
50 fresh IID small/large draws plus 12 standard scenarios, matched between the
run 1 initializer and the new actor, 124 episodes total. Freeze the 62 scenarios
before training with sampler seed 403014 and disjoint noise seeds from the
previous confirmation. Two CPU workers, one world/native/PyTorch thread each,
rotate policy order; check identical observed/physical starting fingerprints and
checkpoint/protocol hashes, and publish trials exclusively. The independent
comparison concerns these two fixed policies at +/-2.5%, not improved performance
on the old +/-5% distribution or the training algorithm across retraining seeds.

Require 62 completed matched pairs and worker exit codes [0, 0] before interpreting
the exact two-sided McNemar test (one comparison, alpha .05). Conservative 95%
delta bounds subtract two exact .975 win/loss probability intervals. Keep the
ten-percentage-point material-gain threshold. Failure to reject is not equality.
Any intrusion, fall, too-low or invalid-fixture outcome, per-size pickup
regression, or standard placement regression blocks the combined improvement
claim and promotion. No observed failures would not prove hardware safety.

Preflight: 30 focused external transfer/size/timeout controls passed, followed by
54 confirmation/framework checks; the two new orchestration controls were
observed failing before implementation. Ruff and whitespace checks passed.
The unchanged full-framework baseline remains 1,277 passed, 28 skipped and 11
unrelated failures. These checks validate the experiment machinery, not learning
success. Results were pending when this protocol was frozen; measured outcomes
are recorded below. `experiment.json`, stage logs/results and the
confirmation plan/trials/report are the evidence, not wall-clock runtime alone.

### Completed narrower continuation: no demonstrated improvement

On the Ryzen 9 3900X/native CPU, training completed successfully in 1,536.54 s
(25.61 minutes), followed by evaluation in 647.80 s (10.80 minutes), inside both
budgets. All 62 frozen matched pairs / 124 episodes completed with worker exits
[0, 0]. The raw-trial audit reproduced the report, identical starting-state
fingerprints, checkpoint/protocol hashes, preregistered contexts and independent
exact p-value. The old training queue and confirmation were not restarted.

Collection qualified 20/32 small, 12/32 standard and 5/32 large demonstrations.
After three whole-episode holdouts per size, fitting had 17 small, nine standard
and only two large episodes: 28 training episodes and nine validation episodes.
More attempts did not create a large balanced qualified dataset. Of 600 fitting
epochs, the saved best-MSE actor came from epoch 27 (holdout MSE .068981);
the final epoch's holdout MSE was .138243. The trainer's six-episode smoke checks
gave 5/6 placements and 5/6 grasp retentions, not evidence of matched improvement.

The independent endpoint-size primary comparison found **28/50 (56%) put-aways
for the new actor versus 34/50 (68%) for its run 1 initializer**. There were five
candidate-only wins and 11 initializer-only wins. The observed delta was -12
percentage points, exact two-sided McNemar p = .210114, with conservative 95%
delta bounds of **-35.16 to +13.19 points**. Placement improvement is
**inconclusive**, not proven worse or equal. A material gain of ten points is
not ruled out by these bounds, but improvement is not supported by this test.

| Size | Trials | Initializer placements / pickups / intrusions | New placements / pickups / intrusions |
|---|---:|---:|---:|
| Small (.975) | 27 | 17 / 22 / 1 | 12 / 12 / 1 |
| Standard (1.0) | 12 | 5 / 10 / 0 | 9 / 9 / 0 |
| Large (1.025) | 23 | 17 / 22 / 0 | 16 / 20 / 0 |

The safety/retention guard **fails**: the new actor has one small-object chassis
intrusion (maximum penetration 2.024944 mm) and fewer observed pickups in every
size stratum, especially small objects (12/27 versus 22/27). The initializer also
had one small-object intrusion, at 2.224742 mm; it is a comparison baseline, not
a hardware-safe policy. The new actor improved standard placements descriptively,
but that does not override the primary result or the failed guard.

Decision: preserve the new actor, all snapshots, dataset, trials and previous
policies; **do not promote, widen, deploy or launch a replacement run**. This
single continuation does not show that smaller variation plus longer fitting
solves pickup. Any future authorised work should first address small-object grasp
retention and obtain more qualified examples, rather than infer effectiveness
from imitation loss or additional epochs. Cause has not been established by this
comparison. No retention claim is made on the old +/-5% range, other shapes,
positions, the right claw, camera hardware or independent retraining seeds.

Final focused confirmation/framework checks passed (54), as did the raw-evidence
audit, diagnostic Ruff and whitespace checks. The full framework was not rerun
for this documentation-only review; its last recorded baseline remains 1,277
passed, 28 skipped and 11 unrelated failures. Evidence is preserved in this run's
`verification/` and `size-narrow-long-evidence.zip`; the final review removes the
monitoring heartbeat without deleting experiment artifacts.

### Small-object grasp recovery protocol (2026-10-14)

After the failed narrower continuation, the user authorised further work on
small-object grasp retention and qualified examples. One new experiment is
`logs/jumper/jumper.pickup/small-grasp-recovery-2026-10-14/`, driven externally by
`/tmp/jumper_grasp_recovery.py`. All prior artifacts and policies are preserved.
The hypothesis is that fresh qualified small-object examples plus witnessed
pickup-stage rehearsal may retain grasp skill better than whole-placement-only
fitting. This is a bounded hypothesis test, not an established root cause.

First compare the run 1 initializer and failed narrower actor on eight matched
small-object diagnostic scenarios (seeds 610017–610024). Record first dual-pad
pinch, first qualified unsupported lift, minimum grip-point/object-centre
distance, raw-action overshoot and existing completion/failure outcomes. Verify
identical starting fingerprints. Diagnostic witnesses remain privileged logging
only, never actor inputs; the canonical evaluator drives the unassisted policies.
These scenarios are diagnosis data, not independent confirmation evidence.

Then collect up to 32 fresh small-object teacher attempts (seeds 710017–710048),
stopping after 12 qualified whole placements. Persist each attempt report and
qualified trajectory separately, including a physically witnessed grasp time;
failed/colliding attempts never contribute labels. If the target is missed,
preserve partial evidence and stop before fitting, without retries or relaxed
criteria. The teacher is collection-only and is absent during NN evaluation.

Keep the prior narrower run's qualified whole-placement dataset and its original
28-train/nine-holdout partition. Add nine fresh whole episodes and their nine
pickup prefixes to training; reserve the other three fresh episodes and their
prefixes for validation. A prefix ends at the actual qualifying grasp witness,
not an arbitrary percentage of the trajectory. Full episode and prefix always
belong to the same split; repeated prefixes are intentional phase weighting,
not extra independent episodes. The old shortage of large-object training
episodes remains a limitation; this experiment targets small-object retention.

Transfer only run 1's actor, explicitly declaring source variation .05, into
the existing .025 destination contract. Reuse the existing fitter for up to
150 epochs, batch size 512, learning rate .001, seed 710017, with frozen source
normalisation and fresh learner state. Select only the best-holdout-MSE final
saved actor; verify a cold-load output match. Change no production code, rewards,
success criteria, physics, servo limits, sensors, action feedback or inputs.

Allow 45 minutes total for diagnostics, collection and fitting, then a separate
45-minute evaluation-only budget. Freeze 50 fresh IID small/large draws plus
12 standard scenarios with sampler seed 503014 before any fitting, disjoint
from previous confirmations and the diagnostic/collection seeds. Compare the
saved actor with run 1 in 62 matched pairs / 124 episodes, using the existing
fixed-policy evaluator protocol, two CPU workers and one world/thread each.
Require complete records, matching starts, unchanged hashes and worker exits
[0, 0]. Use the exact paired two-sided placement test at alpha .05, conservative
95% delta bounds, and the existing ten-point material-gain threshold. Any
candidate intrusion/fall/invalid/too-low event, per-size pickup regression or
standard placement regression fails the combined guard. No automatic promotion,
widening, hardware deployment or replacement run is authorised.

[Deep Learning, chapter 11: Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html)
supports using application-level metrics and instrumentation before incremental
changes. Here the application metrics are physical pickup, put-away and failures,
not imitation MSE alone. The preceding rising holdout error is consistent with
overfitting but does not establish the cause of closed-loop grasp loss.

The two new external recovery controls were observed failing before implementation;
71 focused recovery/rehearsal/transfer/confirmation/framework checks then passed,
as did Ruff. No new task-specific tests or production changes were added to the
repository. Results remain pending. The last full-framework baseline is unchanged:
1,277 passed, 28 skipped and 11 unrelated failures.

#### Recovery collection stopped before fitting

On the Ryzen 9 3900X/native CPU, the bounded diagnostic/collection worker stopped
after 721.93 s (12.03 minutes), exit code 1 without timeout. All 32 permitted
small-object collection attempts completed, yielding only nine qualified whole
placements against the preregistered target of 12. The explicit insufficient-data
guard stopped fitting; no new actor, fit report or confirmation was produced.
The planned 124 confirmation episodes were **not attempted**, not failed. All
nine qualified trajectories and every attempted collection report are preserved.

All 16 diagnostic NN episodes completed with matched initial fingerprints:
the run 1 initializer pinched and picked up in 8/8 cases, placing 6/8; the failed
narrower actor pinched in 8/8 but picked up and placed in only 4/8. Both had only
placement-success or timeout terminations in this diagnostic set. This identifies
the pinch-to-qualified-lift transition as a useful investigation target; a brief
dual-pad contact is not proof of a stable unsupported grasp. It does not establish
the physical cause or an independent statistical improvement claim.

Collection recorded qualifying grasp witnesses in 24/32 attempts but whole
placement in 9/32. Thus 15 attempts reached a grasp witness without qualifying
placement. The collector did not store carry/release phase transitions, final
contact witnesses or failure termination names per attempted teacher episode;
its reports cannot distinguish slipping, poor tray alignment, early release,
collision or failure to settle. That evidence gap must remain explicit. Mean
raw-action overshoot fractions in the NN diagnostics were .12729 for run 1 and
.07656 for the failed narrower actor; overshoot alone does not explain why the
newer actor qualified fewer lifts.

Decision: preserve all policies and partial collection evidence; do not relax
qualification, fit nine examples in place of the target, automatically retry,
promote or deploy. Any next authorised diagnosis should examine contact duration,
actual jaw angle/effort, ground/rim support and object slip through the lift,
then carry alignment, release timing and settling. Side-by-side replay and
phase/contact traces are more informative than another blind longer fit. The
old shortage of large-object examples remains unresolved.

The final artifact audit checked frozen input/source hashes, all eight diagnostic
pairs, all 32 collection reports and nine restricted-decoded qualified records.
Focused recovery controls and whitespace checks passed; no production code or
physics changed and the full framework was not rerun. Results are published in
`small-grasp-recovery-evidence.zip`; the completed-run heartbeat is removed.

### Pinch/lift video and contact diagnosis (2026-10-14)

The user authorised side-by-side replay and phase/contact traces, not another
fit. Recordings use the repository's canonical `ReplayVideo` through the existing
NN-only evaluator. The actors, CAD, physics, sensor noise, failure criteria and
action decoding are unchanged. The previously observed first lift-discordant
small-object case (seed 610018) and a successful control (610017) were selected
for illustration, **not independent confirmation or checkpoint selection**.
The wider-camera failure comparison is in
`logs/jumper/jumper.pickup/pinch-lift-video-2026-10-14-wide/`; the four-episode
recording/trace set, including the control, is in `pinch-lift-video-2026-10-14-r3/`.

In the wider failure comparison, run 1 qualified a pickup at 6.28 s, reached
37.627 mm object-floor clearance and placed at 15.84 s. The failed narrower
actor moved its arm by up to 1.049 rad and maintained dual-pad contact for up to
4.32 s, but reached only 1.901 mm clearance. Non-claw object contact existed in
1,498/1,500 control frames; there was no qualifying unsupported lift, and it
timed out at 30 s. No chassis intrusion occurred in these illustrative replays;
this does not establish hardware safety or generalise beyond the selected cases.

The failure trace exposes a concrete transition: at 6.20 s the requested jaw
target was -.370 rad, actual angle -.429 rad and simulated actuator torque
.594 Nm. By 6.30 s, the requested target was -.456 rad, actual angle -.443 rad,
torque approximately zero and dual-pad contact lost. The object remained
supported. A lower jaw-angle target is opening-directed here: the open angle is
`.10 - pi/2`, closed angle .10. The learned claw channel specifies an **absolute
target angle**, not closing velocity; even a positive raw claw output can reduce
closure. Later it closes without recapturing the object. Peak absolute actuator
torque was 1.155 Nm versus run 1's .908 Nm, so peak torque alone is not the
failure explanation. These are simulated joint torques, not measured gripping
forces or permissible hardware loads.

This supports investigating premature grip relaxation and the concurrent arm
lift path. It does not isolate their causal contributions: no intervention held
the claw closed or changed the arm trajectory. Such a counterfactual would be an
explicitly assisted diagnostic, not NN-only performance. The successful control
placed with both actors, so the new actor is not universally unable to lift.

Initial observation/qpos/qvel fingerprints match the earlier diagnostics and
each other, and success/failure classes reproduce. Subsequent trajectories are
**not bit-exact**: run 1's failure-case completion shifted +.02 s; the control
completions shifted -.40 s and -.64 s. Those differences are recorded, with no
claim that recorder instrumentation is numerically invisible. The rendering and
trace observation code changed, not production physics or learned commands.
The side-by-side video explicitly labels held final frames after an episode ends.

An initial recorder attempt failed because native MuJoCo does not expose the
high-level `qfrc_actuator` field; all its artifacts were preserved. The external
probe now reads existing native actuator forces and projects direct hinge
actuators through their actual model gear, rather than changing the backend.
A second recording stopped on a stricter exact-duration comparison; it too is
preserved. Three failing-first external trace/mapping controls passed after
implementation; 44 focused trace/framework checks passed, alongside Ruff and
whitespace checks. The full framework was not rerun. No new production code,
training, hardware deployment or policy promotion was performed. CSV traces,
contact plot, both matched videos, source identities and failed attempts are
preserved for the next diagnosis.

### Closed-claw counterfactual: closure alone does not rescue lift

The user authorised following the causal diagnostic. One bounded experiment at
`logs/jumper/jumper.pickup/claw-counterfactual-2026-10-14/` completed six episodes:
two previously observed small-object cases, each under NN baseline, unchanged
command-tape replay (sham), and assisted closed-claw replay. Native CPU physics,
actors, observations, sensor noise, servo limits and success/failure criteria
were unchanged. No fitting or production changes were made.

For each case, record the newer actor's complete raw command tape first. Replay
the four raw arm channels exactly; the assisted condition changes only the claw
channel to +1 after first dual-pad contact, then keeps that command latched. This
uses a **privileged simulator contact oracle**, not a deployable sensor frontend
or learned policy. Once the tape ends, zero arm commands hold the current servo
targets; no arm velocity is extrapolated. The primary outcome is qualified
unsupported pickup within the original tape horizon (30 s for failure case
610018, 13.74 s for successful control 610017). The full episode remains bounded
by the existing 30-second limit and physical failure terminations.

The independent trace audit checked every raw arm component against the saved
tape, the zero-command tail, each claw override, identical initial fingerprints
and unchanged checkpoint hashes. Both sham replays reproduced their NN baseline
outcome classes. Equal commands do not imply identical physical arm poses:
object contact and the unchanged perception gate can mediate their response.
This is a command-level ablation, not a prescribed-pose simulation.

| Case / condition | Qualified pickup within tape | Maximum clearance | Placement |
|---|---|---:|---|
| 610018 NN baseline | No | 1.902 mm | No |
| 610018 tape sham | No | 1.902 mm | No |
| 610018 assisted closed claw | No | 7.718 mm | No |
| 610017 NN baseline | Yes | 28.316 mm | Yes |
| 610017 tape sham | Yes | 28.316 mm | Yes |
| 610017 assisted closed claw | Yes | 28.316 mm | No |

In failure case 610018, the intervention began at 1.96 s. Maximum sustained
dual-pad contact was 4.34 s versus 4.32 s unassisted, with no qualifying
unsupported lift. Commanding the claw closed helped clearance briefly, but did
not reach the unchanged 15 mm/unsupported/sustained-grip requirement. This
contradicts the narrow hypothesis that preventing grip relaxation alone suffices
to rescue this arm command sequence; it does not prove grip relaxation never
contributes. All six episodes avoided recorded intrusion/fall/invalid/too-low
failures, which is not hardware safety evidence.

The successful control still qualified a lift under assistance at 6.70 s.
Placement did not complete because the assisted claw deliberately never released;
that is an intervention consequence, not learned-policy improvement or a new
placement defect. Holding the claw closed indefinitely is not a cleanup solution.

Read-only comparison with run 1's earlier working trace shows a divergence
before the jaw relaxation. At 6.20 s, run 1 was already unsupported with
17.711 mm clearance, LF_J1 at -2.220 rad and LF_J3 at -.947 rad. The newer
arm was still supported at approximately -.649 mm clearance, LF_J1 at
-1.736 rad and LF_J3 at -.572 rad, including under closed-claw assistance.
Those differences identify coordinated lift motion/contact geometry as the next
investigation target; they do not isolate a single faulty joint or justify
changing a joint limit, friction, effort cap, success threshold or observation.
The working trace is context from an earlier replay, not another controlled
condition of this experiment.

Decision: preserve all tapes, CSVs, videos and actors; no promotion or hardware
deployment. A next authorised arm-path ablation can distinguish learned arm
motion from grasp/contact geometry while keeping the jaw rule fixed. These two
selected cases provide local diagnostic evidence, not population statistics,
independent confirmation, or proof of a training improvement. The two new
intervention controls were observed failing first, then passed; 46 focused
intervention/trace/framework checks, the command-tape audit, Ruff and whitespace
checks passed. The full framework was not rerun.

## Single-arm scripted/learned lift comparison (2026-10-14)

The user chose to continue with one claw rather than introducing two-arm
coordination. The bounded diagnostic at
`logs/jumper/jumper.pickup/single-arm-teacher-check-2026-10-14/` ran six
episodes: the unchanged canonical `LeftPlacementTeacher`, the frozen run 1
initializer and the frozen failed narrower actor, each on previously observed
small-object seeds 610018 and 610017. This is diagnosis on selected cases, not
fresh confirmation, training or checkpoint selection. No fitting followed it.

Ryzen 9 3900X, native CPU, one world/native/PyTorch thread, 200 Hz physics and
50 Hz control; LF near-start fixed layout, one soap shape at scale .975,
existing synthetic noisy detections and all physical failure/success criteria
unchanged. A 20-minute invocation limit bounded the six recordings; it exited
zero without timeout. All three controllers have identical initial actor
observation/qpos/qvel fingerprints within each case. Source and checkpoint
hashes were checked after completion. The teacher adapter starts its canonical
reset-relative clock at zero, including the normal settling period, rather
than shifting the pickup/lift schedule by a second.

| Case / controller | Qualified pickup | Qualified placement | First pickup (s) | Maximum whole-hull clearance (mm) | Episode end (s) |
|---|---|---|---:|---:|---:|
| 610018 / scripted teacher | yes | yes | 5.18 | 36.544 | 12.68 |
| 610018 / run 1 NN | yes | yes | 6.28 | 37.583 | 15.84 |
| 610018 / newer NN | no | no | — | 1.901 | 30.00 timeout |
| 610017 / scripted teacher | yes | yes | 4.58 | 33.501 | 11.14 |
| 610017 / run 1 NN | yes | yes | 4.80 | 37.270 | 15.60 |
| 610017 / newer NN | yes | yes | 6.70 | 28.316 | 13.74 |

All six episodes had zero recorded maximum chassis penetration and no fall,
too-low, invalid-fixture or intrusion termination. This limited observation
does not establish hardware safety or general robustness. The teacher uses
privileged simulator state for IK and phase transitions; neither NN receives
it. The reused evaluator prints `NN_EVALUATION` even for the supplied teacher
adapter, so that log prefix alone is not evidence of neural control. CSVs,
reports and video labels explicitly identify the scripted condition.

For 610018 the teacher held a qualifying unsupported grasp for a longest
3.34-second interval, transitioned to carry at 5.20 s, lower at 10.98 s and
release at 11.60 s. Run 1 held a longest 8.54-second qualifying interval;
the newer NN had none. At 5.00 s the teacher was already unsupported at
16.986 mm clearance with LF_J1/LF_J3 at -2.173/-.042 rad. Run 1 followed
a different successful route (-1.916/-.800 rad, 5.948 mm clearance then),
whereas the newer actor remained supported (-1.907/-.480 rad, -.383 mm
clearance) and subsequently reversed LF_J1 motion before qualifying a lift.
Two different successful routes mean a single joint-angle prescription is
not justified. These observations do not prove that imitation averaged modes,
that a particular joint is faulty, or that claw/arm input coupling caused the
failure. The preceding held-claw intervention already showed closure alone
was insufficient to rescue this case.

NN outcome classes matched the earlier diagnostic; durations differ from the
original by +.02/-.40 s for run 1 and 0/-.64 s for the newer actor on
610018/610017 respectively. These are matched starts and reproduced outcomes,
not bit-exact trajectory reproduction. Videos hold and label the final frame
after each episode ends. The recorder is the repository's `ReplayVideo`.

Decision: retain run 1 as the promising reference and all existing policies.
Target subsequent demonstration work at coordinated grasp-to-lift motion and
closed-claw retention across varied arm poses, with release examples kept
distinct and whole-episode qualification intact. Verify example sufficiency
before fitting, then use fresh matched NN-only evaluations to test improvement;
do not infer it from these scripted successes or a lower imitation loss.
No observations, rewards, physics, effort caps, joint limits, production code,
hardware deployment or two-arm task were changed. Two adapter controls failed
first, then passed; 32 focused adapter/trace/intervention/framework checks,
Ruff, the six-trajectory/hash/phase audit and whitespace checks passed. The full
framework suite was not rerun; its documented unrelated failures remain.
The evidence directory preserves six CSV/MP4/report sets, the frozen plan,
completed report and audit, two three-panel videos and the lift-path plot.
`single-arm-teacher-evidence.zip` packages these with external controls,
logs, frozen source copies and both unchanged actor checkpoints.

## Grip location audit and rejected centring recipes (2026-10-14)

The user observed that even the scripted teacher grips near an edge, and
requested better placement of the grasp around the object's middle before
training. Contact position, not just pickup/placement success, is now part of
the diagnostic evidence. The canonical teacher targets its fixed-pad point
using `GRASP_OFFSET = (-.029345020, .019968078, .034235365)` world metres
relative to the object's origin, scaled with the object. That is an edge/upper
target, not a two-pad midpoint or the object's centre of mass.

Two preserved external recipe studies ran on the Ryzen 9 3900X, native CPU,
one world/native/PyTorch thread, 200 Hz physics/50 Hz control, LF near-start
fixed stance, zero XY jitter and soap scales .975/1/1.025. The first study,
`logs/jumper/jumper.pickup/center-grasp-teacher-2026-10-14/`, compared legacy
and a middle-of-the-+y-face prototype on seeds 610018/610017 at each size
(12 episodes, 20-minute invocation limit). The prototype waits for alignment
before closure, holds the grasp pose and lifts its grip point vertically at
20 mm/s using the existing IK branch and servo authority. This is a recipe
comparison, not an isolation of target, timing or lift speed.

After inspecting bounded read-only IK queries, a second predeclared six-episode
study at `logs/jumper/jumper.pickup/centerward-quarter-teacher-2026-10-14/`
tested the same prototype recipe with a target only one-quarter of the way
from the legacy target toward the middle-face target (12-minute invocation
limit). Its physical initial qpos/qvel fingerprints match each first-study
legacy case; sources were frozen and checked for both studies. These are
selected development seeds, not independent confirmation or NN learning.
Both invocations exited zero without timeout. Neither ran fitting or changed
production teachers, actors, observations, physics, rewards or safety criteria.

| Scripted recipe | Episodes | Qualified pickups | Qualified placements | Verified middle grasps | Eligible middle-grip examples |
|---|---:|---:|---:|---:|---:|
| Legacy | 6 | 6 | 4 | 0 | 0 |
| Full middle-face target | 6 | 0 | 0 | 0 | 0 |
| Quarter-centreward target | 6 | 6 | 5 | 0 | 0 |

The extra placement for the quarter recipe is a descriptive result on tuning
cases, not statistically verified improvement. All 18 episodes had zero
recorded chassis penetration and no fall, too-low, invalid-fixture or intrusion
termination. Timeouts remain failures; this is not hardware safety evidence.
The earlier progress update stating four quarter-recipe placements was corrected
by the completed audit to five.

The native `env_mjdata(0)` read-only interface supplies actual pad/object contact
positions. Contact centroids for each insert are transformed into object
coordinates as `p_object = R_world_object.T @ (p_world - object_origin)`.
The diagnostic middle region requires BOTH pad centroids within .35 of the
object half-width/half-height on its x/z face axes throughout the eleven-frame
first qualified unsupported-pickup witness. This additional quality bar was
declared before the initial trial, is not a change to task success, and does
not prove an optimal grasp or force closure. The plotted rectangle is the x/z
bounding extent, not a replacement collision shape. Simulator contact truth
is evidence/teacher-only and is never an actor observation.

The first prototype's exploratory classifier initially also inspected samples
after the picked flag had latched. The conservative review instead checks the
eleven-frame FIRST witness, because a latched pickup does not prove current
unsupported contact. Failing-first controls pin this distinction. Original
reports are preserved; `final_review.json` carries authoritative audited
eligibility. All recipes produced zero eligible middle-grip examples and no
new demonstration tensor was admitted for training.

At the first legacy small-object lift on seed 610018, the fixed pad centroid
was (-3.388, 16.531, 36.118) mm and the moving pad (-27.737, -16.582,
25.226) mm in the object frame. The moving pad was about .862 of the x
half-width from centre; the fixed pad was also high on the object. The
quarter-centreward recipe moved the fixed contact lower, but its moving
contact remained near the x edge, so target-point movement alone did not
produce an opposed middle grip.

Read-only scratch IK queries explain why simply commanding the middle was
unsuccessful. The small +y-face target at (0.259175, 0.099500, 0.049775) m
left 25.406--25.502 mm position residual across four declared solver starts.
The nearer -x-face centre was position-reachable (residual below .001 mm),
but its estimated closing axis was 29.5--83.1 degrees away from the opposing
face normal. Thirty-two position-plus-axis queries across four faces, both
axis signs and four starts found none below the declared 4 mm/10 degree
screen; the best-position result still had 12.881 mm/17.455 degree errors.
An interpolation screen found the quarter target position-reachable; halfway
to the centre left 5.889 mm error, explaining the conservative second recipe.
These finite numerical searches are not global impossibility proofs, collision
checks, path validation or hardware calibration. The scratch queries never
wrote robot/object state. The full-centre prototype never reached pinch or
carry, so those unvisited phases are not validated by its completed run.

Decision: reject both prototypes as a fix for middle-grip training labels;
preserve the legacy code, actors, all failed trials and source identities.
Do not train more edge-grip labels and call that better grip placement. The
next design work must select a feasible opposed-contact pose AND an approach
that can reach it, potentially including body positioning rather than only
five fixed-stance claw actions. A change to body positioning or reset layout
would be a new task/distribution boundary, not a hidden fix to these frozen
comparisons. No such change was made. Five new geometry/quality/target controls
failed first, then passed; 37 focused controls/framework checks, Ruff, the
18-episode fingerprint/source/eligibility audit and whitespace checks passed.
The full framework suite was not rerun. A scoped evidence ZIP preserves both
studies, contact CSVs/plot, read-only query results, source copies, probes and
logs; no books, secrets or unrelated changes are included.

### Insert-normal and servo-positioning development audit (2026-10-14)

Decision: retain the production teacher and all actors. No neural-network
training, qualified training labels or policy promotion resulted from this
development audit. The middle-grip step is **not complete**. The following
external prototypes concern one selected LF small-soap case (seed 610018,
scale .975), not held-out policy improvement, all objects or hardware safety.
Sources and failed results are preserved under
`logs/jumper/jumper.pickup/insert-alignment-2026-10-14` and
`central-positioning-pilot-2026-10-14` with distinct revision suffixes.
The host was the local Ryzen 9 3900X, native CPU MuJoCo 3.11.0, one world and
one native/PyTorch thread, 200 Hz physics and 50 Hz control.

Collision-mesh facet measurements found the fixed insert outward normal in
palm coordinates to be `[-.000056693, -.174105885, -.984726935]`. It differs
from the legacy visual-derived axis by 9.717 degrees. The old contact reference
is **on the facet plane**: its 16.374 mm displacement from the facet centroid
is tangential, not evidence that the point is invalid. Neither discrepancy
alone proves the cause of the old edge grips. The soap's broad faces are +/-Y.
For a fixed pad approaching its +Y face, the signed insert-normal target is
world -Y; a face-normal plane constraint would instead make it tangent.
The selected facets have 36.396 mm normal separation and 15.949 degrees
opposition error at jaw -.587223 rad. These finite-patch measurements are not
force-closure proof or a requirement for exactly parallel pads.

The corrected scratch screen accepted 18 of 243 point/orientation queries,
all at hypothetical body X +60 mm, none at X 0 or +30 mm. It used one start
per target and one valid reference point; it does not prove global
unreachability, collision-free travel or necessity of body movement. Its first
version failed JSON serialisation; the corrected report is in `screen-r2`.
Four retreat candidates passed separate bounded arm-path and support-foot
scratch checks. Physical pilots subsequently moved the body using support-leg
servo targets only, never post-reset robot/object pose writes, welds or forces.
Global geometry, physics, perception, rewards and the actor's 117-input /
five-action contract remained unchanged. The additional support-leg commands
are privileged teacher controls, **not compatible five-action training labels**.

Measured development outcomes:

| Pilot | Simulated time | Outcome |
|---|---:|---|
| initial | 1.02 s | Upward stow IK residual rejected |
| r2 | 14.56 s | Body advanced 60.93 mm; XYZ tracking timeout |
| r3 | 10.72 s | Bounded body feedback reached 1.639 mm XYZ error; single grasp IK screen rejected |
| r4 | 17.42 s | Multiple actual-pose IK starts found a sampled corridor; runtime collision guard stopped approach |
| r5 | 18.74 s | Phase-specific contact correction passed the earlier stop; approach timeout, no pinch |
| r5-observed2 | 18.74 s | Same controller/result; joint logging exposed an infeasible time budget |
| r6 | 20.32 s | Reached measured alignment and entered closure; finger-body/object closure screen rejected |

The first observability repeat failed after simulation while serialising a
tuple of waypoints; its report/trace are incomplete and its video/log remain
preserved. A regression and output-only correction preceded the completed
repeat. It is not an independent evaluation episode. An initial suspicion
that the servo target was limit-clipped was disproved: the no-physics limit
audit in `limits.json` found matching native/action ranges (LF J2 -2.2..1.65
rad). The target was still rate-limited and progressing. The measured path's
nominal speed lower bound was 8.324 s, already longer than the original 8 s
approach deadline before settling and perception pauses. R6 used a calculated
15 s deadline with the same .45 rad/s speed, capped at 25 s per approach and
45 s per pilot. It did not increase servo speed or bypass detection gates.

R6 entered closure for three physical control frames, with zero dual-pad pinch,
zero central-contact frames and zero qualified lifts. The support scratch
collision query then predicted 4.469 mm penetration between
`LF_finger_link_meshcol2` and the small soap during closure, exceeding its
2 mm claw-subtree/object guard. This is a **predicted overlap in the support
query**, not a measured chassis intrusion or a hardware force. Maximum
support-foot drift was 7.867 mm under the unchanged 8 mm guard; no existing
episode failure termination fired. This is not proof of safe closure. The
first canonical 11-frame witness, both actual pad centroids within .35 of
object X/Z half-extents, and continued unsupported central lift remain required.

Next design work must check both finite insert patches and the swept finger
body through closure, then choose a contact-aware approach that establishes
dual-pad contact without first pushing/interfering with the object. A reachable
point and normal alone are insufficient. Do not loosen the centrality/contact
guards or collect more edge-grip labels as a substitute. All production code,
old training artifacts and unrelated dirty changes were preserved. New external
geometry, serialization, servo, feedback, candidate, contact-phase and deadline
controls were observed failing before implementation; 47 focused external /
framework checks and Ruff passed. The full framework suite was not rerun.
`positioning_review_r6.json` records measured outcomes beside this decision;
the scoped evidence archive contains plans, sources, probes, traces, logs and
videos, without books, credentials or unrelated changes.

### Closure-aware rejection prototype (2026-10-14)

Decision: keep the production teacher and all checkpoints unchanged. An
external teacher-local closure prediction now rejects an unqualified closing
path **before approach**, rather than relying on point/orientation IK followed
by a blind close. It is a conservative candidate filter, not a central-grip
fix, physical force-closure proof or a new training contract. No NN training,
qualified examples, policy promotion, physics changes or hardware work ran.
The local Ryzen 9 3900X/native CPU MuJoCo 3.11.0 setup remained one world and
one native/PyTorch thread, 200 Hz physics and 50 Hz control.

The preserved `closure-aware-screen-2026-10-14` study tested 72 scratch IK
queries: nine object X/Z targets, four starts, and the valid legacy on-facet
reference versus the measured facet centroid. At hypothetical body X +60 mm,
six passed the unchanged 4 mm/10 degree IK/alignment screen. None survived
65 sampled jaw angles against active collision hulls. Both pad signed
distances had to be within +/-2 mm, both contact-centroid or closest-point
proxies within .35 X/Z half-extents, and no earlier non-insert object contact
over 2 mm, LF/floor penetration over .2 mm or protected robot penetration over
2 mm was permitted. These are scratch candidate conditions; canonical
first-witness physical qualification still applies separately.

To distinguish arm reach from rigid jaw fit, `closure-shape-fit-2026-10-14`
tested 108 virtual rigid placements using the facet-centroid reference and
12 rolls about the object-face normal. Its `-r2` study tested 972 placements,
adding -7/0/+7 degree tilts around two world axes and centroid X/Z offsets
-0.6/0/+0.6 of object half-extents. The centroid need not be central; the
**contact** centrality threshold stayed .35. The second study kept the whole
fixed insert at least .5 mm outside the object's +Y face initially. Neither
study found a surviving sampled closure. They deliberately ignored support,
floor and other-robot contacts when classifying rigid object fit; they are
not reachable or safe teacher poses. All transforms were applied only to
scratch `MjData`, never live robot/object state. Four independent MuJoCo
transform spot checks measured at most 1.567e-8 m point error and 4.417e-8
rotation-matrix error. Every query was preserved separately and all reports
completed at their predeclared counts without physics steps.

The bounded physical `closure-aware-pilot-2026-10-14` used the existing
servo-driven positioning scaffold on selected small SOAP seed 610018, scale
.975. At its measured body pose, 19 of 36 IK candidates received a closure
sweep; all were rejected. The pilot stopped after 536 frames / 10.72 simulated
seconds, before any approach or closure frames. Maximum support-foot drift
was 6.688 mm under the unchanged 8 mm guard and no existing episode failure
termination fired. Zero pickups or central grips were produced. This verifies
the prototype's conservative rejection path, not successful manipulation.

Important limitation: a closest point between flat faces can be nonunique.
Positive-distance points are **not actual pad contact centroids**, and a
frozen object does not capture its displacement, rotation or friction during
closure. Thus these finite failures do not prove that the claw cannot grip
centrally or that CAD/physics is wrong. The representative `closure-path.png`
separates scratch collision-contact samples from closest-point proxies and
marks the first rejected closure region. Do not reduce the acceptance bar or
turn these geometric queries into training examples. The next diagnostic
should examine finite insert contact regions and swept finger hulls against
the source geometry, addressing closest-point ambiguity before further pose
selection; any retained pose still needs an actual canonical central-lift
witness and independent physical cases before training it as reliable.

Six new policy/frame controls failed before implementation; seven new controls
passed, including a native MuJoCo positive-control gripper so the sweep cannot
pass by rejecting every model. In total 54 focused external/framework checks,
four transform checks, completed-count/frozen-source/physical rejection audits,
Ruff and whitespace checks passed. The full framework suite was not rerun.
All old code, policies, artifacts and unrelated dirty changes remain preserved.
The scoped evidence archive contains all three finite searches, pilot video
and trace, source identities, probes, logs, review and plot, not books or secrets.

### TDD finite-region fix and unresolved physical grip (2026-10-14)

Decision: the software closest-point defect is corrected, but **central physical
grasping remains unresolved**. Commit `4bb7882` adds the generic
`mjrl.contact_regions` functional core and its BDD-style framework tests. A
centred 52 x 40 mm planar pad region, .5 mm from a 60 x 94 mm object face, has
geometric central fraction zero; the old nearest-corner proxy rejects it with
fraction .867. An external reduction actually failed against the preserved old
rule. The corrected function clips the finite region to the selected object
face and gap band, then reports projected area and geometric centroid. It never
reports that centroid as an observed physical contact or a successful grip.
Immutable `FaceWindow`/`ContactRegion` contracts keep units and ownership explicit.

The prototype uses a geometry **Strategy** behind a MuJoCo **Adapter**: candidate
measurement can change without changing servo execution or canonical evidence.
The geometric core has no simulator dependency; the adapter supplies collision
mesh vertices through geom frames in object coordinates. The planned-aperture
adapter encodes the existing absolute finger action, not a new actuator contract.
The cost is an extra candidate-measurement boundary that needs its own frame and
positive-control tests. No production controller refactoring was mixed with this
feature, and global `Claw`, rewards, physics and sensor contracts were unchanged.

On the local Ryzen 9 3900X/native CPU MuJoCo 3.11.0 setup (one world and one
native/PyTorch thread), `contact-regions-screen-2026-10-14` completed 72 queries;
six passed IK/alignment, zero survived the finite-region closure screen. The
first `contact-regions-shape-fit-2026-10-14` attempt saved 90 of 108 queries,
then its adapter failed extracting a polygon from a merged reference plane.
That report is incomplete and preserved; it is not evidence of physical failure.
An exact-plane extraction regression passed in the `-r2` adapter. Its 108-query
rigid fit and a separate 972-query expanded fit completed with zero survivors.
Selecting a single facet still discards adjacent usable surface area, so the
next Strategy area-integrated all facing convex-hull triangles independently.
Its roof-shaped regression confirms the complete projected area and stable
centre; `contact-regions-surface-fit-2026-10-14` completed 108 virtual fits,
again with zero survivors. These virtual placements ignore support/floor/reach
and freeze the object; they are neither hardware proof nor a global
impossibility claim. No physics steps occurred in these geometry studies.

The bounded `contact-regions-pilot-2026-10-14` trial used the integrated-surface
Strategy at the measured robot pose, and contact-aware reference target offsets
of +/- .6 object X/Z half-extents. Reference-point offsets need not themselves
be central; the actual-contact .35 requirement was not changed. A screened
aperture and .45 rad/s teacher request ramp replaced blind full-close in the
prototype, through the unchanged squeeze limiter. However, that branch was
**unvisited**: all 16 IK-passing candidates out of 36 failed the closure screen.
The pilot stopped after 536 control frames / 10.72 simulated seconds, before
approach or closure, with maximum support-foot drift 6.688 mm under the 8 mm
guard and no existing failure termination. No sampled admissible central closure
was present even after the first blocking sample. First blockers included fixed
palm/finger pieces against SOAP and palm-pad/palm pieces against the floor.
This validates rejection, not gripping. There were zero qualified pickups,
zero training examples, no NN training and no policy promotion.

Twelve generic BDD tests pass, including clipping, tilted planes, opposing faces,
axis permutation, immutable ownership and invalid input. The full framework at
the initial ten-test snapshot had 1,287 passes, 28 skips and the same 11 known
unrelated failures; the two later ownership/axis tests also passed. The final
focused run including the four CAD controls had 66 passes; adapter-plane, integrated-surface and aperture
controls were observed red before their implementations. Frozen-source/count /
physical rejection audits, Ruff and whitespace checks passed. All older code,
checkpoints and failed attempts remain preserved, and unrelated dirty changes
retain their original hash. The evidence archive includes the incomplete study,
all successful reports, probes, logs, sources, video and the regression plot.

The bounded `saved-pose-cad-2026-10-14-r2` diagnostic then inspected five saved
scratch poses / nine contacts, without physics steps or live-state/asset changes.
The first diagnostic version reprocessed visual vertices after world rotation,
making topology pose-dependent; preserve that result but exclude its classification.
R2 processes each visual mesh once in its geom-local frame, inverse-transforms
the query point, and uses existing trimesh triangle queries with a vectorised
bounds-index Adapter (the optional `rtree` package is unavailable). Three ray
parities must agree; non-watertight/inconsistent meshes retain ambiguity. Hooked
versus solid-jaw controls, an open-mesh control and a transformed-frame control
passed. This is ray parity, not a winding-number calculation or a universal
mesh-validity certificate.

Four moving-insert collision witnesses were .97--1.20 mm **outside** their
watertight visual surface while native insert penetration grew from 1.28 to
6.04 mm. This establishes a local collision/visual discrepancy, but these
different distances cannot be subtracted to infer clearance. Finger-piece
visual distances of 3.71 and .76 mm remain ambiguous because those source meshes
are not watertight. At the preserved r6 blocking query, a fingertip witness was
inside its watertight visual surface (native penetration 3.34 mm), and the
insert witness lay on its visual surface despite 10.31 mm native penetration.
Local witness checks therefore do not justify attributing every obstruction to
a false hull or establish whole-object clearance. No collision geometry changed.

A read-only root-cause committee reconciled the remaining evidence. Both
reviewers confirm the saved-pose diagnostic is complete and recommend testing
one coordinated seating hypothesis, rather than repeating grids or regenerating
meshes. At one sampled transition the fixed insert remains .66 mm away while the
moving geometric region becomes central only after more than 4 mm penetration.
A frozen object cannot model contact-driven seating, but that limitation is not
permission to bypass collision guards or blindly close. Any next trajectory
must be validated as actually executed, retain physical first-witness acceptance,
stop on missing evidence/timeout/guard failure, and stay separate from training.
The shared region helper explicitly models a box-face candidate window; its use
for bevelled SOAP remains an approximation, not exact hull contact geometry.
No safety/centrality weakening, source-geometry modification, actor contract
expansion or hardware deployment is authorised by these results.

### One coordinated-seating path: rejected before physical replay (2026-10-14)

Decision: preserve this failed path; do not replay it physically, train on it,
weaken contact guards or start another parameter grid. The reviewers withdrew
claims that the moving region was absent at every aperture, that exact pad
parallelism was required, and that the finite studies established a global
mouth-geometry obstruction. Query 0 contains moving geometric centralities
.804, .335 and .074 at apertures -.6363, -.5381 and -.4890 rad. These are
candidate regions, not qualified physical contacts. The fixed insert remains
.661 mm away as the moving insert penetrates more deeply. The corrected
closest-point defect was insufficient to produce a grip; it is not proved to
have never contributed to earlier failures.

The user authorised one bounded coordinated-seating investigation. The external
`/tmp/jumper_seating_path.py` prototype freezes query 0 and the measured
`failed_planning_qpos` from the previous pilot. Its single path preshapes at the
saved stowed arm to -.8 rad, approaches the original query at that aperture,
then combines closure to sample 34 (-.6363 rad) with a single midpoint-derived
arm correction. The object-frame correction is (+14.989, -1.000, -5.397) mm;
it translates the reference point using the candidate region midpoint, not a
claimed contact oracle. No pose grid or additional candidate selection was used.
The only seating IK solve returned 1.568 mm / 4.388 degrees, within the existing
4 mm / 10 degree screen. That solve is not a collision-free path or grasp proof.

`coordinated-seating-2026-10-14/plan.json` records the protocol and source hashes.
Five-joint waypoints use <=.009 rad sampling, corresponding to the unchanged
.45 rad/s request limit and .02 s control period. The MuJoCo Adapter checks
every intended sample, keeps every predicate result and concurrent contact
failure, and does not veto a path using unused fully-open-at-target states.
Object-contact permissions remain unchanged: preshape permits no penetration,
approach permits inserts only up to 1 mm, close/lift permit the claw subtree
up to 2 mm. Pad penetration, LF floor and robot/chassis guards remain active.
Finite-region geometry remains a candidate Strategy, not physical evidence.
No new pattern framework or production controller refactoring was added.

On the same local Ryzen 9 3900X/native CPU MuJoCo 3.11.0 setup, the bounded
scratch validation completed after **55 samples**. Sample index 54 rejected
the **preshape** phase at jaw -.987775 rad: the moving grip insert intersects
SOAP by **.480801 mm**. Joint limits, LF floor, other-robot, object-chassis and
2 mm pad-depth predicates passed at that sample, but the no-object-contact
preshape predicate failed. The saved stowed pose is therefore not a clear
preshape location for this path. Approach and coordinated seating were unvisited;
this is not evidence that all coordinated-seating motions are impossible.
The report is complete, timed out false, physical replay admitted false,
physics steps zero and live state writes false. Only private scratch `MjData`
was changed after the ordinary reset; the live environment state was asserted
unchanged. All source hashes and the unrelated tracked-diff hash were preserved.

Eleven external behavioural controls were observed red before implementation
(the missing prototype), then green: unused-path obstruction isolation, all
concurrent blockers retained, missing evidence/deadline rejection, geometry
without actual dual contact cannot qualify lift, 11-frame/15 mm boundaries,
bounded owned waypoint interpolation and invalid boundary inputs. The combined
focused run passed **63 tests**, with 35 warnings, in 6.67 s. Ruff and whitespace
checks passed. No new production source changed, so the earlier full-framework
baseline remains a historical result, not a newly rerun suite. No training,
policy promotion, model/physics/sensor changes, resets into favourable grasp
poses or hardware actuation occurred. All old policies and code remain intact.

Next investigation, requiring a separately bounded path rather than a retry of
this one: establish a genuinely object-clear retreat/preshape trajectory through
existing servo commands, then revalidate approach and coordinated closure.
This result identifies an early path obstruction, not a need for more NN epochs
or permission to remove collision geometry.

## Clear preparation and measured-path continuation (2026-10-14)

On the local native CPU simulator, the preserved stowed-arm preshape reproduced
the premature moving-insert contact (0.481 mm penetration). Moving the open arm
to the preserved preparation pose before preshaping passed 509 sampled native
collision checks without writing live state. This is a path-screen result, not
a physical grip. The subsequent frozen-object seating edge still failed at
2.344 mm palm/soap penetration; it was not executed.

The shared `shortcut_joint_path` helper removes unnecessary stops only through
sampled, checked corridors and fails closed on exhausted budgets. Its isolated
implementation and regression tests are commit `d525df5`. The preparation route
was reduced from 15 arm waypoints to three, followed by a separate preshape
waypoint. External teacher controls additionally pin synchronized joint steps,
floating-point rate-limit boundaries, and measured-pose corridor continuation;
final alignment and collision predicates were not relaxed.

Physical replays under `clear-preshape-fix-2026-10-14/` preserve every failure:

| Replay | Measured result |
| --- | --- |
| Initial | Incomplete after a floating-point rate-boundary exception; video retained. |
| R2 | 1,500 steps / 30 s, timeout at an intermediate tracking stop. |
| R3 | 1,537 steps / 30.74 s, approach timeout at the tracking-lag pause. |
| R4 | 1,537 steps / 30.74 s, approach timeout with approximately 0.064 rad steady joint error. |
| R5 | 696 steps / 13.92 s, stopped when RF support contact disappeared, leaving LR/RM/RR. |

R5 passed every recorded actual preparation collision check and advanced past
the intermediate stop through a checked measured-pose corridor. Maximum measured
support-foot drift was 7.217 mm (8 mm limit). Nevertheless preparation failed:
the unchanged four-support-contact requirement stopped the episode. This does
not establish whether the lost contact is caused by support tracking, dynamic
load transfer, or the contact detector. No central grip or placement succeeded.
The next prerequisite is diagnosing this support loss before another grasp
attempt, not training on these failed paths or weakening the support guard.

Validation: 56 focused framework checks passed (35 warnings, 6.83 s), nine
external prototype controls passed (38 warnings, 10.55 s), and Ruff passed.
Red regressions and all intermediate sources/logs are preserved beside the run.
The full-framework baseline was not rerun. No NN training, policy promotion,
physics/reward/sensor changes, favourable live-state resets or hardware actions
were performed. Privileged support commands are not five-action actor labels;
old code, demonstrations and policies remain preserved.

## Chassis collision boundary

### Detected self-contact qualification and remaining clearance gap (2026-10-14)

Pickup now separately latches detected robot-to-robot contact as a failed episode:
any current matched contact, or negative signed distance recorded in a physics
substep, prevents qualified pickup and placement. The latch survives separation
and is cleared only for reset environments. A whole-robot/whole-robot sensor
excludes terrain and task objects by its matching scope. Chassis penetration
remains a separate diagnostic; its historical 2 mm tolerance no longer grants
permission for a currently detected robot self-contact. Zero-distance transient
touch that has resolved before the control sample is not established by the
distance history, whose absent entries are also zero. Positive clearance still
requires a separate planning check.

This is a qualification fix, **not complete self-collision coverage**: masks,
parent filtering, CAD, physics, walking configurations and actor inputs are
unchanged. Same-arm masked pairs remain invisible to contact sensors. Checkpoint
contracts now include `self_contact_guard`; old policies are preserved but cannot
be silently loaded as if qualified under this stricter contract. No new training
or policy promotion was performed.

A native CPU mask-independent audit queried 863 front-arm/robot geometry pairs
in each of three scratch states, with zero physics steps and unchanged live
state. Same-rigid-body pairs were reported separately; adjacent bodies were
measured, not silently declared safe. The 5 mm search cutoff is diagnostic, not
an approved clearance margin. In the saved preparation-start pose, masked LF
finger pieces overlap the forearm collision hull by up to **9.024 mm** and the
upper-arm hull by **8.701 mm**. The supporting RF finger/forearm hull overlap is
**3.732 mm**, present also in the saved LF preparation-goal state. Reset has two
non-adjacent pairs within the cutoff, both positive base/upper-arm distances
(3.535 mm LF and 3.733 mm RF). These are discrete collision-hull measurements,
not proof of corresponding CAD intersection or continuous path clearance.

Evidence: `logs/jumper/jumper.pickup/self-clearance-audit-2026-10-14/report.json`.
The saved stowed/support poses must not be presented as scrape-free. Next work
must validate the implicated geometry and define explicit non-mating pair
coverage/clearance before reusing those poses; globally enabling all pairs is
not justified by this audit.

Six external controls failed before implementation; nine expanded controls then
passed (38 warnings, 7.39 s). They cover
clear/touching fixtures, resolved substep penetration, completion veto, checkpoint
identity, latch persistence and partial reset, with a two-world native clear
fixture. Focused framework validation passed 48 tests (35 warnings, 5.21 s);
Ruff and whitespace checks passed. Task-specific controls and the audit script
remain outside the framework test suite, following repository policy.

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
