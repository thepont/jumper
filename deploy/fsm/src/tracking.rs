//! Receding-horizon supervision; planning and geometry remain host-owned Strategies.
//! Alignment is not contact, capture, a pickup witness, or hardware safety proof.

use crate::joint_motion::{JointMotion, SceneMotion, SceneStamp};
use crate::kinematics::ToolFrame;
use std::time::Duration;

#[derive(Clone)]
/// Desired tool contact reference and direction, WORLD frame, metres/unit axis.
/// Capture identity is immutable: repeated stamps must carry identical contents.
pub struct Observation {
    pub target: ToolFrame,
    pub scene: SceneStamp,
}

/// Requested and measured scalar-joint positions (radians or metres), and the
/// measured tool frame from the host's existing FK authority. Never commanded FK.
pub struct Feedback<'a> {
    pub current: &'a [f64],
    pub measured: &'a [f64],
    pub tool: ToolFrame,
    pub now: Duration,
    pub dt: f64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub rate: f64,
    pub timeout: Duration,
    pub freshness: Duration,
    pub position_tolerance: f64,
    pub axis_tolerance: f64,
    /// Optional dwell inside RELATIVE alignment limits; never target stationarity.
    pub settle: Duration,
}

/// None means temporarily infeasible: hold, then retry on a NEW observation.
/// Err means a broken planning boundary, not a recoverable IK rejection.
pub trait Planner {
    fn plan(
        &mut self,
        observation: &Observation,
        feedback: &Feedback<'_>,
        remaining: Duration,
    ) -> Result<Option<Vec<Vec<f64>>>, String>;
}

#[derive(Debug, PartialEq)]
pub enum Status {
    Following,
    Holding,
    /// Current relative pose is within limits. NOT permission to close: the
    /// grasp phase still owns closure-path checks, timing and contact witnesses.
    Aligned,
}

pub struct Command {
    pub joints: Vec<f64>,
    pub status: Status,
}

pub struct Tracking<P> {
    policy: Policy,
    planner: P,
    deadline: Duration,
    remaining: Duration,
    last_now: Option<Duration>,
    last_observation: Option<Observation>,
    motion: Option<SceneMotion>,
    aligned_since: Option<Duration>,
    stopped: bool,
}

impl<P: Planner> Tracking<P> {
    pub fn new(policy: Policy, planner: P, start: Duration) -> Result<Self, String> {
        JointMotion::new(
            vec![policy.lower.clone()],
            policy.lower.clone(),
            policy.upper.clone(),
            policy.rate,
            policy.timeout.as_secs_f64(),
        )?;
        if policy.freshness.is_zero()
            || !policy.position_tolerance.is_finite()
            || policy.position_tolerance <= 0.
            || !policy.axis_tolerance.is_finite()
            || policy.axis_tolerance <= 0.
            || policy.axis_tolerance > std::f64::consts::PI
            || policy.settle > policy.timeout
        {
            return Err("invalid tracking policy".into());
        }
        let deadline = start
            .checked_add(policy.timeout)
            .ok_or("deadline overflow")?;
        let remaining = policy.timeout;
        Ok(Self {
            policy,
            planner,
            deadline,
            remaining,
            last_now: None,
            last_observation: None,
            motion: None,
            aligned_since: None,
            stopped: false,
        })
    }

    pub fn step(
        &mut self,
        feedback: Feedback<'_>,
        observation: Option<Observation>,
        mut clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Command, String> {
        if self.stopped {
            return Err("tracking stopped".into());
        }
        // Every error remains latched, including a planner/geometry exception.
        self.stopped = true;
        self.check_feedback(&feedback)?;
        let available = self
            .deadline
            .checked_sub(feedback.now)
            .ok_or("tracking deadline expired")?
            .min(self.remaining);
        if available.is_zero() || available.as_secs_f64() + 1e-12 < feedback.dt {
            return Err("tracking deadline expired".into());
        }
        if self
            .last_now
            .is_some_and(|last| feedback.now - last > Duration::from_secs_f64(feedback.dt + 1e-9))
        {
            self.aligned_since = None;
        }
        self.last_now = Some(feedback.now);
        let fresh =
            match observation {
                Some(observation) => {
                    check_frame(&observation.target)?;
                    let age = feedback
                        .now
                        .checked_sub(observation.scene.observed_at)
                        .ok_or("future target observation")?;
                    if let Some(last) = &self.last_observation {
                        if observation.scene.observed_at == last.scene.observed_at
                            && (observation.scene.revision != last.scene.revision
                                || observation.target.position != last.target.position
                                || observation.target.axis != last.target.axis)
                        {
                            return Err("observation identity reused with changed contents".into());
                        }
                    }
                    if age > self.policy.freshness
                        || self.last_observation.as_ref().is_some_and(|last| {
                            observation.scene.observed_at < last.scene.observed_at
                        })
                    {
                        None
                    } else {
                        if self.last_observation.as_ref().is_none_or(|last| {
                            observation.scene.observed_at > last.scene.observed_at
                        }) {
                            // Discard the old route BEFORE trying a new target/scene.
                            self.motion = None;
                            if let Some(points) =
                                self.planner.plan(&observation, &feedback, available)?
                            {
                                let motion = JointMotion::new(
                                    points,
                                    self.policy.lower.clone(),
                                    self.policy.upper.clone(),
                                    self.policy.rate,
                                    available.as_secs_f64(),
                                )?;
                                self.motion = Some(SceneMotion::new(
                                    motion,
                                    observation.scene.revision,
                                    self.policy.freshness,
                                )?);
                            }
                            self.last_observation = Some(observation.clone());
                        }
                        Some(observation)
                    }
                }
                None => None,
            };
        let command = match (self.motion.as_mut(), fresh) {
            (Some(motion), Some(observation)) => {
                let joints = motion.step(
                    feedback.current,
                    feedback.measured,
                    feedback.dt,
                    feedback.now,
                    observation.scene,
                    &mut clear,
                )?;
                let error = observation
                    .target
                    .position
                    .iter()
                    .zip(feedback.tool.position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let angle = observation
                    .target
                    .axis
                    .iter()
                    .zip(feedback.tool.axis)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    .clamp(-1., 1.)
                    .acos();
                let aligned =
                    error <= self.policy.position_tolerance && angle <= self.policy.axis_tolerance;
                let ready = if aligned {
                    feedback.now - *self.aligned_since.get_or_insert(feedback.now)
                        >= self.policy.settle
                } else {
                    self.aligned_since = None;
                    false
                };
                Command {
                    joints,
                    status: if ready {
                        Status::Aligned
                    } else {
                        Status::Following
                    },
                }
            }
            _ => {
                self.motion = None;
                self.aligned_since = None;
                // Holding is not a cached safe pose. Check the CURRENT measured-to-held
                // corridor even without a target; the geometry Strategy owns its freshness.
                let mut hold = JointMotion::new(
                    vec![feedback.current.to_vec()],
                    self.policy.lower.clone(),
                    self.policy.upper.clone(),
                    self.policy.rate,
                    available.as_secs_f64(),
                )?;
                Command {
                    joints: hold.step(
                        feedback.current,
                        feedback.measured,
                        feedback.dt,
                        &mut clear,
                    )?,
                    status: Status::Holding,
                }
            }
        };
        self.remaining = available.saturating_sub(Duration::from_secs_f64(feedback.dt));
        self.stopped = false;
        Ok(command)
    }

    fn check_feedback(&self, feedback: &Feedback<'_>) -> Result<(), String> {
        let bounded = |values: &[f64]| {
            values.len() == self.policy.lower.len()
                && values
                    .iter()
                    .zip(&self.policy.lower)
                    .zip(&self.policy.upper)
                    .all(|((q, lo), hi)| q.is_finite() && q >= lo && q <= hi)
        };
        if !bounded(feedback.current)
            || !bounded(feedback.measured)
            || !feedback.dt.is_finite()
            || feedback.dt <= 0.
            || feedback.dt > 0.02
            || feedback.now < self.deadline - self.policy.timeout
            || self.last_now.is_some_and(|last| feedback.now <= last)
        {
            return Err("invalid tracking feedback or clock".into());
        }
        check_frame(&feedback.tool)
    }
}

fn check_frame(frame: &ToolFrame) -> Result<(), String> {
    if frame
        .position
        .iter()
        .chain(&frame.axis)
        .any(|x| !x.is_finite())
        || (frame.axis.iter().map(|x| x * x).sum::<f64>() - 1.).abs() > 1e-8
    {
        Err("invalid world-frame target or measured tool".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LinearPlan;
    impl Planner for LinearPlan {
        fn plan(
            &mut self,
            observation: &Observation,
            _: &Feedback<'_>,
            _: Duration,
        ) -> Result<Option<Vec<Vec<f64>>>, String> {
            let q = observation.target.position[0] / 0.1;
            Ok((q.abs() <= 1.).then(|| vec![vec![q]]))
        }
    }
    fn frame(q: f64) -> ToolFrame {
        ToolFrame {
            position: [q * 0.1, 0., 0.],
            axis: [1., 0., 0.],
        }
    }
    fn tracker(timeout: u64) -> Tracking<LinearPlan> {
        Tracking::new(
            Policy {
                lower: vec![-1.],
                upper: vec![1.],
                rate: 0.45,
                timeout: Duration::from_millis(timeout),
                freshness: Duration::from_millis(250),
                position_tolerance: 0.005,
                axis_tolerance: 10_f64.to_radians(),
                settle: Duration::from_millis(180),
            },
            LinearPlan,
            Duration::ZERO,
        )
        .unwrap()
    }
    fn tick(t: &mut Tracking<LinearPlan>, ms: u64, q: f64, target: Option<f64>) -> Command {
        t.step(
            Feedback {
                current: &[q],
                measured: &[q],
                tool: frame(q),
                now: Duration::from_millis(ms),
                dt: 0.02,
            },
            target.map(|q| Observation {
                target: frame(q),
                scene: SceneStamp {
                    revision: 0,
                    observed_at: Duration::from_millis(ms),
                },
            }),
            |_| Ok(true),
        )
        .unwrap()
    }

    #[test]
    fn given_a_moving_reachable_target_when_tracking_then_intercept_without_waiting_for_it_to_stop()
    {
        let mut t = tracker(4000);
        let mut q = 0.;
        let mut aligned_while_moving = false;
        for i in 1..100 {
            let target = 0.2 + i as f64 * 0.002;
            let command = tick(&mut t, i * 20, q, Some(target));
            assert!((command.joints[0] - q).abs() <= 0.009 + 1e-12);
            if command.status == Status::Aligned {
                assert!((q - target).abs() * 0.1 <= 0.005);
                aligned_while_moving = true;
            }
            q = command.joints[0]; // Deterministic ideal plant, not physical capture.
        }
        assert!(
            aligned_while_moving,
            "must follow and align, not just hold safely"
        );
        assert!(q > 0.35);
    }

    #[test]
    fn given_an_infeasible_first_plan_when_target_returns_then_reacquire_in_the_same_session() {
        let mut t = tracker(1000);
        assert_eq!(tick(&mut t, 20, 0., Some(2.)).status, Status::Holding);
        let recovered = tick(&mut t, 40, 0., Some(0.3));
        assert_eq!(recovered.status, Status::Following);
        assert!(recovered.joints[0] > 0.);
    }

    #[test]
    fn given_arrival_when_target_moves_away_then_alignment_is_revoked_and_motion_resumes() {
        let mut t = tracker(2000);
        for i in 1..10 {
            assert_ne!(tick(&mut t, i * 20, 0., Some(0.)).status, Status::Aligned);
        }
        assert_eq!(tick(&mut t, 200, 0., Some(0.)).status, Status::Aligned);
        let moved = tick(&mut t, 220, 0., Some(-0.3));
        assert_eq!(moved.status, Status::Following);
        assert!(moved.joints[0] < 0.);
    }

    #[test]
    fn no_target_stationarity_or_mandatory_dwell_is_required() {
        let mut t = tracker(1000);
        t.policy.settle = Duration::ZERO;
        assert_eq!(tick(&mut t, 20, 0., Some(0.01)).status, Status::Aligned);
        assert_eq!(tick(&mut t, 40, 0.01, Some(0.02)).status, Status::Aligned);
    }

    #[test]
    fn lost_target_holds_then_fresh_observation_resumes_tracking() {
        let mut t = tracker(1000);
        assert!(tick(&mut t, 20, 0., Some(0.3)).joints[0] > 0.);
        let held = tick(&mut t, 40, 0.009, None);
        assert_eq!(held.status, Status::Holding);
        assert_eq!(held.joints, vec![0.009]);
        assert!(tick(&mut t, 60, 0.009, Some(0.4)).joints[0] > 0.009);
    }

    #[test]
    fn retries_never_reset_the_absolute_deadline() {
        let mut t = tracker(400);
        for i in 1..20 {
            tick(&mut t, i * 20, 0., Some(2.));
        }
        assert!(t
            .step(
                Feedback {
                    current: &[0.],
                    measured: &[0.],
                    tool: frame(0.),
                    now: Duration::from_millis(400),
                    dt: 0.02
                },
                None,
                |_| panic!("expired session must not propose commands")
            )
            .is_err());
        assert!(t
            .step(
                Feedback {
                    current: &[0.],
                    measured: &[0.],
                    tool: frame(0.),
                    now: Duration::from_millis(420),
                    dt: 0.02
                },
                None,
                |_| panic!("deadline remains latched")
            )
            .is_err());
    }

    #[test]
    fn an_obstruction_during_hold_is_not_assumed_safe() {
        let mut t = tracker(1000);
        assert!(t
            .step(
                Feedback {
                    current: &[0.],
                    measured: &[0.],
                    tool: frame(0.),
                    now: Duration::from_millis(20),
                    dt: 0.02
                },
                None,
                |_| Ok(false)
            )
            .is_err());
        assert!(t
            .step(
                Feedback {
                    current: &[0.],
                    measured: &[0.],
                    tool: frame(0.),
                    now: Duration::from_millis(40),
                    dt: 0.02
                },
                None,
                |_| panic!("unsafe session cannot resume")
            )
            .is_err());
    }

    #[test]
    fn stale_or_reordered_observations_hold_without_following_the_old_route() {
        for observed in [0, 99] {
            let mut t = tracker(2000);
            tick(&mut t, 100, 0., Some(0.4));
            let command = t
                .step(
                    Feedback {
                        current: &[0.009],
                        measured: &[0.009],
                        tool: frame(0.009),
                        now: Duration::from_millis(400),
                        dt: 0.02,
                    },
                    Some(Observation {
                        target: frame(-0.4),
                        scene: SceneStamp {
                            revision: 0,
                            observed_at: Duration::from_millis(observed),
                        },
                    }),
                    |_| Ok(true),
                )
                .unwrap();
            assert_eq!(command.status, Status::Holding);
            assert_eq!(command.joints, vec![0.009]);
            assert!(tick(&mut t, 420, 0.009, Some(-0.4)).joints[0] < 0.009);
        }
    }

    #[test]
    fn a_fresh_scene_revision_replaces_the_plan_instead_of_abandoning_the_session() {
        let mut t = tracker(1000);
        tick(&mut t, 20, 0., Some(0.4));
        let command = t
            .step(
                Feedback {
                    current: &[0.009],
                    measured: &[0.009],
                    tool: frame(0.009),
                    now: Duration::from_millis(40),
                    dt: 0.02,
                },
                Some(Observation {
                    target: frame(-0.4),
                    scene: SceneStamp {
                        revision: 1,
                        observed_at: Duration::from_millis(40),
                    },
                }),
                |_| Ok(true),
            )
            .unwrap();
        assert!(command.joints[0] < 0.009);
    }

    #[test]
    fn arrival_does_not_mean_alignment_and_wrong_axis_cannot_authorise_closure() {
        let mut t = tracker(1000);
        for i in 1..20 {
            let mut tool = frame(0.);
            tool.axis = [0., 1., 0.];
            let command = t
                .step(
                    Feedback {
                        current: &[0.],
                        measured: &[0.],
                        tool,
                        now: Duration::from_millis(i * 20),
                        dt: 0.02,
                    },
                    Some(Observation {
                        target: frame(0.),
                        scene: SceneStamp {
                            revision: 0,
                            observed_at: Duration::from_millis(i * 20),
                        },
                    }),
                    |_| Ok(true),
                )
                .unwrap();
            assert_eq!(command.status, Status::Following);
        }
    }

    #[test]
    fn invalid_feedback_latches_before_invoking_geometry() {
        for (q, dt) in [(f64::NAN, 0.02), (1.01, 0.02), (0., 0.), (0., 0.021)] {
            let mut t = tracker(1000);
            assert!(t
                .step(
                    Feedback {
                        current: &[q],
                        measured: &[0.],
                        tool: frame(0.),
                        now: Duration::from_millis(20),
                        dt
                    },
                    None,
                    |_| panic!("malformed feedback")
                )
                .is_err());
            assert!(t
                .step(
                    Feedback {
                        current: &[0.],
                        measured: &[0.],
                        tool: frame(0.),
                        now: Duration::from_millis(40),
                        dt: 0.02
                    },
                    None,
                    |_| panic!("malformed input cannot resume")
                )
                .is_err());
        }
    }

    #[test]
    fn new_infeasible_target_cannot_continue_the_old_route() {
        let mut t = tracker(1000);
        tick(&mut t, 20, 0., Some(0.5));
        let command = tick(&mut t, 40, 0.009, Some(2.));
        assert_eq!(command.status, Status::Holding);
        assert_eq!(command.joints, vec![0.009]);
        assert!(tick(&mut t, 60, 0.009, Some(-0.4)).joints[0] < 0.009);
    }

    #[test]
    fn unsampled_time_is_not_evidence_of_relative_alignment() {
        let mut t = tracker(1000);
        assert_eq!(tick(&mut t, 20, 0., Some(0.)).status, Status::Following);
        assert_eq!(tick(&mut t, 300, 0., Some(0.)).status, Status::Following);
    }

    #[test]
    fn repeated_infeasible_packets_do_not_busy_retry_the_planner() {
        struct Counter(usize);
        impl Planner for Counter {
            fn plan(
                &mut self,
                _: &Observation,
                _: &Feedback<'_>,
                _: Duration,
            ) -> Result<Option<Vec<Vec<f64>>>, String> {
                self.0 += 1;
                Ok(None)
            }
        }
        let mut t = Tracking::new(tracker(1000).policy, Counter(0), Duration::ZERO).unwrap();
        for ms in [20, 40, 60] {
            let command = t
                .step(
                    Feedback {
                        current: &[0.],
                        measured: &[0.],
                        tool: frame(0.),
                        now: Duration::from_millis(ms),
                        dt: 0.02,
                    },
                    Some(Observation {
                        target: frame(0.3),
                        scene: SceneStamp {
                            revision: 0,
                            observed_at: Duration::from_millis(20),
                        },
                    }),
                    |_| Ok(true),
                )
                .unwrap();
            assert_eq!(command.status, Status::Holding);
        }
        assert_eq!(
            t.planner.0, 1,
            "retry requires new evidence, not a reset loop"
        );
    }

    #[test]
    fn malformed_and_future_observations_do_not_reach_geometry() {
        for case in 0..3 {
            let mut t = tracker(1000);
            let mut target = frame(0.2);
            if case == 0 {
                target.position[0] = f64::NAN;
            }
            if case == 1 {
                target.axis = [0.; 3];
            }
            assert!(t
                .step(
                    Feedback {
                        current: &[0.],
                        measured: &[0.],
                        tool: frame(0.),
                        now: Duration::from_millis(20),
                        dt: 0.02
                    },
                    Some(Observation {
                        target,
                        scene: SceneStamp {
                            revision: 0,
                            observed_at: Duration::from_millis(if case == 2 { 21 } else { 20 })
                        }
                    }),
                    |_| panic!("malformed observation")
                )
                .is_err());
        }
    }

    #[test]
    fn a_reused_capture_identity_cannot_silently_change_its_target() {
        let mut t = tracker(1000);
        tick(&mut t, 20, 0., Some(0.3));
        assert!(t
            .step(
                Feedback {
                    current: &[0.009],
                    measured: &[0.009],
                    tool: frame(0.009),
                    now: Duration::from_millis(40),
                    dt: 0.02
                },
                Some(Observation {
                    target: frame(-0.4),
                    scene: SceneStamp {
                        revision: 0,
                        observed_at: Duration::from_millis(20)
                    }
                }),
                |_| panic!("mutated capture")
            )
            .is_err());
    }

    #[test]
    fn planner_failure_is_distinct_from_an_infeasible_plan() {
        struct Broken;
        impl Planner for Broken {
            fn plan(
                &mut self,
                _: &Observation,
                _: &Feedback<'_>,
                _: Duration,
            ) -> Result<Option<Vec<Vec<f64>>>, String> {
                Err("planner unavailable".into())
            }
        }
        let mut t = Tracking::new(tracker(1000).policy, Broken, Duration::ZERO).unwrap();
        assert!(t
            .step(
                Feedback {
                    current: &[0.],
                    measured: &[0.],
                    tool: frame(0.),
                    now: Duration::from_millis(20),
                    dt: 0.02
                },
                Some(Observation {
                    target: frame(0.3),
                    scene: SceneStamp {
                        revision: 0,
                        observed_at: Duration::from_millis(20)
                    }
                }),
                |_| panic!("planner failure cannot fall back to unchecked motion")
            )
            .is_err());
    }
}
