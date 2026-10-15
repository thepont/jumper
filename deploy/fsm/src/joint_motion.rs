//! Feedback-gated joint segments. Geometry and actuator ownership stay with the host.

use std::time::Duration;

#[derive(Clone, Copy, PartialEq)]
enum Progress {
    Advance,
    Hold,
}

struct Feedback<'a> {
    current: &'a [f64],
    measured: &'a [f64],
    dt: f64,
}

/// Host scene identity and observation time, in the host's monotonic clock domain.
/// A revision changes whenever geometry relevant to the admitted plan changes.
#[derive(Clone, Copy)]
pub struct SceneStamp {
    pub revision: u64,
    pub observed_at: Duration,
}

/// Opt-in scene guard composed with the existing feedback/geometry authority.
pub struct SceneMotion {
    motion: JointMotion,
    revision: u64,
    max_age: Duration,
    last_now: Option<Duration>,
    last_observation: Option<Duration>,
}

impl SceneMotion {
    pub fn new(motion: JointMotion, revision: u64, max_age: Duration) -> Result<Self, String> {
        if max_age.is_zero() {
            return Err("scene freshness budget must be positive".into());
        }
        Ok(Self {
            motion,
            revision,
            max_age,
            last_now: None,
            last_observation: None,
        })
    }

    pub fn step(
        &mut self,
        current: &[f64],
        measured: &[f64],
        dt: f64,
        now: Duration,
        scene: SceneStamp,
        clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Vec<f64>, String> {
        self.propose(
            Feedback {
                current,
                measured,
                dt,
            },
            now,
            scene,
            Progress::Advance,
            clear,
        )
    }

    /// Check the existing target without advancing any waypoint. Time still runs.
    pub fn hold(
        &mut self,
        current: &[f64],
        measured: &[f64],
        dt: f64,
        now: Duration,
        scene: SceneStamp,
        clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Vec<f64>, String> {
        self.propose(
            Feedback {
                current,
                measured,
                dt,
            },
            now,
            scene,
            Progress::Hold,
            clear,
        )
    }

    fn propose(
        &mut self,
        feedback: Feedback<'_>,
        now: Duration,
        scene: SceneStamp,
        progress: Progress,
        mut clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Vec<f64>, String> {
        // Run inside JointMotion's guard so every scene failure uses its Stop
        // latch, including an already-arrived motion. Never fall back to a path
        // checked against another scene.
        self.motion.propose(feedback, progress, |candidate| {
            if scene.revision != self.revision
                || now
                    .checked_sub(scene.observed_at)
                    .is_none_or(|age| age > self.max_age)
                || self.last_now.is_some_and(|last| now < last)
                || self
                    .last_observation
                    .is_some_and(|last| scene.observed_at < last)
            {
                return Err(
                    "scene changed, expired or out of order; new checked plan required".into(),
                );
            }
            self.last_now = Some(now);
            self.last_observation = Some(scene.observed_at);
            clear(candidate)
        })
    }

    pub fn arrived(&self) -> bool {
        self.motion.arrived()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Moving,
    Arrived,
    Stopped,
}

pub struct JointMotion {
    points: Vec<Vec<f64>>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    rate: f64,
    remaining: f64,
    index: usize,
    state: State,
}

fn bounded(q: &[f64], lower: &[f64], upper: &[f64]) -> bool {
    q.len() == lower.len()
        && q.iter()
            .zip(lower)
            .zip(upper)
            .all(|((q, lo), hi)| q.is_finite() && q >= lo && q <= hi)
}

impl JointMotion {
    pub fn new(
        points: Vec<Vec<f64>>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        rate: f64,
        timeout: f64,
    ) -> Result<Self, String> {
        if lower.is_empty()
            || lower.len() > 32
            || upper.len() != lower.len()
            || lower.iter().zip(&upper).any(|(lo, hi)| {
                !lo.is_finite() || !hi.is_finite() || hi <= lo || !(hi - lo).is_finite()
            })
            || points.is_empty()
            || points.len() > 128
            || points.iter().any(|q| !bounded(q, &lower, &upper))
            || !rate.is_finite()
            || rate <= 0.
            || rate > std::f64::consts::PI
            || !timeout.is_finite()
            || timeout <= 0.
            || timeout > 60.
        {
            return Err("invalid bounded joint motion".into());
        }
        Ok(Self {
            points,
            lower,
            upper,
            rate,
            remaining: timeout,
            index: 0,
            state: State::Moving,
        })
    }
    /// Caller checks the entire measured-to-command corridor, support and sensing.
    /// A rejected or failed check latches Stop; resuming requires a new checked plan.
    pub fn step(
        &mut self,
        current: &[f64],
        measured: &[f64],
        dt: f64,
        clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Vec<f64>, String> {
        self.propose(
            Feedback {
                current,
                measured,
                dt,
            },
            Progress::Advance,
            clear,
        )
    }

    fn propose(
        &mut self,
        feedback: Feedback<'_>,
        progress: Progress,
        mut clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Vec<f64>, String> {
        let Feedback {
            current,
            measured,
            dt,
        } = feedback;
        if self.state == State::Stopped {
            return Err("motion stopped".into());
        }
        // Every error path stays stopped, including a host callback exception.
        self.state = State::Stopped;
        if !bounded(current, &self.lower, &self.upper)
            || !bounded(measured, &self.lower, &self.upper)
            || !dt.is_finite()
            || dt <= 0.
            || dt > 0.02
            || self.remaining + 1e-12 < dt
        {
            return Err("invalid feedback, timestep or expired motion".into());
        }
        let mut index = self.index;
        let reached = |target: &[f64]| {
            current
                .iter()
                .zip(measured)
                .zip(target)
                .all(|((c, m), t)| (c - t).abs() <= 1e-6 && (m - t).abs() <= 0.02)
        };
        if progress == Progress::Advance
            && reached(&self.points[index])
            && index + 1 < self.points.len()
        {
            index += 1;
        }
        let target: &[f64] = match progress {
            Progress::Advance => &self.points[index],
            Progress::Hold => current,
        };
        let arrived =
            progress == Progress::Advance && index + 1 == self.points.len() && reached(target);
        let extent = current
            .iter()
            .zip(target)
            .map(|(c, t)| (t - c).abs())
            .fold(0., f64::max);
        let fraction = if extent > 0. {
            (self.rate * dt / extent).min(1.)
        } else {
            0.
        };
        let next: Vec<f64> = current
            .iter()
            .zip(target)
            .map(|(c, t)| c + fraction * (t - c))
            .collect();
        if !bounded(&next, &self.lower, &self.upper) || !clear(&next)? {
            return Err("command corridor rejected".into());
        }
        self.index = index;
        self.remaining -= dt;
        self.state = if arrived {
            State::Arrived
        } else {
            State::Moving
        };
        Ok(next)
    }
    pub fn arrived(&self) -> bool {
        self.state == State::Arrived
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn motion(points: Vec<Vec<f64>>) -> JointMotion {
        JointMotion::new(points, vec![-2.; 2], vec![2.; 2], 0.45, 2.).unwrap()
    }
    fn scene_motion() -> SceneMotion {
        SceneMotion::new(motion(vec![vec![1., 0.5]]), 7, Duration::from_millis(250)).unwrap()
    }
    #[test]
    fn scene_hold_does_not_advance_a_reached_waypoint() {
        let mut m = SceneMotion::new(
            motion(vec![vec![0.1, 0.], vec![0.2, 0.2]]),
            7,
            Duration::from_millis(250),
        )
        .unwrap();
        let held = m
            .hold(
                &[0.1, 0.],
                &[0.1, 0.],
                0.02,
                Duration::from_millis(20),
                stamp(7, 0),
                |_| Ok(true),
            )
            .unwrap();
        assert_eq!(held, vec![0.1, 0.]);
        assert!(!m.arrived());
        let next = m
            .step(
                &[0.09, 0.],
                &[0.1, 0.],
                0.02,
                Duration::from_millis(40),
                stamp(7, 0),
                |_| Ok(true),
            )
            .unwrap();
        assert_eq!(next[1], 0.);
    }
    fn stamp(revision: u64, ms: u64) -> SceneStamp {
        SceneStamp {
            revision,
            observed_at: Duration::from_millis(ms),
        }
    }
    #[test]
    fn changed_scene_blocks_geometry_and_cannot_resume_an_old_plan() {
        let mut m = scene_motion();
        assert!(m
            .step(
                &[0., 0.],
                &[0., 0.],
                0.02,
                Duration::from_millis(100),
                stamp(8, 100),
                |_| panic!("obsolete path must not be queried")
            )
            .is_err());
        assert!(m
            .step(
                &[0., 0.],
                &[0., 0.],
                0.02,
                Duration::from_millis(120),
                stamp(7, 120),
                |_| panic!("old revision cannot resume")
            )
            .is_err());
        assert!(!m.arrived());
    }
    #[test]
    fn missing_freshness_or_future_observation_blocks_commands() {
        for observed in [749, 1001] {
            let mut m = scene_motion();
            assert!(m
                .step(
                    &[0., 0.],
                    &[0., 0.],
                    0.02,
                    Duration::from_millis(1000),
                    stamp(7, observed),
                    |_| panic!("invalid scene")
                )
                .is_err());
        }
    }
    #[test]
    fn reordered_observations_and_rewound_clocks_latch_stop() {
        for (now, observed) in [(120, 99), (99, 99)] {
            let mut m = scene_motion();
            m.step(
                &[0., 0.],
                &[0., 0.],
                0.02,
                Duration::from_millis(100),
                stamp(7, 100),
                |_| Ok(true),
            )
            .unwrap();
            assert!(m
                .step(
                    &[0., 0.],
                    &[0., 0.],
                    0.02,
                    Duration::from_millis(now),
                    stamp(7, observed),
                    |_| panic!("out of order")
                )
                .is_err());
        }
    }
    #[test]
    fn unchanged_fresh_scene_reuses_joint_motion_and_geometry_failure_latches() {
        let mut m = scene_motion();
        // A slower sensor may repeat a packet, including the exact age boundary.
        for now in [100, 350] {
            let q = m
                .step(
                    &[0., 0.],
                    &[0., 0.],
                    0.02,
                    Duration::from_millis(now),
                    stamp(7, 100),
                    |_| Ok(true),
                )
                .unwrap();
            assert!((q[0] - 0.009).abs() < 1e-12);
        }
        assert!(m
            .step(
                &[0., 0.],
                &[0., 0.],
                0.02,
                Duration::from_millis(360),
                stamp(7, 360),
                |_| Ok(false)
            )
            .is_err());
        assert!(m
            .step(
                &[0., 0.],
                &[0., 0.],
                0.02,
                Duration::from_millis(380),
                stamp(7, 380),
                |_| panic!("latched")
            )
            .is_err());
        assert!(SceneMotion::new(motion(vec![vec![0., 0.]]), 7, Duration::ZERO).is_err());
    }
    #[test]
    fn unequal_joints_follow_one_segment_not_independent_clipping() {
        let mut m = motion(vec![vec![1., 0.5]]);
        let q = m.step(&[0., 0.], &[0., 0.], 0.02, |_| Ok(true)).unwrap();
        assert!((q[0] - 0.009).abs() < 1e-12);
        assert!((q[1] - 0.0045).abs() < 1e-12);
        assert!(!m.arrived());
    }
    #[test]
    fn measured_arrival_does_not_cut_a_commanded_corner() {
        let mut m = motion(vec![vec![0.1, 0.], vec![0.1, 0.2]]);
        let q = m.step(&[0.09, 0.], &[0.1, 0.], 0.02, |_| Ok(true)).unwrap();
        assert_eq!(q[1], 0.);
        let q = m.step(&[0.1, 0.], &[0.1, 0.], 0.02, |_| Ok(true)).unwrap();
        assert!(q[1] > 0.);
        m.step(&[0.1, 0.2], &[0.1, 0.19], 0.02, |_| Ok(true))
            .unwrap();
        assert!(m.arrived());
    }
    #[test]
    fn rejected_geometry_or_callback_failure_latches_stop() {
        for verdict in [Ok(false), Err("unavailable".into())] {
            let mut m = motion(vec![vec![1., 0.5]]);
            assert!(m
                .step(&[0., 0.], &[0., 0.], 0.02, |_| verdict.clone())
                .is_err());
            assert!(m
                .step(&[0., 0.], &[0., 0.], 0.02, |_| panic!("must stay stopped"))
                .is_err());
            assert!(!m.arrived());
        }
    }
    #[test]
    fn invalid_feedback_and_deadline_cannot_emit_a_command() {
        let mut m = motion(vec![vec![1., 0.5]]);
        assert!(m
            .step(&[0., 0.], &[f64::NAN, 0.], 0.02, |_| panic!(
                "invalid input"
            ))
            .is_err());
        assert!(m
            .step(&[0., 0.], &[0., 0.], 0.02, |_| panic!("latched"))
            .is_err());
        let mut m = JointMotion::new(vec![vec![1.]], vec![-2.], vec![2.], 0.45, 0.02).unwrap();
        m.step(&[0.], &[0.], 0.02, |_| Ok(true)).unwrap();
        assert!(m.step(&[0.], &[0.], 0.02, |_| panic!("deadline")).is_err());
    }
    #[test]
    fn invalid_plan_never_constructs() {
        assert!(JointMotion::new(vec![], vec![], vec![], 0.45, 1.).is_err());
        assert!(JointMotion::new(vec![vec![3.]], vec![-2.], vec![2.], 0.45, 1.).is_err());
        assert!(JointMotion::new(vec![vec![0.]], vec![-2.], vec![2.], f64::NAN, 1.).is_err());
    }
}
