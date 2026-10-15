//! Feedback-gated joint segments. Geometry and actuator ownership stay with the host.

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
        mut clear: impl FnMut(&[f64]) -> Result<bool, String>,
    ) -> Result<Vec<f64>, String> {
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
        if reached(&self.points[index]) && index + 1 < self.points.len() {
            index += 1;
        }
        let target = &self.points[index];
        let arrived = index + 1 == self.points.len() && reached(target);
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
