//! Capture phase policy. Tracking and geometric/physical evidence are external ports.
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Tracking,
    Closing,
    Lifting,
    Verified,
    Stopped,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub approach: Duration,
    pub close: Duration,
    pub lift: Duration,
    pub recovery: Duration,
    pub loaded_samples: usize,
    pub loaded_duration: Duration,
}

pub struct Evidence {
    pub now: Duration,
    pub fresh: bool,
    pub plan: bool,
    pub aligned: bool,
    pub contact: bool,
    pub loaded: bool,
    pub pickup: bool,
    pub unsafe_scene: bool,
}

pub struct Decision {
    pub phase: Phase,
    pub advance_jaw: bool,
}

pub struct Capture {
    policy: Policy,
    phase: Phase,
    entered: Duration,
    last: Option<Duration>,
    lost: Option<Duration>,
    loaded_since: Option<Duration>,
    count: usize,
}

impl Capture {
    pub fn new(policy: Policy, start: Duration) -> Result<Self, String> {
        if [policy.approach, policy.close, policy.lift]
            .iter()
            .any(|t| t.is_zero() || *t > Duration::from_secs(60))
            || policy.recovery > policy.close
            || policy.recovery > Duration::from_millis(500)
            || policy.loaded_samples == 0
            || policy.loaded_samples > 1000
            || policy.loaded_duration > policy.close
            || start
                .checked_add(policy.approach)
                .and_then(|t| t.checked_add(policy.close))
                .and_then(|t| t.checked_add(policy.lift))
                .is_none()
        {
            return Err("invalid capture policy".into());
        }
        Ok(Self {
            policy,
            phase: Phase::Tracking,
            entered: start,
            last: None,
            lost: None,
            loaded_since: None,
            count: 0,
        })
    }
    pub fn advance(&mut self, e: Evidence) -> Result<Decision, String> {
        let phase = self.phase;
        self.phase = Phase::Stopped;
        if phase == Phase::Stopped
            || e.now < self.entered
            || self.last.is_some_and(|last| e.now <= last)
            || e.unsafe_scene
            || (e.loaded && !e.contact)
        {
            return Err("capture stopped: invalid evidence, clock or safety guard".into());
        }
        if self
            .last
            .is_some_and(|last| e.now - last > Duration::from_millis(40))
        {
            self.count = 0;
            self.loaded_since = None;
        }
        self.last = Some(e.now);
        let usable = e.fresh && e.plan && e.aligned;
        let next = match phase {
            Phase::Tracking => {
                if e.contact {
                    return Err("unexpected contact before admitted closure".into());
                }
                if e.now - self.entered >= self.policy.approach {
                    return Err("capture tracking deadline expired".into());
                }
                if usable {
                    Phase::Closing
                } else {
                    Phase::Tracking
                }
            }
            Phase::Closing => {
                if e.now - self.entered >= self.policy.close {
                    return Err("loaded capture deadline expired".into());
                }
                if !usable {
                    self.count = 0;
                    self.loaded_since = None;
                    let lost = *self.lost.get_or_insert(e.now);
                    if e.contact || e.now - lost > self.policy.recovery {
                        return Err("alignment or perception lost during contact/closure".into());
                    }
                    Phase::Closing
                } else {
                    self.lost = None;
                    if e.loaded {
                        self.count += 1;
                        let since = *self.loaded_since.get_or_insert(e.now);
                        if self.count >= self.policy.loaded_samples
                            && e.now - since >= self.policy.loaded_duration
                        {
                            Phase::Lifting
                        } else {
                            Phase::Closing
                        }
                    } else {
                        self.count = 0;
                        self.loaded_since = None;
                        Phase::Closing
                    }
                }
            }
            Phase::Lifting => {
                if !e.loaded || !e.fresh || !e.plan {
                    return Err("loaded grip or checked lift plan lost".into());
                }
                if e.now - self.entered >= self.policy.lift {
                    return Err("physical pickup deadline expired".into());
                }
                if e.pickup {
                    Phase::Verified
                } else {
                    Phase::Lifting
                }
            }
            Phase::Verified => {
                if !e.loaded || !e.fresh || !e.plan {
                    return Err("verified pickup lost its secure hold".into());
                }
                Phase::Verified
            }
            Phase::Stopped => unreachable!("stopped handled before transition"),
        };
        if next != phase {
            self.entered = e.now;
            self.count = 0;
            self.loaded_since = None;
            self.lost = None;
        }
        self.phase = next;
        Ok(Decision {
            phase: next,
            advance_jaw: next == Phase::Closing && usable && !e.loaded,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capture() -> Capture {
        Capture::new(
            Policy {
                approach: Duration::from_secs(15),
                close: Duration::from_secs(5),
                lift: Duration::from_secs(5),
                recovery: Duration::from_millis(500),
                loaded_samples: 11,
                loaded_duration: Duration::from_millis(200),
            },
            Duration::ZERO,
        )
        .unwrap()
    }
    fn evidence(ms: u64) -> Evidence {
        Evidence {
            now: Duration::from_millis(ms),
            fresh: true,
            plan: true,
            aligned: true,
            contact: false,
            loaded: false,
            pickup: false,
            unsafe_scene: false,
        }
    }
    #[test]
    fn moving_target_alignment_can_start_closing_without_stationarity_dwell() {
        let d = capture().advance(evidence(20)).unwrap();
        assert_eq!(d.phase, Phase::Closing);
        assert!(d.advance_jaw);
    }
    #[test]
    fn loaded_capture_then_physical_witness_is_required_for_verified_pickup() {
        let mut c = capture();
        c.advance(evidence(20)).unwrap();
        for i in 2..=12 {
            let mut e = evidence(i * 20);
            e.contact = true;
            e.loaded = true;
            let d = c.advance(e).unwrap();
            assert!(!d.advance_jaw);
            assert_eq!(
                d.phase,
                if i == 12 {
                    Phase::Lifting
                } else {
                    Phase::Closing
                }
            );
        }
        let mut e = evidence(260);
        e.contact = true;
        e.loaded = true;
        assert_eq!(c.advance(e).unwrap().phase, Phase::Lifting);
        let mut e = evidence(280);
        e.contact = true;
        e.loaded = true;
        e.pickup = true;
        assert_eq!(c.advance(e).unwrap().phase, Phase::Verified);
    }
    #[test]
    fn temporary_precontact_alignment_loss_pauses_jaw_then_resumes() {
        let mut c = capture();
        c.advance(evidence(20)).unwrap();
        let mut e = evidence(40);
        e.aligned = false;
        let d = c.advance(e).unwrap();
        assert_eq!(d.phase, Phase::Closing);
        assert!(!d.advance_jaw);
        assert!(c.advance(evidence(60)).unwrap().advance_jaw);
    }

    #[test]
    fn unexpected_contact_during_tracking_cannot_start_closure() {
        let mut e = evidence(20);
        e.contact = true;
        assert!(capture().advance(e).is_err());
    }

    #[test]
    fn early_pickup_flag_cannot_skip_capture_or_lift() {
        let mut e = evidence(20);
        e.pickup = true;
        assert_eq!(capture().advance(e).unwrap().phase, Phase::Closing);
    }

    #[test]
    fn unsafe_feedback_wins_over_aligned_pose_and_latches_stop() {
        let mut c = capture();
        let mut e = evidence(20);
        e.unsafe_scene = true;
        assert!(c.advance(e).is_err());
        assert!(c.advance(evidence(40)).is_err());
    }

    #[test]
    fn precontact_recovery_does_not_reset_the_closing_deadline() {
        let mut c = capture();
        c.advance(evidence(20)).unwrap();
        for ms in [
            100, 500, 900, 1300, 1700, 2100, 2500, 2900, 3300, 3700, 4100, 4500, 4900,
        ] {
            c.advance(evidence(ms)).unwrap();
        }
        assert!(c.advance(evidence(5020)).is_err());
    }

    #[test]
    fn missing_plan_during_approach_keeps_recovery_possible() {
        let mut c = capture();
        let mut e = evidence(20);
        e.plan = false;
        assert_eq!(c.advance(e).unwrap().phase, Phase::Tracking);
        assert_eq!(c.advance(evidence(40)).unwrap().phase, Phase::Closing);
    }

    #[test]
    fn contact_prohibits_blind_alignment_recovery() {
        let mut c = capture();
        c.advance(evidence(20)).unwrap();
        let mut e = evidence(40);
        e.contact = true;
        e.aligned = false;
        assert!(c.advance(e).is_err());
    }

    #[test]
    fn a_verified_pickup_does_not_hide_subsequent_loss_of_hold() {
        let mut c = capture();
        c.advance(evidence(20)).unwrap();
        for i in 2..=12 {
            let mut e = evidence(i * 20);
            e.contact = true;
            e.loaded = true;
            c.advance(e).unwrap();
        }
        let mut e = evidence(260);
        e.contact = true;
        e.loaded = true;
        e.pickup = true;
        assert_eq!(c.advance(e).unwrap().phase, Phase::Verified);
        assert!(c.advance(evidence(280)).is_err());
    }
}
