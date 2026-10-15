//! Non-actuating serial hinge-chain queries. No planner or motor authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// T_parent_body: p_parent = translation + R(wxyz) * p_body, metres.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pose {
    pub translation: [f64; 3],
    pub rotation_wxyz: [f64; 4],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Hinge {
    pub name: String,
    pub origin: Pose,
    pub anchor: [f64; 3],
    pub axis: [f64; 3],
    pub reference_rad: f64,
    pub limits_rad: [f64; 2],
}

/// Ordered parent-to-child hinges; tool point/direction are in final body frame.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArmSpec {
    pub joints: Vec<Hinge>,
    pub tool_point: [f64; 3],
    pub tool_axis: [f64; 3],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFrame {
    pub position: [f64; 3],
    pub axis: [f64; 3],
}

/// Adapter to OpenRR's `k`; owns validated geometry, never actuator state.
pub struct SerialArm {
    spec: ArmSpec,
}

fn finite(values: &[f64]) -> Result<(), String> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err("non-finite kinematics input".into())
    }
}

fn unit(values: &[f64]) -> Result<(), String> {
    finite(values)?;
    if (values.iter().map(|v| v * v).sum::<f64>() - 1.).abs() <= 1e-8 {
        Ok(())
    } else {
        Err("expected a unit direction or quaternion".into())
    }
}

fn pose(value: &Pose) -> Result<k::Isometry3<f64>, String> {
    finite(&value.translation)?;
    unit(&value.rotation_wxyz)?;
    let [w, x, y, z] = value.rotation_wxyz;
    Ok(k::Isometry3::from_parts(
        k::Translation3::from(value.translation),
        k::UnitQuaternion::new_normalize(k::nalgebra::Quaternion::new(w, x, y, z)),
    ))
}

impl SerialArm {
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.len() > 1_048_576 {
            return Err("arm descriptor too large".into());
        }
        let spec: ArmSpec = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if spec.joints.is_empty() || spec.joints.len() > 64 {
            return Err("expected 1..64 serial hinges".into());
        }
        finite(&spec.tool_point)?;
        unit(&spec.tool_axis)?;
        let mut names = std::collections::BTreeSet::new();
        for joint in &spec.joints {
            if joint.name.trim().is_empty() || joint.name.len() > 128 || !names.insert(&joint.name)
            {
                return Err("invalid or duplicate joint name".into());
            }
            pose(&joint.origin)?;
            finite(&joint.anchor)?;
            unit(&joint.axis)?;
            finite(&joint.limits_rad)?;
            finite(&[joint.reference_rad])?;
            let [lo, hi] = joint.limits_rad;
            if lo >= hi
                || !(lo - joint.reference_rad).is_finite()
                || !(hi - joint.reference_rad).is_finite()
            {
                return Err("invalid joint limits or reference".into());
            }
        }
        Ok(Self { spec })
    }

    pub fn forward(
        &self,
        base: &Pose,
        positions: &BTreeMap<String, f64>,
    ) -> Result<ToolFrame, String> {
        let base = pose(base)?;
        if positions.len() != self.spec.joints.len() {
            return Err("joint names must match the descriptor exactly".into());
        }
        let angles = self
            .spec
            .joints
            .iter()
            .map(|joint| {
                let q = *positions.get(&joint.name).ok_or("missing joint position")?;
                if !q.is_finite() || q < joint.limits_rad[0] || q > joint.limits_rad[1] {
                    return Err("joint position outside finite limits");
                }
                Ok(q - joint.reference_rad)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Query-local library state prevents stale caches and concurrent-query mixing.
        // T_body = T_origin * T_anchor * R(q-reference) * T_-anchor.
        let root = k::NodeBuilder::new().origin(base).into_node();
        let mut end = root.clone();
        for (joint, angle) in self.spec.joints.iter().zip(angles) {
            let anchor = k::Vector3::from(joint.anchor);
            let hinge = k::NodeBuilder::new()
                .origin(
                    pose(&joint.origin)? * k::Isometry3::translation(anchor.x, anchor.y, anchor.z),
                )
                .joint_type(k::JointType::Rotational {
                    axis: k::nalgebra::Unit::new_normalize(k::Vector3::from(joint.axis)),
                })
                .into_node();
            hinge.set_parent(&end);
            hinge.set_joint_position(angle).map_err(|e| e.to_string())?;
            let body = k::NodeBuilder::new()
                .origin(k::Isometry3::translation(-anchor.x, -anchor.y, -anchor.z))
                .into_node();
            body.set_parent(&hinge);
            end = body;
        }
        let chain = k::Chain::from_root(root);
        chain.update_transforms();
        let transform = end.world_transform().ok_or("missing tool transform")?;
        let position = transform.transform_point(&k::nalgebra::Point3::from(self.spec.tool_point));
        let axis = transform.rotation * k::Vector3::from(self.spec.tool_axis);
        finite(position.coords.as_slice())?;
        finite(axis.as_slice())?;
        Ok(ToolFrame {
            position: position.coords.into(),
            axis: axis.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> Pose {
        Pose {
            translation: [0.; 3],
            rotation_wxyz: [1., 0., 0., 0.],
        }
    }
    fn spec() -> ArmSpec {
        ArmSpec {
            joints: vec![Hinge {
                name: "elbow".into(),
                origin: identity(),
                anchor: [0.; 3],
                axis: [0., 0., 1.],
                reference_rad: 0.,
                limits_rad: [-2., 2.],
            }],
            tool_point: [1., 0., 0.],
            tool_axis: [1., 0., 0.],
        }
    }
    fn arm(spec: &ArmSpec) -> SerialArm {
        SerialArm::from_json(&serde_json::to_string(spec).unwrap()).unwrap()
    }
    fn close(a: [f64; 3], b: [f64; 3]) {
        for (a, b) in a.into_iter().zip(b) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }
    #[test]
    fn given_a_hinge_when_rotated_then_tool_point_and_direction_follow() {
        let result = arm(&spec())
            .forward(
                &identity(),
                &BTreeMap::from([("elbow".into(), std::f64::consts::FRAC_PI_2)]),
            )
            .unwrap();
        close(result.position, [0., 1., 0.]);
        close(result.axis, [0., 1., 0.]);
    }
    #[test]
    fn anchor_reference_and_world_transform_are_not_discarded() {
        let mut s = spec();
        s.joints[0].anchor = [0.5, 0., 0.];
        s.joints[0].reference_rad = 0.2;
        let base = Pose {
            translation: [2., 3., 4.],
            rotation_wxyz: [0., 0., 0., 1.],
        };
        let result = arm(&s)
            .forward(
                &base,
                &BTreeMap::from([("elbow".into(), 0.2 + std::f64::consts::FRAC_PI_2)]),
            )
            .unwrap();
        close(result.position, [1.5, 2.5, 4.]);
        close(result.axis, [0., -1., 0.]);
    }
    #[test]
    fn invalid_input_does_not_poison_a_later_valid_query() {
        let a = arm(&spec());
        for q in [f64::NAN, f64::INFINITY, 2.01] {
            assert!(a
                .forward(&identity(), &BTreeMap::from([("elbow".into(), q)]))
                .is_err());
        }
        assert!(a.forward(&identity(), &BTreeMap::new()).is_err());
        assert!(a
            .forward(&identity(), &BTreeMap::from([("other".into(), 0.)]))
            .is_err());
        close(
            a.forward(&identity(), &BTreeMap::from([("elbow".into(), 0.)]))
                .unwrap()
                .position,
            [1., 0., 0.],
        );
    }
    #[test]
    fn malformed_model_is_rejected_before_library_construction() {
        for mutate in [0, 1, 2, 3, 4, 5] {
            let mut s = spec();
            match mutate {
                0 => s.joints[0].axis = [0.; 3],
                1 => s.joints[0].origin.rotation_wxyz = [2., 0., 0., 0.],
                2 => s.joints[0].limits_rad = [2., -2.],
                3 => s.joints.push(s.joints[0].clone()),
                4 => s.joints.clear(),
                _ => s.tool_axis = [0.; 3],
            }
            assert!(SerialArm::from_json(&serde_json::to_string(&s).unwrap()).is_err());
        }
        assert!(SerialArm::from_json("{}").is_err());
    }

    #[test]
    fn serial_order_is_not_alphabetical_and_origins_rotate_with_the_parent() {
        let mut s = spec();
        s.joints[0].name = "z_shoulder".into();
        let mut second = s.joints[0].clone();
        second.name = "a_wrist".into();
        second.origin.translation = [2., 0., 0.];
        second.origin.rotation_wxyz = [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0., 0.];
        s.joints.push(second);
        let result = arm(&s)
            .forward(
                &identity(),
                &BTreeMap::from([
                    ("a_wrist".into(), std::f64::consts::FRAC_PI_2),
                    ("z_shoulder".into(), std::f64::consts::FRAC_PI_2),
                ]),
            )
            .unwrap();
        close(result.position, [0., 2., 1.]);
        close(result.axis, [0., 0., 1.]);
    }

    #[test]
    fn exact_limits_are_valid_but_bad_base_and_extra_joints_are_not() {
        let a = arm(&spec());
        for q in [-2., 2.] {
            let result = a
                .forward(&identity(), &BTreeMap::from([("elbow".into(), q)]))
                .unwrap();
            close(result.position, [q.cos(), q.sin(), 0.]);
        }
        let joints = BTreeMap::from([("elbow".into(), 0.)]);
        let mut base = identity();
        base.translation[0] = f64::NAN;
        assert!(a.forward(&base, &joints).is_err());
        base = identity();
        base.rotation_wxyz = [0.; 4];
        assert!(a.forward(&base, &joints).is_err());
        assert!(a
            .forward(
                &identity(),
                &BTreeMap::from([("elbow".into(), 0.), ("extra".into(), 0.)])
            )
            .is_err());
        close(
            a.forward(&identity(), &joints).unwrap().position,
            [1., 0., 0.],
        );
    }

    #[test]
    fn descriptor_rejects_unknown_fields_and_resource_exhaustion() {
        let mut value = serde_json::to_value(spec()).unwrap();
        value["units"] = serde_json::json!("millimetres");
        assert!(SerialArm::from_json(&value.to_string()).is_err());
        assert!(SerialArm::from_json(&" ".repeat(1_048_577)).is_err());
        let mut s = spec();
        s.joints = (0..65)
            .map(|i| {
                let mut j = spec().joints.remove(0);
                j.name = format!("joint_{i}");
                j
            })
            .collect();
        assert!(SerialArm::from_json(&serde_json::to_string(&s).unwrap()).is_err());
    }
}
