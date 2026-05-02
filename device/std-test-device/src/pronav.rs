use std::f32::consts::PI;

use common::{
    nalgebra::{Matrix3, Point2, Rotation3, UnitQuaternion, UnitVector3, Vector3},
    tasks::eskf::EskfEstimate,
};

pub struct CameraModel {
    pub width: f32,
    pub height: f32,
    pub f_x: f32,
    pub f_y: f32,
    pub c_x: f32,
    pub c_y: f32,
    pub pitch_offset: f32,
}

impl CameraModel {
    pub fn new(width: f32, height: f32, vfov_rad: f32, pitch_offset: f32) -> Self {
        let f_y = height / (2.0 * (vfov_rad / 2.0).tan());
        let f_x = f_y; // Assume square pixels

        Self {
            width,
            height,
            f_x,
            f_y,
            c_x: width / 2.0,
            c_y: height / 2.0,
            pitch_offset,
        }
    }

    /// Emulates the camera sensor, returning Some(u,v) pixel coordinate if the target is in view.
    pub fn project_to_pixel(
        &self,
        los_global: Vector3<f32>,
        att: UnitQuaternion<f32>,
    ) -> Option<(f32, f32)> {
        let los_body = att.inverse_transform_vector(&los_global);

        // Apply camera pitch (pitching up is positive rotation around body Y)
        let pitch_quat = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.pitch_offset);
        let los_cam_aligned = pitch_quat.inverse_transform_vector(&los_body);

        // Map Body coordinates (Forward-Right-Down) to standard Camera Image coordinates (Right-Down-Forward)
        // Body X (Forward) -> Cam Z (Forward)
        // Body Y (Right) -> Cam X (Right)
        // Body Z (Down) -> Cam Y (Down)
        let z_c = los_cam_aligned.x;
        let x_c = los_cam_aligned.y;
        let y_c = los_cam_aligned.z;

        // If target is behind the camera plane, it cannot be seen!
        if z_c <= 0.0 {
            return None;
        }

        // Pinhole projection
        let u = self.f_x * (x_c / z_c) + self.c_x;
        let v = self.f_y * (y_c / z_c) + self.c_y;

        // Check if the target is actually inside the field of view bounds
        if u < 0.0 || u > self.width || v < 0.0 || v > self.height {
            return None;
        }

        Some((u, v))
    }

    /// Converts a pixel measurement (if available) back to a global LOS vector
    pub fn pixel_to_global(&self, pixel: Point2<f32>, att: UnitQuaternion<f32>) -> Vector3<f32> {
        let [[u, v]] = pixel.coords.data.0;

        // Map pixel back to a 3D ray in the camera frame
        let x_c = (u - self.c_x) / self.f_x;
        let y_c = (v - self.c_y) / self.f_y;
        let z_c = 1.0;

        // Map Camera coordinates (Right-Down-Forward) back to Body coordinates (Forward-Right-Down)
        let los_cam_aligned = Vector3::new(z_c, x_c, y_c).normalize();

        // Apply inverse of camera pitch - must match the pitch used in project_to_pixel!
        let pitch_rot = Rotation3::from_euler_angles(0.0, self.pitch_offset, 0.0);

        let los_body = pitch_rot.matrix() * los_cam_aligned;

        // Body to Global
        att.transform_vector(&los_body)
    }
}

pub struct AlphaBetaLos {
    alpha: f32, // Gain for position (0.0 to 1.0)
    beta: f32,  // Gain for velocity (0.0 to 1.0)
    unit_est: Option<Vector3<f32>>,
    rate_est: Vector3<f32>,
}

impl AlphaBetaLos {
    /// Construct a new alpha-beta filter.
    ///
    /// Larger values of `alpha` will make the filter more responsive, but also potentially noisier.
    pub const fn new(alpha: f32) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            alpha,
            beta: (alpha * alpha) / (2.0 - alpha),
            unit_est: None,
            rate_est: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    /// Reset the filters internal state.
    pub const fn reset(&mut self) {
        self.unit_est = None;
        self.rate_est = Vector3::new(0.0, 0.0, 0.0);
    }

    /// Update the filter with a new LOS-vector measurement (if we have a visual track).
    /// If `measurement` is None, it coasts/dead-reckons based on the previous rate.
    pub fn update(
        &mut self,
        measurement: Option<Vector3<f32>>,
        dt: f32,
    ) -> (Vector3<f32>, Vector3<f32>) {
        if let Some(unit_est) = self.unit_est {
            // Predict the value based on prior state and velocity
            let val_pred = (unit_est + (self.rate_est * dt)).normalize();

            if let Some(meas) = measurement {
                // We have a track: apply prediction + correction
                let residual = meas - val_pred;

                let new_est = (val_pred + (self.alpha * residual)).normalize();

                // Update the rate, then strip any component that goes *along* the LOS vector
                // to prevent rate_est from accumulating length-changing velocity on the sphere
                let mut new_rate = self.rate_est + (self.beta * residual) / dt;
                new_rate = new_rate - new_est * new_est.dot(&new_rate);

                self.rate_est = new_rate;
                self.unit_est = Some(new_est);

                (new_est, self.rate_est)
            } else {
                // Tracking lost: Coast forward using dead-reckoning prediction only
                self.unit_est = Some(val_pred);

                // Keep the rate orthogonal
                let mut coast_rate = self.rate_est;
                coast_rate = coast_rate - val_pred * val_pred.dot(&coast_rate);
                self.rate_est = coast_rate;

                (val_pred, self.rate_est)
            }
        } else {
            // No prior state initialized
            if let Some(meas) = measurement {
                self.unit_est = Some(meas);
                (meas, self.rate_est)
            } else {
                // Completely blind from the start. Return Forward direction as a safe default.
                (Vector3::new(1.0, 0.0, 0.0), Vector3::zeros())
            }
        }
    }
}

enum FlightPhase {
    Cruise,
    Terminal,
}

pub struct ProNav {
    /// Mass of the drone in kilo-grams [kg]
    drone_mass: f32,
    /// The gain of the ProNav control law
    pronav_gain: f32,
    /// The gain of the ProNav integral action
    intnav_gain: f32,
    /// The gain to encourage pure pursuit
    pursuit_gain: f32,
    /// The gain to push the target along the PN desired direction
    velocity_gain: f32,
    velocity_target: f32,
    velocity_comp: f32,
    camera_pitch: f32,
    fov_limit: f32,
    fov_penalty_gain: f32,
    fov_leak_rate: f32,
    fov_integral: f32,
    los_vel_integral: Vector3<f32>,
    prev_target: Option<Vector3<f32>>,
    phase: FlightPhase,
}

impl ProNav {
    pub const fn new(drone_mass: f32) -> Self {
        Self {
            drone_mass,
            pronav_gain: 5.0,
            intnav_gain: 0.0,
            pursuit_gain: 0.0,
            velocity_gain: 0.0,
            velocity_target: 35.0,
            velocity_comp: 0.0,
            camera_pitch: 0.0,
            fov_limit: 45.0f32.to_radians(),
            fov_penalty_gain: 10.0,
            fov_leak_rate: 1.0,
            fov_integral: 0.0,
            los_vel_integral: Vector3::new(0.0, 0.0, 0.0),
            prev_target: None,
            phase: FlightPhase::Cruise,
        }
    }

    pub const fn camera_pitch(mut self, camera_pitch: f32) -> Self {
        self.camera_pitch = camera_pitch;
        self
    }

    pub const fn fov_limit(mut self, fov_limit: f32) -> Self {
        self.fov_limit = fov_limit;
        self
    }

    pub const fn fov_penalty_gain(mut self, fov_penalty_gain: f32) -> Self {
        self.fov_penalty_gain = fov_penalty_gain;
        self
    }

    pub const fn pronav_gain(mut self, pronav_gain: f32) -> Self {
        self.pronav_gain = pronav_gain.max(0.0);
        self
    }

    pub const fn intnav_gain(mut self, intnav_gain: f32) -> Self {
        self.intnav_gain = intnav_gain.max(0.0);
        self
    }

    pub const fn pursuit_gain(mut self, pursuit_gain: f32) -> Self {
        self.pursuit_gain = pursuit_gain.max(0.0);
        self
    }

    pub const fn velocity_gain(mut self, velocity_gain: f32) -> Self {
        self.velocity_gain = velocity_gain.max(0.0);
        self
    }

    pub const fn velocity_target(mut self, velocity_target: f32) -> Self {
        self.velocity_target = velocity_target.max(0.0);
        self
    }

    pub fn update(
        &mut self,
        closing_vel: f32,
        los_unit: Vector3<f32>,
        los_unit_vel: Vector3<f32>,
        attitude: UnitQuaternion<f32>,
        dt: f32,
    ) -> (UnitQuaternion<f32>, f32) {
        if matches!(self.phase, FlightPhase::Cruise) {
            let los_elevation_abs = los_unit.z.abs().asin();
            if los_elevation_abs > 20.0_f32.to_radians() {
                log::warn!("[pronav] Entering terminal phase");
                self.phase = FlightPhase::Terminal;
            }
        }

        // =============================================================
        // This section applies a fairly standard "True ProNav" strategy
        // =============================================================

        // Acceleration contribution of the PN guidance law [m/s^2]
        let pronav_accel = self.pronav_gain * closing_vel * los_unit_vel;

        match self.phase {
            FlightPhase::Cruise => {
                let mut los_unit_vel_cruise = los_unit_vel;
                los_unit_vel_cruise.z = 0.0;
                self.los_vel_integral += los_unit_vel_cruise * dt;
            }
            FlightPhase::Terminal => {
                self.los_vel_integral += los_unit_vel * dt;
            }
        }

        let intnav_accel = self.intnav_gain * closing_vel * self.los_vel_integral;

        // Acceleration contribution of pure pursuit [m/s^2]
        let pursuit_accel = self.pursuit_gain * los_unit;

        // Desired scceleration [m/s^2]
        let mut desired_accel = pronav_accel + intnav_accel + pursuit_accel;

        // Disallow the pronav control law from setting the altitude acceleration in cruise mode
        if matches!(self.phase, FlightPhase::Cruise) {
            desired_accel.z = 0.0;
        }

        // =================================================================
        // This section converts the desired accel into an attitude + thrust
        // =================================================================

        // Add gravity back into the desired accel
        let gravity_vector = Vector3::z() * 9.81;
        let mut global_accel_target = desired_accel - gravity_vector;

        // FOV Constraint Penalty using Leaky Integrator
        // 1. Calculate camera boresight global vector
        let boresight_body = Vector3::new(self.camera_pitch.cos(), 0.0, -self.camera_pitch.sin());
        let boresight_global = attitude.transform_vector(&boresight_body);

        // 2. Measure violation outside of FOV cone
        let angle = boresight_global.angle(&los_unit);
        let violation = angle - self.fov_limit;

        // 3. Continuous Leaky Integrator
        // Input is the severity of the FOV violation (0 if safely inside)
        let penalty_input = violation.max(0.0);

        // Standard leaky integrator: dx/dt = input - leak_rate * state
        // This ensures a continuous gradient and asymptotic decay without harsh switching
        self.fov_integral += (penalty_input - self.fov_leak_rate * self.fov_integral) * dt;
        self.fov_integral = self.fov_integral.max(0.0);

        // 4. Apply restorative rotation
        if self.fov_integral > 0.0 && angle > 1e-4 {
            // Find rotation axis that pushes the camera boresight directly toward the LOS unit vector
            let rot_axis = boresight_global.cross(&los_unit);
            if let Some(rot_axis_unit) = rot_axis.try_normalize(1e-4) {
                // Convert integrated penalty to a physical angle (capped at 90 deg to avoid flipping entirely)
                let penalty_angle = (self.fov_integral * self.fov_penalty_gain).min(PI / 2.0);

                // Construct a quaternion that rotates the entire acceleration demand
                let penalty_quat = UnitQuaternion::from_axis_angle(
                    &UnitVector3::new_unchecked(rot_axis_unit),
                    penalty_angle,
                );
                global_accel_target = penalty_quat * global_accel_target;
            }
        }

        let desired_thrust_dir = global_accel_target
            .try_normalize(1e-6)
            .unwrap_or(-Vector3::z());

        // Determine the alignment between the desired attitude and the current one.
        // Use that to scale down the force target while ill-aligned
        let direction = attitude.transform_vector(&-Vector3::z());
        let alignment_factor = direction.dot(&desired_thrust_dir).clamp(0.0, 1.0);

        // Thrust (body -Z) points toward desired_thrust_dir
        let b_z = -desired_thrust_dir;

        // Nose (Body +X) should point toward target
        let los_norm = los_unit;
        let b_y = b_z.cross(&los_norm).try_normalize(1e-6).unwrap_or_else(|| {
            b_z.cross(&Vector3::x())
                .try_normalize(1e-6)
                .unwrap_or(Vector3::y())
        });

        // Compute the last orthonormal basis vector
        let b_x = b_y.cross(&b_z);

        // Create rotation matrix from basis vectors [x, y, z]
        let rotation_matrix = Matrix3::from_columns(&[b_x, b_y, b_z]);
        let att_target = UnitQuaternion::from_matrix(&rotation_matrix);

        // Compensate for additional air resistance at higher speeds
        let force_target = self.drone_mass * alignment_factor;

        (att_target, force_target)
    }
}
