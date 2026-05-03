use std::collections::VecDeque;
use std::f32::consts::PI;

use common::nalgebra::{Matrix3, Point2, Rotation3, SMatrix, UnitQuaternion, UnitVector3, Vector3};
use embassy_time::Instant;

pub struct CameraModel {
    pub vfov_rad: f32,
    pub width: f32,
    pub height: f32,
    pub f_x: f32,
    pub f_y: f32,
    pub c_x: f32,
    pub c_y: f32,
    pub pitch_rad: f32,
}

impl CameraModel {
    pub fn new(width: f32, height: f32, vfov_rad: f32, pitch_rad: f32) -> Self {
        let f_y = height / (2.0 * (vfov_rad / 2.0).tan());
        let f_x = f_y; // Assume square pixels

        Self {
            vfov_rad,
            width,
            height,
            f_x,
            f_y,
            c_x: width / 2.0,
            c_y: height / 2.0,
            pitch_rad,
        }
    }

    /// Emulates the camera sensor, returning Some(u,v) pixel coordinate if the target is in view.
    pub fn project_to_pixel(
        &self,
        los_global: Vector3<f32>,
        attitude: UnitQuaternion<f32>,
    ) -> Option<(f32, f32)> {
        let los_global = los_global.normalize();
        let los_body = attitude.inverse_transform_vector(&los_global);

        // Apply camera pitch (pitching up is positive rotation around body Y)
        let pitch_quat = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.pitch_rad);
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
        let pitch_rot = Rotation3::from_euler_angles(0.0, self.pitch_rad, 0.0);

        let los_body = pitch_rot.matrix() * los_cam_aligned;

        // Body to Global
        att.transform_vector(&los_body)
    }
}

// =============================================================================
// Attitude ring buffer for interpolated lookups at arbitrary past timestamps
// =============================================================================

#[derive(Clone)]
pub struct AttitudeEntry {
    timestamp: Instant,
    attitude: UnitQuaternion<f32>,
    angular_vel: Vector3<f32>,
}

pub struct AttitudeBuffer {
    pub entries: VecDeque<AttitudeEntry>,
    capacity: usize,
}

impl AttitudeBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Insert an attitude sample. Handles out-of-order arrivals by inserting
    /// at the correct chronological position.
    pub fn push(
        &mut self,
        timestamp: Instant,
        attitude: UnitQuaternion<f32>,
        angular_vel: Vector3<f32>,
    ) {
        let entry = AttitudeEntry {
            timestamp,
            attitude,
            angular_vel,
        };

        // Fast path: most arrivals are in-order (append to back)
        if self
            .entries
            .back()
            .map_or(true, |e| e.timestamp <= timestamp)
        {
            self.entries.push_back(entry);
        } else {
            // Out-of-order: binary search for the correct insertion point
            let pos = self
                .entries
                .binary_search_by(|e| e.timestamp.cmp(&timestamp))
                .unwrap_or_else(|i| i);
            self.entries.insert(pos, entry);
        }

        // Evict oldest entries if over capacity
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    /// Returns the latest attitude and angular velocity in the buffer.
    pub fn latest(&self) -> Option<(UnitQuaternion<f32>, Vector3<f32>)> {
        self.entries.back().map(|e| (e.attitude, e.angular_vel))
    }

    /// SLERP-interpolate attitude and LERP-interpolate angular velocity
    /// at the requested timestamp. Returns `None` if the buffer is empty or
    /// the timestamp is before the first stored entry.
    pub fn interpolate_at(
        &self,
        timestamp: Instant,
    ) -> Option<(UnitQuaternion<f32>, Vector3<f32>)> {
        if self.entries.is_empty() {
            return None;
        }

        let first = self.entries.front().unwrap();
        let last = self.entries.back().unwrap();

        // Clamp: if before the earliest entry, return the earliest
        if timestamp <= first.timestamp {
            return Some((first.attitude, first.angular_vel));
        }

        // Clamp: if at or after the latest entry, extrapolate using angular velocity
        if timestamp >= last.timestamp {
            let dt = (timestamp - last.timestamp).as_micros() as f32 * 1e-6;
            if dt < 1e-9 {
                return Some((last.attitude, last.angular_vel));
            }
            // Small-angle extrapolation: apply angular velocity as a rotation
            let delta_angle = last.angular_vel * dt;
            let mag = delta_angle.norm();
            let extrapolated = if mag > 1e-9 {
                let axis = UnitVector3::new_normalize(delta_angle);
                let rot = UnitQuaternion::from_axis_angle(&axis, mag);
                last.attitude * rot
            } else {
                last.attitude
            };
            return Some((extrapolated, last.angular_vel));
        }

        // Binary search for the two bracketing entries
        let upper = self
            .entries
            .binary_search_by(|e| e.timestamp.cmp(&timestamp))
            .unwrap_or_else(|i| i);

        // Exact match
        if upper < self.entries.len() && self.entries[upper].timestamp == timestamp {
            let e = &self.entries[upper];
            return Some((e.attitude, e.angular_vel));
        }

        let lower = upper.saturating_sub(1);
        let a = &self.entries[lower];
        let b = &self.entries[upper.min(self.entries.len() - 1)];

        let span = (b.timestamp - a.timestamp).as_micros() as f32;
        if span < 1.0 {
            return Some((a.attitude, a.angular_vel));
        }

        let t = (timestamp - a.timestamp).as_micros() as f32 / span;
        let t = t.clamp(0.0, 1.0);

        let att = a.attitude.slerp(&b.attitude, t);
        let ang = a.angular_vel.lerp(&b.angular_vel, t);

        Some((att, ang))
    }
}

// =============================================================================
// Rewindable Alpha-Beta LOS filter with out-of-sequence measurement support
// =============================================================================

/// A snapshot of the filter state at a given point in time.
#[derive(Clone)]
struct LosFilterSnapshot {
    timestamp: Instant,
    unit_est: Option<Vector3<f32>>,
    rate_est: Vector3<f32>,
}

/// A measurement (or coast/predict step) that was applied to the filter.
#[derive(Clone)]
struct LosFilterInput {
    timestamp: Instant,
    measurement: Option<Vector3<f32>>, // None = coast/predict step
}

pub struct AlphaBetaLos {
    alpha: f32,
    beta: f32,
    /// Current filter state
    unit_est: Option<Vector3<f32>>,
    rate_est: Vector3<f32>,
    /// Timestamp of the current state
    current_time: Option<Instant>,
    /// History of filter states for rollback
    snapshots: VecDeque<LosFilterSnapshot>,
    /// History of inputs for replay after rollback
    inputs: VecDeque<LosFilterInput>,
    /// Maximum number of history entries to retain
    max_history: usize,
}

impl AlphaBetaLos {
    /// Construct a new alpha-beta filter.
    ///
    /// Larger values of `alpha` will make the filter more responsive, but also potentially noisier.
    pub fn new(alpha: f32) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            alpha,
            beta: (alpha * alpha) / (2.0 - alpha),
            unit_est: None,
            rate_est: Vector3::new(0.0, 0.0, 0.0),
            current_time: None,
            snapshots: VecDeque::with_capacity(256),
            inputs: VecDeque::with_capacity(256),
            max_history: 256,
        }
    }

    /// Reset the filter's internal state and history.
    pub fn reset(&mut self) {
        self.unit_est = None;
        self.rate_est = Vector3::new(0.0, 0.0, 0.0);
        self.current_time = None;
        self.snapshots.clear();
        self.inputs.clear();
    }

    /// Internal single-step predict-correct cycle. This is the original `update()` logic.
    fn step(&mut self, measurement: Option<Vector3<f32>>, dt: f32) -> (Vector3<f32>, Vector3<f32>) {
        if let Some(unit_est) = self.unit_est {
            // Predict the value based on prior state and velocity
            let val_pred = (unit_est + (self.rate_est * dt)).normalize();

            if let Some(meas) = measurement {
                // We have a track: apply prediction + correction
                let residual = meas - val_pred;

                let new_est = (val_pred + (self.alpha * residual)).normalize();

                // Update the rate, then strip any component that goes *along* the LOS vector
                // to prevent rate_est from accumulating length-changing velocity on the sphere
                let mut new_rate = if dt > 1e-9 {
                    self.rate_est + (self.beta * residual) / dt
                } else {
                    self.rate_est
                };
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

    /// Save the current state as a snapshot at the given timestamp.
    fn save_snapshot(&mut self, timestamp: Instant) {
        self.snapshots.push_back(LosFilterSnapshot {
            timestamp,
            unit_est: self.unit_est,
            rate_est: self.rate_est,
        });
        while self.snapshots.len() > self.max_history {
            self.snapshots.pop_front();
        }
    }

    /// Record an input for potential future replay.
    fn record_input(&mut self, timestamp: Instant, measurement: Option<Vector3<f32>>) {
        // Insert in chronological order (fast-path: append)
        let input = LosFilterInput {
            timestamp,
            measurement,
        };
        if self
            .inputs
            .back()
            .map_or(true, |i| i.timestamp <= timestamp)
        {
            self.inputs.push_back(input);
        } else {
            let pos = self
                .inputs
                .binary_search_by(|i| i.timestamp.cmp(&timestamp))
                .unwrap_or_else(|i| i);
            self.inputs.insert(pos, input);
        }
        while self.inputs.len() > self.max_history {
            self.inputs.pop_front();
        }
    }

    /// Restore filter state from a snapshot.
    fn restore_snapshot(&mut self, snapshot: &LosFilterSnapshot) {
        self.unit_est = snapshot.unit_est;
        self.rate_est = snapshot.rate_est;
        self.current_time = Some(snapshot.timestamp);
    }

    /// Fuse a measurement at its true timestamp. If the measurement is older
    /// than the current filter time, this performs a rollback-and-replay to
    /// incorporate it at the correct chronological point.
    ///
    /// Returns the updated (los_unit, los_rate) estimate at the latest time.
    pub fn fuse(
        &mut self,
        timestamp: Instant,
        measurement: Option<Vector3<f32>>,
    ) -> (Vector3<f32>, Vector3<f32>) {
        let is_oosm = self.current_time.map_or(false, |t| timestamp < t);

        if !is_oosm {
            // In-order measurement: simple forward step
            let dt = self
                .current_time
                .map(|t| (timestamp - t).as_micros() as f32 * 1e-6)
                .unwrap_or(0.0);

            let result = self.step(measurement, dt);
            self.current_time = Some(timestamp);

            // Save state AFTER the step has been applied
            self.save_snapshot(timestamp);
            self.record_input(timestamp, measurement);
            return result;
        }

        // === Out-of-sequence measurement: rollback and replay ===

        // 1. Record the new input in chronological order
        self.record_input(timestamp, measurement);

        // 2. Find the latest snapshot strictly BEFORE the measurement timestamp
        let rollback_idx = self.snapshots.iter().rposition(|s| s.timestamp < timestamp);

        let replay_start_time = if let Some(idx) = rollback_idx {
            let snap = self.snapshots[idx].clone();
            self.restore_snapshot(&snap);
            // Discard all snapshots after the rollback point (they'll be regenerated)
            self.snapshots.truncate(idx + 1);
            snap.timestamp
        } else {
            // No snapshot old enough — reset to initial state and replay everything
            self.unit_est = None;
            self.rate_est = Vector3::zeros();
            self.current_time = None;
            self.snapshots.clear();
            // Replay from the very first input
            self.inputs
                .front()
                .map(|i| i.timestamp)
                .unwrap_or(timestamp)
        };

        // 3. Collect inputs to replay (all inputs strictly after replay_start_time)
        let inputs_to_replay: Vec<LosFilterInput> = self
            .inputs
            .iter()
            .filter(|i| i.timestamp > replay_start_time)
            .cloned()
            .collect();

        // 4. Replay all inputs in chronological order
        let mut result = self.current_estimate();
        for input in &inputs_to_replay {
            let dt = self
                .current_time
                .map(|t| {
                    if input.timestamp > t {
                        (input.timestamp - t).as_micros() as f32 * 1e-6
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);

            result = self.step(input.measurement, dt);
            self.current_time = Some(input.timestamp);
            self.save_snapshot(input.timestamp);
        }

        result
    }

    /// Coast/predict the filter forward to `now` without incorporating a
    /// measurement. Use this from the controller tick to get the latest estimate
    /// extrapolated to the current time.
    ///
    /// This does **not** record a coast step in the input history, so it won't
    /// interfere with future OOSM replays. It advances `current_time`.
    pub fn predict_to(&mut self, now: Instant) -> (Vector3<f32>, Vector3<f32>) {
        let dt = self
            .current_time
            .map(|t| {
                if now > t {
                    (now - t).as_micros() as f32 * 1e-6
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);

        if dt < 1e-9 {
            return self.current_estimate();
        }

        // Predict forward without recording — this is a "peek" extrapolation
        if let Some(unit_est) = self.unit_est {
            let val_pred = (unit_est + (self.rate_est * dt)).normalize();

            // Keep rate orthogonal to the predicted direction
            let mut coast_rate = self.rate_est;
            coast_rate = coast_rate - val_pred * val_pred.dot(&coast_rate);

            (val_pred, coast_rate)
        } else {
            (Vector3::new(1.0, 0.0, 0.0), Vector3::zeros())
        }
    }

    /// Read the current best estimate without mutating state.
    pub fn current_estimate(&self) -> (Vector3<f32>, Vector3<f32>) {
        (
            self.unit_est.unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0)),
            self.rate_est,
        )
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
    camera_pitch: f32,
    fov_limit: f32,
    fov_penalty_gain: f32,
    fov_leak_rate: f32,
    fov_integral: f32,
    los_vel_integral: Vector3<f32>,
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
            camera_pitch: 0.0,
            fov_limit: 45.0f32.to_radians(),
            fov_penalty_gain: 10.0,
            fov_leak_rate: 1.0,
            fov_integral: 0.0,
            los_vel_integral: Vector3::new(0.0, 0.0, 0.0),
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

        // Acceleration contribution of the PN guidance law [m/s^2]
        let pronav_accel = self.pronav_gain * closing_vel * los_unit_vel;
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
        let force_target = self.drone_mass * global_accel_target.norm() * alignment_factor;

        (att_target, force_target)
    }
}

/// Run this ONLY on initialization to establish the initial tangent plane
fn compute_initial_basis(u: &Vector3<f32>) -> SMatrix<f32, 3, 2> {
    let mut v = Vector3::new(1.0, 0.0, 0.0);
    if u.x.abs() > 0.9 {
        v = Vector3::new(0.0, 1.0, 0.0);
    }
    let b1 = u.cross(&v).normalize();
    let b2 = u.cross(&b1).normalize();
    SMatrix::from_columns(&[b1, b2])
}

pub struct EskfLos {
    /// Global nominal 3D unit LOS vector
    pub u_hat: Vector3<f32>,
    /// Global nominal 3D LOS angular rate
    pub omega_hat: Vector3<f32>,
    /// Basis for the tangent space of the unit sphere at u_hat
    pub basis: SMatrix<f32, 3, 2>,

    /// 4x4 Error state covariance matrix [d_theta_1, d_theta_2, d_omega_1, d_omega_2]
    pub p_cov: SMatrix<f32, 4, 4>,
    /// 4x4 Process noise covariance matrix
    pub q_cov: SMatrix<f32, 4, 4>,
    /// 3x3 Baseline pixel measurement noise covariance (in global frame)
    pub r_pixel: SMatrix<f32, 3, 3>,
}

impl EskfLos {
    /// Construct a new ESKF LOS tracker.
    pub fn new(q_cov: SMatrix<f32, 4, 4>, r_pixel: f32) -> Self {
        let u_initial = Vector3::new(1.0, 0.0, 0.0);
        Self {
            u_hat: u_initial,
            omega_hat: Vector3::zeros(),
            basis: compute_initial_basis(&u_initial),
            p_cov: SMatrix::identity(),
            q_cov,
            r_pixel: SMatrix::identity() * r_pixel,
        }
    }

    /// Reset the filter's internal state
    pub fn reset(&mut self) {
        self.u_hat = Vector3::new(1.0, 0.0, 0.0);
        self.omega_hat = Vector3::zeros();
        self.p_cov = SMatrix::identity();
    }

    /// Predict the state forward in time (dead-reckoning).
    /// Call this at your system's base loop rate, regardless of whether a measurement arrived.
    pub fn predict(&mut self, dt: f32) {
        // 1. Nominal State Propagation (Rodrigues' rotation formula)
        let theta = self.omega_hat.norm() * dt;
        if theta > 1e-6 {
            let k = self.omega_hat.normalize();
            let k_skew = skew_symmetric(&k);
            let r =
                Matrix3::identity() + k_skew * theta.sin() + k_skew * k_skew * (1.0 - theta.cos());

            // Rotate the nominal LOS vector
            self.u_hat = r * self.u_hat;

            // Rotate the basis vectors to follow the LOS smoothly (Parallel Transport)
            self.basis = r * self.basis;
        }

        // 2. Error Covariance Propagation
        let mut f = SMatrix::<f32, 4, 4>::identity();
        f[(0, 2)] = dt;
        f[(1, 3)] = dt;

        self.p_cov = f * self.p_cov * f.transpose() + self.q_cov;
    }

    /// Update the filter with a new visual LOS measurement.
    /// `z_u`: The measured 3D unit vector, already rotated into the global inertial frame.
    /// `p_attitude`: The 3x3 covariance matrix from the host vehicle's attitude estimator.
    pub fn update(&mut self, z_u: Vector3<f32>, p_attitude: Matrix3<f32>) {
        // Measurement Jacobian H (3x4)
        let mut h = SMatrix::<f32, 3, 4>::zeros();
        h.fixed_columns_mut::<2>(0).copy_from(&self.basis);

        // Measurement Residual
        let residual = z_u - self.u_hat;

        // Measurement Covariance R (incorporating host attitude uncertainty)
        let z_skew = skew_symmetric(&z_u);
        let r_cov = self.r_pixel + z_skew * p_attitude * z_skew.transpose();

        // Innovation Covariance S and Kalman Gain K
        let s = h * self.p_cov * h.transpose() + r_cov;

        if let Some(s_inv) = s.try_inverse() {
            let k = self.p_cov * h.transpose() * s_inv;

            // Compute Error State (dx = K * residual)
            let dx = k * residual;
            let d_theta = dx.fixed_rows::<2>(0);
            let d_omega = dx.fixed_rows::<2>(2);

            // Inject Error State into Nominal State
            let theta_mag = d_theta.norm();
            if theta_mag > 1e-6 {
                let (theta_sin, theta_cos) = theta_mag.sin_cos();
                let axis = (self.basis * d_theta) / theta_mag;
                self.u_hat = (self.u_hat * theta_cos + axis * theta_sin).normalize();
            }

            // Apply rate correction before refining the basis, and ensure manifold constraint
            self.omega_hat += self.basis * d_omega;
            self.omega_hat -= self.u_hat * self.u_hat.dot(&self.omega_hat);

            // Update Covariance and Reset Error State (implicitly reset by not storing dx)
            self.p_cov = (SMatrix::identity() - k * h) * self.p_cov;

            // Refine the Basis for the NEXT iteration
            let b0 = self.basis.column(0);
            let b1_new = (b0 - self.u_hat * self.u_hat.dot(&b0)).normalize();
            let b2_new = self.u_hat.cross(&b1_new).normalize();
            self.basis = SMatrix::<f32, 3, 2>::from_columns(&[b1_new, b2_new]);
        } else {
            log::error!("ESKF innovation covariance matrix is singular");
        }
    }
}

/// Helper: Creates a skew-symmetric cross-product matrix for a given 3D vector.
#[rustfmt::skip]
fn skew_symmetric(v: &Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0
    )
}

/// Helper: Generates a 3x2 matrix whose columns form an orthonormal basis
/// for the tangent plane perpendicular to the given unit vector `u`.
fn compute_tangent_basis(u: &Vector3<f32>) -> SMatrix<f32, 3, 2> {
    // Pick an arbitrary vector that is not perfectly aligned with `u`
    let mut v = Vector3::new(1.0, 0.0, 0.0);
    if u.x.abs() > 0.9 {
        v = Vector3::new(0.0, 1.0, 0.0);
    }

    // Gram-Schmidt / Cross-product to find two orthogonal basis vectors
    let b1 = u.cross(&v).normalize();
    let b2 = u.cross(&b1).normalize();

    SMatrix::from_columns(&[b1, b2])
}
