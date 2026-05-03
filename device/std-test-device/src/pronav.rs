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
    pub fn project_to_pixel(&self, los_body: Vector3<f32>) -> Option<Point2<f32>> {
        let los_body = los_body.normalize();

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

        Some(Point2::new(u, v))
    }

    /// Converts a pixel measurement (if available) back to a global LOS vector
    pub fn pixel_to_los_body(&self, pixel: Point2<f32>) -> Vector3<f32> {
        let [[u, v]] = pixel.coords.data.0;

        // Map pixel back to a 3D ray in the camera frame
        let x_c = (u - self.c_x) / self.f_x;
        let y_c = (v - self.c_y) / self.f_y;
        let z_c = 1.0;

        // Map Camera coordinates (Right-Down-Forward) back to Body coordinates (Forward-Right-Down)
        let los_cam_aligned = Vector3::new(z_c, x_c, y_c).normalize();

        // Apply inverse of camera pitch - must match the pitch used in project_to_pixel!
        let pitch_rot = Rotation3::from_euler_angles(0.0, self.pitch_rad, 0.0);

        pitch_rot.matrix() * los_cam_aligned
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
    pub fn interpolate_at(&self, timestamp: Instant) -> Option<UnitQuaternion<f32>> {
        if self.entries.is_empty() {
            return None;
        }

        // This is okay due to the !empty check
        let first = self.entries.front().unwrap();
        let last = self.entries.back().unwrap();

        // If before the earliest entry, return the earliest
        if timestamp <= first.timestamp {
            return Some(first.attitude);
        }

        // If at or after the latest entry, extrapolate using angular velocity
        if timestamp >= last.timestamp {
            let dt = (timestamp - last.timestamp).as_micros() as f32 * 1e-6;

            // Apply angular velocity as a rotation
            let delta_angle = last.angular_vel * dt;
            let delta_angle_norm = delta_angle.norm();
            let extrapolated = if delta_angle_norm > 1e-9 {
                let axis = UnitVector3::new_normalize(delta_angle);
                let rot = UnitQuaternion::from_axis_angle(&axis, delta_angle_norm);
                last.attitude * rot
            } else {
                last.attitude
            };
            return Some(extrapolated);
        }

        // Binary search for the two bracketing entries
        let upper = self
            .entries
            .binary_search_by(|e| e.timestamp.cmp(&timestamp))
            .unwrap_or_else(|i| i);

        // Exact match
        if upper < self.entries.len() && self.entries[upper].timestamp == timestamp {
            let e = &self.entries[upper];
            return Some(e.attitude);
        }

        let lower = upper.saturating_sub(1);
        let a = &self.entries[lower];
        let b = &self.entries[upper.min(self.entries.len() - 1)];

        let span = (b.timestamp - a.timestamp).as_micros() as f32;
        if span < 1.0 {
            return Some(a.attitude);
        }

        let t = (timestamp - a.timestamp).as_micros() as f32 / span;
        let t = t.clamp(0.0, 1.0);

        let att = a.attitude.slerp(&b.attitude, t);

        Some(att)
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
            fov_leak_rate: 0.25,
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
        // Traisition phase based on LOS elevation
        if matches!(self.phase, FlightPhase::Cruise) {
            let los_elevation_abs = los_unit.z.abs().asin();
            if los_elevation_abs > 20.0_f32.to_radians() {
                log::warn!("[pronav] Entering terminal phase");
                self.phase = FlightPhase::Terminal;
            }
        }

        // =================================================================
        // 1. Evaluate Camera FOV Constraints first
        // =================================================================
        let boresight_body = Vector3::new(self.camera_pitch.cos(), 0.0, -self.camera_pitch.sin());
        let boresight_global = attitude.transform_vector(&boresight_body);

        let angle = boresight_global.angle(&los_unit);
        let violation = angle - self.fov_limit;
        let penalty_input = violation.max(0.0);

        // =============================================================
        // This section applies a fairly standard "True ProNav" strategy
        // =============================================================

        if penalty_input == 0.0 {
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
        }

        // Acceleration contribution of the PN guidance law [m/s^2]
        let pronav_accel = self.pronav_gain * closing_vel * los_unit_vel;
        let intnav_accel = self.intnav_gain * closing_vel * self.los_vel_integral;

        let velocity_accel = self.velocity_gain * (self.velocity_target - closing_vel) * los_unit;

        // Acceleration contribution of pure pursuit [m/s^2]
        let pursuit_accel = self.pursuit_gain * los_unit;

        // Desired scceleration [m/s^2]
        let mut desired_accel = pronav_accel + intnav_accel + pursuit_accel + velocity_accel;

        // Disallow the pronav control law from setting the altitude acceleration in cruise mode
        if matches!(self.phase, FlightPhase::Cruise) {
            desired_accel.z = 0.0;
        }

        // Add gravity back into the desired accel
        let gravity_vector = Vector3::z() * 9.81;
        let mut global_accel_target = desired_accel - gravity_vector;

        // =================================================================
        // This section converts the desired accel into an attitude + thrust
        // =================================================================

        // Update FOV Leaky Integrator
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
fn initial_basis(u: &Vector3<f32>) -> SMatrix<f32, 3, 2> {
    let mut v = Vector3::new(1.0, 0.0, 0.0);
    if u.x.abs() > 0.9 {
        v = Vector3::new(0.0, 1.0, 0.0);
    }
    let b1 = u.cross(&v).normalize();
    let b2 = u.cross(&b1).normalize();
    SMatrix::from_columns(&[b1, b2])
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

#[derive(Debug, Clone)]
pub struct EskfLos {
    /// Global nominal 3D unit LOS vector
    pub los_hat: Vector3<f32>,
    /// Global nominal 3D LOS angular rate
    pub omega_hat: Vector3<f32>,
    /// Basis for the tangent space of the unit sphere at u_hat
    pub basis: SMatrix<f32, 3, 2>,
    /// 4x4 Error state covariance matrix
    pub p_cov: SMatrix<f32, 4, 4>,
    /// 4x4 Process noise covariance matrix
    pub q_cov: SMatrix<f32, 4, 4>,
}

impl EskfLos {
    /// Construct a new ESKF LOS tracker.
    pub fn new(q_cov: SMatrix<f32, 4, 4>) -> Self {
        let los_initial = Vector3::new(1.0, 0.0, 0.0);
        Self {
            los_hat: los_initial,
            omega_hat: Vector3::zeros(),
            basis: initial_basis(&los_initial),
            p_cov: SMatrix::identity(),
            q_cov,
        }
    }

    /// Reset the filter's internal state using some initial LOS vector
    pub fn reset(&mut self, los: Vector3<f32>) {
        self.los_hat = los;
        self.basis = initial_basis(&los);
        self.omega_hat = Vector3::zeros();
        self.p_cov = SMatrix::identity();
    }

    /// Predict the state forward in time (dead-reckoning).
    /// Call this at your system's base loop rate, regardless of whether a measurement arrived.
    pub fn predict(&mut self, dt: f32) {
        // State Propagation (Rodrigues' rotation formula)
        // https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
        let theta = self.omega_hat.norm() * dt;
        if theta > 1e-6 {
            let u_dot_dir = self.omega_hat.normalize();
            let k = self.los_hat.cross(&u_dot_dir).normalize();
            let k_skew = skew_symmetric(&k);
            let rot =
                Matrix3::identity() + k_skew * theta.sin() + k_skew * k_skew * (1.0 - theta.cos());

            // Rotate to follow the LOS smoothly
            self.los_hat = rot * self.los_hat;
            self.omega_hat = rot * self.omega_hat;
            self.basis = rot * self.basis;
        }

        // Error Covariance Propagation
        let mut f = SMatrix::<f32, 4, 4>::identity();
        f[(0, 2)] = dt;
        f[(1, 3)] = dt;

        self.p_cov = f * self.p_cov * f.transpose() + self.q_cov * dt;
    }

    /// Update the filter with a new visual LOS measurement.
    /// `z_u`: The measured 3D unit vector, already rotated into the global inertial frame.
    /// `p_attitude`: The 3x3 covariance matrix from the host vehicle's attitude estimator.
    /// `r_pixel`: The baseline pixel measurement noise covariance (in global frame).
    pub fn update(&mut self, meas: &Measure) {
        // Measurement Jacobian H (3x4)
        let mut h = SMatrix::<f32, 3, 4>::zeros();
        h.fixed_columns_mut::<2>(0).copy_from(&self.basis);

        // Measurement Residual
        let residual = meas.los_unit - self.los_hat;

        // Measurement Covariance R (incorporating host attitude uncertainty)
        let los_skew = skew_symmetric(&meas.los_unit);
        let r_cov =
            SMatrix::identity() * meas.r_pixel + los_skew * meas.p_attitude * los_skew.transpose();

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
                self.los_hat = (self.los_hat * theta_cos + axis * theta_sin).normalize();
            }

            // Apply rate correction before refining the basis, and ensure manifold constraint
            self.omega_hat += self.basis * d_omega;
            self.omega_hat -= self.los_hat * self.los_hat.dot(&self.omega_hat);

            // Update Covariance and Reset Error State (implicitly reset by not storing dx)
            self.p_cov = (SMatrix::identity() - k * h) * self.p_cov;

            // Refine the tangent basis for the next iteration
            let b0 = self.basis.column(0);
            let b1_new = (b0 - self.los_hat * self.los_hat.dot(&b0)).normalize();
            let b2_new = self.los_hat.cross(&b1_new).normalize();
            self.basis = SMatrix::<f32, 3, 2>::from_columns(&[b1_new, b2_new]);
        } else {
            log::error!("ESKF innovation covariance matrix is singular");
        }
    }
}

// =============================================================================
// OOSM wrapper for EskfLos
// =============================================================================

#[derive(Clone)]
pub struct Timed<T> {
    timestamp: Instant,
    data: T,
}

/// All parameters needed to call `EskfLos::update`.
#[derive(Clone)]
pub struct Measure {
    los_unit: Vector3<f32>,
    p_attitude: Matrix3<f32>,
    r_pixel: f32,
}

/// Wraps `EskfLos` with out-of-sequence measurement (OOSM) support.
///
/// On every call to `fuse` the wrapper either steps forward (in-order) or
/// rolls back to the nearest prior snapshot, inserts the new measurement in
/// chronological order, and replays all subsequent inputs — regenerating
/// snapshots as it goes.  `predict_to` extrapolates the current state to an
/// arbitrary future time without touching the persistent history.
pub struct OosmEskfLos {
    /// The live filter state (always at `current_time` after each call).
    filter: EskfLos,
    /// Monotonically non-decreasing timestamp of the last committed step.
    current_time: Option<Instant>,
    /// Ring of post-step filter snapshots, oldest first.
    filter_snapshot: VecDeque<Timed<EskfLos>>,
    /// Ring of all inputs (predict + update), kept in chronological order.
    measurements: VecDeque<Timed<Measure>>,
    /// Maximum number of snapshots / inputs to retain.
    capacity: usize,
}

impl OosmEskfLos {
    /// Create a new wrapper with the given inner filter and ring-buffer capacity.
    ///
    /// `capacity` controls how far back OOSM rollback can reach.  At 100 Hz
    /// a capacity of 100 covers 1 second of history.
    pub fn new(filter: EskfLos, capacity: usize) -> Self {
        Self {
            filter,
            current_time: None,
            filter_snapshot: VecDeque::with_capacity(capacity),
            measurements: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn save_snapshot(&mut self, timestamp: Instant) {
        if self.filter_snapshot.len() == self.capacity {
            self.filter_snapshot.pop_front();
        }
        self.filter_snapshot.push_back(Timed {
            timestamp,
            data: self.filter.clone(),
        });
    }

    fn save_measurement_ordered(&mut self, input: Timed<Measure>) {
        if self.measurements.len() == self.capacity {
            self.measurements.pop_front();
        }
        // Insert in chronological order (binary search on timestamp).
        let pos = self
            .measurements
            .partition_point(|i| i.timestamp <= input.timestamp);
        self.measurements.insert(pos, input);
    }

    fn measurement_update(&mut self, input: &Timed<Measure>) {
        let dt = self.current_time.map_or(0.0, |t| {
            if input.timestamp > t {
                (input.timestamp - t).as_micros() as f32 * 1e-6
            } else {
                0.0
            }
        });

        self.filter.predict(dt);

        self.filter.update(&input.data);

        self.current_time = Some(input.timestamp);
        self.save_snapshot(input.timestamp);
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Fuse a new LOS measurement at its true capture timestamp.
    ///
    /// If the measurement is in-order (timestamp ≥ current_time) this is a
    /// simple predict+update forward step.  If it is out-of-sequence the
    /// wrapper rolls back to the nearest snapshot before `timestamp`, inserts
    /// the new measurement into the sorted input history, and replays all
    /// inputs from that point to the present.
    pub fn fuse(
        &mut self,
        timestamp: Instant,
        los_unit: Vector3<f32>,
        p_attitude: Matrix3<f32>,
        r_pixel: f32,
    ) {
        let timed = Timed {
            timestamp,
            data: Measure {
                los_unit,
                p_attitude,
                r_pixel,
            },
        };

        // Measurement is in sequence, fuse it normally
        if self.current_time.map_or(true, |t| timestamp > t) {
            self.measurement_update(&timed);
            self.measurements.push_back(timed);
            return;
        }

        // Insert the new measurement in chronological order.
        self.save_measurement_ordered(timed);

        // Find the latest snapshot before the measurement time.
        let rollback_idx = self
            .filter_snapshot
            .iter()
            .rposition(|s| s.timestamp < timestamp);

        // Roll back to the filter snapshot before the new measurement
        if let Some(idx) = rollback_idx {
            let snap = self.filter_snapshot[idx].clone();

            self.filter = snap.data;
            self.current_time = Some(snap.timestamp);
            self.filter_snapshot.truncate(idx + 1);

            // 3. Collect and replay every input strictly after the rollback point.
            //    We clone indices to avoid borrowing `self` while mutating it.
            let to_replay: Vec<Timed<Measure>> = self
                .measurements
                .iter()
                .filter(|i| i.timestamp > snap.timestamp)
                .cloned()
                .collect();

            for input in &to_replay {
                self.measurement_update(input);
            }
        } else {
            log::warn!("Measurement too old for OOSM, discarding.");
            return;
        }
    }

    /// Extrapolate the current filter state to `now` without modifying history.
    ///
    /// Use this from the controller tick to get the best current estimate
    /// without committing a coast step that would pollute future replays.
    pub fn predict_to(&self, now: Instant) -> (Vector3<f32>, Vector3<f32>) {
        let dt = self.current_time.map_or(0.0, |t| {
            if now > t {
                (now - t).as_micros() as f32 * 1e-6
            } else {
                0.0
            }
        });

        if dt < 1e-9 {
            return self.current_estimate();
        }

        // Clone and step — no side effects on `self`.
        let mut tmp = self.filter.clone();
        tmp.predict(dt);
        (tmp.los_hat, tmp.omega_hat)
    }

    /// The current committed estimate (at `current_time`, not extrapolated).
    pub fn current_estimate(&self) -> (Vector3<f32>, Vector3<f32>) {
        (self.filter.los_hat, self.filter.omega_hat)
    }
}
