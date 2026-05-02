use common::nalgebra::{Point2, Rotation3, UnitQuaternion, Vector3};

pub struct CameraSensor {
    pub width: f32,
    pub height: f32,
    pub f_x: f32,
    pub f_y: f32,
    pub c_x: f32,
    pub c_y: f32,
    pub pitch_offset: f32,
}

impl CameraSensor {
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
        target_pos: Vector3<f32>,
        focal_point: Vector3<f32>,
        att: UnitQuaternion<f32>,
    ) -> Option<(f32, f32)> {
        let los_global = target_pos - focal_point;
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
    pub fn pixel_to_global(
        &self,
        pixel: Point2<f32>,
        att: UnitQuaternion<f32>,
    ) -> Vector3<f32> {
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
    pub fn update(&mut self, measurement: Option<Vector3<f32>>, dt: f32) -> (Vector3<f32>, Vector3<f32>) {
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