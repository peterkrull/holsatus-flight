use std::{collections::VecDeque, f32::consts::PI};

use common::nalgebra::{self, Rotation3, UnitQuaternion, Vector3};
use embassy_time::Instant;
use holsatus_sim::{Sim, SimHandle};
use rerun::{Arrows3D, Color, LineStrips3D, Points3D, RecordingStream, Scalars, Vec3D, components::RotationQuat};

pub fn setup(
    handle: SimHandle,
    subsample: usize,
    test_name: &str,
) -> Result<RerunLogger, Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new(test_name).spawn()?;

    rerun::Logger::new(rec.clone())
        .with_filter("off, common=debug, holsatus_sim=debug, std_device=debug")
        .init()?;

    RerunLogger::new(rec, handle, subsample)
}

pub struct RerunLogger {
    pub rec: RecordingStream,
    pos_trail: VecDeque<Vec3D>,
    trail_len: usize,
    handle: SimHandle,
    subsample_cfg: usize,
    subsample_count: usize,
}

impl RerunLogger {
    pub fn new(
        rec: RecordingStream,
        handle: SimHandle,
        subsample: usize
    ) -> Result<Self, Box<dyn std::error::Error>> {
        rec.set_time("embassy-time", std::time::Duration::from_nanos(0));
        rec.log_static("/", &rerun::ViewCoordinates::FRD())?;

        let drone = rerun::Asset3D::from_file_path("sprinter.gltf")?;
        rec.log("target", &drone)?;

        let cam = rerun::Pinhole::from_fov_and_aspect_ratio(110.0f32.to_radians(), 16.0 / 9.0).with_image_plane_distance(0.5);
        rec.log("target/camera", &cam)?;

        let cam_rot = nalgebra::UnitQuaternion::from_euler_angles(3.0 * PI/2.0, 0.0, PI);
        rec.log("target/camera", &rerun::Transform3D::from_translation([0.0, -1.7, 2.0]).with_quaternion(cam_rot.coords.data.0[0]))?;

        let drone = rerun::Asset3D::from_file_path("fpv-drone-2.gltf")?;
        rec.log("drone", &drone)?;

        let cam = rerun::Pinhole::from_fov_and_aspect_ratio(90.0f32.to_radians(), 4.0 / 3.0).with_image_plane_distance(0.5);
        rec.log("drone/camera", &cam)?;

        let camera_pitch = 35.0f32.to_radians();
        let cam_rot = nalgebra::UnitQuaternion::from_euler_angles(PI/2.0 + camera_pitch, 0.0, PI/2.0);
        rec.log("drone/camera", &rerun::Transform3D::from_translation([0.1, 0.0, 0.0]).with_quaternion(cam_rot.coords.data.0[0]))?;

        Ok(RerunLogger {
            rec,
            pos_trail: VecDeque::with_capacity(1000),
            trail_len: 1000,
            handle,
            subsample_cfg: subsample,
            subsample_count: 0,
        })
    }

    pub fn log_subsampled(&mut self) -> Result<(), Box<dyn std::error::Error>> {

        self.subsample_count += 1;
        if self.subsample_count < self.subsample_cfg {
            return Ok(())
        }
        self.subsample_count = 0;

        self.log_now()
    }


    pub fn log_now(&mut self) -> Result<(), Box<dyn std::error::Error>> {

        let state = self.handle.vehicle_state();
        let pos = state.position;
        let rot = state.rotation;

        if self.pos_trail.len() >= self.trail_len {
            _ = self.pos_trail.pop_back();
        }
        self.pos_trail.push_front(Vec3D(pos.into()));

        self.rec.set_time(
            "embassy-time",
            std::time::Duration::from_nanos(Instant::now().as_nanos()),
        );

        log_drone_motors(&self.rec, self.handle.clone())?;

        if let Some(imu_data) = common::signals::CAL_MULTI_IMU_DATA[0].try_get() {
            self.rec
                .log("sim/firmware/acc", &Scalars::new(imu_data.acc))?;
            self.rec
                .log("sim/firmware/gyr", &Scalars::new(imu_data.gyr))?;
        }

        if let Some((target_pos, target_vel)) = crate::TARGET_POSE.try_get() {
            let yaw = -target_vel[0].atan2(target_vel[1]);
            let rotation = UnitQuaternion::from_euler_angles(PI, 0.0, yaw);


            self.rec.log(
                "target",
                &rerun::Transform3D::from_translation(target_pos).with_quaternion(rotation.coords.data.0[0])
            )?;
        }
        
        if let Some(gyr_data) = common::signals::COMP_FUSE_GYR.try_get() {
            self.rec
                .log("sim/firmware/gyr_comp", &Scalars::new(gyr_data))?;
        }
        
        if let Some(estimate) = common::signals::ESKF_ESTIMATE.try_get() {
            self.rec.log(
                "sim/firmware/eskf/pos",
                &Scalars::new(estimate.pos.map(|x| x as f64).iter().cloned()),
            )?;

            self.rec.log(
                "sim/firmware/eskf/vel",
                &Scalars::new(estimate.vel.map(|x| x as f64).iter().cloned()),
            )?;

            let (roll, pitch, yaw) = estimate.att.euler_angles();
            self.rec.log(
                "sim/firmware/eskf/att",
                &Scalars::new([roll as f64, pitch as f64, yaw as f64]),
            )?;

            self.rec.log(
                "sim/firmware/eskf/gyr_bias",
                &Scalars::new(estimate.gyr_bias.map(|x| x as f64).iter().cloned()),
            )?;

            self.rec.log(
                "sim/firmware/eskf/acc_bias",
                &Scalars::new(estimate.acc_bias.map(|x| x as f64).iter().cloned()),
            )?;
        }

        if let Some(rate_sp) = common::signals::TRUE_RATE_SP.try_get() {
            self.rec
                .log("sim/firmware/rate_sp", &Scalars::new(rate_sp))?;
        }

        if let Some(attitude_q_sp) = common::signals::TRUE_ATTITUDE_Q_SP.try_get() {
            let (roll, pitch, yaw) = attitude_q_sp.euler_angles();
            self.rec
                .log("sim/firmware/angl_sp", &Scalars::new([roll, pitch, yaw]))?;
        }

        if let Some(rate_sp) = common::tasks::controller_rate::RATE_REF_FILTERED.try_get() {
            self.rec
                .log("sim/firmware/rate_ref_filt", &Scalars::new(rate_sp))?;
        }

        if let Some(rate_sp) = common::tasks::controller_rate::RATE_FF_PREDICT.try_get() {
            self.rec
                .log("sim/firmware/rate_ff_pred", &Scalars::new(rate_sp))?;
        }

        if let Some(rate_sp) = common::tasks::controller_rate::RATE_COMP_FUSE.try_get() {
            self.rec
                .log("sim/firmware/rate_comp_fuse", &Scalars::new(rate_sp))?;
        }

        if let Some(rate_sp) = common::signals::AHRS_ATTITUDE.try_get() {
            self.rec
                .log("sim/firmware/ahrs_attitude", &Scalars::new(rate_sp))?;
        }

        if let Some([pid_x, pid_y, pid_z]) = common::signals::RATE_PID_TERMS.try_get() {
            self.rec.log(
                "sim/firmware/rate_pid/x/p",
                &Scalars::single(pid_x.p_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/x/i",
                &Scalars::single(pid_x.i_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/x/dr",
                &Scalars::single(pid_x.dr_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/x/dm",
                &Scalars::single(pid_x.dm_out as f64),
            )?;

            self.rec.log(
                "sim/firmware/rate_pid/y/p",
                &Scalars::single(pid_y.p_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/y/i",
                &Scalars::single(pid_y.i_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/y/dr",
                &Scalars::single(pid_y.dr_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/y/dm",
                &Scalars::single(pid_y.dm_out as f64),
            )?;

            self.rec.log(
                "sim/firmware/rate_pid/z/p",
                &Scalars::single(pid_z.p_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/z/i",
                &Scalars::single(pid_z.i_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/z/dr",
                &Scalars::single(pid_z.dr_out as f64),
            )?;
            self.rec.log(
                "sim/firmware/rate_pid/z/dm",
                &Scalars::single(pid_z.dm_out as f64),
            )?;
        }
        if let Some(quat) = common::signals::AHRS_ATTITUDE_Q.try_get() {
            let (x, y, z) = quat.euler_angles();
            self.rec
                .log("sim/firmware/ahrs_att", &Scalars::new([x, y, z]))?;
        }

        self.rec
            .log("sim/position/trail", &LineStrips3D::new([&self.pos_trail]))
            .unwrap();

        self.rec
            .log("sim/position", &Scalars::new(pos.data.0[0]))
            .unwrap();

        let (roll, pitch, yaw) = rot.euler_angles();
        self.rec
            .log("sim/rotation", &Scalars::new([roll, pitch, yaw]))
            .unwrap();

        self.rec.log(
            "drone",
            &rerun::Transform3D::from_translation(pos.data.0[0])
                .with_quaternion(rot.coords.data.0[0]),
        )?;

        if let Some((target_pos, _)) = crate::TARGET_POSE.try_get() {
            // Determine the world-position of the drone-cameras focal point (0.1 in front of center)
            let focal_point = pos + rot.transform_vector(&[0.1, 0.0, 0.0].into());
            let line_of_sight = rerun::LineStrip3D::from_iter([target_pos, focal_point.data.0[0]]);

            self.rec.log(
                "line_of_sight",
                &LineStrips3D::new([line_of_sight]),
            )?;
        }


        Ok(())
    }
}

pub fn log_drone_motors(
    rec: &RecordingStream,
    handle: SimHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    // Make the arrows nice colors
    let make_color = |s: f32| -> Color {
        let g = ((3.5 - s) * 400.0) as u8;
        let r = ((s - 1.0) * 300.0) as u8;
        Color::from_rgb(r, g, 0)
    };

    // Combine the parameters and states of the motors
    let motors = handle
        .vehicle_params()
        .motors
        .iter()
        .cloned()
        .zip(handle.vehicle_state().motors)
        .collect::<Vec<_>>();

    rec.log(
        "drone/motor/force",
        &Scalars::new(motors.iter().map(|motors| motors.1.force)),
    )?;

    rec.log(
        "drone/motor/command",
        &Scalars::new(motors.iter().map(|motors| motors.1.command)),
    )?;

    let mut arrow_lengths = Vec::with_capacity(motors.len());
    let mut arrow_origins = Vec::with_capacity(motors.len());
    let mut arrow_colors = Vec::with_capacity(motors.len());

    for (params, state) in motors.iter() {
        // Shortening by 5 seems to work fine for small quads
        let direction = params.normal.normalize() * state.force / 5.0;

        arrow_lengths.push(Vec3D(direction.data.0[0]));
        arrow_origins.push(Vec3D(params.position.data.0[0]));
        arrow_colors.push(make_color(state.command));
    }

    rec.log(
        "drone/motors",
        &Arrows3D::from_vectors(arrow_lengths)
            .with_origins(arrow_origins)
            .with_colors(arrow_colors)
            .with_radii([0.01; 4]),
    )?;

    Ok(())
}
