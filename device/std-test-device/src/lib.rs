use std::{
    f32::consts::PI, sync::{
        LazyLock, atomic::{AtomicBool, Ordering}
    }
};

use clap::Parser;
use common::{nalgebra::{Matrix3, Point2, UnitQuaternion, UnitVector3, Vector3}, sync::watch::Watch, tasks::eskf::EskfEstimate};
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Timer};
use holsatus_sim::{Resources, Sim, SimHandle};
use rand_distr::{Distribution, Normal};
use tokio::runtime::Runtime;

use crate::resources::simulated_vicon;

pub mod lockstep;
pub mod resources;
pub mod los_estimator;

#[cfg(feature = "rerun")]
pub mod rerun_logger;

pub static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    let runtime = Runtime::new().expect("Unable to create tokio Runtime");
    Box::leak(Box::new(runtime.enter()));
    runtime
});

static RUNNING: AtomicBool = AtomicBool::new(true);

#[derive(clap::Parser)]
pub struct Args {
    /// Path to the configuration file for the simulation
    #[clap(default_value = "sim_config.toml")]
    #[clap(short, long)]
    pub config: String,
}

const SIM_FREQUENCY: u64 = 1000;

pub fn test_entry(
    limit_seconds: u64,
    #[allow(unused)]
    test_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _enter = RUNTIME.enter();

    let args = Args::parse();
    let config = holsatus_sim::config::load_from_file_path(&args.config)?;
    let (r, sitl) = holsatus_sim::initialize(config.clone())?;

    // Setup logging to run at 50 Hz
    #[cfg(feature = "rerun")]
    let mut logger = rerun_logger::setup(sitl.clone(), 10, test_name)?;

    // Sometimes rerun can take a split second to start receiving
    #[cfg(feature = "rerun")]
    std::thread::sleep(std::time::Duration::from_millis(500));

    let fw_sitl = sitl.clone();
    lockstep::lockstep_with(
        move |spawner| firmware_entry(spawner, r, fw_sitl),
        move || {
            #[cfg(feature = "rerun")]
            logger.log_subsampled().unwrap();

            assert!(Instant::now().as_secs() < limit_seconds);

            let step_size = embassy_time::Duration::from_hz(SIM_FREQUENCY);
            sitl.step(step_size.as_micros() as f32 * 1e-6);
            RUNNING.load(Ordering::Relaxed).then_some(step_size)
        },
    );

    // Sometimes dropping the handle early can result in lost recs
    #[cfg(feature = "rerun")]
    std::thread::sleep(std::time::Duration::from_millis(100));

    Ok(())
}

fn firmware_entry(spawner: Spawner, r: Resources, sim: SimHandle) {
    log::debug!("Firmware entry started");

    common::signals::CONTROL_FREQUENCY.store(SIM_FREQUENCY as u16, Ordering::Relaxed);

    // Might as well start the parameter storage module to get things loaded
    spawner.spawn(resources::param_storage(r.flash).unwrap());

    // ------------------ high-priority tasks -------------------

    // These take direct ownership of their hardware to avoid additional complexity
    spawner.spawn(resources::imu_reader(r.imu).unwrap());
    spawner.spawn(resources::motor_governor(r.motors).unwrap());

    spawner.spawn(common::tasks::rc_binder::main().unwrap());
    spawner.spawn(common::tasks::signal_router::main().unwrap());
    spawner.spawn(common::tasks::controller_rate::main().unwrap());

    // ----------------- medium-priority tasks ------------------

    spawner.spawn(common::tasks::commander::main().unwrap());
    spawner.spawn(common::tasks::att_estimator::main().unwrap());
    spawner.spawn(common::tasks::controller_angle::main().unwrap());

    // ------------------- Low-priority tasks -------------------

    spawner.spawn(common::tasks::calibrator::main().unwrap());
    spawner.spawn(common::tasks::arm_blocker::main().unwrap());
    spawner.spawn(common::tasks::eskf::main().unwrap());
    spawner.spawn(common::tasks::controller_mpc::main().unwrap());

    spawner.spawn(flight_test_task().unwrap());
    spawner.spawn(simulated_vicon(sim).unwrap());
}

#[embassy_executor::task]
async fn flight_test_task() {
    use common::tasks::commander::*;

    Timer::after_secs(1).await;

    log::warn!("Sending arming command");
    PROCEDURE
        .send(Request {
            command: Command::ArmDisarm {
                arm: true,
                force: true,
            }
            .into(),
            origin: Origin::Automatic,
        })
        .await;

    log::warn!("Sending control mode command");
    PROCEDURE
        .send(Request {
            command: Command::SetControlMode(ControlMode::Angle),
            origin: Origin::Automatic,
        })
        .await;

    log::debug!("================================================");
    log::debug!("============= Starting flight test =============");
    log::debug!("================================================");

    let mut rcv_eskf_estimate = common::signals::ESKF_ESTIMATE.receiver();
    let mut rcv_motors_state = common::signals::MOTORS_STATE.receiver();
    let mut snd_attitude_sp = common::signals::TRUE_ATTITUDE_Q_SP.sender();
    let mut snd_z_thrust_sp = common::signals::TRUE_Z_THRUST_SP.sender();
    rcv_motors_state.get_and(|state| state.is_armed()).await;

    log::info!("Thrusting upwards");
    
    for _ in 0..350 {
        snd_attitude_sp.send(UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0));
        snd_z_thrust_sp.send(20.0);
        Timer::after_millis(10).await;
    }

    for _ in 0..300 {
        snd_attitude_sp.send(UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0));
        snd_z_thrust_sp.send(4.0);
        Timer::after_millis(10).await;
    }

    snd_attitude_sp.send(UnitQuaternion::from_euler_angles(0.0, -0.4, 0.0));
    Timer::after_millis(1000).await;

    log::info!("Starting ProNav");

    let mut pronav = ProNav::new(0.64)
        .pronav_gain(4.0)
        .intnav_gain(1.0)
        .pursuit_gain(1.0)
        .velocity_gain(2.0)
        .velocity_target(30.0)
        .camera_pitch(35.0_f32.to_radians())
        .fov_limit(40.0_f32.to_radians())
        .fov_penalty_gain(5.0);

    let camera = los_estimator::CameraSensor::new(1440.0, 1080.0, 90.0_f32.to_radians(), 35.0_f32.to_radians());
    let mut ab_filter = los_estimator::AlphaBetaLos::new(0.05); // Tuned down to suppress quantization noise from camera resolution

    let delta = Duration::from_hz(60);
    let dt = delta.as_micros() as f32 * 1e-6;

    let pos_gen = |index| {
        let x = 600.0 - dt * index as f32 * 20.0;
        let y = 100.0 - dt * index as f32 * 20.0;
        Vector3::new(x, y, 0.0)
    };

    let pixel_disr = Normal::new(0.0, 2.0).unwrap();
    let mut rng = rand::rng();
    
    let mut index = 0;
    let mut break_index = usize::MAX;
    while index < break_index {
        let estimate = rcv_eskf_estimate.get().await;

        let mut target_pos = pos_gen(index);
        let target_vel = (target_pos - pos_gen(index - 1)) * dt.recip();

        TARGET_POSE.send((target_pos.into(), target_vel.into()));

        // Raise target pos artificially for now
        target_pos[2] -= 2.0;
        
        if (estimate.pos - target_pos).norm() < 2.0 && break_index == usize::MAX {
            log::info!("Target struck!");
            break_index = index + 10;
        }

        // 1. Vision emulation: Get target in global space, find its pixel
        let focal_point = estimate.pos + estimate.att.transform_vector(&[0.1, 0.0, 0.0].into());
        let exact_pixel = camera.project_to_pixel(target_pos, focal_point, estimate.att);

        // Add noise to sensor coordinate
        let noisy_pixel = exact_pixel.map(|(u, v)| (u + pixel_disr.sample(&mut rng) , v  + pixel_disr.sample(&mut rng)) );

        // 2. Vision processing: Convert back to a measured LOS and run through the AB filter
        let measured_los = noisy_pixel.map(|p| camera.pixel_to_global(Point2::new(p.0, p.1), estimate.att));
        let (los_unit, los_rate) = ab_filter.update(measured_los, dt);

        if noisy_pixel.is_none() {
            log::warn!("Target lost! Dead reckoning LOS.");
        }

        // 3. Guidance: Update ProNav with our filtered LOS estimations
        let (att, force) = pronav.update(target_pos, los_unit, los_rate, estimate, dt);

        // Ensure we have attitude authority
        let force = force.min(25.0);

        snd_attitude_sp.send(att);
        snd_z_thrust_sp.send(force);

        Timer::after(delta).await;
        index += 1;
    }

    RUNNING.store(false, Ordering::Relaxed);

    // Start LOS-rate control
    log::warn!("Sending disarm command");
    PROCEDURE
        .send(Request {
            command: Command::ArmDisarm {
                arm: false,
                force: true,
            }
            .into(),
            origin: Origin::Automatic,
        })
        .await;

    Timer::after_secs(5).await;

    RUNNING.store(false, Ordering::Relaxed);
}

pub static TARGET_POSE: Watch<([f32; 3], [f32; 3])> = Watch::new();

enum FlightPhase {
    Cruise,
    Terminal,
}

struct ProNav {
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
            fov_limit: PI / 4.0,
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
        target_pos: Vector3<f32>,
        los_unit: Vector3<f32>,
        los_unit_vel: Vector3<f32>,
        estimate: EskfEstimate,
        dt: f32
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

        // Global target velocity [m/s]
        let target_vel = self.prev_target.map(|prev_target| {
            (target_pos - prev_target) / dt.max(1e-4)
        }).unwrap_or_default();
        self.prev_target = Some(target_pos);

        // Relative velocity [m/s]
        let relative_vel = estimate.vel - target_vel ;

        // Closing Velocity [m/s]
        let closing_vel = los_unit.dot(&relative_vel);

        // Acceleration contribution of the PN guidance law [m/s^2]
        let pronav_accel = self.pronav_gain * closing_vel * los_unit_vel;

        match self.phase {
            FlightPhase::Cruise => {
                let mut los_unit_vel_cruise = los_unit_vel;
                los_unit_vel_cruise.z = 0.0;
                self.los_vel_integral += los_unit_vel_cruise * dt;
            },
            FlightPhase::Terminal => {
                self.los_vel_integral += los_unit_vel * dt;
            },
        }

        let intnav_accel = self.intnav_gain * closing_vel * self.los_vel_integral;

        // Calculate the alignment between where we are going and where the target is
        let vel_unit = estimate.vel.try_normalize(1e-3).unwrap_or(los_unit);
        let vel_los_alignment = vel_unit.dot(&los_unit).max(0.0);

        // Acceleration contribution of alisnment-based velocity law [m/s^2]
        let velocity_error = self.velocity_target - estimate.vel.norm();
        let axial_accel = velocity_error * self.velocity_gain * vel_los_alignment * los_unit.normalize();

        // Acceleration contribution of pure pursuit [m/s^2]
        let pursuit_accel = self.pursuit_gain * los_unit;

        // Desired scceleration [m/s^2]
        let mut desired_accel = pronav_accel + intnav_accel + axial_accel + pursuit_accel;

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
        let boresight_global = estimate.att.transform_vector(&boresight_body);
        
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
                let penalty_quat = UnitQuaternion::from_axis_angle(&UnitVector3::new_unchecked(rot_axis_unit), penalty_angle);
                global_accel_target = penalty_quat * global_accel_target;
            }
        }

        let desired_thrust_dir = global_accel_target.try_normalize(1e-6).unwrap_or(-Vector3::z());

        // Determine the alignment between the desired attitude and the current one.
        // Use that to scale down the force target while ill-aligned
        let direction = estimate.att.transform_vector(&-Vector3::z());
        let alignment_factor = direction.dot(&desired_thrust_dir).clamp(0.0, 1.0);

        // Thrust (body -Z) points toward desired_thrust_dir
        let b_z = -desired_thrust_dir;

        // Nose (Body +X) should point toward target
        let los_norm = los_unit; 
        let b_y = b_z.cross(&los_norm).try_normalize(1e-6).unwrap_or_else(|| {
            b_z.cross(&Vector3::x()).try_normalize(1e-6).unwrap_or(Vector3::y())
        });

        // Compute the last orthonormal basis vector
        let b_x = b_y.cross(&b_z);

        // Create rotation matrix from basis vectors [x, y, z]
        let rotation_matrix = Matrix3::from_columns(&[b_x, b_y, b_z]);
        let att_target = UnitQuaternion::from_matrix(&rotation_matrix); 

        // Compensate for additional air resistance at higher speeds
        let comp_accel_target = global_accel_target.norm() * (1.0 + (estimate.vel.norm() * self.velocity_comp).powi(2));
        let force_target = self.drone_mass * comp_accel_target * alignment_factor;

        (att_target, force_target)
    }
}
