use std::{
    f32::consts::PI, sync::{
        LazyLock, atomic::{AtomicBool, Ordering}
    }
};

use clap::Parser;
use common::{consts::GRAVITY, nalgebra::{Matrix3, UnitQuaternion, UnitVector3, Vector3}, sync::watch::Watch, tasks::eskf::EskfEstimate};
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Timer};
use holsatus_sim::{Resources, Sim, SimHandle};
use tokio::runtime::Runtime;

use crate::resources::simulated_vicon;

pub mod lockstep;
pub mod resources;

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

const SIM_FREQUENCY: u64 = 500;

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
    std::thread::sleep(std::time::Duration::from_millis(100));

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

fn millis_in_future(millis: u64) -> common::embassy_time::Instant {
    let now = common::embassy_time::Instant::now();
    now + common::embassy_time::Duration::from_millis(millis)
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
    
    for _ in 0..450 {
        snd_attitude_sp.send(UnitQuaternion::from_euler_angles(0.0, -0.2, 0.0));
        snd_z_thrust_sp.send(20.0);
        Timer::after_millis(10).await;
    }

        for _ in 0..300 {
        snd_attitude_sp.send(UnitQuaternion::from_euler_angles(0.0, -0.0, 0.0));
        snd_z_thrust_sp.send(4.0);
        Timer::after_millis(10).await;
    }

    log::info!("Starting ProNav");

    let mut pronav = ProNav::new(0.64)
        .pronav_gain(8.0)
        .pursuit_gain(4.0)
        .velocity_gain(1.0)
        .velocity_target(30.0)
        .velocity_comp(0.025);

    let delta = Duration::from_hz(100);

    let pos_gen = |index| Vector3::new((index as f32 / 185.0).sin() * 50.0 -200.0, index as f32 / 5.0 -400.0, 0.0);

    let mut index = 0;
    loop {
        let estimate = rcv_eskf_estimate.get().await;
        let target_pos = pos_gen(index);
        let target_vel = (target_pos - pos_gen(index - 1)) * delta.as_micros() as f32 * 1e6 ;

        TARGET_POSE.send((target_pos.into(), target_vel.into()));
        
        if (estimate.pos - target_pos).norm() < 2.0 {
            log::info!("Target struck!");
            RUNNING.store(false, Ordering::Relaxed);
            break;
        }

        let (att, force) = pronav.update(target_pos, estimate, delta.as_micros() as f32 * 1e-6);

        snd_attitude_sp.send(att);
        snd_z_thrust_sp.send(force);

        Timer::after(delta).await;
        index += 1;
    }

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

struct ProNav {
    /// Mass of the drone in kilo-grams [kg]
    drone_mass: f32,
    /// The gain of the ProNav control law
    pronav_gain: f32,
    /// The gain to encourage pure pursuit
    pursuit_gain: f32,
    /// The gain to push the target along the PN desired direction
    velocity_gain: f32,
    velocity_target: f32,
    velocity_comp: f32,
    prev_los_unit: Option<Vector3<f32>>,
    prev_target: Option<Vector3<f32>>,
}

impl ProNav {
    pub const fn new(drone_mass: f32) -> Self {
        Self {
            drone_mass,
            pronav_gain: 5.0,
            pursuit_gain: 0.0,
            velocity_gain: 0.0,
            velocity_target: 35.0,
            velocity_comp: 0.0,
            prev_los_unit: None,
            prev_target: None,
        }
    }

    pub const fn pronav_gain(mut self, pronav_gain: f32) -> Self {
        self.pronav_gain = pronav_gain.max(0.0);
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

    pub const fn velocity_comp(mut self, velocity_comp: f32) -> Self {
        self.velocity_comp = velocity_comp;
        self
    }

    pub fn update(&mut self, mut target_pos: Vector3<f32>, estimate: EskfEstimate, dt: f32) -> (UnitQuaternion<f32>, f32) {

        // =============================================================
        // This section applies a fairly standard "True ProNav" strategy
        // =============================================================
        
        // The global position of the camera focal point
        let focal_point = estimate.pos + estimate.att.transform_vector(&[0.1, 0.0, 0.0].into());

        // Raise target pos artificially for now
        target_pos[2] -= 2.0;

        // LOS Unit Vector [-]
        let los_unit = (target_pos - focal_point).normalize();
        
        // LOS Rate Vector [rad/s]
        let los_unit_vel = self.prev_los_unit.map(|prev_los_unit| {
            (los_unit - prev_los_unit) / dt.max(1e-4)
        }).unwrap_or_default();
        self.prev_los_unit = Some(los_unit);

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

        // Calculate the alignment between where we are going and where the target is
        let vel_unit = estimate.vel.try_normalize(1e-3).unwrap_or(los_unit);
        let vel_los_alignment = vel_unit.dot(&los_unit).max(0.0);

        // Acceleration contribution of alisnment-based velocity law [m/s^2]
        let velocity_error = self.velocity_target - estimate.vel.norm();
        let axial_accel = velocity_error * self.velocity_gain * vel_los_alignment * los_unit;

        // Acceleration contribution of pure pursuit [m/s^2]
        let pursuit_accel = self.pursuit_gain * los_unit;

        // Desired scceleration [m/s^2]
        let desired_accel = pronav_accel + axial_accel + pursuit_accel;

        // =================================================================
        // This section converts the desired accel into an attitude + thrust
        // =================================================================

        // Add gravity back into the desired accel
        let gravity_vector = Vector3::z() * 9.81;
        let global_accel_target = desired_accel - gravity_vector;
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

struct AlphaBetaLos {
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

    /// Update the filter with a new LOS-vector measurement.
    /// 
    /// Returns the unit vector, and its rate of change.
    pub fn update(&mut self, measurement: Vector3<f32>, dt: f32) -> (Vector3<f32>, Vector3<f32>) {
        // Predict the value based on prior state and velocity
        let val_pred = self.unit_est.map(|unit_est| {
            unit_est + (self.rate_est * dt)
        }).unwrap_or(measurement);
        
        // The residual between measurement and predicted value
        let residual = measurement - val_pred;
        
        // Do alpha-beta filtering step to update both value and velocity
        let unit_est = (val_pred + (self.alpha * residual)).normalize();
        self.rate_est = self.rate_est + (self.beta * residual) / dt;
        
        // Keep the unit vector a unit vector
        self.unit_est = Some(unit_est);

        (unit_est, self.rate_est)
    }
}