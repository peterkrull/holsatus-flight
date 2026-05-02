use std::sync::{
    atomic::{AtomicBool, Ordering},
    LazyLock,
};

use clap::Parser;
use common::{
    embassy_futures::select::{Either, select}, nalgebra::{Point2, UnitQuaternion, Vector3}, sync::{channel::Channel, watch::Watch}, tasks::eskf::EskfEstimate
};
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Ticker, Timer};
use holsatus_sim::{Resources, Sim, SimHandle};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use tokio::runtime::Runtime;

use crate::resources::simulated_vicon;

pub mod lockstep;
pub mod pronav;
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

const SIM_FREQUENCY: u64 = 1000;

pub fn test_entry(
    limit_seconds: u64,
    #[allow(unused)] test_name: &str,
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

    let mut pronav = pronav::ProNav::new(0.64)
        .pronav_gain(4.0)
        .intnav_gain(1.0)
        .pursuit_gain(1.0)
        .velocity_gain(2.0)
        .velocity_target(30.0)
        .camera_pitch(35.0_f32.to_radians())
        .fov_limit(40.0_f32.to_radians())
        .fov_penalty_gain(5.0);

    let mut ab_filter = pronav::AlphaBetaLos::new(0.05); // Tuned down to suppress quantization noise from camera resolution

    let delta = Duration::from_hz(100);
    let dt = delta.as_micros() as f32 * 1e-6;

    let mut ticker = Ticker::every(delta);

    loop {

        match select(EVENTS.receive(), ticker.next()).await {
            Either::First(event) => {
                match EVENTS.receive().await {
                    Event::Camera {
                        timestamp,
                        target_pixel,
                    } => {
                        let attitude = ; // Interpolate / estimate attitude at timestamp
                        let measured_los = CAMERA.pixel_to_global(target_pixel, attitude);
                        let (los_unit, los_rate) = ab_filter.update(Some(measured_los), dt);
                    }
                    Event::Attitude {
                        timestamp,
                        attitude,
                        angular_vel,
                    } => {}
                    Event::Quit => break,
                }
            }
            Either::Second(()) => {
                let closing_vel = 30.0; // Assume we have an air speed sensor
                let attitude = ; // Interpolate / estimate attitude at timestamp
                let (att, force) = pronav.update(closing_vel, los_unit, los_rate, attitude, dt);

                // Ensure we have attitude authority
                let force = force.min(25.0);

                snd_attitude_sp.send(att);
                snd_z_thrust_sp.send(force);
            }
        }
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

const CAMERA: LazyLock<pronav::CameraModel> = LazyLock::new(|| {
    pronav::CameraModel::new(1440.0, 1080.0, 90.0_f32.to_radians(), 35.0_f32.to_radians())
});

pub enum Event {
    Quit,
    Camera {
        timestamp: Instant,
        target_pixel: Point2<f32>,
    },
    Attitude {
        timestamp: Instant,
        attitude: UnitQuaternion<f32>,
        angular_vel: Vector3<f32>,
    },
}

static EVENTS: Channel<Event, 2> = Channel::new();

#[embassy_executor::task]
async fn camera_task() {
    let mut rcv_eskf_estimate = common::signals::ESKF_ESTIMATE.receiver();

    let delta = Duration::from_hz(50);
    let dt = delta.as_micros() as f32 * 1e-6;

    let pos_gen = |index| {
        let x = 600.0 - dt * index as f32 * 20.0;
        let y = 100.0 - dt * index as f32 * 20.0;
        Vector3::new(x, y, 0.0)
    };

    let pixel_disr = Normal::new(0.0, 2.0).unwrap();
    let mut rng = rand::rng();

    let mut index = 0;
    let mut ticker = Ticker::every(delta);
    loop {
        ticker.next().await;

        let estimate = rcv_eskf_estimate.get().await;

        let mut target_pos = pos_gen(index);
        let target_vel = (target_pos - pos_gen(index - 1)) * dt.recip();

        // Publish so visualization can show target
        TARGET_POSE.send((target_pos.into(), target_vel.into()));

        // Raise target pos artificially for better centering
        target_pos[2] -= 2.0;

        if (estimate.pos - target_pos).norm() < 2.0 {
            log::info!("Target struck!");
            EVENTS.send(Event::Quit).await;
            break;
        }

        // 1. Vision emulation: Get target in global space, find its pixel
        let focal_point = estimate.pos + estimate.att.transform_vector(&[0.1, 0.0, 0.0].into());
        if let Some(exact_pixel) = CAMERA.project_to_pixel(target_pos - focal_point, estimate.att) {
            let noisy_pixel = Point2::new(
                exact_pixel.0 + pixel_disr.sample(&mut rng),
                exact_pixel.1 + pixel_disr.sample(&mut rng),
            );

            let event = Event::Camera {
                timestamp: Instant::now(),
                target_pixel: noisy_pixel,
            };

            // Add time delay and random jitter: 30 + 0..10 ms
            Timer::after_micros(30_000 + rng.next_u64() % 10_000).await;

            EVENTS.send(event).await;
        }

        index += 1;
    }
}

#[embassy_executor::task]
async fn attitude_task() {
    let mut rcv_eskf_estimate = common::signals::ESKF_ESTIMATE.receiver();

    let mut rng = rand::rng();

    let delta = Duration::from_hz(100);
    let mut ticker = Ticker::every(delta);

    loop {
        ticker.next().await;

        let estimate = rcv_eskf_estimate.get().await;

        let event = Event::Attitude {
            timestamp: Instant::from_micros(estimate.timestamp_us),
            attitude: estimate.att.into(),
            angular_vel: estimate.ang_vel.into(),
        };

        // Add time delay and random jitter: 10 + 0..2 ms
        Timer::after_micros(10_000 + rng.next_u64() % 2_000).await;

        EVENTS.send(event).await;
    }
}

pub static TARGET_POSE: Watch<([f32; 3], [f32; 3])> = Watch::new();
