use std::sync::{
    atomic::{AtomicBool, Ordering},
    LazyLock,
};

use clap::Parser;
use common::{
    embassy_futures::select::{select, Either},
    nalgebra::{Matrix3, Point2, SMatrix, UnitQuaternion, Vector3},
    sync::{channel::Channel, watch::Watch},
};
use embassy_executor::{SendSpawner, Spawner};
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
static PRONAV_READY: Watch<bool> = Watch::new();

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
    spawner.spawn(camera_task().unwrap());
    spawner.spawn(attitude_task().unwrap());
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
        snd_z_thrust_sp.send(25.0);
        Timer::after_millis(20).await;
    }

    log::info!("Starting ProNav");

    // Signal producer tasks that the consumer loop is ready
    PRONAV_READY.send(true);

    let mut pronav = pronav::ProNav::new(0.64)
        .pronav_gain(3.0)
        .intnav_gain(1.0)
        .pursuit_gain(1.0)
        .velocity_gain(1.0)
        .velocity_target(30.0)
        .camera_pitch(CAMERA.pitch_rad)
        .fov_limit(CAMERA.vfov_rad / 3.0)
        .fov_penalty_gain(10.0);

    let q_cov = SMatrix::from_diagonal(&[1e-4, 1e-4, 1e-2, 1e-2].into());
    let filter = pronav::EskfLos::new(q_cov);
    let mut eskf_filter = pronav::OosmEskfLos::new(filter, 100);

    let mut att_buffer = pronav::AttitudeBuffer::new(100); // ~2 seconds at 100 Hz

    let delta = Duration::from_hz(100);
    let dt = delta.as_micros() as f32 * 1e-6;

    let mut ticker = Ticker::every(delta);

    loop {
        match select(EVENTS.receive(), ticker.next()).await {
            Either::First(event) => {
                match event {
                    Event::Camera {
                        timestamp,
                        target_pixel,
                    } => {
                        // Look up attitude at the camera's *capture* timestamp
                        if let Some(att_at_capture) = att_buffer.interpolate_at(timestamp) {
                            let body_los = CAMERA.pixel_to_los_body(target_pixel);
                            let global_los = att_at_capture.transform_vector(&body_los);
                            let p_attitude = SMatrix::identity() * 1e-4;
                            let r_pixel = 1e-3; // Variance for 5 pixel std dev
                            eskf_filter.fuse(timestamp, global_los, p_attitude, r_pixel);
                        } else {
                            log::error!(
                                "Failed to interpolate attitude at {} us (buffer size: {})",
                                timestamp.as_micros(),
                                att_buffer.entries.len()
                            );
                        }
                    }
                    Event::Attitude {
                        timestamp,
                        attitude,
                        angular_vel,
                    } => {
                        att_buffer.push(timestamp, attitude, angular_vel);
                    }
                }
            }
            Either::Second(()) => {
                let now = Instant::now();

                // Get filter estimate extrapolated to the current time
                let (los_unit, los_rate) = eskf_filter.predict_to(now);

                let estimate = rcv_eskf_estimate.get().await;

                // Visualize the image LOS observation
                let los_vector_local = estimate.att.inverse_transform_vector(&los_unit) + CAM_TRANS;

                LOS_VECTOR.send((los_vector_local.into(), los_rate.into()));

                // Get latest attitude for the controller
                let attitude = att_buffer
                    .interpolate_at(now)
                    .unwrap_or_else(UnitQuaternion::identity);

                let closing_vel = estimate.vel.norm(); // Assume we have an air speed sensor
                let (att, force) = pronav.update(closing_vel, los_unit, los_rate, attitude, dt);

                // Ensure we have attitude authority
                let force = force.min(25.0);

                snd_attitude_sp.send(att);
                snd_z_thrust_sp.send(force);
            }
        }
    }
}

pub const CAM_TRANS: Vector3<f32> = Vector3::new(0.1, 0.0, 0.0);
pub static LOS_VECTOR: Watch<([f32; 3], [f32; 3])> = Watch::new();
pub static TARGET_POSE: Watch<([f32; 3], [f32; 3])> = Watch::new();

const CAMERA: LazyLock<pronav::CameraModel> = LazyLock::new(|| {
    pronav::CameraModel::new(1440.0, 1080.0, 70.0_f32.to_radians(), 25.0_f32.to_radians())
});

pub enum Event {
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
    // Wait until the ProNav loop is ready to consume events
    PRONAV_READY.receiver().get().await;

    let mut rcv_eskf_estimate = common::signals::ESKF_ESTIMATE.receiver();

    let delta = Duration::from_hz(50);
    let dt = delta.as_micros() as f32 * 1e-6;

    let trajectory = Trajectory::new(vec![
        Waypoint {
            time: 0.0,
            pos: Vector3::new(1000.0, 0.0, 0.0),
        },
        Waypoint {
            time: 10.0,
            pos: Vector3::new(600.0, 0.0, 0.0),
        },
        Waypoint {
            time: 10.0,
            pos: Vector3::new(400.0, -100.0, 0.0),
        },
        Waypoint {
            time: 8.0,
            pos: Vector3::new(200.0, 100.0, 0.0),
        },
        Waypoint {
            time: 20.0,
            pos: Vector3::new(100.0, 200.0, 0.0),
        },
    ]);

    let pixel_disr = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let mut elapsed_time = 0.0;
    let mut break_time = f32::MAX;
    let mut ticker = Ticker::every(delta);
    let mut prev_target_pos = trajectory.smoothed_position_at(0.0, 1.0);
    loop {
        ticker.next().await;

        // Exact target position and velocity
        let target_pos = trajectory.smoothed_position_at(elapsed_time, 1.0);
        let target_vel = (target_pos - prev_target_pos) / dt;
        prev_target_pos = target_pos;

        // Publish so visualization can show target
        TARGET_POSE.send((target_pos.into(), target_vel.into()));

        // Raise target pos artificially for better centering
        let target_pos = target_pos - Vector3::z() * 2.0;

        // Determine pixel which corresponds to the target in the viewport
        let estimate = rcv_eskf_estimate.get().await;
        let focal_point = estimate.pos + estimate.att.transform_vector(&CAM_TRANS);
        let los_global = target_pos - focal_point;
        let los_body = estimate.att.inverse_transform_vector(&los_global);
        if let Some(exact_pixel) = CAMERA.project_to_pixel(los_body) {
            let noisy_pixel = Point2::new(
                exact_pixel.x + pixel_disr.sample(&mut rng),
                exact_pixel.y + pixel_disr.sample(&mut rng),
            );

            // Visualize the noisy image LOS observation
            LOS_VECTOR_LOCAL.send(CAMERA.pixel_to_los_body(noisy_pixel) + CAM_TRANS);

            let event = Event::Camera {
                timestamp: Instant::now(),
                target_pixel: noisy_pixel,
            };

            let sleep_dur = Duration::from_micros(50_000 + rng.next_u64() % 10_000);
            if let Ok(task_handle) = delayed_event_task(event, sleep_dur) {
                let spawner = SendSpawner::for_current_executor().await;
                spawner.spawn(task_handle);
            } else {
                log::error!("Failed to spawn delayed_event_task");
            }
        }

        if los_global.norm() < 2.0 && break_time == f32::MAX {
            log::info!("Target struck!");
            break_time = elapsed_time + 0.1;
        }

        if elapsed_time > break_time {
            RUNNING.store(false, Ordering::Relaxed)
        }

        elapsed_time += dt;
    }
}

static LOS_VECTOR_LOCAL: Watch<Vector3<f32>> = Watch::new();

#[embassy_executor::task]
async fn attitude_task() {
    // Wait until the ProNav loop is ready to consume events
    PRONAV_READY.receiver().get().await;

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

        let sleep_dur = Duration::from_micros(10_000 + rng.next_u64() % 5_000);
        if let Ok(task_handle) = delayed_event_task(event, sleep_dur) {
            let spawner = SendSpawner::for_current_executor().await;
            spawner.spawn(task_handle);
        } else {
            log::error!("Failed to spawn delayed_event_task");
        }
    }
}

#[embassy_executor::task(pool_size = 1000)]
async fn delayed_event_task(event: Event, duration: Duration) {
    Timer::after(duration).await;
    EVENTS.send(event).await;
}

#[derive(Clone, Debug)]
pub struct Waypoint {
    pub time: f32,
    pub pos: Vector3<f32>,
}

pub struct Trajectory {
    waypoints: Vec<Waypoint>,
}

impl Trajectory {
    /// Creates a new trajectory. Waypoints should ideally be ordered by time.
    pub fn new(mut waypoints: Vec<Waypoint>) -> Self {
        // Ensure waypoints are strictly sorted by time to allow safe interpolation
        let mut time_accum = 0.0;
        for waypoint in waypoints.iter_mut() {
            time_accum += waypoint.time;
            waypoint.time = time_accum;
        }
        Self { waypoints }
    }

    /// Evaluates the target position at a given time `t` (in seconds).
    pub fn position_at(&self, t: f32) -> Vector3<f32> {
        if self.waypoints.is_empty() {
            return Vector3::zeros();
        }

        // Clamp to the first waypoint if 't' is before the start
        if t <= self.waypoints.first().unwrap().time {
            return self.waypoints.first().unwrap().pos;
        }

        // Clamp to the last waypoint if 't' is after the end
        if t >= self.waypoints.last().unwrap().time {
            return self.waypoints.last().unwrap().pos;
        }

        // Find the segment that contains time 't' and interpolate
        for window in self.waypoints.windows(2) {
            let wp1 = &window[0];
            let wp2 = &window[1];

            if t >= wp1.time && t <= wp2.time {
                // Calculate how far along the segment we are (0.0 to 1.0)
                let progress = (t - wp1.time) / (wp2.time - wp1.time);

                // Linear interpolation (Lerp)
                return wp1.pos + (wp2.pos - wp1.pos) * progress;
            }
        }

        // Fallback (should be unreachable due to boundary checks above)
        self.waypoints.last().unwrap().pos
    }

    /// Evaluates a smoothed target position by applying a moving average
    /// over the window [t - window, t + window].
    /// A window of 1.0 means it averages over a 2-second spread.
    pub fn smoothed_position_at(&self, t: f32, window: f32) -> Vector3<f32> {
        if window <= 0.001 {
            return self.position_at(t);
        }

        // 20 samples is plenty for a very smooth interpolation
        const SAMPLES: usize = 20;
        let mut sum = Vector3::zeros();

        let start_t = t - window;
        let end_t = t + window;
        let step = (end_t - start_t) / (SAMPLES as f32 - 1.0);

        for i in 0..SAMPLES {
            let sample_t = start_t + (i as f32) * step;
            sum += self.position_at(sample_t);
        }

        sum / (SAMPLES as f32)
    }
}
