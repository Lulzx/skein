//! Decentralized drone swarm simulator (Avian3D + Bevy)
//! ------------------------------------------------------
//! A study harness for autonomous swarms in dynamic, GPS-denied indoor
//! environments under uncertainty. Every drone runs the SAME local control law
//! using ONLY what it can sense within a perception radius — there is no central
//! coordinator and no shared world model.
//!
//! Concepts demonstrated:
//!   * Decentralized control  — each agent acts on local neighbor observations.
//!   * GPS-denied navigation  — control uses RELATIVE bearings/ranges, not global
//!                              coordinates. Only a few "informed" drones know the
//!                              goal (Couzin et al. informed-minority model); the
//!                              swarm still navigates collectively.
//!   * Dynamic environment    — a moving obstacle sweeps through the arena.
//!   * Uncertainty            — sensor noise on perceived neighbor positions and
//!                              actuation noise on thrust (toggle with N).
//!   * Comms tolerance        — neighbor "packets" are dropped probabilistically;
//!                              behavior degrades gracefully, never collapses.
//!
//! Controls:  drag/scroll orbit+zoom   [Space] pause      [G] goal-seek   [L] links
//!            [N] noise   [E] estimator view   [O] avoidance   [C] cooperative
//!            [P] perception   [R] re-scatter   [Esc] quit

use avian3d::prelude::*;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use rand::Rng;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tunable constants
// ---------------------------------------------------------------------------
const N_DRONES: usize = 48;
const INFORMED_FRACTION: f32 = 0.18; // fraction that knows the goal

const ROOM_HALF: f32 = 14.0; // half-extent on X and Z
const FLOOR_Y: f32 = 1.0;
const CEIL_Y: f32 = 16.0;

const DRONE_RADIUS: f32 = 0.28;
const MAX_SPEED: f32 = 7.0;
const ROTOR_SPEED: f32 = 40.0; // rad/s, visual rotor-blade spin

// Quadrotor dynamics: thrust acts along the (tilt-limited) body axis and must
// fight real gravity; the battery caps available thrust and drains with use.
const GRAVITY_ACCEL: f32 = 9.81; // m/s^2
const THRUST_MAX: f32 = 26.0; // m/s^2 of thrust at full battery (~2.6 g)
const TILT_ACCEL_RATE: f32 = 45.0; // m/s^3, how fast horizontal thrust can change (attitude lag)
const VEL_KP: f32 = 4.0; // velocity-controller gain (1/s)
const MAX_LATERAL_ACCEL: f32 = 14.0; // m/s^2 cap on commanded steering accel
const BATTERY_DRAIN: f32 = 0.0016; // per (m/s^2 thrust)·s
const BATTERY_RECHARGE: f32 = 0.012; // per s (idle trickle / resupply)
const BATTERY_MIN: f32 = 0.55; // floor — still leaves enough thrust to hover

// Behavior weights (acceleration magnitudes, m/s^2)
// Preferred-velocity blend weights (separation/obstacles are handled by the
// velocity-obstacle avoidance stage, not here).
const W_ALIGNMENT: f32 = 0.9; // match the flock's average velocity
const W_COHESION: f32 = 0.7; // pull toward the local centroid
const W_GOAL: f32 = 1.7; // cruise toward the (believed) goal
const W_BOUNDARY: f32 = 10.0; // firm inward steer near walls (no hard walls exist)

const GOAL_ARRIVE_RADIUS: f32 = 4.0;
const CRUISE_SPEED: f32 = 5.5;

// ORCA-style collision avoidance (deterministic half-plane projection).
const VO_TIME_HORIZON: f32 = 2.5; // s, how far ahead collisions are anticipated
const VO_SIGMA_INFLATE: f32 = 2.0; // safety radius added per 1-sigma of estimate error
const VO_BASE_MARGIN: f32 = 0.9; // extra clearance beyond the two body radii (m)

const PACKET_LOSS: f32 = 0.08; // probability a neighbor measurement is dropped
const ACTUATION_NOISE_STD: f32 = 2.0; // m/s^2, on applied thrust

// Range + bearing sensor model (the realistic GPS-denied measurement).
// Range (e.g. UWB) is accurate; bearing (e.g. vision) is noisy -> anisotropic.
const RANGE_STD: f32 = 0.12; // m
const BEARING_STD: f32 = 0.07; // rad (~4 deg) for azimuth and elevation

// Per-drone neighbor estimator (6-state constant-velocity EKF).
const KF_ACCEL_VAR: f32 = 9.0; // process noise: assumed neighbor accel variance (m/s^2)^2
const KF_INIT_VEL_VAR: f32 = MAX_SPEED * MAX_SPEED; // prior variance on unknown neighbor velocity
const TRACK_TIMEOUT: f32 = 2.0; // s without a measurement before a track is dropped

// Self-localization: each drone estimates its OWN pose (no GPS) by fusing
// drifting odometry (IMU/VIO) with ranges to fixed anchors (e.g. UWB beacons).
const ANCHOR_RANGE: f32 = 22.0; // m, max ranging distance (leaves a coverage gap)
const ANCHOR_RANGE_STD: f32 = 0.10; // m, anchor range measurement noise
const ODOM_BIAS_MAX: f32 = 0.10; // m/s, fixed per-drone odometry bias (source of drift)
const ODOM_NOISE_STD: f32 = 0.15; // m/s, white odometry velocity noise
const SELF_PROCESS_VAR: f32 = 0.08; // self-EKF covariance growth per second (m^2/s)
const COOP_REL_STD: f32 = 0.3; // m, position-equiv. noise of a relative neighbor fix

// Cooperative SLAM: the environment has landmarks at UNKNOWN positions. Drones
// observe them (range+bearing) and cooperatively triangulate a shared map, then
// localize against mapped landmarks where anchors are out of reach.
const N_LANDMARKS: usize = 9;
const LANDMARK_OBS_RANGE: f32 = 8.0; // m, how close a drone must be to observe one
const LANDMARK_USABLE_VAR: f32 = 1.0; // map estimate good enough to localize against

// Distributed goal consensus (gossip): only informed drones observe the goal;
// every drone holds a belief and fuses neighbors' beliefs over the comms graph.
const GOAL_MEAS_STD: f32 = 0.5; // m, an informed drone's direct goal observation noise
const GOAL_DECAY_VAR: f32 = 0.6; // belief variance growth per second (the goal moves -> info stales)
const GOAL_RELAY_PENALTY: f32 = 0.3; // variance added per comms hop (m^2)
const GOAL_USABLE_VAR: f32 = 9.0; // below this (1σ < 3 m) the belief is good enough to navigate by
const GOAL_INIT_VAR: f32 = 400.0; // initial "no idea where the goal is" variance

// ---------------------------------------------------------------------------
// Components & resources
// ---------------------------------------------------------------------------
#[derive(Component)]
struct Drone {
    id: usize,
    informed: bool,
    period: u64, // re-plans control every `period` ticks (1 = every tick)...
    phase: u64,  // ...offset so drones don't all re-plan on the same tick
}

/// Quadrotor actuation state: the current thrust direction (which lags commands,
/// since the airframe must physically tilt) and remaining battery charge.
#[derive(Component)]
struct Quad {
    horiz: Vec3,       // current (attitude-lagged) horizontal thrust acceleration
    battery: f32,
    last_thrust: Vec3, // held and re-applied on ticks the drone doesn't re-plan
}

/// Per-drone material whose color encodes the drone's goal-knowledge state.
/// Lives on the parent entity; the visual sub-meshes share this handle.
#[derive(Component)]
struct ColorMat(Handle<StandardMaterial>);

/// Marker for a spinning rotor mesh (a child of a drone).
#[derive(Component)]
struct Rotor;

#[derive(Component)]
struct MovingObstacle;

#[derive(Component)]
struct GoalMarker;

#[derive(Component)]
struct HudText;

/// One agent's observable state for this timestep.
#[derive(Clone, Copy, Default)]
struct AgentObs {
    pos: Vec3,
    vel: Vec3,
    alive: bool,
}

/// Simultaneous ground-truth snapshot of the swarm. Used to *synthesize each
/// drone's noisy measurements* and for HUD metrics — NOT read by the control law.
#[derive(Resource)]
struct Snapshot(Vec<AgentObs>);

type Vec6 = nalgebra::SVector<f32, 6>;
type Mat6 = nalgebra::SMatrix<f32, 6, 6>;
type Vec3n = nalgebra::SVector<f32, 3>;
type Mat3n = nalgebra::SMatrix<f32, 3, 3>;
type Mat36 = nalgebra::SMatrix<f32, 3, 6>;

/// One drone's belief about one neighbor: a 6-state (position + velocity)
/// Extended Kalman Filter tracking a neighbor in the world frame.
///
/// The measurement is RANGE + BEARING relative to the ego drone's own pose —
/// the realistic GPS-denied sensor (e.g. UWB ranging + a vision bearing), not a
/// direct position read. Range is accurate, bearing is noisy, so the position
/// covariance is genuinely ANISOTROPIC (tight radially, loose tangentially).
#[derive(Clone)]
struct Track {
    x: Vec6,  // [px, py, pz, vx, vy, vz] in world frame
    p: Mat6,  // 6x6 covariance
    last_seen: f32,
}

impl Track {
    /// Initialize from the first range+bearing measurement and the ego pose.
    fn new(ego: Vec3, z: Vec3n, now: f32) -> Self {
        let pos = ego + relative_from_meas(z);
        let mut x = Vec6::zeros();
        x[0] = pos.x;
        x[1] = pos.y;
        x[2] = pos.z;
        // Large, diagonal prior: unsure of position, very unsure of velocity.
        let mut p = Mat6::zeros();
        for i in 0..3 {
            p[(i, i)] = 4.0;
        }
        for i in 3..6 {
            p[(i, i)] = KF_INIT_VEL_VAR;
        }
        Self { x, p, last_seen: now }
    }

    /// EKF predict: constant-velocity motion model, covariance inflated by the
    /// process noise. Runs every step — measured or not — so beliefs coast
    /// through comms dropouts instead of vanishing.
    fn predict(&mut self, dt: f32) {
        let mut f = Mat6::identity();
        for i in 0..3 {
            f[(i, i + 3)] = dt;
        }
        self.x = f * self.x;

        // Discrete constant-acceleration-noise Q, per axis.
        let q = KF_ACCEL_VAR;
        let (q11, q12, q22) = (
            q * dt.powi(4) / 4.0,
            q * dt.powi(3) / 2.0,
            q * dt.powi(2),
        );
        let mut qm = Mat6::zeros();
        for i in 0..3 {
            qm[(i, i)] = q11;
            qm[(i, i + 3)] = q12;
            qm[(i + 3, i)] = q12;
            qm[(i + 3, i + 3)] = q22;
        }
        self.p = f * self.p * f.transpose() + qm;
    }

    /// EKF update: fuse a nonlinear range+bearing measurement taken from `ego`.
    fn update(&mut self, ego: Vec3, z: Vec3n, r: &Mat3n, now: f32) {
        let pos = self.pos();
        let d = pos - ego;
        let h_x = meas_from_relative(d); // predicted measurement
        let big_h = measurement_jacobian(d); // 3x6 Jacobian at the estimate

        // Innovation, with azimuth wrapped to [-pi, pi].
        let mut y = z - h_x;
        y[1] = wrap_angle(y[1]);

        let s = big_h * self.p * big_h.transpose() + r;
        let Some(s_inv) = s.try_inverse() else { return };
        let k = self.p * big_h.transpose() * s_inv; // 6x3 Kalman gain
        self.x += k * y;
        let i = Mat6::identity();
        self.p = (i - k * big_h) * self.p;
        self.last_seen = now;
    }

    fn pos(&self) -> Vec3 {
        Vec3::new(self.x[0], self.x[1], self.x[2])
    }
    fn vel(&self) -> Vec3 {
        Vec3::new(self.x[3], self.x[4], self.x[5])
    }
    /// 3x3 position covariance block.
    fn pos_cov(&self) -> Mat3n {
        self.p.fixed_view::<3, 3>(0, 0).into()
    }
    /// 1-sigma positional uncertainty along the worst (largest) axis (m).
    fn pos_sigma(&self) -> f32 {
        self.pos_cov().symmetric_eigenvalues().max().max(0.0).sqrt()
    }
}

/// Map a relative position into a (range, azimuth, elevation) measurement.
fn meas_from_relative(d: Vec3) -> Vec3n {
    let r = d.length().max(1e-4);
    let az = d.x.atan2(d.z); // azimuth in the XZ plane
    let el = (d.y / r).clamp(-1.0, 1.0).asin();
    Vec3n::new(r, az, el)
}

/// Invert a (range, azimuth, elevation) measurement back into a relative vector.
fn relative_from_meas(z: Vec3n) -> Vec3 {
    let (r, az, el) = (z[0], z[1], z[2]);
    let horiz = r * el.cos();
    Vec3::new(horiz * az.sin(), r * el.sin(), horiz * az.cos())
}

/// Jacobian of the range+bearing measurement w.r.t. the 6-state (∂h/∂x).
/// Only the position columns are non-zero.
fn measurement_jacobian(d: Vec3) -> Mat36 {
    let (dx, dy, dz) = (d.x, d.y, d.z);
    let r = d.length().max(1e-4);
    let rxz2 = (dx * dx + dz * dz).max(1e-6);
    let rxz = rxz2.sqrt();
    let mut h = Mat36::zeros();
    // d(range)/d(pos)
    h[(0, 0)] = dx / r;
    h[(0, 1)] = dy / r;
    h[(0, 2)] = dz / r;
    // d(azimuth)/d(pos)
    h[(1, 0)] = dz / rxz2;
    h[(1, 2)] = -dx / rxz2;
    // d(elevation)/d(pos)
    h[(2, 0)] = -dy * dx / (rxz * r * r);
    h[(2, 1)] = rxz / (r * r);
    h[(2, 2)] = -dy * dz / (rxz * r * r);
    h
}

fn wrap_angle(a: f32) -> f32 {
    let mut a = a;
    while a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    }
    while a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// Each drone's decentralized world model: estimates of currently-tracked
/// neighbors, keyed by neighbor id. This is what the control law actually reads.
#[derive(Component, Default)]
struct Beliefs(HashMap<usize, Track>);

/// Each drone's belief about the (moving) goal's location, with a scalar
/// confidence (variance). Informed drones refresh it by direct observation;
/// everyone else only learns the goal through gossip with neighbors.
#[derive(Component)]
struct GoalBelief {
    pos: Vec3,
    var: f32,
}

impl Default for GoalBelief {
    fn default() -> Self {
        Self { pos: Vec3::new(0.0, 8.0, 0.0), var: GOAL_INIT_VAR }
    }
}

/// Bayesian fusion of two independent position beliefs (inverse-variance weighted).
fn fuse_belief(a: (Vec3, f32), b: (Vec3, f32)) -> (Vec3, f32) {
    let (pa, va) = a;
    let (pb, vb) = b;
    let var = 1.0 / (1.0 / va + 1.0 / vb);
    let pos = (pa / va + pb / vb) * var;
    (pos, var)
}

/// Each drone's estimate of its OWN pose — a 3-state position EKF fusing
/// drifting odometry with ranges to fixed anchors. Without this, control
/// would secretly depend on ground-truth position; with it, the loop is fully
/// GPS-denied (ground truth is used only to synthesize sensor readings).
#[derive(Component, Clone)]
struct SelfLoc {
    x: Vec3,    // believed own position (world frame)
    p: Mat3n,   // 3x3 covariance
    bias: Vec3, // fixed odometry bias — the systematic source of drift
}

impl SelfLoc {
    fn new(start: Vec3, bias: Vec3) -> Self {
        Self {
            x: start, // assume the launch pose is known; error accumulates from here
            p: Mat3n::identity() * 0.1,
            bias,
        }
    }

    /// Predict from odometry (dead reckoning): integrate velocity, grow covariance.
    fn predict(&mut self, odom_vel: Vec3, dt: f32) {
        self.x += odom_vel * dt;
        self.p += Mat3n::identity() * (SELF_PROCESS_VAR * dt);
    }

    /// EKF update from a single anchor range measurement (nonlinear, scalar).
    fn update_range(&mut self, anchor: Vec3, z_range: f32, r_var: f32) {
        let d = self.x - anchor;
        let rng = d.length().max(1e-3);
        let h = Vec3n::new(d.x / rng, d.y / rng, d.z / rng); // ∂range/∂pos
        let pht = self.p * h; // 3x1
        let s = h.dot(&pht) + r_var;
        if s < 1e-6 {
            return;
        }
        let k = pht / s; // 3x1 Kalman gain
        let x = Vec3n::new(self.x.x, self.x.y, self.x.z) + k * (z_range - rng);
        self.x = Vec3::new(x[0], x[1], x[2]);
        self.p = (Mat3n::identity() - k * h.transpose()) * self.p;
    }

}

/// World obstacles as the drones' proximity sensors would report them
/// (position + effective radius). Static pillars + one moving obstacle.
#[derive(Resource, Default)]
struct Obstacles {
    pillars: Vec<(Vec3, f32)>,
    moving: (Vec3, f32),
    moving_vel: Vec3,
}

#[derive(Resource)]
struct Goal {
    pos: Vec3,
}

/// Monotonic physics-step counter, used to drive per-drone asynchronous rates.
#[derive(Resource, Default)]
struct Tick(u64);

/// Orbit-camera state: left-drag rotates, scroll zooms, idle slowly auto-rotates.
#[derive(Resource)]
struct OrbitCam {
    yaw: f32,
    pitch: f32,
    radius: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self { yaw: 0.0, pitch: 0.38, radius: 20.0 }
    }
}

/// Fixed beacons at known world positions (e.g. UWB anchors), used by every
/// drone's self-localization EKF to bound odometry drift.
#[derive(Resource, Default)]
struct Anchors(Vec<Vec3>);

/// Ground-truth landmark positions — UNKNOWN to the drones. Used only to
/// synthesize observations (like the snapshot).
#[derive(Resource, Default)]
struct Landmarks(Vec<Vec3>);

/// One landmark's estimate in the swarm's shared, cooperatively-built map.
#[derive(Clone, Copy)]
struct LandmarkEst {
    pos: Vec3,
    cov: Mat3n,
    seen: bool,
}

/// The shared SLAM map: the swarm's current estimate of each landmark, fused
/// from many drones' observations via Covariance Intersection.
#[derive(Resource, Default)]
struct LandmarkMap(Vec<LandmarkEst>);

/// What each drone broadcasts to the swarm: its believed pose and its full 3x3
/// position covariance, indexed by `Drone::id`. Refreshed each step (one step of
/// comms latency), so cooperative localization reads it instead of live state.
#[derive(Resource)]
struct Broadcast(Vec<(Vec3, Mat3n)>);

/// What each drone gossips about the goal: (believed position, variance),
/// indexed by `Drone::id`. One step of comms latency, like `Broadcast`.
#[derive(Resource)]
struct GoalBroadcast(Vec<(Vec3, f32)>);

#[derive(Resource)]
struct SimState {
    paused: bool,
    goal_seek: bool,
    show_links: bool,
    noise: bool,
    show_estimator: bool,
    avoidance: bool,
    cooperative: bool,
    perception: f32,
}

impl Default for SimState {
    fn default() -> Self {
        Self {
            paused: false,
            goal_seek: true,
            show_links: false,
            noise: true,
            show_estimator: false,
            avoidance: true,
            cooperative: true,
            perception: 5.0,
        }
    }
}

/// Drone whose estimator beliefs are visualized (belief vs. ground truth).
const EGO_DRONE: usize = 0;

/// Live, smoothed metrics shown in the HUD.
#[derive(Resource, Default)]
struct Metrics {
    avg_neighbors: f32,
    avg_speed: f32,
    frac_at_goal: f32,
    nearest_pair: f32,
    est_error: f32,   // RMS neighbor-position estimation error across the swarm (m)
    self_error: f32,  // RMS self-localization error across the swarm (m)
    goal_coverage: f32, // fraction of the swarm that currently knows the goal
    battery: f32,     // mean battery charge across the swarm (0..1)
    map_built: usize, // landmarks mapped well enough to localize against
    map_error: f32,   // RMS map landmark position error (m)
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Avian3D — Decentralized Drone Swarm".into(),
                    resolution: (1280u32, 800u32).into(),
                    ..default()
                }),
                ..default()
            }),
            PhysicsPlugins::default(),
        ))
        // Real gravity now acts on the drones; they must thrust to stay aloft.
        .insert_resource(Gravity(Vec3::new(0.0, -GRAVITY_ACCEL, 0.0)))
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.05)))
        .insert_resource(Snapshot(vec![AgentObs::default(); N_DRONES]))
        .insert_resource(Obstacles::default())
        .insert_resource(Goal { pos: Vec3::new(0.0, 8.0, 0.0) })
        .init_resource::<Anchors>()
        .init_resource::<OrbitCam>()
        .init_resource::<Landmarks>()
        .insert_resource(LandmarkMap(vec![
            LandmarkEst { pos: Vec3::ZERO, cov: Mat3n::identity() * 100.0, seen: false };
            N_LANDMARKS
        ]))
        .init_resource::<Tick>()
        .insert_resource(Broadcast(vec![(Vec3::ZERO, Mat3n::identity()); N_DRONES]))
        .insert_resource(GoalBroadcast(vec![
            (Vec3::new(0.0, 8.0, 0.0), GOAL_INIT_VAR);
            N_DRONES
        ]))
        .init_resource::<SimState>()
        .init_resource::<Metrics>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_input,
                move_goal,
                move_obstacle,
                orbit_camera,
                draw_gizmos,
                draw_estimator,
                update_drone_visuals,
                spin_rotors,
                update_hud,
            ),
        )
        // Control runs in the fixed (physics) schedule, before Avian's step in
        // FixedPostUpdate. Ground-truth sensing -> per-drone estimation -> control.
        .add_systems(
            FixedUpdate,
            (
                advance_tick,
                sense_swarm,
                broadcast_estimates,
                self_localize,
                slam_mapping,
                estimate_neighbors,
                goal_consensus,
                actuate_swarm,
            )
                .chain(),
        )
        .run();
}

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut obstacles: ResMut<Obstacles>,
) {
    // Camera (with per-camera ambient light — AmbientLight is a component in Bevy 0.18)
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 18.0, 30.0).looking_at(Vec3::new(0.0, 7.0, 0.0), Vec3::Y),
        AmbientLight {
            color: Color::srgb(0.6, 0.7, 0.9),
            brightness: 220.0,
            ..default()
        },
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 6000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 24.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Floor (visual only)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0 * ROOM_HALF + 4.0, 0.3, 2.0 * ROOM_HALF + 4.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.08, 0.09, 0.12),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, FLOOR_Y - 0.15, 0.0),
    ));

    // Static pillar obstacles
    let pillar_positions = [
        Vec3::new(-7.0, 0.0, -6.0),
        Vec3::new(6.5, 0.0, 4.0),
        Vec3::new(-5.0, 0.0, 7.5),
        Vec3::new(8.0, 0.0, -8.0),
    ];
    let pillar_radius = 1.2;
    let pillar_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.27, 0.33),
        perceptual_roughness: 0.8,
        ..default()
    });
    let pillar_mesh = meshes.add(Cylinder::new(pillar_radius, CEIL_Y - FLOOR_Y));
    for p in pillar_positions {
        let pos = Vec3::new(p.x, (FLOOR_Y + CEIL_Y) * 0.5, p.z);
        commands.spawn((
            Mesh3d(pillar_mesh.clone()),
            MeshMaterial3d(pillar_mat.clone()),
            Transform::from_translation(pos),
            RigidBody::Static,
            Collider::cylinder(pillar_radius, CEIL_Y - FLOOR_Y),
        ));
        // Effective avoidance radius = physical radius + drone clearance.
        obstacles.pillars.push((Vec3::new(p.x, 0.0, p.z), pillar_radius + 1.6));
    }

    // Moving obstacle (kinematic) — the "dynamic environment"
    let mob_half = Vec3::new(1.4, 2.2, 1.4);
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(mob_half.x * 2.0, mob_half.y * 2.0, mob_half.z * 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.15, 0.2),
            emissive: LinearRgba::new(0.8, 0.05, 0.05, 1.0),
            ..default()
        })),
        Transform::from_xyz(0.0, 8.0, 0.0),
        RigidBody::Kinematic,
        Collider::cuboid(mob_half.x * 2.0, mob_half.y * 2.0, mob_half.z * 2.0),
        MovingObstacle,
    ));
    obstacles.moving = (Vec3::new(0.0, 8.0, 0.0), mob_half.x + 1.8);

    // Goal marker (visual only, glowing)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.6))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 1.0, 0.5, 0.6),
            emissive: LinearRgba::new(0.1, 1.6, 0.5, 1.0),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 8.0, 0.0),
        GoalMarker,
    ));

    // Fixed localization anchors (UWB beacons). Only three, at varied heights
    // (so vertical position is observable) and clustered toward +X — this leaves
    // a deliberate coverage GRADIENT: the far/spawn corner is anchor-poor and
    // must rely on cooperative localization to stay bounded.
    let anchor_positions = vec![
        Vec3::new(11.0, 3.0, -10.0),
        Vec3::new(12.0, 13.0, 9.0),
        Vec3::new(2.0, 8.0, 12.0),
    ];
    let anchor_mesh = meshes.add(Cuboid::new(0.5, 0.5, 0.5));
    let anchor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.8, 1.0),
        emissive: LinearRgba::new(0.1, 0.8, 1.4, 1.0),
        ..default()
    });
    for a in &anchor_positions {
        commands.spawn((
            Mesh3d(anchor_mesh.clone()),
            MeshMaterial3d(anchor_mat.clone()),
            Transform::from_translation(*a),
        ));
    }
    commands.insert_resource(Anchors(anchor_positions));

    // Unknown landmarks (features the drones will discover and map cooperatively).
    let mut rng = rand::thread_rng();
    let landmark_mesh = meshes.add(Cuboid::new(0.4, 0.4, 0.4));
    let landmark_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.4, 0.9),
        emissive: LinearRgba::new(0.5, 0.2, 0.8, 1.0),
        ..default()
    });
    let mut landmark_positions = Vec::new();
    for _ in 0..N_LANDMARKS {
        let pos = Vec3::new(
            rng.gen_range(-ROOM_HALF + 2.0..ROOM_HALF - 2.0),
            rng.gen_range(FLOOR_Y + 1.0..CEIL_Y - 1.0),
            rng.gen_range(-ROOM_HALF + 2.0..ROOM_HALF - 2.0),
        );
        commands.spawn((
            Mesh3d(landmark_mesh.clone()),
            MeshMaterial3d(landmark_mat.clone()),
            Transform::from_translation(pos),
        ));
        landmark_positions.push(pos);
    }
    commands.insert_resource(Landmarks(landmark_positions));

    // Drones — built as little quadrotors: a glowing body, an X of arms, and
    // four spinning rotors. Shared meshes/material across all drones; only the
    // state-color material is per-drone (so it can be recolored live).
    let body_mesh = meshes.add(Cuboid::new(0.42, 0.14, 0.42));
    let arm_mesh = meshes.add(Cuboid::new(0.92, 0.06, 0.085));
    let hub_mesh = meshes.add(Cylinder::new(0.19, 0.03));
    let blade_mesh = meshes.add(Cuboid::new(0.36, 0.018, 0.05));
    let dark_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.04, 0.05, 0.07),
        metallic: 0.8,
        perceptual_roughness: 0.35,
        ..default()
    });
    let arm = 0.32; // rotor offset from the hub
    let rotor_offsets = [
        Vec3::new(arm, 0.085, arm),
        Vec3::new(arm, 0.085, -arm),
        Vec3::new(-arm, 0.085, arm),
        Vec3::new(-arm, 0.085, -arm),
    ];
    let f4 = std::f32::consts::FRAC_PI_4;

    let n_informed = (N_DRONES as f32 * INFORMED_FRACTION).round() as usize;
    for id in 0..N_DRONES {
        let informed = id < n_informed;
        // Scatter near one corner so the swarm has to organize and traverse.
        let pos = Vec3::new(
            rng.gen_range(-11.0..-6.0),
            rng.gen_range(4.0..11.0),
            rng.gen_range(-11.0..-6.0),
        );
        let color_mat = materials.add(StandardMaterial {
            base_color: drone_color(informed, informed, 0.0),
            emissive: drone_emissive(informed, informed, 0.0),
            ..default()
        });
        commands
            .spawn((
                Transform::from_translation(pos),
                Visibility::default(),
                RigidBody::Dynamic,
                Collider::sphere(DRONE_RADIUS),
                LockedAxes::ROTATION_LOCKED, // keep the quad level and readable
                LinearDamping(0.6),
                Restitution::new(0.1),
                LinearVelocity(Vec3::ZERO),
                Drone {
                    id,
                    informed,
                    period: rng.gen_range(1..=4), // asynchronous: 64 Hz down to 16 Hz
                    phase: rng.gen_range(0..4),
                },
                Beliefs::default(),
                GoalBelief::default(),
                Quad {
                    horiz: Vec3::ZERO,
                    battery: 1.0,
                    last_thrust: Vec3::Y * GRAVITY_ACCEL,
                },
                ColorMat(color_mat.clone()),
                SelfLoc::new(
                    pos,
                    Vec3::new(
                        rng.gen_range(-ODOM_BIAS_MAX..ODOM_BIAS_MAX),
                        rng.gen_range(-ODOM_BIAS_MAX..ODOM_BIAS_MAX),
                        rng.gen_range(-ODOM_BIAS_MAX..ODOM_BIAS_MAX),
                    ),
                ),
            ))
            .with_children(|c| {
                // Glowing central body — carries the state color.
                c.spawn((Mesh3d(body_mesh.clone()), MeshMaterial3d(color_mat.clone())));
                // Two crossed arms (X configuration).
                c.spawn((
                    Mesh3d(arm_mesh.clone()),
                    MeshMaterial3d(dark_mat.clone()),
                    Transform::from_rotation(Quat::from_rotation_y(f4)),
                ));
                c.spawn((
                    Mesh3d(arm_mesh.clone()),
                    MeshMaterial3d(dark_mat.clone()),
                    Transform::from_rotation(Quat::from_rotation_y(-f4)),
                ));
                // A dark hub + a colored spinning blade at each rotor.
                for off in rotor_offsets {
                    c.spawn((
                        Mesh3d(hub_mesh.clone()),
                        MeshMaterial3d(dark_mat.clone()),
                        Transform::from_translation(off),
                    ));
                    c.spawn((
                        Mesh3d(blade_mesh.clone()),
                        MeshMaterial3d(color_mat.clone()),
                        Transform::from_translation(off + Vec3::Y * 0.02),
                        Rotor,
                    ));
                }
            });
    }

    // HUD
    commands.spawn((
        Text::new(""),
        TextFont { font_size: 15.0, ..default() },
        TextColor(Color::srgb(0.85, 0.9, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        HudText,
    ));
}

// ---------------------------------------------------------------------------
// Sensing: build the simultaneous snapshot (read-only over the swarm)
// ---------------------------------------------------------------------------
/// Advance the global physics-step counter (drives per-drone async rates).
fn advance_tick(mut tick: ResMut<Tick>, sim: Res<SimState>) {
    if !sim.paused {
        tick.0 = tick.0.wrapping_add(1);
    }
}

fn sense_swarm(mut snap: ResMut<Snapshot>, query: Query<(&Position, &LinearVelocity, &Drone)>) {
    for obs in snap.0.iter_mut() {
        obs.alive = false;
    }
    for (pos, vel, drone) in &query {
        snap.0[drone.id] = AgentObs {
            pos: pos.0,
            vel: vel.0,
            alive: true,
        };
    }
}

/// Publish each drone's self-estimate and goal belief for neighbors to use.
fn broadcast_estimates(
    mut bc: ResMut<Broadcast>,
    mut gbc: ResMut<GoalBroadcast>,
    query: Query<(&Drone, &SelfLoc, &GoalBelief)>,
) {
    for (drone, sl, gb) in &query {
        bc.0[drone.id] = (sl.x, sl.p);
        gbc.0[drone.id] = (gb.pos, gb.var);
    }
}

/// Covariance Intersection: consistently fuse two estimates whose correlation is
/// unknown. `P_ci⁻¹ = ω·Pa⁻¹ + (1-ω)·Pb⁻¹`, with ω ∈ [0,1] chosen to minimize
/// the fused covariance (smallest determinant). Never overconfident, for any
/// correlation — and ω→1 when the other estimate is worse, so it self-gates.
fn covariance_intersection(xa: Vec3, pa: Mat3n, xb: Vec3, pb: Mat3n) -> (Vec3, Mat3n) {
    let (Some(pa_inv), Some(pb_inv)) = (pa.try_inverse(), pb.try_inverse()) else {
        return (xa, pa);
    };
    let mut best_w = 1.0;
    let mut best_det = f32::INFINITY;
    for i in 0..=10 {
        let w = i as f32 / 10.0;
        if let Some(p) = (pa_inv * w + pb_inv * (1.0 - w)).try_inverse() {
            let det = p.determinant();
            if det < best_det {
                best_det = det;
                best_w = w;
            }
        }
    }
    let info = pa_inv * best_w + pb_inv * (1.0 - best_w);
    let Some(p) = info.try_inverse() else { return (xa, pa) };
    let xa_n = Vec3n::new(xa.x, xa.y, xa.z);
    let xb_n = Vec3n::new(xb.x, xb.y, xb.z);
    let x = p * (pa_inv * xa_n * best_w + pb_inv * xb_n * (1.0 - best_w));
    (Vec3::new(x[0], x[1], x[2]), p)
}

// ---------------------------------------------------------------------------
// Distributed goal consensus: the goal is observed only by the informed
// minority; every drone gossips its belief so the knowledge diffuses across the
// comms graph, one hop per step. Beliefs decay (the goal moves), so the swarm
// must keep re-propagating fresh observations — agreement is never "done".
// ---------------------------------------------------------------------------
fn goal_consensus(
    snap: Res<Snapshot>,
    gbc: Res<GoalBroadcast>,
    goal: Res<Goal>,
    sim: Res<SimState>,
    time: Res<Time>,
    mut metrics: ResMut<Metrics>,
    mut query: Query<(&Drone, &mut GoalBelief)>,
) {
    if sim.paused {
        return;
    }
    let dt = time.delta_secs().max(1e-4);
    let perception_sq = sim.perception * sim.perception;
    let meas_var = (GOAL_MEAS_STD * GOAL_MEAS_STD).max(1e-4);
    let mut rng = rand::thread_rng();

    let mut known = 0.0f32;
    let mut total = 0.0f32;

    for (drone, mut gb) in &mut query {
        let truth = snap.0[drone.id];
        if !truth.alive {
            continue;
        }
        // 1) Staleness: confidence decays because the goal keeps moving.
        gb.var += GOAL_DECAY_VAR * dt;

        // 2) Informed drones refresh from a direct (noisy) observation.
        if drone.informed {
            let std = if sim.noise { GOAL_MEAS_STD } else { 0.0 };
            let z = goal.pos
                + Vec3::new(gauss(&mut rng, std), gauss(&mut rng, std), gauss(&mut rng, std));
            let (pos, var) = fuse_belief((gb.pos, gb.var), (z, meas_var));
            gb.pos = pos;
            gb.var = var;
        }

        // 3) Gossip: fuse the single most confident neighbor belief in range,
        //    penalized for the extra hop. Fusing only the best (not all) avoids
        //    double-counting shared information; the gate keeps it monotone.
        let mut best: Option<(Vec3, f32)> = None;
        for (j, other) in snap.0.iter().enumerate() {
            if !other.alive || j == drone.id {
                continue;
            }
            if truth.pos.distance_squared(other.pos) > perception_sq {
                continue;
            }
            if sim.noise && rng.gen::<f32>() < PACKET_LOSS {
                continue; // dropped gossip packet
            }
            let (npos, nvar) = gbc.0[j];
            let relayed = nvar + GOAL_RELAY_PENALTY;
            if relayed < best.map_or(f32::INFINITY, |(_, v)| v) {
                best = Some((npos, relayed));
            }
        }
        if let Some(cand) = best {
            if cand.1 < gb.var {
                let (pos, var) = fuse_belief((gb.pos, gb.var), cand);
                gb.pos = pos;
                gb.var = var;
            }
        }

        total += 1.0;
        if gb.var < GOAL_USABLE_VAR {
            known += 1.0;
        }
    }

    if total > 0.0 {
        metrics.goal_coverage += 0.1 * (known / total - metrics.goal_coverage);
    }
}

// ---------------------------------------------------------------------------
// Self-localization: each drone estimates its OWN pose with no GPS, by dead-
// reckoning drifting odometry and correcting against (a) ranges to fixed anchors
// and (b) COOPERATIVE fixes — a neighbor broadcasts its own pose estimate and the
// drone measures the relative offset to it, so localization information relays
// across the comms graph from anchor-rich regions into anchor-poor ones.
// After this runs, NOTHING in the control path reads ground-truth position.
// ---------------------------------------------------------------------------
fn self_localize(
    snap: Res<Snapshot>,
    anchors: Res<Anchors>,
    landmarks: Res<Landmarks>,
    map: Res<LandmarkMap>,
    broadcast: Res<Broadcast>,
    sim: Res<SimState>,
    time: Res<Time>,
    mut metrics: ResMut<Metrics>,
    mut query: Query<(&Drone, &mut SelfLoc)>,
) {
    if sim.paused {
        return;
    }
    let dt = time.delta_secs().max(1e-4);
    let mut rng = rand::thread_rng();
    let odom_std = if sim.noise { ODOM_NOISE_STD } else { 0.0 };
    let range_std = if sim.noise { ANCHOR_RANGE_STD } else { 0.0 };
    let r_var = (range_std * range_std).max(1e-4);

    let mut err_sum = 0.0f32;
    let mut count = 0.0f32;

    for (drone, mut sl) in &mut query {
        let truth = snap.0[drone.id];
        if !truth.alive {
            continue;
        }
        // Odometry = true velocity corrupted by a fixed bias + white noise.
        let bias = if sim.noise { sl.bias } else { Vec3::ZERO };
        let odom = truth.vel
            + bias
            + Vec3::new(
                gauss(&mut rng, odom_std),
                gauss(&mut rng, odom_std),
                gauss(&mut rng, odom_std),
            );
        sl.predict(odom, dt);

        // Correct against every anchor currently in range (bounds the drift).
        for &a in anchors.0.iter() {
            let true_range = (truth.pos - a).length();
            if true_range > ANCHOR_RANGE {
                continue;
            }
            let z = true_range + gauss(&mut rng, range_std);
            sl.update_range(a, z, r_var);
        }

        // Map-based localization: range against well-mapped landmarks in view —
        // this is what carries localization into the anchor-poor far side.
        for (lid, ltrue) in landmarks.0.iter().enumerate() {
            let true_range = (truth.pos - *ltrue).length();
            if true_range > LANDMARK_OBS_RANGE {
                continue;
            }
            let est = map.0[lid];
            let map_var = est.cov.symmetric_eigenvalues().max().max(0.0);
            if !est.seen || map_var > LANDMARK_USABLE_VAR {
                continue; // not mapped confidently enough to trust yet
            }
            let z = true_range + gauss(&mut rng, range_std);
            // Treat the landmark's own map uncertainty as extra range noise.
            sl.update_range(est.pos, z, r_var + map_var);
        }

        // Cooperative localization via COVARIANCE INTERSECTION. A neighbor's
        // broadcast estimate is correlated with ours (information we relayed to
        // it can come back), so naive Kalman fusion would be overconfident. CI is
        // provably consistent under *unknown* correlation, and its optimal weight
        // ω automatically down-weights neighbors worse than us — no manual gate.
        if sim.cooperative {
            let perception_sq = sim.perception * sim.perception;
            let coop_std = if sim.noise { COOP_REL_STD } else { 0.0 };
            let meas_cov = Mat3n::identity() * (COOP_REL_STD * COOP_REL_STD).max(1e-4);
            for (j, other) in snap.0.iter().enumerate() {
                if !other.alive || j == drone.id {
                    continue;
                }
                if truth.pos.distance_squared(other.pos) > perception_sq {
                    continue;
                }
                if sim.noise && rng.gen::<f32>() < PACKET_LOSS {
                    continue; // dropped comms packet
                }
                let (neighbor_est, neighbor_cov) = broadcast.0[j];
                // Measure the relative offset to the neighbor (noisy), then place
                // ourselves relative to its broadcast estimate. Its covariance
                // (plus the relative-measurement noise) is the fix's uncertainty.
                let d_meas = (other.pos - truth.pos)
                    + Vec3::new(
                        gauss(&mut rng, coop_std),
                        gauss(&mut rng, coop_std),
                        gauss(&mut rng, coop_std),
                    );
                let z_self = neighbor_est - d_meas;
                let (x_new, p_new) =
                    covariance_intersection(sl.x, sl.p, z_self, neighbor_cov + meas_cov);
                sl.x = x_new;
                sl.p = p_new;
            }
        }

        err_sum += sl.x.distance(truth.pos);
        count += 1.0;
    }

    if count > 0.0 {
        metrics.self_error += 0.1 * (err_sum / count - metrics.self_error);
    }
}

// ---------------------------------------------------------------------------
// Cooperative SLAM: drones observe unknown landmarks (range+bearing from their
// own estimated pose) and fuse each observation into a SHARED map via Covariance
// Intersection. Many partial views from different drones triangulate the map,
// which then feeds back into self-localization (loop closure).
// ---------------------------------------------------------------------------
fn slam_mapping(
    snap: Res<Snapshot>,
    landmarks: Res<Landmarks>,
    sim: Res<SimState>,
    mut map: ResMut<LandmarkMap>,
    mut metrics: ResMut<Metrics>,
    query: Query<(&Drone, &SelfLoc)>,
) {
    if sim.paused {
        return;
    }
    let mut rng = rand::thread_rng();
    let (range_std, bearing_std) = if sim.noise {
        (RANGE_STD, BEARING_STD)
    } else {
        (0.0, 0.0)
    };

    for (drone, sl) in &query {
        let me_true = snap.0[drone.id].pos;
        for (lid, ltrue) in landmarks.0.iter().enumerate() {
            let d_true = *ltrue - me_true;
            let true_range = d_true.length();
            if true_range > LANDMARK_OBS_RANGE {
                continue;
            }
            // Observe (range+bearing), then place the landmark in the world using
            // the drone's own ESTIMATED pose -> the observation inherits self error.
            let z = meas_from_relative(d_true)
                + Vec3n::new(
                    gauss(&mut rng, range_std),
                    gauss(&mut rng, bearing_std),
                    gauss(&mut rng, bearing_std),
                );
            let world_est = sl.x + relative_from_meas(z);
            // Observation covariance ≈ self-pose covariance + measurement spread.
            let spread = RANGE_STD.max(true_range * BEARING_STD).powi(2).max(1e-4);
            let obs_cov = sl.p + Mat3n::identity() * spread;

            let cur = map.0[lid];
            map.0[lid] = if cur.seen {
                let (pos, cov) = covariance_intersection(cur.pos, cur.cov, world_est, obs_cov);
                LandmarkEst { pos, cov, seen: true }
            } else {
                LandmarkEst { pos: world_est, cov: obs_cov, seen: true }
            };
        }
    }

    // Map metrics: how many landmarks are confidently mapped, and how accurate.
    let mut built = 0;
    let mut err_sq = 0.0f32;
    for (lid, ltrue) in landmarks.0.iter().enumerate() {
        let est = map.0[lid];
        if est.seen && est.cov.symmetric_eigenvalues().max() <= LANDMARK_USABLE_VAR {
            built += 1;
            err_sq += est.pos.distance_squared(*ltrue);
        }
    }
    metrics.map_built = built;
    if built > 0 {
        metrics.map_error = (err_sq / built as f32).sqrt();
    }
}

// ---------------------------------------------------------------------------
// Estimation: each drone runs its OWN Kalman filter per perceived neighbor.
// Ground truth (the snapshot) is used only to synthesize noisy measurements;
// the drone never reads it directly. This is the heart of GPS-denied autonomy:
// acting on an inferred, uncertain world model rather than perfect information.
// ---------------------------------------------------------------------------
fn estimate_neighbors(
    snap: Res<Snapshot>,
    sim: Res<SimState>,
    time: Res<Time>,
    mut metrics: ResMut<Metrics>,
    mut query: Query<(&Drone, &SelfLoc, &mut Beliefs)>,
) {
    if sim.paused {
        return;
    }
    let dt = time.delta_secs().max(1e-4);
    let now = time.elapsed_secs();
    let perception_sq = sim.perception * sim.perception;
    let mut rng = rand::thread_rng();

    // Measurement-noise stds collapse to ~0 when uncertainty is toggled off.
    let (range_std, bearing_std) = if sim.noise {
        (RANGE_STD, BEARING_STD)
    } else {
        (0.0, 0.0)
    };
    let r_cov = Mat3n::from_diagonal(&Vec3n::new(
        (range_std * range_std).max(1e-6),
        (bearing_std * bearing_std).max(1e-6),
        (bearing_std * bearing_std).max(1e-6),
    ));

    let mut err_sq_sum = 0.0f32;
    let mut err_count = 0.0f32;

    for (drone, sl, mut beliefs) in &mut query {
        let me_true = snap.0[drone.id].pos; // physical sensor origin (ground truth)
        let ego_est = sl.x; // where the drone BELIEVES it is — anchors the EKF

        // 1) Predict: coast every existing track forward and grow its covariance.
        for track in beliefs.0.values_mut() {
            track.predict(dt);
        }

        // 2) Update: for each neighbor within sensing range, synthesize a noisy
        //    range+bearing measurement (subject to packet loss) and fuse it.
        for (j, other) in snap.0.iter().enumerate() {
            if !other.alive || j == drone.id {
                continue;
            }
            if me_true.distance_squared(other.pos) > perception_sq {
                continue;
            }
            // Comms tolerance: the measurement packet may not arrive this step.
            if sim.noise && rng.gen::<f32>() < PACKET_LOSS {
                continue;
            }
            // The PHYSICAL measurement is the true relative geometry + noise...
            let true_meas = meas_from_relative(other.pos - me_true);
            let z = true_meas
                + Vec3n::new(
                    gauss(&mut rng, range_std),
                    gauss(&mut rng, bearing_std),
                    gauss(&mut rng, bearing_std),
                );
            // ...but the filter interprets it from the ESTIMATED self-pose, so
            // self-localization error propagates into the neighbor estimate.
            beliefs
                .0
                .entry(j)
                .and_modify(|t| t.update(ego_est, z, &r_cov, now))
                .or_insert_with(|| Track::new(ego_est, z, now));
        }

        // 3) Prune stale tracks (neighbor out of range / silent too long).
        beliefs.0.retain(|_, t| now - t.last_seen <= TRACK_TIMEOUT);

        // Metric: how far each belief is from the truth right now.
        for (&j, t) in beliefs.0.iter() {
            if snap.0[j].alive {
                err_sq_sum += t.pos().distance_squared(snap.0[j].pos);
                err_count += 1.0;
            }
        }
    }

    if err_count > 0.0 {
        let rms = (err_sq_sum / err_count).sqrt();
        metrics.est_error += 0.1 * (rms - metrics.est_error);
    }
}

/// Approximately-Gaussian noise via the sum of two uniforms.
fn gauss(rng: &mut rand::rngs::ThreadRng, std: f32) -> f32 {
    if std <= 0.0 {
        0.0
    } else {
        (rng.gen_range(-1.0..1.0) + rng.gen_range(-1.0..1.0)) * std
    }
}

// ---------------------------------------------------------------------------
// Actuation: two stages, both on LOCAL estimated info only.
//   (a) compute a PREFERRED velocity from flocking + goal + boundary, then
//   (b) project it to the nearest velocity that avoids collisions, using
//       reciprocal velocity obstacles whose radii are inflated by the EKF's
//       position uncertainty (act conservatively when unsure).
// ---------------------------------------------------------------------------

/// One collision constraint, later linearized into a velocity half-plane.
struct VoConstraint {
    rel: Vec3,   // estimated position of the other body, relative to the drone
    vel: Vec3,   // the other body's (estimated) velocity (0 for static obstacles)
    radius: f32, // combined safety radius
    planar: bool, // true for tall pillars (ignore the vertical component)
}

fn actuate_swarm(
    snap: Res<Snapshot>,
    sim: Res<SimState>,
    obstacles: Res<Obstacles>,
    time: Res<Time>,
    tick: Res<Tick>,
    mut metrics: ResMut<Metrics>,
    mut query: Query<(&Drone, &Beliefs, &SelfLoc, &GoalBelief, &mut Quad, Forces)>,
) {
    if sim.paused {
        return;
    }
    let perception_sq = sim.perception * sim.perception;
    let dt = time.delta_secs().max(1e-3);
    let mut rng = rand::thread_rng();
    let mut battery_sum = 0.0f32;
    let mut battery_n = 0.0f32;

    for (drone, beliefs, selfloc, goal_belief, mut quad, mut forces) in &mut query {
        let me = snap.0[drone.id];
        if !me.alive {
            continue;
        }

        // Asynchronous control: each drone re-plans only on its own ticks.
        // Between, it holds (zero-order hold) its last thrust command.
        if (tick.0 + drone.phase) % drone.period != 0 {
            quad.battery = (quad.battery - BATTERY_DRAIN * quad.last_thrust.length() * dt
                + BATTERY_RECHARGE * dt)
                .clamp(BATTERY_MIN, 1.0);
            forces.apply_linear_acceleration(quad.last_thrust);
            battery_sum += quad.battery;
            battery_n += 1.0;
            continue;
        }
        // Control uses the BELIEVED self-position (velocity is well-observed by
        // onboard IMU/VIO, so it's taken as the true value). No GPS here.
        let p = selfloc.x;
        let v_cur = me.vel;

        // --- (a) Preferred velocity: flocking coherence + goal + boundary ------
        let mut align = Vec3::ZERO;
        let mut center = Vec3::ZERO;
        let mut n_neighbors = 0.0f32;
        let mut constraints: Vec<VoConstraint> = Vec::new();

        for track in beliefs.0.values() {
            let track_pos = track.pos();
            let rel = track_pos - p; // estimated relative position
            if rel.length_squared() > perception_sq {
                continue;
            }
            n_neighbors += 1.0;
            align += track.vel();
            center += track_pos;

            // Safety radius grows with this neighbor's estimate uncertainty.
            let radius = 2.0 * DRONE_RADIUS + VO_BASE_MARGIN + VO_SIGMA_INFLATE * track.pos_sigma();
            constraints.push(VoConstraint {
                rel,
                vel: track.vel(),
                radius,
                planar: false,
            });
        }

        let mut v_pref = Vec3::ZERO;
        if n_neighbors > 0.0 {
            v_pref += (align / n_neighbors) * W_ALIGNMENT; // match the flock's velocity
            v_pref += (center / n_neighbors - p) * W_COHESION; // close the gap to center
        }
        // Goal seeking uses the drone's GOSSIPED goal belief — but only if it's
        // confident enough. Drones the wave hasn't reached yet just flock.
        if sim.goal_seek && goal_belief.var < GOAL_USABLE_VAR {
            let to_goal = goal_belief.pos - p;
            let d = to_goal.length();
            if d > 0.01 {
                let scale = (d / GOAL_ARRIVE_RADIUS).min(1.0); // ease off near the goal
                v_pref += to_goal.normalize() * CRUISE_SPEED * W_GOAL * scale;
            }
        }
        v_pref += boundary_push(p) * W_BOUNDARY; // steer inward near walls
        v_pref = v_pref.clamp_length_max(MAX_SPEED);

        // Obstacle constraints (non-reciprocal: they won't dodge for us).
        for (c, r) in obstacles.pillars.iter().copied() {
            constraints.push(VoConstraint {
                rel: c - p,
                vel: Vec3::ZERO,
                radius: r,
                planar: true,
            });
        }
        let (mc, mr) = obstacles.moving;
        constraints.push(VoConstraint {
            rel: mc - p,
            vel: obstacles.moving_vel,
            radius: mr,
            planar: false,
        });

        // --- (b) Project the preferred velocity to a collision-free one -------
        let v_new = if sim.avoidance {
            select_safe_velocity(v_pref, v_cur, &constraints)
        } else {
            v_pref
        };

        // Velocity controller: a bounded proportional acceleration toward the
        // target velocity. (Reaching it in one timestep would demand absurd
        // accelerations and permanently saturate the thrust.)
        let act_std = if sim.noise { ACTUATION_NOISE_STD } else { 0.0 };
        let accel = ((v_new - v_cur) * VEL_KP).clamp_length_max(MAX_LATERAL_ACCEL)
            + Vec3::new(
                gauss(&mut rng, act_std),
                gauss(&mut rng, act_std),
                gauss(&mut rng, act_std),
            );

        // --- Quadrotor: decoupled altitude + tilt-limited horizontal thrust ---
        // Vertical (collective) thrust holds altitude precisely and counters
        // gravity; horizontal thrust comes from tilting, which the airframe can
        // only do so fast (attitude lag) — so it slews toward the demand.
        let horiz_demand = Vec3::new(accel.x, 0.0, accel.z);
        let dh = (horiz_demand - quad.horiz).clamp_length_max(TILT_ACCEL_RATE * dt);
        quad.horiz += dh;
        let vert = accel.y + GRAVITY_ACCEL;
        let mut thrust = quad.horiz + Vec3::Y * vert;

        // Battery-limited total thrust; if over budget, shed horizontal (tilt
        // back toward level) to protect altitude.
        let thrust_max = THRUST_MAX * (0.6 + 0.4 * quad.battery);
        if thrust.length() > thrust_max {
            let v = thrust.y.min(thrust_max);
            let h_budget = (thrust_max * thrust_max - v * v).max(0.0).sqrt();
            thrust = quad.horiz.clamp_length_max(h_budget) + Vec3::Y * v;
        }
        quad.last_thrust = thrust; // remembered for the held (async) ticks

        // Battery drains with thrust, trickle-recharges, and never fully dies.
        quad.battery = (quad.battery - BATTERY_DRAIN * thrust.length() * dt
            + BATTERY_RECHARGE * dt)
            .clamp(BATTERY_MIN, 1.0);
        battery_sum += quad.battery;
        battery_n += 1.0;

        // Gravity is applied by the physics engine; we apply only the thrust.
        forces.apply_linear_acceleration(thrust);
    }

    if battery_n > 0.0 {
        metrics.battery += 0.05 * (battery_sum / battery_n - metrics.battery);
    }
}

/// Deterministic ORCA-style avoidance. Each constraint on a collision course
/// becomes a velocity half-plane `n·v ≤ b` that caps the closing speed along the
/// line of centers so the gap can't shrink below the safety radius within the
/// horizon. The preferred velocity is then projected onto the intersection of
/// those half-planes (iterative projection). Satisfying them is collision-free
/// for the horizon (given accurate relative velocities) — a guarantee the old
/// random sampler could only approximate.
fn select_safe_velocity(v_pref: Vec3, v_cur: Vec3, constraints: &[VoConstraint]) -> Vec3 {
    let mut planes: Vec<(Vec3, f32)> = Vec::new(); // (normal n, bound b)
    for c in constraints {
        let (rel, vother) = if c.planar {
            (Vec3::new(c.rel.x, 0.0, c.rel.z), Vec3::new(c.vel.x, 0.0, c.vel.z))
        } else {
            (c.rel, c.vel)
        };
        let dist = rel.length();
        if dist < 1e-3 {
            continue;
        }
        // Only constrain bodies we're actually about to hit within the horizon.
        let w = v_cur - vother;
        if time_to_collision(rel, w, c.radius, VO_TIME_HORIZON).is_infinite() {
            continue;
        }
        let n = rel / dist; // unit vector from the drone toward the other body
        // Keep our closing speed bounded: (v - vother)·n ≤ (dist - radius)/horizon.
        let s_max = (dist - c.radius) / VO_TIME_HORIZON; // may be < 0 if overlapping
        planes.push((n, vother.dot(n) + s_max));
    }
    if planes.is_empty() {
        return v_pref.clamp_length_max(MAX_SPEED);
    }
    // Project v_pref onto the intersection of the half-planes (+ speed cap).
    let mut v = v_pref;
    for _ in 0..16 {
        for &(n, b) in &planes {
            let excess = v.dot(n) - b;
            if excess > 0.0 {
                v -= n * excess; // project onto this half-plane's boundary
            }
        }
        v = v.clamp_length_max(MAX_SPEED);
    }
    v
}

/// Time until a body at relative position `rel` moving with relative velocity `w`
/// enters `radius` of the drone, or +inf if not within `horizon` seconds.
fn time_to_collision(rel: Vec3, w: Vec3, radius: f32, horizon: f32) -> f32 {
    let pp = rel.length_squared();
    let rr = radius * radius;
    if pp <= rr {
        return 0.0; // already overlapping
    }
    let a = w.length_squared();
    if a < 1e-6 {
        return f32::INFINITY; // no relative motion
    }
    // Solve |rel - w t|^2 = radius^2 for the earliest positive root.
    let b = -2.0 * rel.dot(w);
    let c = pp - rr;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return f32::INFINITY;
    }
    let sq = disc.sqrt();
    let t1 = (-b - sq) / (2.0 * a);
    let t2 = (-b + sq) / (2.0 * a);
    let t_enter = if t1 > 0.0 {
        t1
    } else if t2 > 0.0 {
        t2
    } else {
        return f32::INFINITY;
    };
    if t_enter > horizon {
        f32::INFINITY
    } else {
        t_enter
    }
}

/// Inward push that grows as a drone nears a wall, floor, or ceiling.
fn boundary_push(p: Vec3) -> Vec3 {
    let margin = 2.5;
    let mut push = Vec3::ZERO;
    if p.x > ROOM_HALF - margin {
        push.x -= (p.x - (ROOM_HALF - margin)) / margin;
    } else if p.x < -ROOM_HALF + margin {
        push.x += ((-ROOM_HALF + margin) - p.x) / margin;
    }
    if p.z > ROOM_HALF - margin {
        push.z -= (p.z - (ROOM_HALF - margin)) / margin;
    } else if p.z < -ROOM_HALF + margin {
        push.z += ((-ROOM_HALF + margin) - p.z) / margin;
    }
    if p.y > CEIL_Y - margin {
        push.y -= (p.y - (CEIL_Y - margin)) / margin;
    } else if p.y < FLOOR_Y + margin {
        push.y += ((FLOOR_Y + margin) - p.y) / margin;
    }
    push
}

// ---------------------------------------------------------------------------
// Environment dynamics
// ---------------------------------------------------------------------------
fn move_goal(
    time: Res<Time>,
    sim: Res<SimState>,
    mut goal: ResMut<Goal>,
    mut markers: Query<&mut Transform, With<GoalMarker>>,
) {
    if sim.paused {
        return;
    }
    let t = time.elapsed_secs();
    // Lissajous tour of the arena.
    goal.pos = Vec3::new(
        (t * 0.27).sin() * (ROOM_HALF - 3.0),
        8.0 + (t * 0.19).sin() * 3.5,
        (t * 0.33).cos() * (ROOM_HALF - 3.0),
    );
    for mut tf in &mut markers {
        tf.translation = goal.pos;
    }
}

fn move_obstacle(
    time: Res<Time>,
    sim: Res<SimState>,
    mut obstacles: ResMut<Obstacles>,
    mut q: Query<&mut Transform, With<MovingObstacle>>,
) {
    if sim.paused {
        return;
    }
    let t = time.elapsed_secs();
    let pos = Vec3::new((t * 0.5).sin() * 9.0, 8.0 + (t * 0.7).cos() * 3.0, (t * 0.4).cos() * 9.0);
    // Analytic velocity so avoidance can anticipate where it's heading.
    let vel = Vec3::new(
        0.5 * 9.0 * (t * 0.5).cos(),
        -0.7 * 3.0 * (t * 0.7).sin(),
        -0.4 * 9.0 * (t * 0.4).sin(),
    );
    obstacles.moving.0 = pos;
    obstacles.moving_vel = vel;
    for mut tf in &mut q {
        tf.translation = pos;
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<SimState>,
    mut exit: MessageWriter<AppExit>,
    mut drones: Query<(&Drone, &mut Transform, &mut LinearVelocity)>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::Space) {
        sim.paused = !sim.paused;
    }
    if keys.just_pressed(KeyCode::KeyG) {
        sim.goal_seek = !sim.goal_seek;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        sim.show_links = !sim.show_links;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        sim.noise = !sim.noise;
    }
    if keys.just_pressed(KeyCode::KeyE) {
        sim.show_estimator = !sim.show_estimator;
    }
    if keys.just_pressed(KeyCode::KeyO) {
        sim.avoidance = !sim.avoidance;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        sim.cooperative = !sim.cooperative;
    }
    if keys.just_pressed(KeyCode::KeyP) {
        sim.perception = if sim.perception > 6.5 { 3.5 } else { sim.perception + 1.5 };
    }
    if keys.just_pressed(KeyCode::KeyR) {
        let mut rng = rand::thread_rng();
        for (_d, mut tf, mut vel) in &mut drones {
            tf.translation = Vec3::new(
                rng.gen_range(-11.0..-6.0),
                rng.gen_range(4.0..11.0),
                rng.gen_range(-11.0..-6.0),
            );
            vel.0 = Vec3::ZERO;
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers: gizmos, visuals, camera, HUD
// ---------------------------------------------------------------------------
fn draw_gizmos(mut gizmos: Gizmos, sim: Res<SimState>, snap: Res<Snapshot>, goal: Res<Goal>) {
    // Room wireframe.
    draw_room(&mut gizmos);

    // Goal influence sphere.
    gizmos.sphere(
        Isometry3d::from_translation(goal.pos),
        GOAL_ARRIVE_RADIUS,
        Color::srgba(0.2, 1.0, 0.5, 0.25),
    );

    // Comms graph: a faint line for every neighbor pair within perception.
    if sim.show_links {
        let pr_sq = sim.perception * sim.perception;
        for i in 0..snap.0.len() {
            let a = snap.0[i];
            if !a.alive {
                continue;
            }
            for j in (i + 1)..snap.0.len() {
                let b = snap.0[j];
                if !b.alive {
                    continue;
                }
                if a.pos.distance_squared(b.pos) <= pr_sq {
                    gizmos.line(a.pos, b.pos, Color::srgba(0.3, 0.55, 0.9, 0.10));
                }
            }
        }
    }
}

fn draw_room(gizmos: &mut Gizmos) {
    let c = Color::srgba(0.3, 0.4, 0.6, 0.5);
    let h = ROOM_HALF;
    let (y0, y1) = (FLOOR_Y, CEIL_Y);
    let corners = [
        Vec3::new(-h, y0, -h),
        Vec3::new(h, y0, -h),
        Vec3::new(h, y0, h),
        Vec3::new(-h, y0, h),
    ];
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        gizmos.line(a, b, c); // bottom edge
        gizmos.line(a + Vec3::Y * (y1 - y0), b + Vec3::Y * (y1 - y0), c); // top edge
        gizmos.line(a, a + Vec3::Y * (y1 - y0), c); // vertical edge
    }
}

/// Visualize the EGO drone's EKF: for each tracked neighbor draw the belief and
/// its 1-sigma covariance as a true ELLIPSOID (principal axes from the eigen-
/// decomposition of the 3x3 position covariance). With range+bearing sensing the
/// ellipsoid is elongated tangentially — long where bearing is uncertain, short
/// along the accurate range axis. Magenta lines show belief-to-truth error.
fn draw_estimator(
    sim: Res<SimState>,
    snap: Res<Snapshot>,
    anchors: Res<Anchors>,
    landmarks: Res<Landmarks>,
    map: Res<LandmarkMap>,
    mut gizmos: Gizmos,
    drones: Query<(&Drone, &Beliefs, &SelfLoc)>,
) {
    if !sim.show_estimator {
        return;
    }
    // Shared SLAM map: each mapped landmark's estimate (purple cross) and its
    // error line back to the true landmark position.
    for (lid, ltrue) in landmarks.0.iter().enumerate() {
        let est = map.0[lid];
        if !est.seen {
            continue;
        }
        let sigma = est.cov.symmetric_eigenvalues().max().max(0.0).sqrt();
        gizmos.sphere(
            Isometry3d::from_translation(est.pos),
            sigma.clamp(0.1, 3.0),
            Color::srgb(0.7, 0.3, 1.0),
        );
        gizmos.line(est.pos, *ltrue, Color::srgba(0.7, 0.3, 1.0, 0.5));
    }
    for (drone, beliefs, selfloc) in &drones {
        let truth = snap.0[drone.id];
        if !truth.alive {
            continue;
        }
        // Every drone: a faint line from where it IS to where it THINKS it is.
        // The collective length of these is the swarm's self-localization drift.
        gizmos.line(truth.pos, selfloc.x, Color::srgba(1.0, 0.55, 0.1, 0.5));

        if drone.id != EGO_DRONE {
            continue;
        }
        // Ego detail: true pose (white), believed pose (cyan), sensing range.
        gizmos.sphere(Isometry3d::from_translation(truth.pos), 0.6, Color::WHITE);
        gizmos.sphere(Isometry3d::from_translation(selfloc.x), 0.4, Color::srgb(0.2, 0.8, 1.0));
        gizmos.sphere(
            Isometry3d::from_translation(truth.pos),
            sim.perception,
            Color::srgba(1.0, 1.0, 1.0, 0.06),
        );
        // Which anchors are constraining the ego's self-estimate right now.
        for &a in anchors.0.iter() {
            if truth.pos.distance(a) <= ANCHOR_RANGE {
                gizmos.line(selfloc.x, a, Color::srgba(0.2, 0.8, 1.0, 0.35));
            }
        }
        // Neighbor beliefs: covariance ellipsoids + belief-to-truth error lines.
        for (&j, track) in beliefs.0.iter() {
            let belief = track.pos();
            draw_covariance(&mut gizmos, belief, track.pos_cov());
            if snap.0[j].alive {
                gizmos.line(belief, snap.0[j].pos, Color::srgb(1.0, 0.2, 0.7));
            }
        }
    }
}

/// Draw a 1-sigma covariance ellipsoid as its three principal axes.
fn draw_covariance(gizmos: &mut Gizmos, center: Vec3, cov: Mat3n) {
    let eig = cov.symmetric_eigen();
    let color = Color::srgba(1.0, 0.9, 0.2, 0.85);
    for k in 0..3 {
        let sigma = eig.eigenvalues[k].max(0.0).sqrt().clamp(0.05, 5.0);
        let v = eig.eigenvectors.column(k);
        let axis = Vec3::new(v[0], v[1], v[2]) * sigma;
        gizmos.line(center - axis, center + axis, color);
    }
}

/// Recolor drones: orange = informed (direct goal observers), green = a drone the
/// gossip wave has reached (knows the goal), blue = still goal-blind. Brightness
/// tracks speed. This makes the consensus front visible as it sweeps the swarm.
fn update_drone_visuals(
    snap: Res<Snapshot>,
    drones: Query<(&Drone, &GoalBelief, &ColorMat)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (drone, gb, color_mat) in &drones {
        let obs = snap.0[drone.id];
        let speed_t = (obs.vel.length() / MAX_SPEED).clamp(0.0, 1.0);
        let knows = gb.var < GOAL_USABLE_VAR;
        if let Some(mat) = materials.get_mut(&color_mat.0) {
            mat.base_color = drone_color(drone.informed, knows, speed_t);
            mat.emissive = drone_emissive(drone.informed, knows, speed_t);
        }
    }
}

/// Spin the rotor blades so the quads look alive.
fn spin_rotors(time: Res<Time>, sim: Res<SimState>, mut rotors: Query<&mut Transform, With<Rotor>>) {
    if sim.paused {
        return;
    }
    let d = ROTOR_SPEED * time.delta_secs();
    for mut t in &mut rotors {
        t.rotate_local_y(d);
    }
}

fn drone_color(informed: bool, knows_goal: bool, speed_t: f32) -> Color {
    if informed {
        Color::srgb(1.0, 0.55 + 0.3 * speed_t, 0.1) // orange
    } else if knows_goal {
        Color::srgb(0.1, 0.9, 0.3 + 0.3 * speed_t) // green
    } else {
        Color::srgb(0.1 + 0.4 * speed_t, 0.5, 1.0) // blue
    }
}

fn drone_emissive(informed: bool, knows_goal: bool, speed_t: f32) -> LinearRgba {
    let g = 0.4 + 1.6 * speed_t;
    if informed {
        LinearRgba::new(1.4 + speed_t, 0.5 * g, 0.05, 1.0)
    } else if knows_goal {
        LinearRgba::new(0.05, 1.0 + speed_t, 0.3 * g, 1.0)
    } else {
        LinearRgba::new(0.05, 0.4 * g, 1.2 + speed_t, 1.0)
    }
}

fn orbit_camera(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut orbit: ResMut<OrbitCam>,
    mut cam: Query<&mut Transform, With<Camera3d>>,
) {
    let target = Vec3::new(0.0, 8.5, 0.0);
    if buttons.pressed(MouseButton::Left) {
        // Drag to look around.
        orbit.yaw -= motion.delta.x * 0.006;
        orbit.pitch = (orbit.pitch - motion.delta.y * 0.006).clamp(-1.45, 1.45);
    } else {
        // Idle: a gentle automatic sweep.
        orbit.yaw += time.delta_secs() * 0.1;
    }
    // Scroll to zoom.
    orbit.radius = (orbit.radius - scroll.delta.y * 1.5).clamp(6.0, 45.0);

    let cp = orbit.pitch.cos();
    let dir = Vec3::new(orbit.yaw.sin() * cp, orbit.pitch.sin(), orbit.yaw.cos() * cp);
    for mut tf in &mut cam {
        tf.translation = target + dir * orbit.radius;
        tf.look_at(target, Vec3::Y);
    }
}

fn update_hud(
    snap: Res<Snapshot>,
    sim: Res<SimState>,
    goal: Res<Goal>,
    mut metrics: ResMut<Metrics>,
    mut hud: Query<&mut Text, With<HudText>>,
) {
    // Compute instantaneous metrics.
    let pr_sq = sim.perception * sim.perception;
    let mut total_neighbors = 0.0;
    let mut total_speed = 0.0;
    let mut at_goal = 0.0;
    let mut alive: f32 = 0.0;
    let mut nearest = f32::INFINITY;
    for i in 0..snap.0.len() {
        let a = snap.0[i];
        if !a.alive {
            continue;
        }
        alive += 1.0;
        total_speed += a.vel.length();
        if a.pos.distance(goal.pos) <= GOAL_ARRIVE_RADIUS {
            at_goal += 1.0;
        }
        for j in 0..snap.0.len() {
            if i == j {
                continue;
            }
            let b = snap.0[j];
            if !b.alive {
                continue;
            }
            let dsq = a.pos.distance_squared(b.pos);
            if dsq <= pr_sq {
                total_neighbors += 1.0;
            }
            if dsq < nearest {
                nearest = dsq;
            }
        }
    }
    let alive = alive.max(1.0);
    // Exponential smoothing for readability.
    let k = 0.1;
    metrics.avg_neighbors += k * (total_neighbors / alive - metrics.avg_neighbors);
    metrics.avg_speed += k * (total_speed / alive - metrics.avg_speed);
    metrics.frac_at_goal += k * (at_goal / alive - metrics.frac_at_goal);
    if nearest.is_finite() {
        metrics.nearest_pair += k * (nearest.sqrt() - metrics.nearest_pair);
    }

    let onoff = |b: bool| if b { "ON " } else { "OFF" };
    let text = format!(
        "DECENTRALIZED DRONE SWARM  ·  Avian3D + Bevy\n\
         ------------------------------------------------\n\
         drones: {n}   informed (know goal): {inf}\n\
         perception radius: {pr:.1} m\n\
         \n\
         avg neighbors / drone : {nb:.1}\n\
         avg speed             : {sp:.2} m/s\n\
         swarm at goal         : {fg:3.0} %\n\
         nearest pair gap      : {np:.2} m\n\
         self-loc error        : {se:.2} m (odometry + anchors + coop)\n\
         neighbor est. error   : {ee:.2} m (per-drone EKF, range+bearing)\n\
         goal info coverage    : {gc:3.0} % (gossip consensus)\n\
         mean battery          : {bt:3.0} %\n\
         SLAM map built        : {mb}/{ml} landmarks  (err {me_:.2} m)\n\
         \n\
         [Space] {pause}   [G] goal-seek {gs}   [L] links {lk}\n\
         [N] noise {ns}   [E] estimator {es}   [O] avoidance {av}\n\
         [C] cooperative {co}   [P] perception   [R] re-scatter   [Esc] quit",
        n = N_DRONES,
        inf = (N_DRONES as f32 * INFORMED_FRACTION).round() as usize,
        pr = sim.perception,
        nb = metrics.avg_neighbors,
        sp = metrics.avg_speed,
        fg = metrics.frac_at_goal * 100.0,
        np = metrics.nearest_pair,
        se = metrics.self_error,
        ee = metrics.est_error,
        gc = metrics.goal_coverage * 100.0,
        bt = metrics.battery * 100.0,
        mb = metrics.map_built,
        ml = N_LANDMARKS,
        me_ = metrics.map_error,
        pause = if sim.paused { "RESUME" } else { "PAUSE " },
        gs = onoff(sim.goal_seek),
        lk = onoff(sim.show_links),
        ns = onoff(sim.noise),
        es = onoff(sim.show_estimator),
        av = onoff(sim.avoidance),
        co = onoff(sim.cooperative),
    );
    for mut t in &mut hud {
        t.0 = text.clone();
    }
}

// ---------------------------------------------------------------------------
// Tests: verify the filters converge and the avoidance geometry is correct.
// These run headless (`cargo test`) — no window required.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn anchor_ring() -> Vec<Vec3> {
        vec![
            Vec3::new(-12.0, 14.5, -12.0),
            Vec3::new(12.0, 14.5, -12.0),
            Vec3::new(12.0, 14.5, 12.0),
            Vec3::new(-12.0, 14.5, 12.0),
            Vec3::new(0.0, 2.5, 0.0),
        ]
    }

    #[test]
    fn self_loc_drifts_without_anchors() {
        // Dead reckoning with a biased odometer must accumulate error.
        let mut sl = SelfLoc::new(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.1));
        let dt = 1.0 / 64.0;
        for _ in 0..640 {
            // True velocity zero, but odometry reports the bias -> pure drift.
            sl.predict(Vec3::splat(0.1), dt);
        }
        let truth = Vec3::new(0.0, 8.0, 0.0);
        assert!(sl.x.distance(truth) > 0.8, "expected drift, got {}", sl.x.distance(truth));
    }

    #[test]
    fn self_loc_bounded_with_anchors() {
        // The same biased odometer, but corrected by anchor ranges, stays tight.
        let anchors = anchor_ring();
        let truth = Vec3::new(0.0, 8.0, 0.0);
        let mut sl = SelfLoc::new(truth, Vec3::splat(0.1));
        let dt = 1.0 / 64.0;
        let r_var = ANCHOR_RANGE_STD * ANCHOR_RANGE_STD;
        for _ in 0..640 {
            sl.predict(Vec3::splat(0.1), dt); // biased odometry
            for a in &anchors {
                let r = truth.distance(*a);
                if r <= ANCHOR_RANGE {
                    sl.update_range(*a, r, r_var);
                }
            }
        }
        assert!(sl.x.distance(truth) < 0.3, "self-loc diverged: {}", sl.x.distance(truth));
    }

    #[test]
    fn cooperative_localization_rescues_anchor_blind_drone() {
        // Drone A sees an anchor and stays well-localized. Drone B sees NO anchor
        // and has a biased odometer — it can only stay bounded by fusing A's
        // broadcast pose via the relative measurement between them.
        let dt = 1.0 / 64.0;
        // A sees three well-spread anchors -> becomes confidently localized.
        let anchors = [
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(3.0, 4.0, 1.0),
            Vec3::new(-2.0, 7.0, -1.0),
        ];
        let a_truth = Vec3::new(1.0, 6.0, 0.0);
        let b_truth = Vec3::new(3.0, 6.0, 0.0); // both hovering, B is 2 m from A
        let a_seed = SelfLoc::new(a_truth, Vec3::splat(0.05));
        let b_seed = SelfLoc::new(b_truth, Vec3::splat(0.15)); // stronger drift

        let run = |coop: bool| {
            let mut a = a_seed.clone();
            let mut b = b_seed.clone();
            for _ in 0..1280 {
                // A: dead-reckons its bias, corrected by its anchors.
                a.predict(Vec3::splat(0.05), dt);
                for anchor in anchors {
                    a.update_range(anchor, a_truth.distance(anchor), ANCHOR_RANGE_STD.powi(2));
                }
                // B: dead-reckons its bias, NO anchor in range.
                b.predict(Vec3::splat(0.15), dt);
                if coop {
                    // B measures the (true) offset to A (neighbor minus self) and
                    // covariance-intersects with A's broadcast estimate. No gate:
                    // CI itself ignores A when A is the worse estimate.
                    let d = a_truth - b_truth;
                    let z_self = a.x - d;
                    let meas_cov = Mat3n::identity() * COOP_REL_STD.powi(2);
                    let (x, p) = covariance_intersection(b.x, b.p, z_self, a.p + meas_cov);
                    b.x = x;
                    b.p = p;
                }
            }
            b.x.distance(b_truth)
        };

        let without = run(false);
        let with = run(true);
        assert!(without > 2.0, "B should drift badly without coop, got {without}");
        assert!(with < 0.8, "coop should keep B bounded, got {with}");
        assert!(with < without * 0.3, "coop must help a lot: {with} vs {without}");
    }

    #[test]
    fn covariance_intersection_is_consistent_and_complementary() {
        // Two estimates of the same point: A is tight in X but loose in Y/Z,
        // B is tight in Y but loose in X/Z. CI should fuse to something tight in
        // BOTH X and Y (true point is the origin), never overconfident.
        let xa = Vec3::new(0.05, 1.0, 0.0); // good X, bad Y
        let pa = Mat3n::from_diagonal(&Vec3n::new(0.01, 4.0, 4.0));
        let xb = Vec3::new(1.0, 0.05, 0.0); // good Y, bad X
        let pb = Mat3n::from_diagonal(&Vec3n::new(4.0, 0.01, 4.0));

        let (x, p) = covariance_intersection(xa, pa, xb, pb);

        // Fused estimate is close to truth in both well-observed axes.
        assert!(x.x.abs() < 0.6 && x.y.abs() < 0.6, "fused estimate {x:?}");
        // And it's confident in both X and Y (tighter than either input there).
        assert!(p[(0, 0)] < 0.5, "X variance not tightened: {}", p[(0, 0)]);
        assert!(p[(1, 1)] < 0.5, "Y variance not tightened: {}", p[(1, 1)]);

        // Consistency: CI must never be overconfident. Fusing an estimate with
        // ITSELF (maximal correlation) must not shrink the covariance — a naive
        // Kalman update famously would. Determinant should not decrease.
        let (_, p_self) = covariance_intersection(xa, pa, xa, pa);
        assert!(
            p_self.determinant() >= pa.determinant() - 1e-6,
            "CI got overconfident fusing with itself: {} < {}",
            p_self.determinant(),
            pa.determinant()
        );
    }

    #[test]
    fn neighbor_ekf_converges() {
        // Repeated clean range+bearing measurements of a static neighbor should
        // pull the estimate onto the true position.
        let ego = Vec3::ZERO;
        let target = Vec3::new(3.0, 1.0, 2.0);
        let r = Mat3n::from_diagonal(&Vec3n::new(
            RANGE_STD * RANGE_STD,
            BEARING_STD * BEARING_STD,
            BEARING_STD * BEARING_STD,
        ));
        let z = meas_from_relative(target - ego);
        let mut t = Track::new(ego, z, 0.0);
        for _ in 0..500 {
            t.predict(1.0 / 64.0);
            t.update(ego, z, &r, 0.0);
        }
        assert!(t.pos().distance(target) < 0.2, "track off by {}", t.pos().distance(target));
    }

    #[test]
    fn goal_consensus_propagates_along_a_chain() {
        // A line of 6 drones; only #0 observes the goal. Gossip should carry the
        // belief down the chain so the far end ends up knowing where the goal is.
        let n = 6;
        let goal = Vec3::new(10.0, 8.0, -4.0);
        let mut belief = vec![(Vec3::ZERO, GOAL_INIT_VAR); n];
        let dt = 1.0 / 64.0;

        for _ in 0..600 {
            let prev = belief.clone(); // one step of comms latency
            for i in 0..n {
                belief[i].1 += GOAL_DECAY_VAR * dt; // staleness
                if i == 0 {
                    belief[i] = fuse_belief(belief[i], (goal, GOAL_MEAS_STD.powi(2)));
                }
                // Chain topology: i talks only to i-1 and i+1.
                let mut best = (Vec3::ZERO, f32::INFINITY);
                for &j in [i.wrapping_sub(1), i + 1].iter() {
                    if j < n {
                        let cand = (prev[j].0, prev[j].1 + GOAL_RELAY_PENALTY);
                        if cand.1 < best.1 {
                            best = cand;
                        }
                    }
                }
                if best.1 < belief[i].1 {
                    belief[i] = fuse_belief(belief[i], best);
                }
            }
        }

        // Every drone, including the far end, should now know the goal well.
        for i in 0..n {
            assert!(belief[i].1 < GOAL_USABLE_VAR, "drone {i} never learned the goal");
            assert!(
                belief[i].0.distance(goal) < 1.5,
                "drone {i} goal estimate off by {}",
                belief[i].0.distance(goal)
            );
        }
        // Confidence should degrade with distance from the source.
        assert!(belief[n - 1].1 > belief[0].1, "far drone should be less certain");
    }

    #[test]
    fn orca_caps_closing_speed() {
        // Obstacle 5 m straight ahead, radius 1, static; drone wants to fly
        // straight into it at full speed. The projected velocity must not close
        // along the line of centers faster than (dist - radius)/horizon.
        let c = VoConstraint {
            rel: Vec3::new(5.0, 0.0, 0.0),
            vel: Vec3::ZERO,
            radius: 1.0,
            planar: false,
        };
        let v_pref = Vec3::new(MAX_SPEED, 0.0, 0.0);
        let v = select_safe_velocity(v_pref, v_pref, std::slice::from_ref(&c));
        let s_max = (5.0 - 1.0) / VO_TIME_HORIZON;
        assert!(
            v.dot(Vec3::X) <= s_max + 0.05,
            "closing speed {} exceeds safe {s_max}",
            v.dot(Vec3::X)
        );
    }

    #[test]
    fn slam_map_estimate_is_accurate_and_usable() {
        // Two drones observe an unknown landmark from different viewpoints; the
        // CI-fused map estimate should be accurate and confident enough to use.
        let ltrue = Vec3::new(5.0, 6.0, -3.0);
        let self_p = Mat3n::identity() * 0.05;
        let observe = |ego: Vec3| {
            let d = ltrue - ego;
            let world = ego + relative_from_meas(meas_from_relative(d));
            let spread = RANGE_STD.max(d.length() * BEARING_STD).powi(2);
            (world, self_p + Mat3n::identity() * spread)
        };
        let (w1, c1) = observe(Vec3::new(1.0, 5.0, 0.0));
        let (w2, c2) = observe(Vec3::new(8.0, 7.0, -1.0));
        let (pos, cov) = covariance_intersection(w1, c1, w2, c2);
        assert!(pos.distance(ltrue) < 0.3, "map estimate off by {}", pos.distance(ltrue));
        assert!(
            cov.symmetric_eigenvalues().max() <= LANDMARK_USABLE_VAR,
            "map landmark not usable: {}",
            cov.symmetric_eigenvalues().max()
        );
    }

    #[test]
    fn meas_roundtrip() {
        // relative_from_meas should invert meas_from_relative.
        let d = Vec3::new(2.0, -1.5, 3.0);
        let back = relative_from_meas(meas_from_relative(d));
        assert!(d.distance(back) < 1e-3, "roundtrip off by {}", d.distance(back));
    }

    #[test]
    fn ttc_head_on_and_diverging() {
        // Approaching head-on: gap 5, radius 1, closing at 1 m/s -> hit at t=4.
        let ttc = time_to_collision(Vec3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0, 10.0);
        assert!((ttc - 4.0).abs() < 1e-3, "ttc was {ttc}");
        // Receding: never collides.
        let away = time_to_collision(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 1.0, 10.0);
        assert!(away.is_infinite());
        // Already overlapping: immediate.
        let now = time_to_collision(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO, 1.0, 10.0);
        assert_eq!(now, 0.0);
    }
}
