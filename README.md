# Decentralized Drone Swarm — Avian3D + Bevy

A real-time 3D simulator for **autonomous drone swarms operating in dynamic,
GPS-denied indoor environments under uncertainty**. Every drone runs the *same*
local control law on *estimated* state — there is no central coordinator, no
shared world model, and **no ground truth touches the control loop**.

It's a complete, self-contained autonomy stack built on the
[Avian3D](https://crates.io/crates/avian3d) physics engine:

> **self-localization → cooperative localization → SLAM mapping → neighbor
> estimation → goal consensus → collision avoidance → quadrotor control**

…all under sensor noise, actuation noise, comms packet loss, asynchronous update
rates, and a moving goal and obstacle.

## Run

```bash
cargo run --release
```

(First build compiles Bevy + Avian; subsequent runs are instant.) The estimator
and avoidance math has headless unit tests:

```bash
cargo test
```

## Controls

| Key       | Action |
|-----------|--------|
| `Space`   | Pause / resume |
| `G`       | Toggle goal-seeking (emergent flocking vs. directed migration) |
| `L`       | Toggle the **comms graph** overlay (who can sense whom) |
| `N`       | Toggle **uncertainty** (sensor + actuation noise + packet loss) |
| `E`       | Toggle the **estimator overlay** (beliefs vs. ground truth) |
| `O`       | Toggle **collision avoidance** — compare *nearest pair gap* on/off |
| `C`       | Toggle **cooperative localization** — off, far drones drift |
| `P`       | Cycle the perception / comms radius |
| `R`       | Re-scatter the swarm |
| `Esc`     | Quit |

## Reading the scene

- **Drones** — *orange* = informed (direct goal observers), *green* = learned the
  goal via gossip, *blue* = still goal-blind. Watch the green consensus wave sweep
  outward. Brightness encodes speed.
- **Green sphere** = moving goal · **red box** = moving obstacle · **gray
  cylinders** = static obstacles · **cyan cubes** = fixed UWB anchors (clustered
  to one side) · **purple cubes** = unknown landmarks the swarm maps.
- Press `E` for the estimator overlay: a faint orange line from each drone's true
  position to where it *believes* it is (the live drift field); the ego drone's
  neighbor-uncertainty ellipsoids; and the purple SLAM map estimates with error
  lines to the true landmarks.

## What it demonstrates

| Challenge | How it's modeled |
|-----------|------------------|
| **Decentralized control** | Every drone runs the identical local law on only what it senses within its perception radius. No global blackboard. |
| **Self-localization (GPS-denied)** | A 3-state position **EKF** (`self_localize`) fuses **drifting biased odometry** with **ranges to fixed anchors**. Odometry alone diverges; anchors pin it down. |
| **Cooperative localization** | Anchors are clustered to one side. Drones in the gap relay pose info and fuse it with **Covariance Intersection** (`covariance_intersection`) — provably consistent under *unknown* correlation, so it's never overconfident and needs no manual gate. |
| **Cooperative SLAM** | The environment has **unknown landmarks**. Drones observe them (range+bearing) and triangulate a **shared map** (`slam_mapping`, fused via CI), which then feeds back into localization where anchors can't reach — loop closure. |
| **Neighbor estimation** | A per-neighbor 6-state **EKF** (`estimate_neighbors`) fuses nonlinear **range+bearing** measurements taken from the drone's *own estimated pose*, coasting through dropouts. Range is accurate, bearing is noisy → genuinely **anisotropic** uncertainty. |
| **Distributed goal consensus** | Only the informed minority observes the goal; everyone else learns it by **gossip** (`goal_consensus`) over the lossy comms graph. Beliefs decay as the goal moves, so consensus must continually re-propagate. |
| **Collision avoidance** | Deterministic **ORCA-style** half-plane projection (`select_safe_velocity`): each approaching body becomes a velocity constraint capping the closing speed; the preferred velocity is projected onto their intersection. Safety radii are **inflated by estimate uncertainty**. |
| **Quadrotor dynamics** | Real gravity; thrust acts along a **tilt-rate-limited** body axis and is capped by a **draining battery**. Drones must actively thrust to hover. |
| **Asynchrony** | Each drone re-plans on its **own variable rate** (16–64 Hz) and holds its last thrust command between updates (zero-order hold) — agents are not in lockstep. |
| **Uncertainty & comms tolerance** | Gaussian noise on measurements and thrust; neighbor/gossip packets dropped with probability `PACKET_LOSS`; filters predict through the gaps. Toggle `N`. |

## The agent loop

Each fixed step, every drone runs (all systems are decentralized — they iterate
agents independently):

```
advance_tick         global step counter (drives per-drone async rates)
sense_swarm          ground-truth snapshot — used ONLY to synthesize measurements
broadcast_estimates  each drone publishes (pose+covariance, goal belief)
self_localize        EKF: odometry + anchor ranges + map ranges + cooperative (CI) fixes
slam_mapping         fuse landmark observations into the shared map (CI)
estimate_neighbors   per-neighbor range+bearing EKFs from the estimated self-pose
goal_consensus       gossip the goal belief across the comms graph
actuate_swarm        preferred velocity → ORCA projection → tilt/battery-limited thrust
→ Avian physics step (FixedPostUpdate)
```

After `self_localize`, nothing downstream reads a true position. The
matrix math (EKFs, Covariance Intersection) uses stack-allocated `nalgebra`
`SMatrix`, so all the per-agent and per-neighbor filters run with no heap churn.

## How it uses Avian3D

- Drones are `RigidBody::Dynamic` sphere colliders under real `Gravity`; pillars
  are `Static`; the moving obstacle is `Kinematic`. Hard collisions are a physical
  backstop behind the velocity-space avoidance.
- Thrust is applied with `Forces::apply_linear_acceleration`; gravity is handled
  by the engine, so the controller must produce thrust that both hovers and steers.
- Control runs in `FixedUpdate` (before Avian's `FixedPostUpdate` step).

## Honest caveats

This is a learning harness, so a few things are abstracted: the SLAM map is a
shared resource (rather than per-drone maps merged over comms), the asynchrony is
in control (sensing still samples each tick), and the scalar goal-gossip uses
simple inverse-variance fusion. The localization/mapping fusion, by contrast, is
done properly with Covariance Intersection. Natural next steps: a real VIO/SLAM
front-end, per-drone maps with explicit map-merge messaging, full asynchronous
estimation, and an ORCA linear-program (vs. the iterative projection here).

## Layout

Everything is in [`src/main.rs`](src/main.rs) — constants and ECS components up
top, then one system per stage of the agent loop, then the unit tests. Built with
Bevy 0.18, Avian3D 0.6, and nalgebra.
