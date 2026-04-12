# Korea Race Stack (Spliner Overtake Integration)

This repository contains our F1TENTH simulation stack for Korea map testing, focused on a robust local overtake planner (`spliner`) integrated with global pure pursuit.

Current driving modes:
- `GB_TRACK`: follow global racing waypoints
- `OVERTAKE`: follow local spline bypass around an on-raceline hazard

No FTG fallback is active in this branch. Overtake behavior is local-path-driven.

## Quick Start

From repo root:

```bash
bash simulate.sh
```

Then in a second terminal:

```bash
bash run.sh
```

The evaluation launch is:

```bash
ros2 launch state_machine state_machine_launch.py
```

## What This Branch Is For

- Validate spliner overtake logic end-to-end in simulation
- Stress test obstacle bypass and rejoin behavior
- Collect team feedback on path quality, state transitions, and tuning

## System Flow

1. `obstacle_detector` reads `/scan`, filters clusters, and publishes static obstacle status and pose.
2. `state_machine` decides `GB_TRACK` vs `OVERTAKE` using hazard + planner readiness.
3. `spliner` generates a local overtake trajectory (`/planner/local_path`) when hazard is valid.
4. `controller_manager` locks that local path during `OVERTAKE` and tracks it until clear, then returns to global tracking.

## Spliner Logic (Core)

The local path is generated in Frenet coordinates, then mapped back to Cartesian:

1. Convert ego and obstacle to Frenet `(s, d)` on the global raceline.
2. Compute longitudinal gap `s_gap` and reject hazards outside planning horizon.
3. Select overtake side (currently right-preferred in launch parameters).
4. Build a lateral control profile with cubic spline:
   - control points start at `d=0`
   - reach `d_apex` near obstacle
   - return to `d=0` after passing
5. Sample spline points and transform each `(s, d)` back to `(x, y)` for `/planner/local_path`.
6. Controller locks this path as `/planner/locked_overtake_path` and follows it consistently.

Because the profile is constrained to return to `d=0`, rejoin to the global line is natural and smooth, not a hard jump to a waypoint index.

## Visualization (RViz)

- Global track line: `/full_track_path`
- Lookahead point: `/waypoint_markers` (`lookahead_point`)
- Overtake path used by controller: `/planner/locked_overtake_path`
- State display above car:
  - `state_color_dot` and `state_text` on `/waypoint_markers`
  - `GB_TRACK` = green
  - `OVERTAKE` = cyan

## Key Files

- Spliner planner: `src/local_planner/local_planner/spliner.py`
- Frenet conversion for planner: `src/local_planner/local_planner/frenet_converter.py`
- State machine: `src/state_machine/state_machine/state_machine.py`
- Static obstacle detector: `src/state_machine/state_machine/obstacle_detector.py`
- Controller integration: `src/pure_pursuit/pure_pursuit/controller_manager.py`
- Sim config (map, topics, spawn): `src/f1tenth_gym_ros/config/sim.yaml`
- Team launch wiring: `src/state_machine/launch/state_machine_launch.py`

## Main Topics

- Inputs:
  - `/scan`
  - `/ego_racecar/odom`
  - `/obstacle_detected`
  - `/obstacle_distance`
  - `/static_obstacle_pose`
- Planner outputs:
  - `/planner/local_path`
  - `/planner/path_active`
  - `/planner/overtake_feasible`
- State:
  - `/state`
- Drive:
  - `/drive`

## Maps and Raceline

- Korea map files are in `src/f1tenth_gym_ros/maps/`
- Active map is configured in `src/f1tenth_gym_ros/config/sim.yaml`
- Raceline used by planner/controller:
  - `src/pure_pursuit/racelines/korea_mintime_sparse.csv`

## Tuning Notes

Most practical tuning happens through node parameters in:
- `src/state_machine/launch/state_machine_launch.py`
- `src/pure_pursuit/config/controller_manager.yaml`
- Detector sensitivity and track filter thresholds
- Spliner horizon, lateral margins, and smoothing
- State transition timing and overtake commitment
- Controller speed caps, steering rate limits, and overtake behavior

## Team Testing Checklist

- Confirm `GB_TRACK` remains stable when no hazard exists.
- Confirm `OVERTAKE` triggers only for valid on-raceline hazard.
- Confirm locked local path is followed without side switching.
- Confirm clean rejoin to global track after obstacle clearance.
- Report corner cases with logs and screenshots when possible.
