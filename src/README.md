# src Packages

- `f1tenth_gym_ros/`
  - Simulation bridge and map config
  - Main config: `f1tenth_gym_ros/config/sim.yaml`

- `pure_pursuit/`
  - Global tracking and command output
  - Main controller: `pure_pursuit/pure_pursuit/controller_manager.py`
  - Overtake-locked path tracking + global pursuit

- `state_machine/`
  - 2-state logic (`GB_TRACK`, `OVERTAKE`)
  - State node: `state_machine/state_machine/state_machine.py`

- `local_planner/`
  - Local overtake path generator
  - Spliner node: `local_planner/local_planner/spliner.py`

- `particle_filter/`
  - Localization package

- `gap_finder/`
  - Legacy package retained for reference
