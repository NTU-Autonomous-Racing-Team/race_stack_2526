## Overview
Integrated the perception module (`detect.py`) with the local planning module (`spliner.py`). 

## Key Features
- **Unified Coordinate System**: Both nodes utilize `frenet_conversion.frenet_converter` 
- **Standardized Messaging**: Communication between nodes is handled via `f110_msgs.msg`
- **Visualization**: Added RViz  visualization of:
  - Obstacle bounding boxes (red CUBE markers with orientation, somehow not shown yet).
  - Local spline paths (purple Line Strip markers).

## Execution Pipeline

1. **Launch Simulation**:
   ```bash
   bash simulate.sh
   ```
2. **Run Perception Node**:
   ```bash
   ros2 run perception detect
   ```
3. **Run Local Planner Node**:
   ```bash
   ros2 run local_planner spliner
   ```
