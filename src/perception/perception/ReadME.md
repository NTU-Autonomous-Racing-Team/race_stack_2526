# Obstacle Detection Node — `detect.py`
## You can tune the self.boundary_inflation to push the boundaries(yellow lines) towards the raceline.
## Dependencies

Install the required Python packages before running:

```bash
pip install scikit-learn scipy numpy
```


---

## Running the Node

### Step 1 — Launch the simulator and load the map

Start the F1tenth gym simulator with your map as usual. Make sure the map is loaded and the ego car is localised before proceeding.

### Step 2 — Run the detect node

In a new terminal:

```bash
ros2 run perception detect
```
you will get warnings, just ignore those and let it keep running till you see the normal logs. usually you might want to run the detect node before running the pure pursuit(in competition)

### Step 3 — Spawn the opponent car

In another separate terminal, publish a static drive command to activate the opponent car:

```bash
ros2 topic pub /opp_drive ackermann_msgs/msg/AckermannDriveStamped "{drive: {speed: 0.0, steering_angle: 0.0}}" --rate 10
```

This keeps the opponent car stationary on the track so the detector can see it.

---

## Visualising in RViz

Open RViz and add the following three displays:

### 0. change no.of agents to 2 in f1tenth_gym_ros
- under 'Config' --> sim.yaml, 'change num_agent' to 2 
- change sx1 = -2.9, sy1 = 0.26

### 1. Opponent Car (ground truth position)
- Click **Add → By Display Type → RobotModel** or **Add → By Topic**
- Topic: `/opp_racecar/...` (whatever your simulator publishes for the opponent model)

### 2. Raceline Boundary Marker
- Click **Add → By Topic**
- Topic: `/raceline_marker`
- Type: **MarkerArray**
- This shows the green raceline centerline and the yellow left/right track boundaries

### 3. Obstacle Detection Marker
- Click **Add → By Topic**
- Topic: `/obstacle_markers`
- Type: **MarkerArray**
- Detected obstacles appear as **red spheres** on the track

Make sure the **Fixed Frame** in RViz is set to `map`.

---

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/tracked_obstacles` | `Float32MultiArray` | Obstacle data — see format below |
| `/obstacle_markers` | `MarkerArray` | RViz markers for detected obstacles |
| `/raceline_marker` | `MarkerArray` | RViz markers for raceline and track boundaries |

---

## `/tracked_obstacles` Message Format

Each detected obstacle publishes **7 floats** in a flat array. If there are N obstacles, the array contains N × 7 values. Chunk in groups of 7 when reading on the subscriber side.

| Index | Field | Description |
|---|---|---|
| 0 | `s` | Longitudinal position along the raceline (metres) |
| 1 | `d` | Lateral offset from the raceline (metres, negative = right) |
| 2 | `vs` | Velocity along the raceline (m/s) |
| 3 | `vd` | Lateral velocity (m/s) |
| 4 | `size_s` | Bounding box size in the s direction (metres) |
| 5 | `size_d` | Bounding box size in the d direction (metres) |
| 6 | `id` | Unique track ID (integer cast to float) |

Example subscriber code to parse the message:

```python
from std_msgs.msg import Float32MultiArray

def obstacles_cb(self, msg):
    data = msg.data
    num_obstacles = len(data) // 7
    for i in range(num_obstacles):
        s, d, vs, vd, size_s, size_d, obs_id = data[i*7 : i*7+7]
```

---
BTW I changed few lines in the frenet converter.py, it was giving garbage s and d values
