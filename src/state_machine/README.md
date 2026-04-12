# state_machine Package

Main files:
- State transitions and RViz state legend: `state_machine/state_machine.py`
- State enum: `state_machine/drive_state.py`
- Dynamic opponent detector/tracker: `state_machine/opponent_detector.py`
- Static front obstacle detector: `state_machine/obstacle_detector.py`

Published state topic:
- `/state`

RViz marker topic:
- `/state_marker` (MarkerArray)
