### To run ###
```python
ros2 launch state_machine state_machine_launch.py
```

### Changed/ Added: ### 
```
controller_manager, controller_manager.yaml, pure_pursuit_logic_modified, ftg_logic, opp_controller

state_machine.py, drive_state.py, state_machine launchfile

launch_master
```

For new maps: copy korea.png and korea.yaml into f1tenth_gym_ros/maps, and change sim.yaml

csv file has been pushed tgt  
 --note:   
korea_mintime_sparse.csv contains x,y,v  
korea.csv is the wpts of the track's centerline and track width  

**launch_master**
1. a parent launch file to run everything in the future, can modify at your convenience
2. e.g. act as a parent launch file to run other node's launch file

**Controller Manager.py**
1. Only change the parameters in controller_manager.yaml, and pass it to the node via launch file (check statemachine launch.py)
2. set parameter self.reverse_waypoints to True to run your car in anticlockwise direction as in competition, we might race in either direction during head to head race. Should ensure the calculation (especially the part using frenet) works fine in both direction
3. (for simulation) It will spawn 2 cars upon starting, with both running pure pursuit, can adjust the car initial pose by changing the index @ line 123 or 150&151
4. Added Trailing: follow the front car when overtake is not feasible
5. Added safe transition logic: Ensure it navigate safely to the raceline when switch from other states to GB_TRACK as pp is blind 

**pure_pursuit**
1. the modified version use ackermann kinematics model to calculate steering angle and use the intersection of lookahead circle and global wpts as the target wpts
both version work well in sim, but I am thinking that a proper kinematics model will be more suitable to real world as compared to using proportional gain. 
 -- Should test out both version ltr on

**state_machine**
1. wired detect.py into it (local planner not yet)
2. integrated trailng logic into it (still have some TODO to complete)

Both trailing and safe transition is working well when tested alone, but haven't verify and would need more testing after integrated into state machine, as it subjected to the robustness of decision logic of state machine and also missing some input from local planner

**state_estimation**
1. working in progress, ignore it for now
