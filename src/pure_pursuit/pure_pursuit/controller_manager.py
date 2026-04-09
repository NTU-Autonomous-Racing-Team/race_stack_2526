#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
import math
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, PoseStamped
from pure_pursuit.pure_pursuit_logic import PurePursuitLogic
from pure_pursuit.pure_pursuit_logic_modified import PurePursuitLogic as PurePursuitLogicModified
from pure_pursuit.ftg_logic import FTGLogic
from frenet_conversion.frenet_converter import FrenetConverter
from pure_pursuit.opp_controller import OpponentController
from state_machine.drive_state import DriveState


# TODO: Rewire the trailing logic

class ControllerManager(Node):
    def __init__(self):
        super().__init__('controller_manager_node')

        self.declare_parameter("waypoints_path", "./src/pure_pursuit/racelines/korea_mintime_sparse.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("min_lookahead", 0.6)#0.4#2.0 #0.6
        self.declare_parameter("max_lookahead", 1.8)#4.0 #1.8
        self.declare_parameter("lookahead_ratio", 9.0)#8.0 #9 for copy
        self.declare_parameter("K_p", 0.5)
        self.declare_parameter("steering_limit", 25.0) # Degrees
        self.declare_parameter("velocity_percentage", 1.0)
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("local_waypoints_window", 100)
        self.declare_parameter("transition_d_threshold", 0.1)  # meters lateral offset
        self.declare_parameter("transition_s_threshold", 1.0)  # meters ahead/behind nearest waypoint
        self.declare_parameter("reverse_waypoints", False) # Reverse the waypoints to drive in anticlockwise direction
        
        self.path = self.get_parameter("waypoints_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.drive_topic = self.get_parameter("drive_topic").value
        self.min_la = self.get_parameter("min_lookahead").value
        self.max_la = self.get_parameter("max_lookahead").value
        self.la_ratio = self.get_parameter("lookahead_ratio").value
        self.kp = self.get_parameter("K_p").value
        self.steer_limit = np.radians(self.get_parameter("steering_limit").value)
        self.vel_percent = self.get_parameter("velocity_percentage").value
        self.wheelbase = self.get_parameter("wheelbase").value
        self.local_waypoints_window = int(self.get_parameter("local_waypoints_window").value)
        
        # Dynamic Lookahead
        self.la_ratio = self.get_parameter("lookahead_ratio").value
        self.min_la = self.get_parameter("min_lookahead").value
        self.max_la = self.get_parameter("max_lookahead").value

        # 2. Initialize Logic & Data
        if not os.path.exists(self.path):
            self.get_logger().error(f"Waypoints file not found: {self.path}")
            raise FileNotFoundError(self.path)

        self.waypoints = np.loadtxt(self.path, delimiter=',') # Assume x, y, v
        if self.get_parameter("reverse_waypoints").value:
            self.waypoints = self.waypoints[::-1].copy()    
            self.get_logger().info("Waypoints reversed for anticlockwise direction")

        self.pure_pursuit_logic = PurePursuitLogicModified(self.wheelbase, self.waypoints)
        self.pure_pursuit_logic_copy = PurePursuitLogicModified(self.wheelbase, self.waypoints)
        self.ftg_logic = FTGLogic()
        self.curr_velocity = 0.0
        self.current_state = DriveState.GB_TRACK
        self.latest_scan = None

        # 3. Pubs & Subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.state_sub = self.create_subscription(String, '/state', self.state_callback, 10)

        self.viz_pub = self.create_publisher(Marker, '/waypoint_markers', 10)
        self.path_viz_pub = self.create_publisher(Marker, '/full_track_path', 10)
        self.local_waypoints_pub = self.create_publisher(Path, '/local_waypoints', 10)
        # Trigger the path visualization once at the start
        # (Wait a tiny bit for RViz to connect)
        self.create_timer(1.0, self.publish_static_path)

        self.get_logger().info("Pure Pursuit Node Started")

    ### --- Safe Transition to GB_TRACK Parameters --- ###
        self.declare_parameter("max_steering_rate", 0.15) # Max radians the wheels can turn per frame
        self.max_steer_rate = self.get_parameter("max_steering_rate").value
        self.last_steering_angle = 0.0

        self.declare_parameter("transition_frames_required", 30) # Number of consecutive frames needed
        self.required_frames = self.get_parameter("transition_frames_required").value
        self.consecutive_valid_frames = 0

        self.declare_parameter("clearance_distance", 1.5) # Meters ahead that must be clear
        self.declare_parameter("safe_corridor_width", 0.45) # Width of the car + safety margin
        
        self.in_transition = False
        self.has_initialized_idx = False
    ### --- END OF TRANSITION PARAMETERS --- ###

    ### ---- OPPONENT CAR IN SIM ---- ###
        self.declare_parameter("opp_odom_topic", "/opp_racecar/odom")
        self.declare_parameter("opp_drive_topic", "/opp_drive")
        self.opp_odom_topic = self.get_parameter("opp_odom_topic").value
        self.opp_drive_topic = self.get_parameter("opp_drive_topic").value
        self.opp_drive_pub = self.create_publisher(AckermannDriveStamped, self.opp_drive_topic, 10)
        self.opp_controller = OpponentController(
            node=self,
            pure_pursuit_logic=self.pure_pursuit_logic,
            waypoints=self.waypoints,
            opp_drive_pub=self.opp_drive_pub,
            min_la=self.min_la,
            max_la=self.max_la,
            la_ratio=self.la_ratio,
            spawn_waypoint_idx=20,
        )
        self.opp_sub = self.create_subscription(Odometry, self.opp_odom_topic, self.opp_controller.opp_callback, 10)
    ### --- END OF OPPONENT CAR  ###

    ### ---- FRENET CONVERSION FOR TRAILING LOGIC AND TRANSITION CHECKS ---- ###
        diffs = np.sqrt(
            np.diff(self.waypoints[:, 0])**2 + 
            np.diff(self.waypoints[:, 1])**2
        )
        actual_spacing = float(np.mean(diffs))

        wp_x = self.waypoints[:, 0]
        wp_y = self.waypoints[:, 1]

        # If your CSV doesn't have psi, compute it from consecutive waypoints:
        dx = np.diff(wp_x, append=wp_x[0] - wp_x[-1])
        dy = np.diff(wp_y, append=wp_y[0] - wp_y[-1])
        wp_psi = np.arctan2(dy, dx)

        self.frenet_converter = FrenetConverter(wp_x, wp_y, wp_psi)
        #self.frenet_converter.waypoints_distance_m = actual_spacing  # ← patch
        self.track_length = self.frenet_converter.raceline_length
    ### --- END OF FRENET CONVERSION ###

    ### --- SPAWN EGO CAR IN SIMULATION --- ### 
        # Extract starting coordinates from the waypoints array 
        x1, y1 = self.waypoints[0, 0], self.waypoints[0, 1]
        x2, y2 = self.waypoints[1, 0], self.waypoints[1, 1]
        self.start_yaw = math.atan2(y2 - y1, x2 - x1)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        # Use a one-shot timer to publish the pose 1 second after the node starts
        self.has_spawned = False
        self.create_timer(1.0, lambda: self.publish_spawn_pose(x1, y1, self.start_yaw))
    ### --- END OF SPAWN LOGIC --- ###

    def state_callback(self, msg):
        try:
            incoming_state = DriveState(msg.data)
        except ValueError:
            self.get_logger().warn(f"Received unknown state: {msg.data}")
            return

        if incoming_state != self.current_state:
            self.get_logger().info(f"--- STATE SWITCH: {self.current_state.value} -> {incoming_state.value} ---")
            
            # Entering GB_TRACK from another state — use FTG as bridge
            if incoming_state == DriveState.GB_TRACK and self.current_state != DriveState.GB_TRACK:
                self.in_transition = True
                self.get_logger().info("Transition to GB_TRACK: using FTG until on raceline")
                
            if incoming_state == DriveState.GB_TRACK:
                self.ftg_logic.prev_steering = 0.0
                
            # Safely update the state to the Enum
            self.current_state = incoming_state
    
    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        #self.get_logger().info(f"DEBUG: Active Logic: {self.current_state}", throttle_duration_sec=1.0)


        if not self.has_initialized_idx:
            car_x = msg.pose.pose.position.x
            car_y = msg.pose.pose.position.y
            distances = np.linalg.norm(
                self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1
            )
            start_idx = int(np.argmin(distances))
            self.pure_pursuit_logic.current_idx = start_idx
            self.pure_pursuit_logic_copy.current_idx = start_idx
            self.has_initialized_idx = True
            self.get_logger().info(f"Initialized current_idx={start_idx}")
            return

        self.curr_velocity = msg.twist.twist.linear.x
        
        if self.current_state == DriveState.FTGONLY:
            self.execute_ftg_logic()
            return

        # Bridge: use FTG until back on raceline
        elif self.current_state == DriveState.GB_TRACK and self.in_transition:
            self.execute_transition_logic(msg)
            return

        self.execute_pure_pursuit_logic(msg)

        #self.get_logger().info(f"Current Velocity: {self.curr_velocity:.2f} m/s", throttle_duration_sec=1.0)
    
    ### --- SAFE TRANSITION BETWEEN ANY STATE AND GB_TRACK--- ###

    ### Check no obstacles in a corridor ahead before allowing transition to GB_TRACK. 
    ### To prevent the controller from trying to follow the raceline when it's unsafe to do so.
    def is_path_clear(self):
        if self.latest_scan is None:
            return False 

        clear_dist = self.get_parameter("clearance_distance").value
        half_width = self.get_parameter("safe_corridor_width").value / 2.0

        ranges = np.array(self.latest_scan.ranges)
        
        # Calculate the angle for every ray in the scan
        angles = np.linspace(
            self.latest_scan.angle_min,
            self.latest_scan.angle_max,
            len(ranges)
        )

        # Filter: We only care about the front half of the LiDAR (-90 to +90 degrees)
        front_mask = (angles > -np.pi/2) & (angles < np.pi/2)
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        # Ignore inf and nan values
        valid_mask = np.isfinite(front_ranges)
        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        if len(front_ranges) == 0:
            return True

        # Convert polar (range, angle) to local cartesian (x forward, y lateral)
        xs = front_ranges * np.cos(front_angles)
        ys = front_ranges * np.sin(front_angles)

        # Check if any LiDAR points fall inside our forward safety corridor
        # x must be between 0.1m and clear_dist, y must be within the car's width
        in_corridor = (xs > 0.1) & (xs < clear_dist) & (np.abs(ys) < half_width)

        if np.any(in_corridor):
            self.get_logger().info("TRANSITION DENIED: Obstacle in forward corridor!", throttle_duration_sec=0.5)
            return False

        return True

    ### Check if we're close enough to the raceline (in d) to consider ourselves "on" it for transition purposes.
    def is_on_raceline(self, car_x, car_y):
        frenet = self.frenet_converter.get_frenet(
            np.array([car_x]), np.array([car_y])
        )
        s = float(frenet[0][0])
        d = float(frenet[1][0])  # lateral offset from raceline

        d_ok = abs(d) < self.get_parameter("transition_d_threshold").value

        self.get_logger().info(
            f"TRANSITION CHECK: d={d:.3f}m d_ok={d_ok}",
            throttle_duration_sec=0.5
        )
        return d_ok

    ### --- END OF SAFE TRANSITION LOGIC --- ###

    def execute_ftg_logic(self):
        if self.latest_scan is None:
            return
            
        speed, steer = self.ftg_logic.process_lidar(self.latest_scan)
        # self.get_logger().warn(f"FTG Active: Steer={steer:.2f}, Speed={speed:.2f}", throttle_duration_sec=1.0)
        self.publish_drive(steer, speed)

    def execute_pure_pursuit_logic(self, msg):
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)

        current_closest_idx = self.pure_pursuit_logic_copy.current_idx
        track_target_vel = self.waypoints[current_closest_idx, 2]
        # lookahead_dist = np.clip(max_la * self.curr_velocity / la_ratio, min_la, max_la)
        lookahead_dist = np.clip(self.max_la * track_target_vel / self.la_ratio, self.min_la, self.max_la)
        target_pt_car, actual_la, target_idx = self.pure_pursuit_logic_copy.find_target_waypoint(
            car_x, car_y, car_yaw, lookahead_dist
        )

        if target_idx == -1:
            self.publish_drive(0.0, 0.0) 
            return

        self.visualize_lookahead_point(self.waypoints[target_idx])
        self.publish_local_waypoints(window_size=self.local_waypoints_window)
        steer = self.pure_pursuit_logic_copy.calculate_steering(target_pt_car, actual_la)
        target_vel = self.waypoints[target_idx, 2] * self.vel_percent
        
        ### --- Trailing Logic --- ### 
        ### TODO: it uses /opp_racecar/odom as input in simulation which is unavailable in real world. 
        ### TODO: would need to create a logic in state machine
        ego_frenet = self.frenet_converter.get_frenet(np.array([car_x]), np.array([car_y]))
        ego_s = float(ego_frenet[0][0])
        if self.current_state == DriveState.TRAILING and self.opp_controller.opponent_data is not None:
            target_vel = self.execute_trailing_logic(msg, target_vel)
        else:
            self.pure_pursuit_logic_copy.i_gap = 0.0  # reset integrator when not trailing
        ### --- End of Trailing Logic --- ###

        self.publish_drive(steer, target_vel)

    def execute_trailing_logic(self, msg, global_speed_limit):
        opp_frenet = self.frenet_converter.get_frenet(
            np.array([self.opp_controller.opponent_data['x']]),
            np.array([self.opp_controller.opponent_data['y']])
        )
        opp_s = float(opp_frenet[0][0])            
        opp_vel = self.opp_controller.opponent_data['vel']
        target_vel = self.pure_pursuit_logic_copy.trailing_controller(
            ego_s, 
            self.curr_velocity,
            opp_s, 
            opp_vel,
            global_speed_limit,
            self.track_length
        )
        return target_vel

    def execute_transition_logic(self, msg):
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)
        path_clear = self.is_path_clear()
        on_raceline = self.is_on_raceline(car_x, car_y)
        if not path_clear:
            # Obstacle nearby — plain FTG, avoid first
            self.consecutive_valid_frames = 0
            self.get_logger().info("TRANSITION: obstacle detected, plain FTG", throttle_duration_sec=0.5)
            self.execute_ftg_logic()
            return
        # Path is clear — always use goal-directed FTG to pull toward raceline
        distances = np.linalg.norm(
            self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1
        )
        closest_idx = int(np.argmin(distances))
        # Also update current_idx so PP is ready when transition ends
        self.pure_pursuit_logic_copy.current_idx = closest_idx
        # Use waypoints_s at closest_idx as the seed s
        seed_s = float(self.frenet_converter.waypoints_s[closest_idx])
        # Pass seed_s directly to get_frenet to force correct segment
        frenet = self.frenet_converter.get_frenet(
            np.array([car_x]), np.array([car_y]),
            s=np.array([seed_s])  # ← constrain projection to correct segment
        )
        d = float(frenet[1][0])
        next_idx = (closest_idx + 1) % len(self.waypoints)
        raceline_heading = np.arctan2(
            self.waypoints[next_idx, 1] - self.waypoints[closest_idx, 1],
            self.waypoints[next_idx, 0] - self.waypoints[closest_idx, 0]
        )
        heading_error = (raceline_heading - car_yaw + np.pi) % (2 * np.pi) - np.pi
        lateral_correction = np.arctan2(-d, 2.0)
        target_angle = lateral_correction + 0.5 * heading_error
        speed, steer = self.ftg_logic.process_lidar(self.latest_scan, target_angle=target_angle)
        
        # self.get_logger().info(
            #     f"d={d:.3f} "
            #     f"raceline_heading={np.degrees(raceline_heading):.1f} "
            #     f"car_yaw={np.degrees(car_yaw):.1f} "
            #     f"heading_err={np.degrees(heading_error):.1f} "
            #     f"lat_corr={np.degrees(lateral_correction):.1f} "
            #     f"target_angle={np.degrees(target_angle):.1f}"
            # )
        self.publish_drive(steer, speed)
        if on_raceline:
            self.consecutive_valid_frames += 1
            self.get_logger().info(
                f"TRANSITION: valid frame {self.consecutive_valid_frames}/{self.required_frames}"
            )
            # Ensure the car stays on the raceline long enough before switching to pure pursuit
            if self.consecutive_valid_frames >= self.required_frames:
                self.in_transition = False
                self.consecutive_valid_frames = 0
                self.get_logger().info("Transition complete — resuming Pure Pursuit")
                return
        else:
            #if self.consecutive_valid_frames > 0:
                # self.get_logger().info("Transition streak reset")
            self.consecutive_valid_frames = 0
            return  # still off raceline, stay in transition next frame

    def publish_drive(self, steer, vel):
        # 1. Calculate how much the algorithm WANTS to change the steering
        steer_diff = steer - self.last_steering_angle
        # 2. Clip that change to your maximum allowed physical rate
        steer_diff = np.clip(steer_diff, -self.max_steer_rate, self.max_steer_rate)
        # 3. Apply the allowed change to the previous angle
        smoothed_steer = self.last_steering_angle + steer_diff
        # 4. Save for the next frame
        self.last_steering_angle = smoothed_steer

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(vel)
        drive_msg.drive.steering_angle = float(smoothed_steer)
        self.drive_pub.publish(drive_msg)

    def visualize_lookahead_point(self, point):
        """
        Publishes a marker to visualize the current lookahead point in RViz.
        :param point: A list or array [x, y] in the 'map' frame.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_point"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set the scale of the sphere (diameter in meters)
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        
        # Set the color (RGBA) - Bright Red for visibility
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        # Set the position of the marker
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0 # Waypoints are on the 2D plane
        
        # Publish the marker
        self.viz_pub.publish(marker)

    def publish_static_path(self):
        """
        Publishes all waypoints from the CSV as a single continuous green line.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "static_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP # This connects all points in order
        marker.action = Marker.ADD
        
        # Line width
        marker.scale.x = 0.1 
        
        # Color: Green (so it contrasts with your red lookahead dot)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        # Add all waypoints from your loaded CSV to the marker
        for wp in self.waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            marker.points.append(p)
        
        # If it's a loop, connect the last point to the first
        if len(self.waypoints) > 0:
            p_start = Point()
            p_start.x = float(self.waypoints[0][0])
            p_start.y = float(self.waypoints[0][1])
            marker.points.append(p_start)

        self.path_viz_pub.publish(marker)

    def publish_local_waypoints(self, window_size=100):
        """
        Publishes the rolling waypoint window used by pure pursuit as nav_msgs/Path.
        """
        num_waypoints = len(self.waypoints)
        if num_waypoints == 0:
            return

        window_size = int(max(1, min(window_size, num_waypoints)))
        start = int(self.pure_pursuit_logic.current_idx)

        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(window_size):
            idx = (start + i) % num_waypoints
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(self.waypoints[idx][0])
            pose.pose.position.y = float(self.waypoints[idx][1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.local_waypoints_pub.publish(path_msg)

    def publish_spawn_pose(self, x, y, yaw):
        if self.has_spawned:
            return # Ensure it only runs once
            
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set Position
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        
        # Convert Yaw to Quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Publish
        self.initial_pose_pub.publish(msg)
        self.get_logger().info(f"Ego's start Pose -> X: {x:.3f}, Y: {y:.3f}, Yaw: {yaw:.3f}")
        self.has_spawned = True

    def get_yaw_from_quat(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ControllerManager())
    rclpy.shutdown()

if __name__ == '__main__':
    main()