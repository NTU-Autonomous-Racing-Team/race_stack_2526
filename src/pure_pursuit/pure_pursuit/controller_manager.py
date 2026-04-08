#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from f110_msgs.msg import DriveState as DriveStateMsg
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
from pure_pursuit.pure_pursuit_logic import PurePursuitLogic
from state_machine.drive_state import DriveState

class ControllerManager(Node):
    def __init__(self):
        super().__init__('controller_manager_node')

        default_waypoints_path = os.path.join(
            get_package_share_directory('pure_pursuit'),
            'racelines',
            'arc.csv',
        )
        self.declare_parameter("waypoints_path", default_waypoints_path)
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("min_lookahead", 2.0)
        self.declare_parameter("max_lookahead", 4.0)
        self.declare_parameter("lookahead_ratio", 8.0)
        self.declare_parameter("K_p", 0.5)
        self.declare_parameter("steering_limit", 25.0) # Degrees
        self.declare_parameter("velocity_percentage", 0.6)
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("local_waypoints_window", 100)

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

        # 2. Initialize Logic & Data
        if not os.path.exists(self.path):
            self.get_logger().error(f"Waypoints file not found: {self.path}")
            raise FileNotFoundError(self.path)

        self.waypoints = np.loadtxt(self.path, delimiter=',', skiprows=1) # Assume x, y, v
        self.pure_pursuit_logic = PurePursuitLogic(self.wheelbase, self.waypoints)
        self.curr_velocity = 0.0
        self.current_state = DriveState.GB_TRACK

        # 3. Pubs & Subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.state_sub = self.create_subscription(DriveStateMsg, '/state', self.state_callback, 10)

        self.get_logger().info("Pure Pursuit Node Started")
        self.viz_pub = self.create_publisher(Marker, '/waypoint_markers', 10)
        self.path_viz_pub = self.create_publisher(Marker, '/full_track_path', 10)
        self.local_waypoints_pub = self.create_publisher(Path, '/local_waypoints', 10)
        # Trigger the path visualization once at the start
        # (Wait a tiny bit for RViz to connect)
        self.create_timer(1.0, self.publish_static_path)

    def state_callback(self, msg):
        try:
            new_state = DriveState(msg.state)
        except ValueError:
            self.get_logger().warn(f"Unknown state received: {msg.state}")
            return

        if new_state != self.current_state:
            self.get_logger().info(f"--- STATE SWITCH: {self.current_state.value} -> {new_state.value} ---")
        self.current_state = new_state

    def get_yaw_from_quat(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        # self.get_logger().info(f"DEBUG: Active Logic: {self.current_state}", throttle_duration_sec=1.0)
        self.curr_velocity = msg.twist.twist.linear.x

        if self.current_state != DriveState.GB_TRACK:
            return

        self.execute_pure_pursuit_logic(msg)

    def execute_pure_pursuit_logic(self, msg):
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)

        # Dynamic Lookahead
        la_ratio = self.get_parameter("lookahead_ratio").value
        min_la = self.get_parameter("min_lookahead").value
        max_la = self.get_parameter("max_lookahead").value
        lookahead_dist = np.clip(max_la * self.curr_velocity / la_ratio, min_la, max_la)

        target_pt_car, actual_la, target_idx = self.pure_pursuit_logic.find_target_waypoint(
            car_x, car_y, car_yaw, lookahead_dist
        )

        if target_idx == -1:
            self.publish_drive(0.0, 0.0) 
            return

        self.visualize_lookahead_point(self.waypoints[target_idx])
        self.publish_local_waypoints(window_size=self.local_waypoints_window)
        steer = self.pure_pursuit_logic.calculate_steering(target_pt_car, actual_la, self.kp)
        target_vel = self.waypoints[target_idx, 2] * self.vel_percent
        
        self.publish_drive(steer, target_vel)
   
    def publish_drive(self, steer, vel):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(vel)
        drive_msg.drive.steering_angle = float(steer)
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

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ControllerManager())
    rclpy.shutdown()

if __name__ == '__main__':
    main()