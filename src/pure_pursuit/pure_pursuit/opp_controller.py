#!/usr/bin/env python3
import math
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
 
class OpponentController:
    """
    Handles all opponent car logic: storing opponent state and
    driving the opponent via pure pursuit.

    The parent node should:
      1. Instantiate this class and pass a reference to itself (the Node).
      2. Call self.opp_sub = self.create_subscription(
             Odometry, '/opp_racecar/odom', self.opp_controller.opp_callback, 10
         )
    """

    def __init__(self, node, pure_pursuit_logic, waypoints, opp_drive_pub, min_la, max_la, la_ratio, spawn_waypoint_idx=20):        
        """
        Args:
            node:               The parent rclpy Node (used for logging).
            pure_pursuit_logic: The opponent's own PurePursuitLogic instance
            waypoints:          Numpy array of shape (N, 3) — x, y, v.
            opp_drive_pub:      Publisher for AckermannDriveStamped on /opp_drive.
            min_la:             Minimum lookahead distance (metres).
            max_la:             Maximum lookahead distance (metres).
            la_ratio:           Lookahead ratio divisor.
            spawn_waypoint_idx:  Waypoint index the opponent spawns at (default 20).

        """
        self.node = node
        self.pure_pursuit_logic = pure_pursuit_logic
        self.waypoints = waypoints
        self.opp_drive_pub = opp_drive_pub
        self.min_la = min_la
        self.max_la = max_la
        self.la_ratio = la_ratio

        # Latest opponent state — readable by the parent node for trailing logic
        self.opponent_data = None

        # --- Spawn pose ---
        self._has_spawned = False
        opp_x1, opp_y1 = waypoints[spawn_waypoint_idx, 0],     waypoints[spawn_waypoint_idx, 1]
        opp_x2, opp_y2 = waypoints[spawn_waypoint_idx + 1, 0], waypoints[spawn_waypoint_idx + 1, 1]
        self._spawn_x   = opp_x1
        self._spawn_y   = opp_y1
        self._spawn_yaw = math.atan2(opp_y2 - opp_y1, opp_x2 - opp_x1)
  
        self._goal_pose_pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
        node.create_timer(1.0, self._publish_spawn_pose)

    def opp_callback(self, msg: Odometry):
        """ROS2 subscription callback for /opp_racecar/odom."""
        self.opponent_data = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'vel': msg.twist.twist.linear.x,
        }
        self._execute_opp_pure_pursuit(msg)

    def _publish_spawn_pose(self):
        """One-shot timer callback — fires once then becomes a no-op."""
        if self._has_spawned:
            return
 
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(self._spawn_x)
        msg.pose.position.y = float(self._spawn_y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(self._spawn_yaw / 2.0)
        msg.pose.orientation.w = math.cos(self._spawn_yaw / 2.0)
 
        self._goal_pose_pub.publish(msg)
        self.node.get_logger().info(
            f"Opponent's start Pose -> X: {self._spawn_x:.3f}, Y: {self._spawn_y :.3f}, Yaw: {self._spawn_yaw:.3f}")
        self._has_spawned = True


    def _execute_opp_pure_pursuit(self, msg: Odometry):
        """Drive the opponent car using pure pursuit."""
        car_x = msg.pose.pose.position.x
        car_y = msg.pose.pose.position.y
        car_yaw = self._get_yaw_from_quat(msg.pose.pose.orientation)

        current_closest_idx = self.pure_pursuit_logic.current_idx
        track_target_vel = self.waypoints[current_closest_idx, 2]

        lookahead_dist = np.clip(
            self.max_la * track_target_vel / self.la_ratio,
            self.min_la,
            self.max_la,
        )

        target_pt_car, actual_la, target_idx = self.pure_pursuit_logic.find_target_waypoint(
            car_x, car_y, car_yaw, lookahead_dist
        )

        if target_idx == -1:
            self._publish_opp_drive(0.0, 0.0)
            return

        steer = self.pure_pursuit_logic.calculate_steering(target_pt_car, actual_la)
        target_vel = self.waypoints[target_idx, 2] * 0.6
        self._publish_opp_drive(steer, target_vel)

    def _publish_opp_drive(self, steer: float, vel: float):
        drive_msg = AckermannDriveStamped()
        #drive_msg.drive.speed = 0.0
        drive_msg.drive.speed = float(vel)
        drive_msg.drive.steering_angle = float(steer)
        self.opp_drive_pub.publish(drive_msg)

    @staticmethod
    def _get_yaw_from_quat(q) -> float:
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)