#!/usr/bin/env python3
import math
import os
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

from pure_pursuit.frenet_converter import FrenetConverter
from pure_pursuit.pure_pursuit_logic_modified import PurePursuitLogic
from state_machine.drive_state import DriveState


class ControllerManager(Node):
    def __init__(self):
        super().__init__('controller_manager_node')

        self.declare_parameter('waypoints_path', '/sim_ws/src/pure_pursuit/racelines/korea_mintime_sparse.csv')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('state_topic', '/state')

        self.declare_parameter('min_lookahead', 0.8)
        self.declare_parameter('max_lookahead', 2.2)
        self.declare_parameter('lookahead_ratio', 8.5)
        self.declare_parameter('steering_limit', 25.0)
        self.declare_parameter('velocity_percentage', 0.95)
        self.declare_parameter('max_speed_cap', 8.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('local_waypoints_window', 120)
        self.declare_parameter('visualize_lookahead', True)

        self.declare_parameter('max_steering_rate', 0.16)
        self.declare_parameter('offtrack_d_limit', 1.55)
        self.declare_parameter('recovery_speed', 1.6)
        self.declare_parameter('publish_state_markers', True)
        self.declare_parameter('uturn_heading_error_deg', 80.0)
        self.declare_parameter('gb_min_speed', 1.0)
        self.declare_parameter('overtake_min_speed', 1.35)
        self.declare_parameter('dynamic_speed_min_scale', 0.72)
        self.declare_parameter('dynamic_speed_steer_penalty', 0.42)

        self.declare_parameter('planner_path_topic', '/planner/local_path')
        self.declare_parameter('planner_active_topic', '/planner/path_active')
        self.declare_parameter('locked_overtake_path_topic', '/planner/locked_overtake_path')
        self.declare_parameter('overtake_speed_boost', 0.6)
        self.declare_parameter('overtake_speed_cap', 2.0)
        self.declare_parameter('overtake_lookahead_scale', 0.95)
        self.declare_parameter('overtake_steering_limit_deg', 18.0)
        self.declare_parameter('overtake_max_steering_rate', 0.24)
        self.declare_parameter('overtake_min_path_points', 10)
        self.declare_parameter('overtake_reacquire_sec', 1.0)

        self.declare_parameter('opponent_detection_topic', '/opponent_detection')
        self.declare_parameter('opponent_detected_topic', '/opponent_detected')
        self.declare_parameter('opponent_hold_sec', 0.45)

        self.declare_parameter('use_ground_truth_opponent', False)
        self.declare_parameter('opp_odom_topic', '/ego_racecar/opp_odom')

        self.path = self.get_parameter('waypoints_path').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.state_topic = self.get_parameter('state_topic').value

        self.min_la = float(self.get_parameter('min_lookahead').value)
        self.max_la = float(self.get_parameter('max_lookahead').value)
        self.la_ratio = float(self.get_parameter('lookahead_ratio').value)
        self.steer_limit = np.radians(float(self.get_parameter('steering_limit').value))
        self.vel_percent = float(self.get_parameter('velocity_percentage').value)
        self.max_speed_cap = float(self.get_parameter('max_speed_cap').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.local_waypoints_window = int(self.get_parameter('local_waypoints_window').value)
        self.visualize_lookahead = bool(self.get_parameter('visualize_lookahead').value)

        self.max_steer_rate = float(self.get_parameter('max_steering_rate').value)
        self.offtrack_d_limit = float(self.get_parameter('offtrack_d_limit').value)
        self.recovery_speed = float(self.get_parameter('recovery_speed').value)
        self.publish_state_markers = bool(self.get_parameter('publish_state_markers').value)
        self.uturn_heading_error = np.radians(float(self.get_parameter('uturn_heading_error_deg').value))
        self.gb_min_speed = float(self.get_parameter('gb_min_speed').value)
        self.overtake_min_speed = float(self.get_parameter('overtake_min_speed').value)
        self.dynamic_speed_min_scale = float(self.get_parameter('dynamic_speed_min_scale').value)
        self.dynamic_speed_steer_penalty = float(self.get_parameter('dynamic_speed_steer_penalty').value)

        self.planner_path_topic = self.get_parameter('planner_path_topic').value
        self.planner_active_topic = self.get_parameter('planner_active_topic').value
        self.locked_overtake_path_topic = self.get_parameter('locked_overtake_path_topic').value
        self.overtake_speed_boost = float(self.get_parameter('overtake_speed_boost').value)
        self.overtake_speed_cap = float(self.get_parameter('overtake_speed_cap').value)
        self.overtake_lookahead_scale = float(self.get_parameter('overtake_lookahead_scale').value)
        self.overtake_steer_limit = np.radians(float(self.get_parameter('overtake_steering_limit_deg').value))
        self.overtake_max_steer_rate = float(self.get_parameter('overtake_max_steering_rate').value)
        self.overtake_min_path_points = int(self.get_parameter('overtake_min_path_points').value)
        self.overtake_reacquire_sec = float(self.get_parameter('overtake_reacquire_sec').value)

        self.opponent_detection_topic = self.get_parameter('opponent_detection_topic').value
        self.opponent_detected_topic = self.get_parameter('opponent_detected_topic').value
        self.opponent_hold_sec = float(self.get_parameter('opponent_hold_sec').value)

        self.use_ground_truth_opponent = bool(self.get_parameter('use_ground_truth_opponent').value)
        self.opp_odom_topic = self.get_parameter('opp_odom_topic').value

        self.waypoints = self._load_waypoints(self.path)
        self.pp_logic = PurePursuitLogic(self.wheelbase, self.waypoints)
        self.frenet_converter = FrenetConverter(self.waypoints[:, :2])
        self.track_length = self.frenet_converter.total_length

        self.current_state = DriveState.GB_TRACK
        self.has_initialized_idx = False

        self.curr_velocity = 0.0
        self.last_steering_angle = 0.0
        self.last_track_steer = 0.0

        self.latest_scan = None
        self.latest_odom = None

        self.local_path_points = np.empty((0, 2), dtype=float)
        self.planner_active = False
        self.last_local_path_ts = 0.0
        self.locked_overtake_path = np.empty((0, 2), dtype=float)
        self.overtake_path_idx = 0
        self.overtake_lock_ts = 0.0

        self.opponent_detected = False
        self.latest_opp_pose = None
        self.last_opp_seen_ts = 0.0
        self.last_opp_s = None
        self.last_opp_s_ts = None
        self.filtered_opp_vel = 0.0

        self.latest_gt_opp_odom = None

        self.last_lookahead_point = None

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.viz_pub = self.create_publisher(Marker, '/waypoint_markers', 10)
        self.path_viz_pub = self.create_publisher(Marker, '/full_track_path', 10)
        self.local_waypoints_pub = self.create_publisher(Path, '/local_waypoints', 10)
        self.locked_overtake_path_pub = self.create_publisher(Path, self.locked_overtake_path_topic, 10)

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(String, self.state_topic, self.state_callback, 10)
        self.create_subscription(Path, self.planner_path_topic, self.local_path_callback, 10)
        self.create_subscription(Bool, self.planner_active_topic, self.planner_active_callback, 10)
        self.create_subscription(PoseStamped, self.opponent_detection_topic, self.opponent_pose_callback, 10)
        self.create_subscription(Bool, self.opponent_detected_topic, self.opponent_flag_callback, 10)

        if self.use_ground_truth_opponent:
            self.create_subscription(Odometry, self.opp_odom_topic, self.gt_opp_odom_callback, 10)

        self.create_timer(1.0, self.publish_static_path)
        if self.visualize_lookahead:
            self.create_timer(0.1, self.publish_lookahead_timer)
        else:
            self._cleared_lookahead = False
            self.create_timer(1.0, self.clear_lookahead_marker)

        self.get_logger().info('Controller manager started')

    def _load_waypoints(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Waypoints file not found: {path}')

        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            if np.isnan(data).any() or data.ndim != 2:
                raise ValueError
        except Exception:
            data = np.loadtxt(path, delimiter=',')

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[1] < 2:
            raise ValueError('Waypoints must include at least x and y columns')
        if data.shape[1] < 3:
            v_col = np.full((data.shape[0], 1), 3.0, dtype=float)
            data = np.concatenate((data[:, :2], v_col), axis=1)
        return data[:, :3]

    def state_callback(self, msg):
        try:
            incoming = DriveState(msg.data)
        except ValueError:
            self.get_logger().warn(f'Unknown state received: {msg.data}')
            return

        if incoming != self.current_state:
            self.get_logger().info(f'Controller state {self.current_state.value} -> {incoming.value}')
            if self.current_state == DriveState.OVERTAKE and incoming != DriveState.OVERTAKE:
                self._reset_overtake_lock()
            if incoming == DriveState.OVERTAKE:
                self._reset_overtake_lock()
                self._lock_current_local_path()

        self.current_state = incoming

    def scan_callback(self, msg):
        self.latest_scan = msg

    def local_path_callback(self, msg: Path):
        if not msg.poses:
            self.local_path_points = np.empty((0, 2), dtype=float)
            return
        pts = [[float(p.pose.position.x), float(p.pose.position.y)] for p in msg.poses]
        self.local_path_points = np.array(pts, dtype=float)
        self.last_local_path_ts = time.time()
        if (
            self.current_state == DriveState.OVERTAKE
            and self.locked_overtake_path.shape[0] < self.overtake_min_path_points
        ):
            self._lock_current_local_path()

    def planner_active_callback(self, msg: Bool):
        self.planner_active = bool(msg.data)

    def _reset_overtake_lock(self):
        self.locked_overtake_path = np.empty((0, 2), dtype=float)
        self.overtake_path_idx = 0
        self.overtake_lock_ts = 0.0
        self._publish_locked_overtake_path()

    def _lock_current_local_path(self):
        if self.local_path_points.shape[0] < self.overtake_min_path_points:
            return False
        self.locked_overtake_path = self.local_path_points.copy()
        self.overtake_path_idx = 0
        self.overtake_lock_ts = time.time()
        self._publish_locked_overtake_path()
        return True

    def _publish_locked_overtake_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for pt in self.locked_overtake_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.locked_overtake_path_pub.publish(path_msg)

    def _find_locked_overtake_target(self, msg, lookahead):
        if self.locked_overtake_path.shape[0] < self.overtake_min_path_points:
            return None

        car_x = float(msg.pose.pose.position.x)
        car_y = float(msg.pose.pose.position.y)
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)

        path = self.locked_overtake_path
        n = path.shape[0]
        start = max(0, self.overtake_path_idx - 4)
        end = min(n, self.overtake_path_idx + 45)
        if end <= start:
            return None

        segment = path[start:end]
        dists = np.linalg.norm(segment - np.array([car_x, car_y]), axis=1)
        nearest_idx = start + int(np.argmin(dists))
        if nearest_idx > self.overtake_path_idx:
            self.overtake_path_idx = nearest_idx

        target_idx = None
        arc = 0.0
        for i in range(self.overtake_path_idx, n - 1):
            p0 = path[i]
            p1 = path[i + 1]
            arc += float(np.linalg.norm(p1 - p0))
            p1_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, p1)
            if p1_car[0] <= 0.05:
                continue
            if arc >= lookahead:
                target_idx = i + 1
                break

        if target_idx is None:
            best_i = None
            best_x = -1e9
            for i in range(self.overtake_path_idx, n):
                p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, path[i])
                if p_car[0] > best_x:
                    best_x = float(p_car[0])
                    best_i = i
            if best_i is None or best_x <= 0.05:
                return None
            target_idx = best_i

        self.overtake_path_idx = max(self.overtake_path_idx, target_idx)
        target_world = path[target_idx]
        target_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, target_world)
        target_la = float(max(np.linalg.norm(target_car), 0.1))
        return target_car, target_la, target_world

    def opponent_flag_callback(self, msg: Bool):
        self.opponent_detected = bool(msg.data)
        if self.opponent_detected:
            self.last_opp_seen_ts = time.time()

    def opponent_pose_callback(self, msg: PoseStamped):
        self.latest_opp_pose = msg.pose
        self.last_opp_seen_ts = time.time()

    def gt_opp_odom_callback(self, msg: Odometry):
        self.latest_gt_opp_odom = msg

    def odom_callback(self, msg):
        self.latest_odom = msg
        self.curr_velocity = float(msg.twist.twist.linear.x)
        if self.publish_state_markers:
            self.publish_state_color_marker(msg)

        car_x = float(msg.pose.pose.position.x)
        car_y = float(msg.pose.pose.position.y)

        if not self.has_initialized_idx:
            distances = np.linalg.norm(self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1)
            start_idx = int(np.argmin(distances))
            self.pp_logic.current_idx = start_idx
            self.has_initialized_idx = True
            self.get_logger().info(f'Initialized PP index at {start_idx}')
            return

        self.execute_track_logic(msg)

    def get_yaw_from_quat(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def transform_point_to_car_frame(self, car_x, car_y, car_yaw, point):
        dx = point[0] - car_x
        dy = point[1] - car_y
        cos_y = np.cos(-car_yaw)
        sin_y = np.sin(-car_yaw)
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y
        return np.array([local_x, local_y])

    def _compute_track_target(self, msg, lookahead_scale=1.0):
        car_x = float(msg.pose.pose.position.x)
        car_y = float(msg.pose.pose.position.y)
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)

        current_idx = int(self.pp_logic.current_idx)
        current_idx = max(0, min(current_idx, len(self.waypoints) - 1))
        track_target_vel = float(self.waypoints[current_idx, 2])

        lookahead = np.clip(self.max_la * track_target_vel / self.la_ratio, self.min_la, self.max_la)
        lookahead = max(self.min_la, lookahead * lookahead_scale)

        target_pt_car, actual_la, target_idx = self.pp_logic.find_target_waypoint(car_x, car_y, car_yaw, lookahead)
        if target_idx == -1:
            return None

        steer = self.pp_logic.calculate_steering(target_pt_car, actual_la)
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        target_vel = float(self.waypoints[target_idx, 2] * self.vel_percent)

        return target_pt_car, actual_la, target_idx, steer, target_vel

    def _estimate_opponent_frenet_from_lidar(self, msg):
        now = time.time()
        if (
            not self.opponent_detected
            or self.latest_opp_pose is None
            or (now - self.last_opp_seen_ts) > self.opponent_hold_sec
        ):
            return None

        car_x = float(msg.pose.pose.position.x)
        car_y = float(msg.pose.pose.position.y)
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)

        opp_local_x = float(self.latest_opp_pose.position.x)
        opp_local_y = float(self.latest_opp_pose.position.y)

        cos_yaw = math.cos(car_yaw)
        sin_yaw = math.sin(car_yaw)
        opp_map_x = car_x + opp_local_x * cos_yaw - opp_local_y * sin_yaw
        opp_map_y = car_y + opp_local_x * sin_yaw + opp_local_y * cos_yaw

        opp_s, _ = self.frenet_converter.get_frenet(opp_map_x, opp_map_y)
        opp_vel = self.filtered_opp_vel

        if self.last_opp_s is not None and self.last_opp_s_ts is not None:
            dt = now - self.last_opp_s_ts
            if dt > 1e-3:
                ds_raw = opp_s - self.last_opp_s
                ds = (ds_raw + 0.5 * self.track_length) % self.track_length - 0.5 * self.track_length
                instant_vel = max(0.0, ds / dt)
                self.filtered_opp_vel = 0.6 * self.filtered_opp_vel + 0.4 * instant_vel
                opp_vel = self.filtered_opp_vel

        self.last_opp_s = opp_s
        self.last_opp_s_ts = now

        return float(opp_s), float(np.clip(opp_vel, 0.0, 8.0))

    def _estimate_opponent_frenet_from_gt(self):
        if not self.use_ground_truth_opponent or self.latest_gt_opp_odom is None:
            return None
        x = float(self.latest_gt_opp_odom.pose.pose.position.x)
        y = float(self.latest_gt_opp_odom.pose.pose.position.y)
        opp_s, _ = self.frenet_converter.get_frenet(x, y)
        opp_vel = float(self.latest_gt_opp_odom.twist.twist.linear.x)
        return float(opp_s), max(0.0, opp_vel)

    def _get_opponent_frenet(self, msg):
        lidar_est = self._estimate_opponent_frenet_from_lidar(msg)
        if lidar_est is not None:
            return lidar_est
        return self._estimate_opponent_frenet_from_gt()

    def _find_local_path_target(self, msg, lookahead):
        if self.local_path_points.shape[0] == 0:
            return None

        car_x = float(msg.pose.pose.position.x)
        car_y = float(msg.pose.pose.position.y)
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)

        best = None
        best_score = float('inf')
        fallback = None

        for wp in self.local_path_points:
            p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, wp)
            if p_car[0] <= 0.05:
                continue

            dist = float(np.linalg.norm(p_car))
            if fallback is None:
                fallback = (p_car, max(dist, 0.1), wp)

            if dist <= lookahead * 1.4:
                score = abs(dist - lookahead)
                # Penalize targets that require aggressive lateral correction.
                score += 0.45 * abs(p_car[1])
                if score < best_score:
                    best_score = score
                    best = (p_car, max(dist, 0.1), wp)

        if best is not None:
            return best
        if fallback is not None:
            # If no near-lookahead point exists, use nearest forward point.
            return fallback

        # Last resort: no forward points, do not use local path this cycle.
        return None

    def _limit_overtake_steer(self, steer):
        return float(np.clip(steer, -self.overtake_steer_limit, self.overtake_steer_limit))

    def _nearest_track_index(self, x, y):
        distances = np.linalg.norm(self.waypoints[:, :2] - np.array([x, y]), axis=1)
        return int(np.argmin(distances))

    def _track_heading_error(self, car_x, car_y, car_yaw):
        idx = self._nearest_track_index(car_x, car_y)
        nxt = (idx + 1) % len(self.waypoints)
        track_h = np.arctan2(
            self.waypoints[nxt, 1] - self.waypoints[idx, 1],
            self.waypoints[nxt, 0] - self.waypoints[idx, 0],
        )
        return (track_h - car_yaw + np.pi) % (2 * np.pi) - np.pi, idx

    def execute_track_logic(self, msg):
        out = self._compute_track_target(msg)
        if out is None:
            self.publish_drive(0.0, 0.0)
            return

        target_pt_car, actual_la, target_idx, steer, target_vel = out

        self.last_track_steer = steer
        self.last_lookahead_point = self.waypoints[target_idx, :2]

        car_x = float(msg.pose.pose.position.x)
        car_y = float(msg.pose.pose.position.y)
        car_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)
        ego_s, ego_d = self.frenet_converter.get_frenet(car_x, car_y)
        heading_err, nearest_idx = self._track_heading_error(car_x, car_y, car_yaw)

        if abs(ego_d) > self.offtrack_d_limit or abs(heading_err) > self.uturn_heading_error:
            recovery_idx = (nearest_idx + 2) % len(self.waypoints)
            recovery_world = self.waypoints[recovery_idx, :2]
            recovery_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, recovery_world)
            recovery_la = float(np.clip(np.linalg.norm(recovery_car), 0.75, 1.4))
            recovery_steer = 1.35 * self.pp_logic.calculate_steering(recovery_car, recovery_la)
            recovery_steer = float(np.clip(recovery_steer, -0.26, 0.26))
            self.last_lookahead_point = recovery_world
            self.publish_drive(recovery_steer, min(self.recovery_speed, max(0.7, 0.40 * target_vel)))
            return

        if self.current_state == DriveState.OVERTAKE:
            has_locked_path = self.locked_overtake_path.shape[0] >= self.overtake_min_path_points
            if not has_locked_path:
                has_locked_path = self._lock_current_local_path()
            local = None
            if has_locked_path:
                local = self._find_locked_overtake_target(msg, actual_la * self.overtake_lookahead_scale)
                self._publish_locked_overtake_path()
            if local is not None:
                local_pt_car, local_la, local_world = local
                steer = self.pp_logic.calculate_steering(local_pt_car, local_la)
                steer = self._limit_overtake_steer(steer)
                target_vel = min(target_vel + self.overtake_speed_boost, self.overtake_speed_cap)
                self.last_lookahead_point = local_world
            else:
                # Keep line-following behavior in OVERTAKE if local target is temporarily unavailable.
                target_vel = min(target_vel, max(self.gb_min_speed, 0.85 * target_vel))

        self.publish_local_waypoints(window_size=self.local_waypoints_window)
        self.publish_drive(steer, target_vel)

    def publish_drive(self, steer, vel):
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        steer_rate_limit = self.max_steer_rate
        if self.current_state == DriveState.OVERTAKE:
            steer_rate_limit = max(self.max_steer_rate, self.overtake_max_steer_rate)
        steer_diff = steer - self.last_steering_angle
        steer_diff = float(np.clip(steer_diff, -steer_rate_limit, steer_rate_limit))
        smoothed_steer = self.last_steering_angle + steer_diff
        self.last_steering_angle = smoothed_steer

        if vel > 0.0:
            steer_ratio = abs(smoothed_steer) / max(self.steer_limit, 1e-3)
            dyn_scale = 1.0 - self.dynamic_speed_steer_penalty * steer_ratio
            dyn_scale = float(np.clip(dyn_scale, self.dynamic_speed_min_scale, 1.1))
            vel = float(vel * dyn_scale)
            if self.current_state == DriveState.OVERTAKE:
                vel = max(vel, self.overtake_min_speed)
            elif self.current_state == DriveState.GB_TRACK:
                vel = max(vel, self.gb_min_speed)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(min(self.max_speed_cap, max(0.0, vel)))
        drive_msg.drive.steering_angle = float(smoothed_steer)
        self.drive_pub.publish(drive_msg)

    def _state_color(self):
        if self.current_state == DriveState.GB_TRACK:
            return (0.1, 0.95, 0.1)
        if self.current_state == DriveState.OVERTAKE:
            return (0.2, 0.9, 1.0)
        return (1.0, 0.1, 0.1)

    def publish_state_color_marker(self, odom_msg):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'state_color_dot'
        marker.id = 42
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.28
        marker.scale.y = 0.28
        marker.scale.z = 0.28
        marker.color.a = 0.95
        r, g, b = self._state_color()
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.position.x = float(odom_msg.pose.pose.position.x)
        marker.pose.position.y = float(odom_msg.pose.pose.position.y)
        marker.pose.position.z = 0.38
        marker.pose.orientation.w = 1.0
        self.viz_pub.publish(marker)

        text = Marker()
        text.header = marker.header
        text.ns = 'state_text'
        text.id = 43
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = marker.pose.position.x
        text.pose.position.y = marker.pose.position.y
        text.pose.position.z = 0.78
        text.pose.orientation.w = 1.0
        text.scale.z = 0.28
        text.color.a = 0.95
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.text = self.current_state.value
        self.viz_pub.publish(text)

    def visualize_lookahead_point(self, point):
        if not self.visualize_lookahead:
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'lookahead_point'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0).to_msg()
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0
        self.viz_pub.publish(marker)

    def publish_lookahead_timer(self):
        if self.last_lookahead_point is None:
            return
        self.visualize_lookahead_point(self.last_lookahead_point)

    def clear_lookahead_marker(self):
        if getattr(self, '_cleared_lookahead', False):
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'lookahead_point'
        marker.id = 1
        marker.action = Marker.DELETE
        self.viz_pub.publish(marker)
        self._cleared_lookahead = True

    def publish_static_path(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'static_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for wp in self.waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.0
            marker.points.append(p)

        if len(self.waypoints) > 0:
            p_start = Point()
            p_start.x = float(self.waypoints[0][0])
            p_start.y = float(self.waypoints[0][1])
            marker.points.append(p_start)

        self.path_viz_pub.publish(marker)

    def publish_local_waypoints(self, window_size=120):
        num_waypoints = len(self.waypoints)
        if num_waypoints == 0:
            return

        window_size = int(max(1, min(window_size, num_waypoints)))
        start = int(self.pp_logic.current_idx)

        path_msg = Path()
        path_msg.header.frame_id = 'map'
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
    node = ControllerManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
