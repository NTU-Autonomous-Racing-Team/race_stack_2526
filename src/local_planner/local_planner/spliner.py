#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from scipy.interpolate import CubicSpline
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32

from .frenet_converter import FrenetConverter


class SplinerNode(Node):
    def __init__(self):
        super().__init__('spliner')

        self.declare_parameter('waypoints_path', '/sim_ws/src/pure_pursuit/racelines/korea_mintime_sparse.csv')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('opponent_detected_topic', '/opponent_detected')
        self.declare_parameter('opponent_pose_topic', '/opponent_detection')
        self.declare_parameter('obstacle_detected_topic', '/obstacle_detected')
        self.declare_parameter('obstacle_distance_topic', '/obstacle_distance')
        self.declare_parameter('obstacle_pose_topic', '/static_obstacle_pose')

        self.declare_parameter('local_path_topic', '/planner/local_path')
        self.declare_parameter('feasible_topic', '/planner/overtake_feasible')
        self.declare_parameter('path_active_topic', '/planner/path_active')

        self.declare_parameter('static_trigger_distance', 1.25)
        self.declare_parameter('opponent_trigger_distance', 3.4)
        self.declare_parameter('planner_lookahead_horizon', 8.0)
        self.declare_parameter('min_side_clearance', 1.0)
        self.declare_parameter('lane_half_width', 1.35)
        self.declare_parameter('boundary_margin', 0.2)
        self.declare_parameter('apex_lateral_margin', 0.4)
        self.declare_parameter('car_half_width', 0.16)
        self.declare_parameter('obstacle_half_width', 0.18)
        self.declare_parameter('overtake_lateral_buffer', 0.20)
        self.declare_parameter('min_s_gap_for_plan', 1.2)
        self.declare_parameter('static_line_d_threshold', 0.28)
        self.declare_parameter('static_obs_alpha', 0.25)
        self.declare_parameter('vmax', 6.0)
        self.declare_parameter('pre_apex_points', [2.0, 3.0, 4.0])
        self.declare_parameter('post_apex_points', [4.5, 5.0, 5.5])
        self.declare_parameter('resolution', 55)
        self.declare_parameter('update_rate_hz', 20.0)
        self.declare_parameter('path_hold_sec', 1.2)
        self.declare_parameter('side_lock_release_sec', 0.8)
        self.declare_parameter('lateral_smoothing_window', 5)
        self.declare_parameter('prefer_right_overtake', True)

        self.waypoints_path = self.get_parameter('waypoints_path').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.opponent_detected_topic = self.get_parameter('opponent_detected_topic').value
        self.opponent_pose_topic = self.get_parameter('opponent_pose_topic').value
        self.obstacle_detected_topic = self.get_parameter('obstacle_detected_topic').value
        self.obstacle_distance_topic = self.get_parameter('obstacle_distance_topic').value
        self.obstacle_pose_topic = self.get_parameter('obstacle_pose_topic').value

        self.local_path_topic = self.get_parameter('local_path_topic').value
        self.feasible_topic = self.get_parameter('feasible_topic').value
        self.path_active_topic = self.get_parameter('path_active_topic').value

        self.static_trigger_distance = float(self.get_parameter('static_trigger_distance').value)
        self.opponent_trigger_distance = float(self.get_parameter('opponent_trigger_distance').value)
        self.planner_lookahead_horizon = float(self.get_parameter('planner_lookahead_horizon').value)
        self.min_side_clearance = float(self.get_parameter('min_side_clearance').value)
        self.lane_half_width = float(self.get_parameter('lane_half_width').value)
        self.boundary_margin = float(self.get_parameter('boundary_margin').value)
        self.apex_lateral_margin = float(self.get_parameter('apex_lateral_margin').value)
        self.car_half_width = float(self.get_parameter('car_half_width').value)
        self.obstacle_half_width = float(self.get_parameter('obstacle_half_width').value)
        self.overtake_lateral_buffer = float(self.get_parameter('overtake_lateral_buffer').value)
        self.min_s_gap_for_plan = float(self.get_parameter('min_s_gap_for_plan').value)
        self.static_line_d_threshold = float(self.get_parameter('static_line_d_threshold').value)
        self.static_obs_alpha = float(self.get_parameter('static_obs_alpha').value)
        self.vmax = float(self.get_parameter('vmax').value)
        self.pre_apex_points = np.array(self.get_parameter('pre_apex_points').value, dtype=float)
        self.post_apex_points = np.array(self.get_parameter('post_apex_points').value, dtype=float)
        self.resolution = int(self.get_parameter('resolution').value)
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.path_hold_sec = float(self.get_parameter('path_hold_sec').value)
        self.side_lock_release_sec = float(self.get_parameter('side_lock_release_sec').value)
        self.lateral_smoothing_window = int(self.get_parameter('lateral_smoothing_window').value)
        self.prefer_right_overtake = bool(self.get_parameter('prefer_right_overtake').value)

        self.waypoints = self._load_waypoints(self.waypoints_path)
        self.converter = FrenetConverter(self.waypoints[:, :2])

        self.latest_odom = None
        self.latest_scan = None
        self.opponent_detected = False
        self.latest_opponent_pose = None
        self.static_obstacle_detected = False
        self.static_obstacle_distance = float('inf')
        self.latest_static_pose = None
        self.last_valid_points = None
        self.last_valid_ts = 0.0
        self.locked_side = 0.0
        self.last_hazard_ts = 0.0
        self.static_obs_s = None
        self.static_obs_d = None
        self.static_obs_last_ts = 0.0
        self.static_locked_apex_d = None

        self.path_pub = self.create_publisher(Path, self.local_path_topic, 10)
        self.feasible_pub = self.create_publisher(Bool, self.feasible_topic, 10)
        self.active_pub = self.create_publisher(Bool, self.path_active_topic, 10)

        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)
        self.create_subscription(Bool, self.opponent_detected_topic, self._opp_flag_cb, 10)
        self.create_subscription(PoseStamped, self.opponent_pose_topic, self._opp_pose_cb, 10)
        self.create_subscription(Bool, self.obstacle_detected_topic, self._static_flag_cb, 10)
        self.create_subscription(Float32, self.obstacle_distance_topic, self._static_dist_cb, 10)
        self.create_subscription(PoseStamped, self.obstacle_pose_topic, self._static_pose_cb, 10)

        self.create_timer(1.0 / max(self.update_rate_hz, 1.0), self._plan_cycle)
        self.get_logger().info('Local spliner planner started')

    def _load_waypoints(self, path):
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            if data.shape[1] < 2:
                raise ValueError('Waypoints file must have at least x,y columns')
            return data
        except Exception:
            data = np.loadtxt(path, delimiter=',')
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            if data.shape[1] < 2:
                raise ValueError('Waypoints file must have at least x,y columns')
            return data

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def _scan_cb(self, msg):
        self.latest_scan = msg

    def _opp_flag_cb(self, msg):
        self.opponent_detected = bool(msg.data)

    def _opp_pose_cb(self, msg):
        self.latest_opponent_pose = msg.pose

    def _static_flag_cb(self, msg):
        self.static_obstacle_detected = bool(msg.data)

    def _static_dist_cb(self, msg):
        d = float(msg.data)
        self.static_obstacle_distance = d if d > 0.0 else float('inf')

    def _static_pose_cb(self, msg):
        self.latest_static_pose = msg.pose

    def _scan_side_clearance(self):
        if self.latest_scan is None:
            return 0.0, 0.0

        ranges = np.array(self.latest_scan.ranges, dtype=float)
        ranges[np.isnan(ranges)] = 0.0
        inf_fill = self.latest_scan.range_max if self.latest_scan.range_max > 0.1 else 10.0
        ranges[np.isinf(ranges)] = inf_fill

        angles = self.latest_scan.angle_min + np.arange(len(ranges)) * self.latest_scan.angle_increment
        front = (angles > -math.radians(80.0)) & (angles < math.radians(80.0))
        left = front & (angles >= 0.0)
        right = front & (angles < 0.0)

        left_vals = ranges[left]
        right_vals = ranges[right]

        if left_vals.size == 0 or right_vals.size == 0:
            return 0.0, 0.0

        left_clear = float(np.percentile(left_vals, 70))
        right_clear = float(np.percentile(right_vals, 70))
        return left_clear, right_clear

    def _pose_car_to_map(self, local_pose):
        if self.latest_odom is None or local_pose is None:
            return None
        ego_pose = self.latest_odom.pose.pose
        q = ego_pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_local = float(local_pose.position.x)
        y_local = float(local_pose.position.y)
        map_x = float(ego_pose.position.x + x_local * cos_yaw - y_local * sin_yaw)
        map_y = float(ego_pose.position.y + x_local * sin_yaw + y_local * cos_yaw)
        return map_x, map_y

    def _hazard_candidate_dynamic(self, ego_s):
        if not self.opponent_detected or self.latest_opponent_pose is None:
            return None

        opp_x = float(self.latest_opponent_pose.position.x)
        opp_y = float(self.latest_opponent_pose.position.y)
        if opp_x <= 0.05:
            return None

        dist = math.hypot(opp_x, opp_y)
        if dist > self.opponent_trigger_distance:
            return None

        map_xy = self._pose_car_to_map(self.latest_opponent_pose)
        if map_xy is None:
            return None

        obs_s, obs_d = self.converter.get_frenet(map_xy[0], map_xy[1])
        s_gap = (obs_s - ego_s) % self.converter.total_length
        if s_gap < self.min_s_gap_for_plan or s_gap > self.planner_lookahead_horizon:
            return None

        return {
            'kind': 'dynamic',
            'dist': dist,
            's': float(obs_s),
            'd': float(obs_d),
            's_gap': float(s_gap),
        }

    def _hazard_candidate_static(self, ego_s):
        if not self.static_obstacle_detected or self.latest_static_pose is None:
            return None
        if not np.isfinite(self.static_obstacle_distance):
            return None
        if self.static_obstacle_distance > self.static_trigger_distance:
            return None

        static_x = float(self.latest_static_pose.position.x)
        static_y = float(self.latest_static_pose.position.y)
        if static_x <= 0.05:
            return None

        map_xy = self._pose_car_to_map(self.latest_static_pose)
        if map_xy is None:
            return None

        obs_s, obs_d = self.converter.get_frenet(map_xy[0], map_xy[1])
        if abs(obs_d) > self.static_line_d_threshold:
            return None

        now = time.time()
        if self.static_obs_s is None or (now - self.static_obs_last_ts) > self.side_lock_release_sec:
            self.static_obs_s = float(obs_s)
            self.static_obs_d = float(obs_d)
        else:
            alpha = float(np.clip(self.static_obs_alpha, 0.05, 0.95))
            ds_raw = float(obs_s - self.static_obs_s)
            ds = (ds_raw + 0.5 * self.converter.total_length) % self.converter.total_length - 0.5 * self.converter.total_length
            self.static_obs_s = float((self.static_obs_s + alpha * ds) % self.converter.total_length)
            self.static_obs_d = float((1.0 - alpha) * self.static_obs_d + alpha * float(obs_d))
        self.static_obs_last_ts = now

        s_gap = (self.static_obs_s - ego_s) % self.converter.total_length
        if s_gap < self.min_s_gap_for_plan or s_gap > self.planner_lookahead_horizon:
            return None

        return {
            'kind': 'static',
            'dist': float(self.static_obstacle_distance),
            's': float(self.static_obs_s),
            'd': float(self.static_obs_d),
            's_gap': float(s_gap),
        }

    def _choose_hazard(self, ego_s):
        candidates = []
        c_dyn = self._hazard_candidate_dynamic(ego_s)
        if c_dyn is not None:
            candidates.append(c_dyn)
        c_static = self._hazard_candidate_static(ego_s)
        if c_static is not None:
            candidates.append(c_static)

        if not candidates:
            return None
        return min(candidates, key=lambda c: c['dist'])

    def _build_spline_profile(self, ego_speed, hazard, left_clear, right_clear):
        d_obs = float(hazard['d'])
        s_gap = float(hazard['s_gap'])
        max_d = self.lane_half_width - self.boundary_margin
        if max_d <= 0.2:
            return None, None

        required_offset = max(
            self.apex_lateral_margin,
            self.car_half_width + self.obstacle_half_width + self.overtake_lateral_buffer,
        )

        left_space = max_d - d_obs
        right_space = max_d + d_obs
        can_left = left_space >= required_offset
        can_right = right_space >= required_offset
        if not can_left and not can_right:
            return None, None

        if self.prefer_right_overtake:
            # Enforce a single overtake side to prevent direction switches.
            if not can_right:
                return None, None
            side = -1.0
            self.locked_side = side
        else:
            if self.locked_side > 0.5:
                if not can_left:
                    return None, None
                side = 1.0
            elif self.locked_side < -0.5:
                if not can_right:
                    return None, None
                side = -1.0
            elif can_left and not can_right:
                side = 1.0
                self.locked_side = side
            elif can_right and not can_left:
                side = -1.0
                self.locked_side = side
            else:
                left_score = float(left_clear) + 0.35 * float(left_space)
                right_score = float(right_clear) + 0.35 * float(right_space)
                side = 1.0 if left_score >= right_score else -1.0
                self.locked_side = side

        if hazard['kind'] == 'static' and self.static_locked_apex_d is not None:
            d_apex = float(np.clip(self.static_locked_apex_d, -max_d, max_d))
        else:
            d_apex = d_obs + side * required_offset
            d_apex = float(
                np.clip(
                    d_apex,
                    -max_d,
                    max_d,
                )
            )
            if hazard['kind'] == 'static':
                self.static_locked_apex_d = d_apex

        if hazard['kind'] == 'static':
            sigma_v = 1.0
        else:
            sigma_v = 1.0 + min(max(ego_speed, 0.0) / max(self.vmax, 1e-3), 0.6)
        pre = np.array(self.pre_apex_points, dtype=float) * sigma_v
        post = np.array(self.post_apex_points, dtype=float) * sigma_v

        pre_limit = max(0.35, s_gap - 0.35)
        pre = np.clip(pre, 0.25, pre_limit)

        s_ctrl = np.array(
            [
                max(0.4, s_gap - pre[2]),
                max(0.8, s_gap - pre[1]),
                max(1.2, s_gap - pre[0]),
                max(1.6, s_gap),
                s_gap + post[0],
                s_gap + post[1],
                s_gap + post[2],
            ],
            dtype=float,
        )

        for i in range(1, len(s_ctrl)):
            if s_ctrl[i] <= s_ctrl[i - 1] + 0.12:
                s_ctrl[i] = s_ctrl[i - 1] + 0.12

        d_ctrl = np.array([0.0, 0.0, 0.0, d_apex, 0.0, 0.0, 0.0], dtype=float)
        spline = CubicSpline(s_ctrl, d_ctrl, bc_type='clamped')

        s_rel = np.linspace(0.0, s_ctrl[-1], self.resolution)
        d_rel = spline(s_rel)

        # Smooth lateral profile to reduce zig-zagging while keeping the same side.
        win = max(1, int(self.lateral_smoothing_window))
        if win > 1 and d_rel.size >= win:
            kernel = np.ones(win, dtype=float) / float(win)
            d_rel = np.convolve(d_rel, kernel, mode='same')

        max_abs_d = float(np.max(np.abs(d_rel)))
        if max_abs_d > max_d:
            return None, None

        return s_rel, d_rel

    def _publish_bool(self, pub, value):
        msg = Bool()
        msg.data = bool(value)
        pub.publish(msg)

    def _publish_empty_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        self.path_pub.publish(path)

    def _publish_path_from_points(self, points):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for px, py in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _plan_cycle(self):
        if self.latest_odom is None or self.latest_scan is None:
            self._publish_bool(self.feasible_pub, False)
            self._publish_bool(self.active_pub, False)
            self._publish_empty_path()
            return

        x = float(self.latest_odom.pose.pose.position.x)
        y = float(self.latest_odom.pose.pose.position.y)
        ego_speed = float(self.latest_odom.twist.twist.linear.x)
        ego_s, _ = self.converter.get_frenet(x, y)

        hazard = self._choose_hazard(ego_s)
        if hazard is None:
            now = time.time()
            if (now - self.last_hazard_ts) > self.side_lock_release_sec:
                self.locked_side = 0.0
                self.static_locked_apex_d = None
                self.static_obs_s = None
                self.static_obs_d = None
            # No hazard -> planner must stay inactive and clear the local overtake path.
            self._publish_bool(self.feasible_pub, False)
            self._publish_bool(self.active_pub, False)
            self._publish_empty_path()
            return

        self.last_hazard_ts = time.time()
        left_clear, right_clear = self._scan_side_clearance()
        if left_clear < self.min_side_clearance and right_clear < self.min_side_clearance:
            self._publish_bool(self.feasible_pub, False)
            self._publish_bool(self.active_pub, False)
            self._publish_empty_path()
            return

        s_rel, d_rel = self._build_spline_profile(ego_speed, hazard, left_clear, right_clear)
        if s_rel is None or d_rel is None:
            self._publish_bool(self.feasible_pub, False)
            self._publish_bool(self.active_pub, False)
            self._publish_empty_path()
            return

        points = []
        for s_delta, d in zip(s_rel, d_rel):
            s = (ego_s + s_delta) % self.converter.total_length
            px, py = self.converter.get_cartesian(s, d)
            points.append((float(px), float(py)))

        self.last_valid_points = points
        self.last_valid_ts = time.time()
        self._publish_path_from_points(points)
        self._publish_bool(self.feasible_pub, True)
        self._publish_bool(self.active_pub, True)


def main(args=None):
    rclpy.init(args=args)
    node = SplinerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
