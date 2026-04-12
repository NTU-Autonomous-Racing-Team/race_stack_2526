#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


class OpponentDetector(Node):
    def __init__(self):
        super().__init__('opponent_detector')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('waypoints_path', '/sim_ws/src/pure_pursuit/racelines/arc.csv')
        self.declare_parameter('use_track_filter', True)
        self.declare_parameter('track_dist_thresh', 1.2)
        self.declare_parameter('detect_topic', '/opponent_detection')
        self.declare_parameter('detected_flag_topic', '/opponent_detected')
        self.declare_parameter('marker_topic', '/opponent_marker')
        self.declare_parameter('min_cluster_points', 6)
        self.declare_parameter('max_cluster_points', 80)
        self.declare_parameter('base_breakpoint', 0.12)
        self.declare_parameter('range_breakpoint_scale', 0.03)
        self.declare_parameter('min_range', 0.20)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('size_min_x', 0.25)
        self.declare_parameter('size_max_x', 1.20)
        self.declare_parameter('size_min_y', 0.15)
        self.declare_parameter('size_max_y', 0.70)
        self.declare_parameter('min_persist_frames', 3)
        self.declare_parameter('max_stale_time', 0.6)
        self.declare_parameter('vel_smoothing', 0.6)
        self.declare_parameter('dynamic_threshold', 0.6)
        self.declare_parameter('min_ego_speed', 0.8)
        self.declare_parameter('hold_time', 0.4)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.waypoints_path = self.get_parameter('waypoints_path').value
        self.use_track_filter = bool(self.get_parameter('use_track_filter').value)
        self.track_dist_thresh = float(self.get_parameter('track_dist_thresh').value)
        self.detect_topic = self.get_parameter('detect_topic').value
        self.detected_flag_topic = self.get_parameter('detected_flag_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value

        self.min_cluster_points = int(self.get_parameter('min_cluster_points').value)
        self.max_cluster_points = int(self.get_parameter('max_cluster_points').value)
        self.base_breakpoint = float(self.get_parameter('base_breakpoint').value)
        self.range_breakpoint_scale = float(self.get_parameter('range_breakpoint_scale').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.size_min_x = float(self.get_parameter('size_min_x').value)
        self.size_max_x = float(self.get_parameter('size_max_x').value)
        self.size_min_y = float(self.get_parameter('size_min_y').value)
        self.size_max_y = float(self.get_parameter('size_max_y').value)
        self.min_persist_frames = int(self.get_parameter('min_persist_frames').value)
        self.max_stale_time = float(self.get_parameter('max_stale_time').value)
        self.vel_smoothing = float(self.get_parameter('vel_smoothing').value)
        self.dynamic_threshold = float(self.get_parameter('dynamic_threshold').value)
        self.min_ego_speed = float(self.get_parameter('min_ego_speed').value)
        self.hold_time = float(self.get_parameter('hold_time').value)

        self.pub = self.create_publisher(PoseStamped, self.detect_topic, 10)
        self.flag_pub = self.create_publisher(Bool, self.detected_flag_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)

        self.ego_pose = None
        self.ego_speed = 0.0

        self.waypoints_xy = None
        if self.use_track_filter:
            try:
                waypoints = np.loadtxt(self.waypoints_path, delimiter=',', skiprows=1)
                if waypoints.ndim == 2 and waypoints.shape[1] >= 2:
                    self.waypoints_xy = waypoints[:, :2]
                else:
                    self.use_track_filter = False
                    self.get_logger().warn("Waypoints file format invalid, disabling track filter.")
            except Exception as exc:
                self.use_track_filter = False
                self.get_logger().warn(f"Failed to load waypoints from {self.waypoints_path}: {exc}")

        self.last_center = None
        self.last_time = None
        self.rel_vel_ema = np.zeros(2, dtype=float)
        self.persist_count = 0
        self.center_ema = None
        self.confirmed_center = None
        self.confirmed_size = None
        self.last_confirm_time = None

        self.get_logger().info("Opponent detector started")

    def on_odom(self, msg: Odometry):
        self.ego_pose = msg.pose.pose
        self.ego_speed = float(msg.twist.twist.linear.x)

    def get_yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def transform_car_to_map(self, x, y):
        if self.ego_pose is None:
            return None
        yaw = self.get_yaw_from_quat(self.ego_pose.orientation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        map_x = self.ego_pose.position.x + x * cos_yaw - y * sin_yaw
        map_y = self.ego_pose.position.y + x * sin_yaw + y * cos_yaw
        return map_x, map_y

    def distance_to_racing_line(self, map_x, map_y):
        if self.waypoints_xy is None or len(self.waypoints_xy) == 0:
            return None
        diffs = self.waypoints_xy - np.array([map_x, map_y])
        dists = np.linalg.norm(diffs, axis=1)
        return float(np.min(dists))

    def update_tracker(self, center, stamp_sec):
        if self.last_center is None or self.last_time is None:
            self.rel_vel_ema = np.zeros(2, dtype=float)
            self.persist_count = 1
        else:
            dt = stamp_sec - self.last_time
            if dt <= 0.0 or dt > self.max_stale_time:
                self.rel_vel_ema = np.zeros(2, dtype=float)
                self.persist_count = 1
            else:
                rel_vel = (center - self.last_center) / dt
                self.rel_vel_ema = (
                    self.vel_smoothing * rel_vel + (1.0 - self.vel_smoothing) * self.rel_vel_ema
                )
                self.persist_count += 1

        self.last_center = center
        self.last_time = stamp_sec

        dynamic_score = self.rel_vel_ema[0] + self.ego_speed
        is_dynamic = (
            self.ego_speed >= self.min_ego_speed and
            dynamic_score > self.dynamic_threshold and
            self.persist_count >= self.min_persist_frames
        )
        return bool(is_dynamic)

    def on_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = 0.0

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = (ranges > self.min_range) & (ranges < self.max_range)
        idxs = np.where(valid)[0]

        clusters = []
        if idxs.size > 0:
            current = [idxs[0]]
            for i in idxs[1:]:
                prev = current[-1]
                r_prev = ranges[prev]
                r_curr = ranges[i]
                gap = abs(r_curr - r_prev)
                thresh = self.base_breakpoint + self.range_breakpoint_scale * max(r_prev, r_curr)
                if gap > thresh or (i - prev) > 1:
                    clusters.append(current)
                    current = [i]
                else:
                    current.append(i)
            clusters.append(current)

        best = None
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp_sec == 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        for c in clusters:
            if len(c) < self.min_cluster_points or len(c) > self.max_cluster_points:
                continue
            pts = []
            for i in c:
                r = ranges[i]
                a = angles[i]
                x = r * math.cos(a)
                y = r * math.sin(a)
                pts.append((x, y))
            pts = np.array(pts)
            if pts.size == 0:
                continue
            xs = pts[:, 0]
            ys = pts[:, 1]
            if np.mean(xs) < 0.15:
                continue
            size_x = float(xs.max() - xs.min())
            size_y = float(ys.max() - ys.min())
            if not (self.size_min_x <= size_x <= self.size_max_x):
                continue
            if not (self.size_min_y <= size_y <= self.size_max_y):
                continue

            dist_to_line = None
            if self.use_track_filter and self.ego_pose is not None and self.waypoints_xy is not None:
                mapped = self.transform_car_to_map(float(np.mean(xs)), float(np.mean(ys)))
                if mapped is not None:
                    dist_to_line = self.distance_to_racing_line(mapped[0], mapped[1])
                    if dist_to_line is not None and dist_to_line > self.track_dist_thresh:
                        continue

            dist = float(np.linalg.norm([np.mean(xs), np.mean(ys)]))
            score = -dist
            if dist_to_line is not None:
                score -= 0.2 * dist_to_line
            if best is None or score > best['score']:
                best = {
                    'center': (float(np.mean(xs)), float(np.mean(ys))),
                    'size_x': size_x,
                    'size_y': size_y,
                    'score': score,
                }

        detected = False
        output_center = None
        output_size = None
        if best is not None:
            center = np.array(best['center'], dtype=float)
            is_dynamic = self.update_tracker(center, stamp_sec)
            if is_dynamic:
                if self.center_ema is None:
                    self.center_ema = center
                else:
                    self.center_ema = 0.6 * center + 0.4 * self.center_ema
                self.confirmed_center = self.center_ema.copy()
                self.confirmed_size = (best['size_x'], best['size_y'])
                self.last_confirm_time = stamp_sec
                detected = True
                output_center = self.confirmed_center
                output_size = self.confirmed_size

        if not detected and self.last_confirm_time is not None:
            if (stamp_sec - self.last_confirm_time) <= self.hold_time:
                detected = True
                output_center = self.confirmed_center
                output_size = self.confirmed_size

        flag_msg = Bool()
        flag_msg.data = detected
        self.flag_pub.publish(flag_msg)

        marker = Marker()
        marker.header.stamp = msg.header.stamp
        marker.header.frame_id = msg.header.frame_id
        marker.ns = 'opponent'
        marker.id = 0
        if detected:
            cx, cy = output_center
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(output_size[0], 0.35)
            marker.scale.y = max(output_size[1], 0.20)
            marker.scale.z = 0.20
            marker.color.a = 0.90
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.1

            pose = PoseStamped()
            pose.header = marker.header
            pose.pose.position.x = cx
            pose.pose.position.y = cy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            self.pub.publish(pose)
        else:
            marker.action = Marker.DELETE
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
