#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('waypoints_path', '/sim_ws/src/pure_pursuit/racelines/korea_mintime_sparse.csv')
        self.declare_parameter('use_track_filter', True)
        self.declare_parameter('track_dist_thresh', 0.45)
        self.declare_parameter('detect_dist', 1.1)
        self.declare_parameter('status_topic', '/obstacle_detected')
        self.declare_parameter('distance_topic', '/obstacle_distance')
        self.declare_parameter('pose_topic', '/static_obstacle_pose')
        self.declare_parameter('marker_topic', '/obstacle_marker')
        self.declare_parameter('min_cluster_points', 2)
        self.declare_parameter('max_cluster_points', 30)
        self.declare_parameter('base_breakpoint', 0.08)
        self.declare_parameter('range_breakpoint_scale', 0.02)
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('size_min_x', 0.05)
        self.declare_parameter('size_max_x', 0.40)
        self.declare_parameter('size_min_y', 0.05)
        self.declare_parameter('size_max_y', 0.40)
        self.declare_parameter('min_persist_frames', 2)
        self.declare_parameter('hold_time', 0.35)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.waypoints_path = self.get_parameter('waypoints_path').value
        self.use_track_filter = bool(self.get_parameter('use_track_filter').value)
        self.track_dist_thresh = float(self.get_parameter('track_dist_thresh').value)
        self.detect_dist = float(self.get_parameter('detect_dist').value)
        self.status_topic = self.get_parameter('status_topic').value
        self.distance_topic = self.get_parameter('distance_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
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
        self.hold_time = float(self.get_parameter('hold_time').value)

        self.detect_pub = self.create_publisher(Bool, self.status_topic, 10)
        self.distance_pub = self.create_publisher(Float32, self.distance_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)

        self.ego_pose = None
        self.waypoints_xy = None
        self.persist_count = 0
        self.last_confirm_time = None
        self.last_confirm_center = None
        self.last_confirm_size = None
        if self.use_track_filter:
            try:
                waypoints = np.loadtxt(self.waypoints_path, delimiter=',', skiprows=1)
                if waypoints.ndim == 2 and waypoints.shape[1] >= 2:
                    self.waypoints_xy = waypoints[:, :2]
                else:
                    self.use_track_filter = False
                    self.get_logger().warn('Invalid waypoints file; disabling static track filter.')
            except Exception as exc:
                self.use_track_filter = False
                self.get_logger().warn(f'Failed to load waypoints: {exc}. Disabling static track filter.')

        self.get_logger().info('Obstacle detector started')

    def on_odom(self, msg: Odometry):
        self.ego_pose = msg.pose.pose

    def _yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _local_to_map(self, x_local, y_local):
        if self.ego_pose is None:
            return None
        yaw = self._yaw_from_quat(self.ego_pose.orientation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_map = self.ego_pose.position.x + x_local * cos_yaw - y_local * sin_yaw
        y_map = self.ego_pose.position.y + x_local * sin_yaw + y_local * cos_yaw
        return float(x_map), float(y_map)

    def _distance_to_racing_line(self, x_map, y_map):
        if self.waypoints_xy is None or len(self.waypoints_xy) == 0:
            return None
        d = self.waypoints_xy - np.array([x_map, y_map], dtype=float)
        return float(np.min(np.linalg.norm(d, axis=1)))

    def _publish_detection(self, detected, distance, center, size, scan_msg):
        detect_msg = Bool()
        detect_msg.data = bool(detected)
        self.detect_pub.publish(detect_msg)

        dist_msg = Float32()
        dist_msg.data = float(distance if np.isfinite(distance) else 0.0)
        self.distance_pub.publish(dist_msg)

        marker = Marker()
        marker.header.stamp = scan_msg.header.stamp
        marker.header.frame_id = scan_msg.header.frame_id
        marker.ns = 'obstacle'
        marker.id = 0

        if detected and center is not None and size is not None:
            cx, cy = center
            sx, sy = size

            pose_msg = PoseStamped()
            pose_msg.header = marker.header
            pose_msg.pose.position.x = float(cx)
            pose_msg.pose.position.y = float(cy)
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            self.pose_pub.publish(pose_msg)

            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(cx)
            marker.pose.position.y = float(cy)
            marker.pose.position.z = 0.06
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(max(0.10, sx))
            marker.scale.y = float(max(0.10, sy))
            marker.scale.z = 0.12
            marker.color.a = 0.95
            marker.color.r = 0.95
            marker.color.g = 0.2
            marker.color.b = 0.2
        else:
            marker.action = Marker.DELETE

        self.marker_pub.publish(marker)

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
            for idx in idxs[1:]:
                prev = current[-1]
                r_prev = ranges[prev]
                r_curr = ranges[idx]
                gap = abs(r_curr - r_prev)
                thresh = self.base_breakpoint + self.range_breakpoint_scale * max(r_prev, r_curr)
                if gap > thresh or (idx - prev) > 1:
                    clusters.append(current)
                    current = [idx]
                else:
                    current.append(idx)
            clusters.append(current)

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp_sec == 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9

        best = None
        for c in clusters:
            if len(c) < self.min_cluster_points or len(c) > self.max_cluster_points:
                continue

            pts = []
            for i in c:
                r = ranges[i]
                a = angles[i]
                pts.append((r * math.cos(a), r * math.sin(a)))
            pts = np.array(pts)
            if pts.size == 0:
                continue

            xs = pts[:, 0]
            ys = pts[:, 1]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            if cx < 0.12:
                continue

            size_x = float(xs.max() - xs.min())
            size_y = float(ys.max() - ys.min())
            if not (self.size_min_x <= size_x <= self.size_max_x):
                continue
            if not (self.size_min_y <= size_y <= self.size_max_y):
                continue

            dist = float(math.hypot(cx, cy))
            if dist > self.detect_dist:
                continue

            if self.use_track_filter and self.ego_pose is not None and self.waypoints_xy is not None:
                mapped = self._local_to_map(cx, cy)
                if mapped is None:
                    continue
                line_dist = self._distance_to_racing_line(mapped[0], mapped[1])
                if line_dist is None or line_dist > self.track_dist_thresh:
                    continue

            score = -dist
            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'center': (cx, cy),
                    'size': (size_x, size_y),
                    'dist': dist,
                }

        detected = False
        center = None
        size = None
        distance = float('inf')
        if best is not None:
            self.persist_count += 1
            if self.persist_count >= self.min_persist_frames:
                detected = True
                center = best['center']
                size = best['size']
                distance = best['dist']
                self.last_confirm_center = center
                self.last_confirm_size = size
                self.last_confirm_time = stamp_sec
        else:
            self.persist_count = 0

        if not detected and self.last_confirm_time is not None:
            if (stamp_sec - self.last_confirm_time) <= self.hold_time:
                detected = True
                center = self.last_confirm_center
                size = self.last_confirm_size
                distance = float(math.hypot(center[0], center[1]))

        self._publish_detection(detected, distance, center, size, msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
