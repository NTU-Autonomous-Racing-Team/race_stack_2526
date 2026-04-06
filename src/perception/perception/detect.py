#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time, Duration
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
 
import numpy as np
import time
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
from f110_msgs.msg import Obstacle, ObstacleArray
from frenet_conversion import FrenetConverter
 
# -----------------------------
# HELPERS
# -----------------------------
def from_vector3_msg(msg):
    return np.array([msg.x, msg.y, msg.z])
 
def from_quat_msg(msg):
    return Rotation.from_quat([msg.x, msg.y, msg.z, msg.w])
 
def compute_psi(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.arctan2(dy, dx)
 
# -----------------------------
# TRACKED OBSTACLE
# -----------------------------
class TrackedObstacle:
    def __init__(self, obs_id, s, d, size_s, size_d):
        self.id = obs_id
        self.s = s
        self.d = d
        self.vs = 0.0
        self.vd = 0.0
        self.size_s = size_s
        self.size_d = size_d
        self.last_seen = time.time()
 
# -----------------------------
# MAIN NODE
# -----------------------------
class Detect(Node):
    def __init__(self):
        super().__init__('detect_dbscan_tracking')
        self.get_logger().info("Initializing Detect Node...")
 
        # -----------------------------
        # LOAD TRACK CSV
        # -----------------------------
        self.csv_path = "/sim_ws/src/pure_pursuit/racelines/arc.csv"
        data = np.loadtxt(self.csv_path, delimiter=",")
        # --- TEMPORARY DEBUG ---
        data = np.loadtxt(self.csv_path, delimiter=",")
        # print(f"Raw raceline first point: {data[0,0]:.3f}, {data[0,1]:.3f}")
        # print(f"Raw raceline X range: {data[:,0].min():.3f} to {data[:,0].max():.3f}")
        # print(f"Raw raceline Y range: {data[:,1].min():.3f} to {data[:,1].max():.3f}")
        # --- END DEBUG ---

        x = data[:, 0] 
        y = data[:, 1] 
        psi = compute_psi(x, y)
        self.converter = FrenetConverter(x, y, psi)
        self.track_length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        self.get_logger().info("FrenetConverter initialized successfully")
 
        # -----------------------------
        # ROS SETUP
        # -----------------------------
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        # self.pub = self.create_publisher(Float32MultiArray, '/tracked_obstacles', 10)
        self.pub = self.create_publisher(ObstacleArray, '/obstacles', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        self.raceline_pub = self.create_publisher(MarkerArray, '/raceline_marker', 10)
 
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
 
        # -----------------------------
        # PARAMETERS
        # -----------------------------
        self.eps = 0.2
        self.min_samples = 3
        self.max_range = 5.0
        self.track_half_width = 1.8
        self.match_threshold = 0.5
        self.max_age = 0.5
        self.max_obs_size = 0.5   # max physical size in metres (car ~0.3m wide)
        self.min_obs_size = 5    # min number of points to be a valid cluster
 
        # -----------------------------
        # TRACKING STATE
        # -----------------------------
        self.tracked = []
        self.next_id = 0
        self.active_marker_ids = set()
 
        # -----------------------------
        # RACELINE TIMER
        # -----------------------------
        self.create_timer(1.0, self.publish_raceline)
 
        self.get_logger().info("Detect node ready!")
 
    # -----------------------------
    # RECTANGLE FITTING (ETH method)
    # -----------------------------
    def fit_rectangle(self, xy_points):
        """
        Fits a rectangle to a set of 2D points using angle search.
        Returns (center_x, center_y, size, theta_opt)
        Based on the ETH race stack detection method.
        """
        obstacle = np.array(xy_points)  # shape (N, 2)
        min_2_points_dist = 0.01
 
        theta = np.linspace(0, np.pi / 2 - np.pi / 180, 90)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
 
        # Project points onto candidate axes
        distance1 = np.dot(obstacle, [cos_theta, sin_theta])   # (N, 90)
        distance2 = np.dot(obstacle, [-sin_theta, cos_theta])  # (N, 90)
 
        D10 = -distance1 + np.amax(distance1, axis=0)
        D11 = distance1 - np.amin(distance1, axis=0)
        D20 = -distance2 + np.amax(distance2, axis=0)
        D21 = distance2 - np.amin(distance2, axis=0)
 
        min_array = np.argmin(
            [np.linalg.norm(D10, axis=0), np.linalg.norm(D11, axis=0)], axis=0)
        D10 = np.transpose(D10)
        D11 = np.transpose(D11)
        D10[min_array == 1] = D11[min_array == 1]
        D10 = np.transpose(D10)
 
        min_array = np.argmin(
            [np.linalg.norm(D20, axis=0), np.linalg.norm(D21, axis=0)], axis=0)
        D20 = np.transpose(D20)
        D21 = np.transpose(D21)
        D20[min_array == 1] = D21[min_array == 1]
        D20 = np.transpose(D20)
 
        D = np.minimum(D10, D20)
        D[D < min_2_points_dist] = min_2_points_dist
 
        # Find optimal angle
        theta_opt = np.argmax(np.sum(np.reciprocal(D), axis=0)) * np.pi / 180
 
        distances1 = np.dot(obstacle, [np.cos(theta_opt), np.sin(theta_opt)])
        distances2 = np.dot(obstacle, [-np.sin(theta_opt), np.cos(theta_opt)])
 
        max_dist1 = np.max(distances1)
        min_dist1 = np.min(distances1)
        max_dist2 = np.max(distances2)
        min_dist2 = np.min(distances2)
 
        # Compute center from bounding box midpoints
        mid1 = (max_dist1 + min_dist1) / 2
        mid2 = (max_dist2 + min_dist2) / 2
        center_x = np.cos(theta_opt) * mid1 - np.sin(theta_opt) * mid2
        center_y = np.sin(theta_opt) * mid1 + np.cos(theta_opt) * mid2
 
        # Physical size = largest bounding dimension
        size = max(max_dist1 - min_dist1, max_dist2 - min_dist2)
 
        return center_x, center_y, size, theta_opt
 
    # -----------------------------
    # FRENET WRAPAROUND DISTANCE
    # -----------------------------
    def frenet_dist(self, s1, d1, s2, d2):
        ds = abs(s1 - s2)
        ds = min(ds, self.track_length - ds)
        dd = d1 - d2
        return np.sqrt(ds**2 + dd**2)
    # -----------------------------
    # PUBLISH RACELINE BOUNDARY
    # -----------------------------
    def publish_raceline(self):
        data = np.loadtxt(self.csv_path, delimiter=",")
        x = data[:, 0]
        y = data[:, 1] 
        psi = compute_psi(x, y)
 
        center_marker = Marker()
        center_marker.header.frame_id = 'map'
        center_marker.header.stamp = self.get_clock().now().to_msg()
        center_marker.ns = 'raceline'
        center_marker.id = 0
        center_marker.type = Marker.LINE_STRIP
        center_marker.action = Marker.ADD
        center_marker.scale.x = 0.05
        center_marker.color.a = 1.0
        center_marker.color.g = 1.0
 
        left_marker = Marker()
        left_marker.header.frame_id = 'map'
        left_marker.header.stamp = self.get_clock().now().to_msg()
        left_marker.ns = 'raceline'
        left_marker.id = 1
        left_marker.type = Marker.LINE_STRIP
        left_marker.action = Marker.ADD
        left_marker.scale.x = 0.05
        left_marker.color.a = 1.0
        left_marker.color.r = 1.0
        left_marker.color.g = 1.0
 
        right_marker = Marker()
        right_marker.header.frame_id = 'map'
        right_marker.header.stamp = self.get_clock().now().to_msg()
        right_marker.ns = 'raceline'
        right_marker.id = 2
        right_marker.type = Marker.LINE_STRIP
        right_marker.action = Marker.ADD
        right_marker.scale.x = 0.05
        right_marker.color.a = 1.0
        right_marker.color.r = 1.0
        right_marker.color.g = 1.0
 
        for i in range(len(x)):
            p = Point()
            p.x, p.y, p.z = float(x[i]), float(y[i]), 0.0
            center_marker.points.append(p)
 
            perp_x = -np.sin(psi[i])
            perp_y = np.cos(psi[i])
 
            pl = Point()
            pl.x = float(x[i] + self.track_half_width * perp_x)
            pl.y = float(y[i] + self.track_half_width * perp_y)
            pl.z = 0.0
            left_marker.points.append(pl)
 
            pr = Point()
            pr.x = float(x[i] - self.track_half_width * perp_x)
            pr.y = float(y[i] - self.track_half_width * perp_y)
            pr.z = 0.0
            right_marker.points.append(pr)
 
        marker_array = MarkerArray()
        marker_array.markers = [center_marker, left_marker, right_marker]
        self.raceline_pub.publish(marker_array)
 
    # -----------------------------
    # PUBLISH MARKERS
    # -----------------------------
    def publish_obstacle_markers(self, dead_ids=set()):
        marker_array = MarkerArray()
 
        for dead_id in dead_ids:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = dead_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
 
        for t in self.tracked:
            xy = self.converter.get_cartesian(t.s, t.d)
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = t.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = xy[0]
            marker.pose.position.y = xy[1]
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(t.size_s, 0.3)
            marker.scale.y = max(t.size_d, 0.3)
            marker.scale.z = 0.3
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
 
        self.marker_pub.publish(marker_array)
 
    # -----------------------------
    # MAIN CALLBACK
    # -----------------------------
    def scan_cb(self, scan):
        start_time = time.perf_counter()
 
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                scan.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            # self.get_logger().warn(f"TF lookup failed: {e}")
            return
 
        # -----------------------------
        # FILTER INVALID SCAN POINTS
        # -----------------------------
        ranges_raw = np.array(scan.ranges)
        valid_mask = (ranges_raw >= scan.range_min) & (ranges_raw <= self.max_range)
        ranges = ranges_raw[valid_mask]
        angles_full = np.linspace(scan.angle_min, scan.angle_max, len(ranges_raw))
        angles = angles_full[valid_mask]
 
        # self.get_logger().info(f"Valid scan points: {len(ranges)} / {len(ranges_raw)}")
 
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)
        points_local = np.vstack((x_local, y_local)).T
 
        T = from_vector3_msg(transform.transform.translation)
        # print(f"Ego pos: x={T[0]:.3f}, y={T[1]:.3f}")
        R = from_quat_msg(transform.transform.rotation).as_matrix()
        points_global = (R[:2, :2] @ points_local.T).T + T[:2]

        # DEBUG: check coordinate alignment
        # print("Sample global point:", points_global[0])
        # print("Raceline first point:",
                # self.converter.waypoints_x[0],
                # self.converter.waypoints_y[0])
 
        # -----------------------------
        # DBSCAN CLUSTERING
        # -----------------------------
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_global)
        labels = clustering.labels_
 
        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            clusters.append(points_global[mask])
 
        # self.get_logger().info(f"Clusters detected: {len(clusters)}")
 
        # -----------------------------
        # RECTANGLE FITTING + FILTERING
        # -----------------------------
        detections = []
        for xy_points in clusters:
 
            # Min points filter
            if len(xy_points) < self.min_obs_size:
                # self.get_logger().info(f'  -> REJECTED: too few points ({len(xy_points)})')
                continue
 
            # Rectangle fit in Cartesian space
            cx, cy, size, theta = self.fit_rectangle(xy_points)
 
            # Physical size filter — reject walls and large objects
            if size > self.max_obs_size:
                # self.get_logger().info(f'  -> REJECTED by size filter (size={size:.2f}m)')
                continue
 
            # Convert fitted center to Frenet
            s_arr, d_arr = self.converter.get_frenet(np.array([cx]), np.array([cy]))
            s_center = float(s_arr[0])
            d_center = float(d_arr[0])
 
            # self.get_logger().info(
            #     f'cluster: pts={len(xy_points)} size={size:.2f}m '
            #     f's={s_center:.2f} d={d_center:.2f}')
 
            # Track boundary filter
            if abs(d_center) > self.track_half_width:
                # self.get_logger().info(
                #     f'  -> REJECTED by track_half_width (d={d_center:.2f})')
                continue
 
            self.get_logger().info(f'  -> ACCEPTED as detection')
            detections.append((s_center, d_center, size, size))
 
        # self.get_logger().info(f"Valid obstacles: {len(detections)}")
 
        # -----------------------------
        # TRACKING
        # -----------------------------
        current_time = time.time()
        updated_tracks = []
 
        for det in detections:
            s_det, d_det, size_s, size_d = det
            best_match = None
            best_dist = float('inf')


 
            for track in self.tracked:
                dist = self.frenet_dist(track.s, track.d, s_det, d_det)
                if dist < best_dist and dist < self.match_threshold:
                    best_dist = dist
                    best_match = track
 
            if best_match:
                dt = max(current_time - best_match.last_seen, 1e-3)
                ds = s_det - best_match.s
                if ds > self.track_length / 2:
                    ds -= self.track_length
                elif ds < -self.track_length / 2:
                    ds += self.track_length
                best_match.vs = ds / dt
                best_match.vd = (d_det - best_match.d) / dt
                best_match.s = s_det
                best_match.d = d_det
                best_match.size_s = size_s
                best_match.size_d = size_d
                best_match.last_seen = current_time
                updated_tracks.append(best_match)
            else:
                new_track = TrackedObstacle(self.next_id, s_det, d_det, size_s, size_d)
                self.next_id += 1
                updated_tracks.append(new_track)
 
        self.tracked = [t for t in updated_tracks if current_time - t.last_seen < self.max_age]
        new_ids = {t.id for t in self.tracked}
        dead_ids = self.active_marker_ids - new_ids
        self.active_marker_ids = new_ids
 
        # self.get_logger().info(f"Tracking objects: {len(self.tracked)}")
 
        # -----------------------------
        # PUBLISH
        # -----------------------------
        # msg = Float32MultiArray()
        # flat = []
        # for t in self.tracked:
        #     flat.extend([t.s, t.d, t.vs, t.vd, t.size_s, t.size_d, float(t.id)])
        # msg.data = flat
        # self.pub.publish(msg)
 
        # self.publish_obstacle_markers(dead_ids)
        obs_array_msg = ObstacleArray()
        obs_array_msg.header = scan.header
        for t in self.tracked:
            o = Obstacle()
            o.id = int(t.id)
            o.s_center = float(t.s)
            o.d_center = float(t.d)
            o.vs = float(t.vs)
            o.vd = float(t.vd)
            o.size = float(max(t.size_s, t.size_d))
            o.d_left = t.d + (t.size_d / 2.0)
            o.d_right = t.d - (t.size_d / 2.0)
            o.is_static = True #to be modify
            
            obs_array_msg.obstacles.append(o)
            
        self.pub.publish(obs_array_msg)
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        # self.get_logger().info(f"Loop latency: {latency:.2f} ms")
 
 
# -----------------------------
# MAIN
# -----------------------------
def main():
    rclpy.init()
    node = Detect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node")
    node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()