import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import CubicSpline
import os

class SplinerNode(Node):
    def __init__(self):
        super().__init__('spliner_node')

        # Parameters
        self.declare_parameter("waypoints_path", "/ros2_ws/src/pure_pursuit/racelines/arc.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("local_path_topic", "/local_path")
        self.declare_parameter("num_future_waypoints", 5) 
        self.declare_parameter("path_resolution", 20)     

        self.path_to_csv = self.get_parameter("waypoints_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.local_path_topic = self.get_parameter("local_path_topic").value
        self.num_future = self.get_parameter("num_future_waypoints").value
        self.resolution = self.get_parameter("path_resolution").value

        # Load Global Waypoints
        if not os.path.exists(self.path_to_csv):
            self.get_logger().error(f"Waypoints CSV file not found: {self.path_to_csv}")
            self.get_logger().warn("Using default search for CSV...")
            self.path_to_csv = "/sim_ws/src/pure_pursuit/racelines/arc.csv"

        try:
            self.waypoints = np.loadtxt(self.path_to_csv, delimiter=',', skiprows=1)
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            self.waypoints = None

        # State
        self.curr_pos = None

        # Pubs & Subs
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.path_pub = self.create_publisher(Path, self.local_path_topic, 10)

        self.get_logger().info("Spliner Node initialized and waiting for Odometry...")

    def find_nearest_waypoint_idx(self, x, y):
        """ Find the index of the nearest global waypoint. """
        if self.waypoints is None:
            return -1
        dists = np.linalg.norm(self.waypoints[:, :2] - np.array([x, y]), axis=1)
        return np.argmin(dists)

    def odom_callback(self, msg):
        self.curr_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
        if self.waypoints is None:
            return

        nearest_idx = self.find_nearest_waypoint_idx(self.curr_pos[0], self.curr_pos[1])
        if nearest_idx == -1:
            return

        # Points: Current position + next N future waypoints
        spline_points = [self.curr_pos]
        for i in range(1, self.num_future + 1):
            idx = (nearest_idx + i) % len(self.waypoints)
            spline_points.append(self.waypoints[idx, :2])

        spline_points = np.array(spline_points)

        t = np.linspace(0, 1, len(spline_points))
        cs_x = CubicSpline(t, spline_points[:, 0])
        cs_y = CubicSpline(t, spline_points[:, 1])

        # Generate smooth path
        t_smooth = np.linspace(0, 1, self.resolution)
        smooth_x = cs_x(t_smooth)
        smooth_y = cs_y(t_smooth)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for x, y in zip(smooth_x, smooth_y):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SplinerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
