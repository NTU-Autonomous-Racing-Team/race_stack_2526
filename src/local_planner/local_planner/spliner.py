import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import CubicSpline
from visualization_msgs.msg import Marker, MarkerArray
import os
# from .frenet_converter import FrenetConverter
from frenet_conversion.frenet_converter import FrenetConverter
from f110_msgs.msg import ObstacleArray

# -----------------------------
# HELPERS
# -----------------------------
# def from_vector3_msg(msg):
#     return np.array([msg.x, msg.y, msg.z])
 
# def from_quat_msg(msg):
#     return Rotation.from_quat([msg.x, msg.y, msg.z, msg.w])
 
def compute_psi(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.arctan2(dy, dx)

class SplinerNode(Node):
    def __init__(self):
        super().__init__('spliner_node')

        # LOAD TRACK CSV
        self.csv_path = "/sim_ws/src/pure_pursuit/racelines/arc.csv"
        self.waypoints = np.loadtxt(self.csv_path, delimiter=",")

        x = self.waypoints[:, 0] 
        y = self.waypoints[:, 1] 
        psi = compute_psi(x, y)
        self.converter = FrenetConverter(x, y, psi)
        self.track_length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        self.get_logger().info("FrenetConverter initialized successfully")

        # Parameters
        self.declare_parameter("waypoints_path", "/sim_ws/src/pure_pursuit/racelines/arc.csv")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("local_path_topic", "/local_path")
        self.declare_parameter("num_future_waypoints", 5) 
        self.declare_parameter("path_resolution", 20)     
        self.declare_parameter("obs_threshold", 2.0)
        self.declare_parameter("evasion_dist", 0.65)
        self.declare_parameter("spline_bound_mindist", 0.5)

        self.path_to_csv = self.get_parameter("waypoints_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.local_path_topic = self.get_parameter("local_path_topic").value
        self.num_future = self.get_parameter("num_future_waypoints").value
        self.resolution = self.get_parameter("path_resolution").value
        self.obs_threshold = self.get_parameter("obs_threshold").value
        self.evasion_dist = self.get_parameter("evasion_dist").value
        self.spline_bound_mindist = self.get_parameter("spline_bound_mindist").value

        self.local_path_marker_pub = self.create_publisher(MarkerArray, '/local_path_marker', 10)

        # State
        self.curr_pos = None
        self.current_vs=0.0
        self.obstacles=[]

        # Pubs & Subs
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        #subscribe to ObstacleArray when possible
        self.obstacle_sub = self.create_subscription(ObstacleArray, '/obstacles', self.obstacle_callback, 10)
        self.path_pub = self.create_publisher(Path, self.local_path_topic, 10)
        self.get_logger().info("Spliner Node initialized and waiting for Odometry...")

    def obstacle_callback(self, msg):
        self.obstacles = msg.obstacles
    
    def filter_obstacles(self, all_obstacles, ego_s, lookahead_dist=10.0):
        close_obstacles = []
        for obs in all_obstacles:
            dist_s = (obs.s_center - ego_s) % self.track_length 
            
            if dist_s < lookahead_dist and abs(obs.d_center) < self.obs_threshold:
                close_obstacles.append(obs)
                
        return min(close_obstacles, key=lambda o: (o.s_center - ego_s) % self.track_length) if close_obstacles else None

    def decide_evasive_side(self, obstacle, nearest_wpnt):
        buffer = self.evasion_dist
        min_space = buffer + self.spline_bound_mindist
        
        left_gap = abs(nearest_wpnt.d_left - obstacle.d_left)
        right_gap = abs(nearest_wpnt.d_right + obstacle.d_right)

        if left_gap > min_space and right_gap < min_space:
            side = "left"
            d_apex = obstacle.d_left + buffer
        elif right_gap > min_space and left_gap < min_space:
            side = "right"
            d_apex = obstacle.d_right - buffer
        else:
            candidate_left = obstacle.d_left + buffer
            candidate_right = obstacle.d_right - buffer
            
            if abs(candidate_left) <= abs(candidate_right):
                side = "left"
                d_apex = candidate_left
            else:
                side = "right"
                d_apex = candidate_right

        d_apex = np.clip(d_apex, nearest_wpnt.d_right + 0.2, nearest_wpnt.d_left - 0.2)
        
        return side, d_apex

    def generate_spline_points(self, s_apex, d_apex, current_vs):
       
        scale = np.clip(1.0 + current_vs / 10.0, 1.0, 1.5) 
        
        pre_dist = [-5.0 * scale, -2.5 * scale]  # P-2, P-1
        post_dist = [2.5 * scale, 5.0 * scale]   # P1, P2
        
        control_points = []
        
        control_points.append([s_apex + pre_dist[0], 0.0])
        control_points.append([s_apex + pre_dist[1], d_apex * 0.5]) 
        
        control_points.append([s_apex, d_apex])
        
        control_points.append([s_apex + post_dist[0], d_apex * 0.5]) 
        control_points.append([s_apex + post_dist[1], 0.0])
        
        return np.array(control_points)
    
    def publish_path_marker(self, path_msg):
        marker_array = MarkerArray()
        
        line_marker = Marker()
        line_marker.header = path_msg.header
        line_marker.ns = "local_path_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        line_marker.color.r = 1.0
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0
        line_marker.color.a = 1.0
        
        line_marker.scale.x = 0.1
        
        for pose_stamped in path_msg.poses:
            line_marker.points.append(pose_stamped.pose.position)
            
        marker_array.markers.append(line_marker)
        self.local_path_marker_pub.publish(marker_array)

    def odom_callback(self, msg):
        self.curr_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.current_vs = msg.twist.twist.linear.x
        
        if self.converter is None:
            return

        ego_s_arr, ego_d_arr = self.converter.get_frenet(np.array([self.curr_pos[0]]), np.array([self.curr_pos[1]]))
        ego_s = float(ego_s_arr[0])
        ego_d = float(ego_d_arr[0])

        if len(self.obstacles) > 0:
            self.get_logger().info(f"I see {len(self.obstacles)} obstacles!")

        target_obs = self.filter_obstacles(self.obstacles, ego_s)

        if target_obs:
            obs_xy = self.converter.get_cartesian([target_obs.s_center], np.array([0.0]))
            obs_x, obs_y = float(obs_xy[0][0]), float(obs_xy[1][0])
            
            dists = np.linalg.norm(self.waypoints[:, :2] - np.array([obs_x, obs_y]), axis=1)
            nearest_idx = np.argmin(dists)

            class Waypoint:
                def __init__(self, d_left, d_right):
                    self.d_left = d_left
                    self.d_right = d_right
            
            nearest_wpnt = Waypoint(d_left=3.0, d_right=-3.0) 
            
            side, d_apex = self.decide_evasive_side(target_obs, nearest_wpnt)
            
            s_apex = target_obs.s_center

            if s_apex < ego_s:
                s_apex += self.track_length
                
            control_points = self.generate_spline_points(s_apex, d_apex, self.current_vs)
            
            s_ctrl = control_points[:, 0]
            d_ctrl = control_points[:, 1]
            cs_d = CubicSpline(s_ctrl, d_ctrl, bc_type='clamped')
            
            s_smooth = np.linspace(s_ctrl[0], s_ctrl[-1], self.resolution)
            d_smooth = cs_d(s_smooth)
            
        else:
            s_smooth = np.linspace(ego_s, ego_s + 10.0, self.resolution)
            d_smooth = np.zeros_like(s_smooth)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for s, d in zip(s_smooth, d_smooth):
            x, y = self.converter.get_cartesian(s, d)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.publish_path_marker(path_msg)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} points")

def main(args=None):
    rclpy.init(args=args)
    node = SplinerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
