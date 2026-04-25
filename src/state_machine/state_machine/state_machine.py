#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String, Float32MultiArray, Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from rclpy.qos import qos_profile_sensor_data
from frenet_conversion.frenet_converter import FrenetConverter
from state_machine.drive_state import DriveState

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        self.current_state = DriveState.GB_TRACK

        self.opponent_detected = False
        self.overtake_feasible = True
        self.declare_parameter('feasibility_topic', '/planner/overtake_feasible')
        self.feasibility_topic = self.get_parameter('feasibility_topic').value

        # Topic publishing flattened Frenet obstacle points: [s0, d0, s1, d1, ...].
        self.declare_parameter('obs_topic', '/tracked_obstacles')
        # Lateral distance threshold (meters) below which FTG is activated.
        self.declare_parameter('safety_lateral_distance', 0.5)
        # Longitudinal trigger window in Frenet s (meters) around the car.
        self.declare_parameter('trigger_distance', 3.0)
        # Raceline used for transition-completion checks.
        self.declare_parameter('waypoints_path', './src/pure_pursuit/racelines/korea_mintime_sparse.csv')
        # Transition completion thresholds.
        self.declare_parameter('transition_frames_required', 30)
        self.declare_parameter('transition_d_threshold', 0.1)
        self.declare_parameter('clearance_distance', 1.5)
        self.declare_parameter('safe_corridor_width', 0.45)
        # RViz marker topic for live state text.
        self.declare_parameter('state_marker_topic', '/state_marker')
        # TF frame where state text is displayed.
        self.declare_parameter('state_marker_frame', 'roboracer_1')
        self.declare_parameter('scan_topic', '/autodrive/roboracer_1/lidar')
        self.declare_parameter('odom_topic', '/autodrive/roboracer_1/odom')
        self.obs_topic = self.get_parameter('obs_topic').value
        self.safety_lateral_distance = float(self.get_parameter('safety_lateral_distance').value)
        self.trigger_distance = float(self.get_parameter('trigger_distance').value)
        self.waypoints_path = self.get_parameter('waypoints_path').value
        self.transition_frames_required = int(self.get_parameter('transition_frames_required').value)
        self.transition_d_threshold = float(self.get_parameter('transition_d_threshold').value)
        self.clearance_distance = float(self.get_parameter('clearance_distance').value)
        self.safe_corridor_half_width = float(self.get_parameter('safe_corridor_width').value) / 2.0
        self.state_marker_topic = self.get_parameter('state_marker_topic').value
        self.state_marker_frame = self.get_parameter('state_marker_frame').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.latest_scan = None
        self.latest_odom = None
        self.transition_valid_frames = 0

        data = np.loadtxt(self.waypoints_path, delimiter=',')
        wp_x = data[:, 0]
        wp_y = data[:, 1]
        wp_dx = np.diff(wp_x, append=wp_x[0] - wp_x[-1])
        wp_dy = np.diff(wp_y, append=wp_y[0] - wp_y[-1])
        wp_psi = np.arctan2(wp_dy, wp_dx)
        self.frenet_converter = FrenetConverter(wp_x, wp_y, wp_psi)

        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            self.obs_topic,
            self.obs_callback,
            qos_profile_sensor_data,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.feasibility_sub = self.create_subscription(
            Bool, 
            self.feasibility_topic, 
            self.feasibility_callback, 
            10
        )

        self.state_pub = self.create_publisher(String, '/state', 10)
        self.state_marker_pub = self.create_publisher(Marker, self.state_marker_topic, 10)
        # Republish marker so it appears even if RViz starts after this node.
        self.create_timer(0.25, self.publish_state_marker)
        # Periodically re-check whether a transition can finish even if obstacle updates stall.
        self.create_timer(0.1, self.transition_timer_callback)

        self.get_logger().info(
            "State machine started (obs_topic=%s, safety_lateral_distance=%.2f, trigger_distance=%.2f)"
            % (self.obs_topic, self.safety_lateral_distance, self.trigger_distance)
        )
        self.publish_state()

    def publish_state(self):
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)
        self.publish_state_marker()

    def _state_color(self):
        if self.current_state == DriveState.GB_TRACK:
            return (0.1, 0.9, 0.1)
        if self.current_state == DriveState.TRAILING:
            return (1.0, 0.75, 0.0)
        if self.current_state == DriveState.FTGONLY:
            return (1.0, 0.1, 0.1)
        if self.current_state == DriveState.TRANSITION:
            return (0.2, 0.8, 1.0)
        return (1.0, 1.0, 1.0)

    def publish_state_marker(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.state_marker_frame
        marker.ns = 'state_machine'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.1
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.45
        marker.color.a = 1.0
        r, g, b = self._state_color()
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.text = f"STATE: {self.current_state.value}"
        self.state_marker_pub.publish(marker)

    def feasibility_callback(self, msg: Bool):
        """ Updates the state machine with the local planner's feasibility assessment """
        self.overtake_feasible = msg.data

    def parse_obs_wpts(self, msg):
        # raw = np.array(msg.data, dtype=float)
        # if raw.size == 0:
        #     return np.empty((0, 2), dtype=float)
# 
        # if raw.size % 2 != 0:
        #     self.get_logger().warn('obs_wpts payload length is not even, dropping last value')
        #     raw = raw[:-1]
# 
        # obs = raw.reshape((-1, 2))
        # finite_mask = np.isfinite(obs).all(axis=1)
        # return obs[finite_mask]

        raw = np.array(msg.data, dtype=float)
        
        # Expecting 7 values per obstacle from Detect node: 
        # [s, d, vs, vd, size_s, size_d, id]
        NUM_FEATURES = 7
        
        if raw.size == 0:
            return np.empty((0, NUM_FEATURES), dtype=float)

        if raw.size % NUM_FEATURES != 0:
            self.get_logger().warn(f'obs_wpts payload length {raw.size} is not a multiple of {NUM_FEATURES}. Truncating...')
            valid_length = (raw.size // NUM_FEATURES) * NUM_FEATURES
            raw = raw[:valid_length]

        # Reshape into rows of 7 columns
        obs = raw.reshape((-1, NUM_FEATURES))
        
        # Filter out any rows that have infinity or NaN values
        finite_mask = np.isfinite(obs).all(axis=1)
        return obs[finite_mask]

    def get_closest_obstacle(self, obs_wpts):
        if obs_wpts.size == 0:
            return None

        # Prefer obstacles ahead of the car in Frenet s, fallback to nearest by |s|.
        ahead = obs_wpts[obs_wpts[:, 0] >= 0.0]
        candidates = ahead if ahead.size > 0 else obs_wpts
        if ahead.size > 0:
            idx = int(np.argmin(candidates[:, 0]))
        else:
            idx = int(np.argmin(np.abs(candidates[:, 0])))
        return candidates[idx]

    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def is_path_clear(self):
        if self.latest_scan is None:
            return False

        ranges = np.array(self.latest_scan.ranges)
        angles = np.linspace(self.latest_scan.angle_min, self.latest_scan.angle_max, len(ranges))

        front_mask = (angles > -np.pi / 2) & (angles < np.pi / 2)
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        valid_mask = np.isfinite(front_ranges)
        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        if front_ranges.size == 0:
            return True

        xs = front_ranges * np.cos(front_angles)
        ys = front_ranges * np.sin(front_angles)
        in_corridor = (xs > 0.1) & (xs < self.clearance_distance) & (np.abs(ys) < self.safe_corridor_half_width)
        return not np.any(in_corridor)

    def is_on_raceline(self):
        if self.latest_odom is None:
            return False

        car_x = self.latest_odom.pose.pose.position.x
        car_y = self.latest_odom.pose.pose.position.y
        frenet = self.frenet_converter.get_frenet(np.array([car_x]), np.array([car_y]))
        d = float(frenet[1][0])
        return abs(d) < self.transition_d_threshold

    def transition_complete(self):
        if not self.is_path_clear():
            self.transition_valid_frames = 0
            return False
        if self.is_on_raceline():
            self.transition_valid_frames += 1
            return self.transition_valid_frames >= self.transition_frames_required
        self.transition_valid_frames = 0
        return False

    def transition_timer_callback(self):
        if self.current_state != DriveState.TRANSITION:
            return

        if self.transition_complete():
            self.current_state = DriveState.GB_TRACK
            self.transition_valid_frames = 0
            self.publish_state()

    def obs_callback(self, msg):
        obs_wpts = self.parse_obs_wpts(msg)
        closest = self.get_closest_obstacle(obs_wpts)

        desired_state = self.current_state
        if closest is None:
            desired_state = DriveState.GB_TRACK
        else:
            obs_s = float(closest[0])
            obs_d = float(closest[1])
            lateral_close = abs(obs_d) < self.safety_lateral_distance
            within_trigger = abs(obs_s) < self.trigger_distance

            if self.current_state in (DriveState.GB_TRACK, DriveState.TRANSITION):

            ### TODO: SWAP FTGONLY TO OVERTAKE ### 

            # Trigger dynamic state when closest obstacle is laterally too close.
                if lateral_close and within_trigger:
                    if self.overtake_feasible:
                        desired_state = DriveState.FTGONLY
                    else:
                        desired_state = DriveState.TRAILING
                # Return to GB when either obstacle is not near in s or not near in d.
            elif self.current_state == DriveState.TRAILING:
                # If obstacle clears, return to racing line
                if (not within_trigger) or (not lateral_close):
                    desired_state = DriveState.GB_TRACK
                # If we are trailing and overtake becomes feasible, switch to FTGONLY
                elif self.overtake_feasible:
                    desired_state = DriveState.FTGONLY

            elif self.current_state == DriveState.FTGONLY:
                # Return to GB when either obstacle is not near in s or not near in d.
                if (not within_trigger) or (not lateral_close):
                    desired_state = DriveState.GB_TRACK

        # Transition state orchestration happens here, not in controller_manager.
        if self.current_state == DriveState.TRANSITION:
            if desired_state != DriveState.GB_TRACK:
                new_state = desired_state
                self.transition_valid_frames = 0
            elif self.transition_complete():
                new_state = DriveState.GB_TRACK
                self.transition_valid_frames = 0
            else:
                new_state = DriveState.TRANSITION
        else:
            if desired_state == DriveState.GB_TRACK and self.current_state in (DriveState.FTGONLY, DriveState.TRAILING):
                new_state = DriveState.TRANSITION
                self.transition_valid_frames = 0
            else:
                new_state = desired_state

        self.current_state = new_state
        self.publish_state()
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
