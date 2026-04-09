#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String, Float32MultiArray, Bool
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rclpy.qos import qos_profile_sensor_data
from state_machine.drive_state import DriveState

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        self.current_state = DriveState.TRAILING

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
        # Maximum age (seconds) of obs_topic data before lidar fallback is used.
        self.declare_parameter('obs_timeout_sec', 0.5)
        # Lidar topic used only as fallback when Frenet obstacle data is stale.
        self.declare_parameter('scan_topic', '/scan')
        # Fallback trigger distance (meters): closer than this enters FTG.
        self.declare_parameter('scan_trigger_distance', 0.7)
        # Fallback clear distance (meters): farther than this returns to GB_TRACK.
        self.declare_parameter('scan_clear_distance', 1.2)
        # RViz marker topic for live state text.
        self.declare_parameter('state_marker_topic', '/state_marker')
        # TF frame where state text is displayed.
        self.declare_parameter('state_marker_frame', 'ego_racecar/base_link')
        self.obs_topic = self.get_parameter('obs_topic').value
        self.safety_lateral_distance = float(self.get_parameter('safety_lateral_distance').value)
        self.trigger_distance = float(self.get_parameter('trigger_distance').value)
        self.obs_timeout_sec = float(self.get_parameter('obs_timeout_sec').value)
        self.scan_topic = self.get_parameter('scan_topic').value
        self.scan_trigger_distance = float(self.get_parameter('scan_trigger_distance').value)
        self.scan_clear_distance = float(self.get_parameter('scan_clear_distance').value)
        self.state_marker_topic = self.get_parameter('state_marker_topic').value
        self.state_marker_frame = self.get_parameter('state_marker_frame').value
        self.last_obs_msg_time = None

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

        self.get_logger().info(
            "State machine started (obs_topic=%s, safety_lateral_distance=%.2f, trigger_distance=%.2f)"
            % (self.obs_topic, self.safety_lateral_distance, self.trigger_distance)
        )
        self.publish_state()

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

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

    def obs_callback(self, msg):
        self.last_obs_msg_time = self._now_sec()
        obs_wpts = self.parse_obs_wpts(msg)
        closest = self.get_closest_obstacle(obs_wpts)

        new_state = self.current_state
        if closest is None:
            new_state = DriveState.GB_TRACK
        else:
            obs_s = float(closest[0])
            obs_d = float(closest[1])
            lateral_close = abs(obs_d) < self.safety_lateral_distance
            within_trigger = abs(obs_s) < self.trigger_distance

            if self.current_state == DriveState.GB_TRACK:

            ### TODO: SWAP FTGONLY TO OVERTAKE ### 

            # Trigger dynamic state when closest obstacle is laterally too close.
                if lateral_close and within_trigger:
                    if self.overtake_feasible:
                        new_state = DriveState.FTGONLY
                    else:
                        new_state = DriveState.TRAILING
                # Return to GB when either obstacle is not near in s or not near in d.
            elif self.current_state == DriveState.TRAILING:
                # If obstacle clears, return to racing line
                if (not within_trigger) or (not lateral_close):
                    new_state = DriveState.GB_TRACK
                # If we are trailing and overtake becomes feasible, switch to FTGONLY
                elif self.overtake_feasible:
                    new_state = DriveState.FTGONLY

            elif self.current_state == DriveState.FTGONLY:
                # Return to GB when either obstacle is not near in s or not near in d.
                if (not within_trigger) or (not lateral_close):
                    new_state = DriveState.GB_TRACK

            if new_state != self.current_state:
                self.get_logger().info(
                    f"State switch {self.current_state.value} -> {new_state.value} "
                    f"(closest_s={obs_s:.2f}, closest_d={obs_d:.2f}, within_trigger={within_trigger})"
                )

        self.current_state = new_state
        self.publish_state()

    def scan_callback(self, msg):
        # Use lidar as a fallback only when obstacle Frenet data is stale/missing.
        now = self._now_sec()
        if self.last_obs_msg_time is not None and (now - self.last_obs_msg_time) <= self.obs_timeout_sec:
            return
        ranges = np.array(msg.ranges, dtype=float)
        valid = ranges[np.isfinite(ranges) & (ranges > 0.05)]
        if valid.size == 0:
            return

        min_dist = float(np.min(valid))
        new_state = self.current_state

        if min_dist < self.scan_trigger_distance:
            if True:
                new_state = DriveState.FTGONLY
            else:
                new_state = DriveState.TRAILING
        elif self.current_state == DriveState.TRAILING:
            if min_dist > self.scan_clear_distance:
                new_state = DriveState.GB_TRACK
            elif self.overtake_feasible:
                new_state = DriveState.FTGONLY

        elif self.current_state == DriveState.FTGONLY: 
            if min_dist > self.scan_clear_distance:
                new_state = DriveState.GB_TRACK

        if new_state != self.current_state:
            self.get_logger().info(
                f"State switch {self.current_state.value} -> {new_state.value} "
                f"(lidar fallback, min_dist={min_dist:.2f})"
            )
            self.current_state = new_state
            self.publish_state()

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
