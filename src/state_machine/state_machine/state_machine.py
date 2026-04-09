#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from f110_msgs.msg import DriveState as DriveStateMsg
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from state_machine.drive_state import DriveState

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        self.current_state = DriveState.GB_TRACK

        # Topic publishing flattened tracked obstacles from detect.py.
        self.declare_parameter('obs_topic', '/tracked_obstacles')
        # Lateral distance threshold (meters) below which FTG is activated.
        self.declare_parameter('safety_lateral_distance', 1.5)
        # Longitudinal trigger window in Frenet s (meters) around the car.
        self.declare_parameter('trigger_distance', 3.0)
        # Maximum age (seconds) of obs_topic data before lidar fallback is used.
        self.declare_parameter('obs_timeout_sec', 1.0)
        # QoS reliability for obs_topic: reliable|best_effort.
        self.declare_parameter('obs_qos_reliability', 'reliable')
        # QoS queue depth for obs_topic.
        self.declare_parameter('obs_qos_depth', 10)
        # Lidar topic used only as fallback when Frenet obstacle data is stale.
        self.declare_parameter('scan_topic', '/scan')
        # Fallback trigger distance (meters): closer than this enters FTG.
        self.declare_parameter('scan_trigger_distance', 0.7)
        # Fallback clear distance (meters): farther than this returns to GB_TRACK.
        self.declare_parameter('scan_clear_distance', 1.2)
        self.obs_topic = self.get_parameter('obs_topic').value
        self.safety_lateral_distance = float(self.get_parameter('safety_lateral_distance').value)
        self.trigger_distance = float(self.get_parameter('trigger_distance').value)
        self.obs_timeout_sec = float(self.get_parameter('obs_timeout_sec').value)
        obs_qos_reliability = str(self.get_parameter('obs_qos_reliability').value).strip().lower()
        obs_qos_depth = int(self.get_parameter('obs_qos_depth').value)
        self.scan_topic = self.get_parameter('scan_topic').value
        self.scan_trigger_distance = float(self.get_parameter('scan_trigger_distance').value)
        self.scan_clear_distance = float(self.get_parameter('scan_clear_distance').value)
        self.last_obs_msg_time = None
        self._warned_no_obs = False
        self._warned_stale_obs = False

        if obs_qos_reliability == 'best_effort':
            obs_reliability = QoSReliabilityPolicy.BEST_EFFORT
        else:
            obs_reliability = QoSReliabilityPolicy.RELIABLE
            if obs_qos_reliability != 'reliable':
                self.get_logger().warn(
                    f"Unsupported obs_qos_reliability='{obs_qos_reliability}', defaulting to reliable"
                )

        obs_qos_profile = QoSProfile(depth=max(1, obs_qos_depth), reliability=obs_reliability)

        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            self.obs_topic,
            self.obs_callback,
            obs_qos_profile,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.state_pub = self.create_publisher(DriveStateMsg, '/state', 10)

        self.get_logger().info(
            "State machine started (obs_topic=%s, safety_lateral_distance=%.2f, trigger_distance=%.2f, obs_timeout_sec=%.2f, obs_qos=%s/%d)"
            % (
                self.obs_topic,
                self.safety_lateral_distance,
                self.trigger_distance,
                self.obs_timeout_sec,
                'reliable' if obs_reliability == QoSReliabilityPolicy.RELIABLE else 'best_effort',
                max(1, obs_qos_depth),
            )
        )
        self.create_timer(1.0, self.monitor_obs_topic)
        self.publish_state()

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_state(self):
        state_msg = DriveStateMsg()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.header.frame_id = 'state_machine'
        state_msg.state = self.current_state.value
        self.state_pub.publish(state_msg)

    def parse_obs_wpts(self, msg):
        raw = np.array(msg.data, dtype=float)
        if raw.size == 0:
            return np.empty((0, 2), dtype=float)

        # detect.py publishes [s, d, vs, vd, size_s, size_d, id] per obstacle.
        if raw.size % 7 == 0:
            reshaped = raw.reshape((-1, 7))
            obs = reshaped[:, :2]
            finite_mask = np.isfinite(obs).all(axis=1)
            return obs[finite_mask]

        # Fallback: also accept legacy [s0, d0, s1, d1, ...] payloads.
        if raw.size % 2 != 0:
            self.get_logger().warn('tracked_obstacles payload length is invalid, dropping last value')
            raw = raw[:-1]

        obs = raw.reshape((-1, 2))
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
        self._warned_stale_obs = False
        if self._warned_no_obs:
            self.get_logger().info(f"Obstacle stream active on {self.obs_topic}")
            self._warned_no_obs = False

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
                # Trigger FTG when closest obstacle is laterally too close.
                if lateral_close:
                    new_state = DriveState.FTGONLY
            else:
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

        if self.last_obs_msg_time is not None and not self._warned_stale_obs:
            age = now - self.last_obs_msg_time
            self.get_logger().warn(
                f"Obstacle data stale for {age:.2f}s (> {self.obs_timeout_sec:.2f}s). Using lidar fallback."
            )
            self._warned_stale_obs = True

        ranges = np.array(msg.ranges, dtype=float)
        valid = ranges[np.isfinite(ranges) & (ranges > 0.05)]
        if valid.size == 0:
            return

        min_dist = float(np.min(valid))
        new_state = self.current_state
        if self.current_state == DriveState.GB_TRACK and min_dist < self.scan_trigger_distance:
            new_state = DriveState.FTGONLY
        elif self.current_state == DriveState.FTGONLY and min_dist > self.scan_clear_distance:
            new_state = DriveState.GB_TRACK

        if new_state != self.current_state:
            self.get_logger().info(
                f"State switch {self.current_state.value} -> {new_state.value} "
                f"(lidar fallback, min_dist={min_dist:.2f})"
            )
            self.current_state = new_state
            self.publish_state()

    def monitor_obs_topic(self):
        if self.last_obs_msg_time is not None:
            return
        if not self._warned_no_obs:
            self.get_logger().warn(
                f"No obstacle messages received on {self.obs_topic}. "
                "Make sure perception/detect is running."
            )
            self._warned_no_obs = True

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
