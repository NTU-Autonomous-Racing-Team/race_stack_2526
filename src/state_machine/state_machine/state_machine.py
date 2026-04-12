#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from state_machine.drive_state import DriveState


class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('state_topic', '/state')
        self.declare_parameter('opponent_flag_topic', '/opponent_detected')
        self.declare_parameter('opponent_pose_topic', '/opponent_detection')
        self.declare_parameter('obstacle_flag_topic', '/obstacle_detected')
        self.declare_parameter('obstacle_distance_topic', '/obstacle_distance')
        self.declare_parameter('planner_feasible_topic', '/planner/overtake_feasible')
        self.declare_parameter('planner_path_active_topic', '/planner/path_active')
        self.declare_parameter('allow_overtake', True)

        self.declare_parameter('opponent_trigger_distance', 3.0)
        self.declare_parameter('static_trigger_distance', 1.1)
        self.declare_parameter('hard_safety_distance', 0.22)
        self.declare_parameter('clear_distance', 1.25)
        self.declare_parameter('front_window', 35)
        self.declare_parameter('hazard_confirm_cycles', 2)
        self.declare_parameter('required_clear_cycles', 7)
        self.declare_parameter('overtake_timeout_sec', 5.0)
        self.declare_parameter('overtake_min_commit_sec', 1.2)
        self.declare_parameter('overtake_lost_cycles_to_fallback', 8)
        self.declare_parameter('planner_hold_sec', 0.45)
        self.declare_parameter('overtake_engage_distance', 1.9)
        self.declare_parameter('emergency_persist_cycles', 2)
        self.declare_parameter('ignore_opponent', False)

        self.declare_parameter('state_marker_topic', '/state_marker')
        self.declare_parameter('state_marker_frame', 'ego_racecar/base_link')

        self.scan_topic = self.get_parameter('scan_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.opponent_flag_topic = self.get_parameter('opponent_flag_topic').value
        self.opponent_pose_topic = self.get_parameter('opponent_pose_topic').value
        self.obstacle_flag_topic = self.get_parameter('obstacle_flag_topic').value
        self.obstacle_distance_topic = self.get_parameter('obstacle_distance_topic').value
        self.planner_feasible_topic = self.get_parameter('planner_feasible_topic').value
        self.planner_path_active_topic = self.get_parameter('planner_path_active_topic').value
        self.allow_overtake = bool(self.get_parameter('allow_overtake').value)

        self.opponent_trigger_distance = float(self.get_parameter('opponent_trigger_distance').value)
        self.static_trigger_distance = float(self.get_parameter('static_trigger_distance').value)
        self.hard_safety_distance = float(self.get_parameter('hard_safety_distance').value)
        self.clear_distance = float(self.get_parameter('clear_distance').value)
        self.front_window = int(self.get_parameter('front_window').value)
        self.hazard_confirm_cycles = int(self.get_parameter('hazard_confirm_cycles').value)
        self.required_clear_cycles = int(self.get_parameter('required_clear_cycles').value)
        self.overtake_timeout_sec = float(self.get_parameter('overtake_timeout_sec').value)
        self.overtake_min_commit_sec = float(self.get_parameter('overtake_min_commit_sec').value)
        self.overtake_lost_cycles_to_fallback = int(
            self.get_parameter('overtake_lost_cycles_to_fallback').value
        )
        self.planner_hold_sec = float(self.get_parameter('planner_hold_sec').value)
        self.overtake_engage_distance = float(self.get_parameter('overtake_engage_distance').value)
        self.emergency_persist_cycles = int(self.get_parameter('emergency_persist_cycles').value)
        self.ignore_opponent = bool(self.get_parameter('ignore_opponent').value)

        self.state_marker_topic = self.get_parameter('state_marker_topic').value
        self.state_marker_frame = self.get_parameter('state_marker_frame').value

        self.current_state = DriveState.GB_TRACK
        self.last_state_change_ts = time.time()

        self.latest_scan = None
        self.latest_front_min = float('inf')

        self.opponent_detected = False
        self.opponent_distance = None

        self.static_detected = False
        self.static_distance = float('inf')

        self.planner_feasible = False
        self.planner_path_active = False
        self.last_planner_feasible_ts = 0.0
        self.last_planner_active_ts = 0.0

        self.clear_cycles = 0
        self.overtake_lost_cycles = 0
        self.emergency_cycles = 0
        self.static_hazard_cycles = 0
        self.dynamic_hazard_cycles = 0

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.opponent_flag_sub = self.create_subscription(Bool, self.opponent_flag_topic, self.opponent_flag_callback, 10)
        self.opponent_pose_sub = self.create_subscription(PoseStamped, self.opponent_pose_topic, self.opponent_pose_callback, 10)
        self.static_flag_sub = self.create_subscription(Bool, self.obstacle_flag_topic, self.static_flag_callback, 10)
        self.static_distance_sub = self.create_subscription(Float32, self.obstacle_distance_topic, self.static_distance_callback, 10)
        self.feasible_sub = self.create_subscription(Bool, self.planner_feasible_topic, self.feasible_callback, 10)
        self.path_active_sub = self.create_subscription(Bool, self.planner_path_active_topic, self.path_active_callback, 10)

        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.state_marker_pub = self.create_publisher(MarkerArray, self.state_marker_topic, 10)

        self.create_timer(0.05, self.state_update_tick)
        self.create_timer(0.25, self.publish_state_marker)

        if self.allow_overtake:
            self.get_logger().info('State machine started with states: GB_TRACK/OVERTAKE')
        else:
            self.get_logger().info('State machine started with states: GB_TRACK')
        self.publish_state()

    def _state_color(self, state):
        if state == DriveState.GB_TRACK:
            return (0.1, 0.95, 0.1)
        if state == DriveState.OVERTAKE:
            return (0.2, 0.85, 1.0)
        return (0.9, 0.9, 0.9)

    def publish_state_marker(self):
        marker_array = MarkerArray()

        state_marker = Marker()
        state_marker.header.stamp = self.get_clock().now().to_msg()
        state_marker.header.frame_id = self.state_marker_frame
        state_marker.ns = 'state_machine'
        state_marker.id = 0
        state_marker.type = Marker.TEXT_VIEW_FACING
        state_marker.action = Marker.ADD
        state_marker.pose.position.x = 0.0
        state_marker.pose.position.y = 0.0
        state_marker.pose.position.z = 1.15
        state_marker.pose.orientation.w = 1.0
        state_marker.scale.z = 0.40
        state_marker.color.a = 1.0
        r, g, b = self._state_color(self.current_state)
        state_marker.color.r = r
        state_marker.color.g = g
        state_marker.color.b = b
        state_marker.text = f'STATE: {self.current_state.value}'
        marker_array.markers.append(state_marker)

        legend = [('GB_TRACK = GREEN', DriveState.GB_TRACK)]
        if self.allow_overtake:
            legend.append(('OVERTAKE = CYAN', DriveState.OVERTAKE))

        for idx, (text, state) in enumerate(legend, start=1):
            mk = Marker()
            mk.header = state_marker.header
            mk.ns = 'state_legend'
            mk.id = idx
            mk.type = Marker.TEXT_VIEW_FACING
            mk.action = Marker.ADD
            mk.pose.position.x = 0.0
            mk.pose.position.y = -0.75 + (idx - 1) * 0.50
            mk.pose.position.z = 1.65
            mk.pose.orientation.w = 1.0
            mk.scale.z = 0.18
            mk.color.a = 0.95
            cr, cg, cb = self._state_color(state)
            mk.color.r = cr
            mk.color.g = cg
            mk.color.b = cb
            mk.text = text
            marker_array.markers.append(mk)

        self.state_marker_pub.publish(marker_array)

    def publish_state(self):
        msg = String()
        msg.data = self.current_state.value
        self.state_pub.publish(msg)

    def _set_state(self, new_state, reason):
        if new_state == self.current_state:
            return
        old_state = self.current_state
        self.current_state = new_state
        self.last_state_change_ts = time.time()
        self.clear_cycles = 0
        self.overtake_lost_cycles = 0
        self.emergency_cycles = 0
        self.static_hazard_cycles = 0
        self.dynamic_hazard_cycles = 0
        self.get_logger().info(f'State switch {old_state.value} -> {new_state.value} ({reason})')
        self.publish_state()

    def opponent_flag_callback(self, msg: Bool):
        self.opponent_detected = bool(msg.data)

    def opponent_pose_callback(self, msg: PoseStamped):
        dx = float(msg.pose.position.x)
        dy = float(msg.pose.position.y)
        self.opponent_distance = math.sqrt(dx * dx + dy * dy)

    def static_flag_callback(self, msg: Bool):
        self.static_detected = bool(msg.data)

    def static_distance_callback(self, msg: Float32):
        d = float(msg.data)
        self.static_distance = d if d > 0.0 else float('inf')

    def feasible_callback(self, msg: Bool):
        self.planner_feasible = bool(msg.data)
        if self.planner_feasible:
            self.last_planner_feasible_ts = time.time()

    def path_active_callback(self, msg: Bool):
        self.planner_path_active = bool(msg.data)
        if self.planner_path_active:
            self.last_planner_active_ts = time.time()

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isnan(ranges)] = 0.0
        inf_fill = msg.range_max if msg.range_max > 0.1 else 10.0
        ranges[np.isinf(ranges)] = inf_fill

        mid = len(ranges) // 2
        win = max(1, self.front_window)
        start = max(0, mid - win)
        end = min(len(ranges), mid + win)
        front = ranges[start:end]
        valid = front[front > 0.05]

        if valid.size == 0:
            self.latest_front_min = float('inf')
            return
        self.latest_front_min = float(np.min(valid))

    def _hazards(self):
        static_hazard_raw = self.static_detected and self.static_distance < self.static_trigger_distance
        dynamic_hazard_raw = (
            (not self.ignore_opponent)
            and self.opponent_detected
            and self.opponent_distance is not None
            and self.opponent_distance < self.opponent_trigger_distance
        )
        if static_hazard_raw:
            self.static_hazard_cycles += 1
        else:
            self.static_hazard_cycles = 0
        if dynamic_hazard_raw:
            self.dynamic_hazard_cycles += 1
        else:
            self.dynamic_hazard_cycles = 0

        static_hazard = self.static_hazard_cycles >= self.hazard_confirm_cycles
        dynamic_hazard = self.dynamic_hazard_cycles >= self.hazard_confirm_cycles
        return static_hazard, dynamic_hazard, (static_hazard or dynamic_hazard)

    def _planner_ready(self):
        if self.planner_feasible and self.planner_path_active:
            return True
        now = time.time()
        feasible_recent = (now - self.last_planner_feasible_ts) <= self.planner_hold_sec
        active_recent = (now - self.last_planner_active_ts) <= self.planner_hold_sec
        return feasible_recent or active_recent

    def state_update_tick(self):
        static_hazard, dynamic_hazard, hazard = self._hazards()
        planner_ready = self._planner_ready()

        # Emergency is only used as a stronger trigger into OVERTAKE when planner is ready.
        min_front = self.latest_front_min
        emergency = self.latest_front_min < self.hard_safety_distance
        if static_hazard:
            min_front = min(min_front, self.static_distance)
        elif dynamic_hazard and self.opponent_distance is not None:
            min_front = min(min_front, self.opponent_distance)

        if emergency:
            self.emergency_cycles += 1
        else:
            self.emergency_cycles = 0

        state_age = time.time() - self.last_state_change_ts

        if not self.allow_overtake:
            if self.current_state != DriveState.GB_TRACK:
                self._set_state(DriveState.GB_TRACK, 'overtake disabled')
            return

        if self.current_state == DriveState.GB_TRACK:
            if hazard:
                if planner_ready or (emergency and planner_ready):
                    self._set_state(DriveState.OVERTAKE, 'hazard detected with planner ready')
            return

        if self.current_state == DriveState.OVERTAKE:
            if state_age > self.overtake_timeout_sec:
                self._set_state(DriveState.GB_TRACK, 'overtake timeout; return to GB')
                return

            if not hazard:
                if state_age < self.overtake_min_commit_sec:
                    return
                if min_front > self.clear_distance:
                    self.clear_cycles += 1
                    if self.clear_cycles >= self.required_clear_cycles:
                        self._set_state(DriveState.GB_TRACK, 'overtake complete and clear')
                else:
                    self.clear_cycles = 0
                return

            self.clear_cycles = 0
            if not planner_ready:
                self.overtake_lost_cycles += 1
                if self.overtake_lost_cycles >= self.overtake_lost_cycles_to_fallback:
                    self._set_state(DriveState.GB_TRACK, 'planner lost; return to GB')
                    return
            else:
                self.overtake_lost_cycles = 0
            return


def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
