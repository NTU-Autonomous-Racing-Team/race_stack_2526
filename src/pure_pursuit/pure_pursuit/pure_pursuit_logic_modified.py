import numpy as np


class PurePursuitLogic:
    def __init__(self, wheelbase, waypoints):
        self.L = wheelbase
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
        self.current_idx = 0
        self.steering_gain = 1.0

        self.trailing_gap = 1.5
        self.trailing_p_gain = 0.5
        self.trailing_i_gain = 0.05
        self.trailing_d_gain = 0.1
        self.blind_trailing_speed = 2.2

        self.i_gap = 0.0
        self.loop_rate = 20.0

    def trailing_controller(self, ego_s, ego_vel, opp_s, opp_vel, global_speed, track_length):
        raw_gap = opp_s - ego_s
        gap_actual = (raw_gap + (track_length / 2.0)) % track_length - (track_length / 2.0)

        emergency_gap = 0.8
        if gap_actual < emergency_gap:
            return max(0.0, opp_vel * 0.8)

        gap_error = self.trailing_gap - gap_actual
        v_diff = ego_vel - opp_vel

        if abs(gap_error) < 2.0:
            self.i_gap = np.clip(self.i_gap + gap_error / self.loop_rate, -5.0, 5.0)
        else:
            self.i_gap *= 0.9

        p_value = gap_error * self.trailing_p_gain
        i_value = self.i_gap * self.trailing_i_gain
        d_value = v_diff * self.trailing_d_gain

        trailing_speed = np.clip(opp_vel - p_value - i_value - d_value, 0.0, global_speed)
        if gap_actual > self.trailing_gap:
            trailing_speed = max(self.blind_trailing_speed, trailing_speed)

        return trailing_speed

    def transform_point_to_car_frame(self, car_x, car_y, car_yaw, point):
        dx = point[0] - car_x
        dy = point[1] - car_y
        cos_y = np.cos(-car_yaw)
        sin_y = np.sin(-car_yaw)
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y
        return np.array([local_x, local_y])

    def find_target_waypoint(self, car_x, car_y, car_yaw, lookahead_dist):
        start = self.current_idx
        end = (start + 120) % self.num_waypoints

        final_i = -1
        longest_dist = 0.0

        if end < start:
            search_range = list(range(start, self.num_waypoints)) + list(range(0, end))
        else:
            search_range = range(start, end)

        car_pos = np.array([car_x, car_y], dtype=float)

        for i in search_range:
            p_world = self.waypoints[i, :2]
            dist = float(np.linalg.norm(p_world - car_pos))
            p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, p_world)
            if dist <= lookahead_dist and dist >= longest_dist and p_car[0] > 0.0:
                longest_dist = dist
                final_i = i

        if final_i != -1:
            self.current_idx = final_i
            target_pt_car = self.transform_point_to_car_frame(
                car_x, car_y, car_yaw, self.waypoints[final_i, :2]
            )
            return target_pt_car, max(longest_dist, 0.1), final_i

        distances = np.linalg.norm(self.waypoints[:, :2] - car_pos, axis=1)
        nearest_i = int(np.argmin(distances))
        final_i = nearest_i

        for offset in range(1, min(40, self.num_waypoints)):
            candidate_i = (nearest_i + offset) % self.num_waypoints
            p_car = self.transform_point_to_car_frame(
                car_x, car_y, car_yaw, self.waypoints[candidate_i, :2]
            )
            if p_car[0] > 0.0:
                final_i = candidate_i
                break

        self.current_idx = final_i
        target_pt_car = self.transform_point_to_car_frame(
            car_x, car_y, car_yaw, self.waypoints[final_i, :2]
        )
        return target_pt_car, max(float(distances[final_i]), 0.1), final_i

    def calculate_steering(self, target_point, lookahead_dist):
        y = target_point[1]
        safe_la = max(lookahead_dist, 0.1)
        steering_angle = np.arctan((2.0 * self.L * y) / (safe_la ** 2))
        if np.isnan(steering_angle) or np.isinf(steering_angle):
            steering_angle = 0.0
        steering_angle *= self.steering_gain
        return steering_angle
