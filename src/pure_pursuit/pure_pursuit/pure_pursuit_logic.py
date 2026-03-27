import numpy as np

class PurePursuitLogic:
    def __init__(self, wheelbase, waypoints):
        self.L = wheelbase
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
        self.current_idx = 0

    #transform from map frame to baselink frame
    def transform_point_to_car_frame(self, car_x, car_y, car_yaw, point):
        """
        Manually transforms a world-frame point to the car's local frame.
        x > 0: In front of the car
        y > 0: To the left of the car
        """
        dx = point[0] - car_x
        dy = point[1] - car_y
        cos_y = np.cos(-car_yaw)
        sin_y = np.sin(-car_yaw)
        # 2D Rotation matrix calculation
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y
        return np.array([local_x, local_y])

    def find_target_waypoint(self, car_x, car_y, car_yaw, lookahead_dist):
        """
        1. Search within a 100-point window from the last found index.
        2. Handle 'loop around' if the window crosses the end of the array.
        3. Enforce forward-half-plane (target must be in front of the car).
        """
        start = self.current_idx
        # Use modulo to create a circular buffer effect
        end = (start + 100) % self.num_waypoints 
        
        final_i = -1
        longest_dist = 0

        # Define the search range based on whether it loops around the array end
        if end < start:
            # Case: Window crosses the finish line (e.g., from index 950 to 50)
            search_range = list(range(start, self.num_waypoints)) + list(range(0, end))
        else:
            # Case: Normal sequential search
            search_range = range(start, end)

        for i in search_range:
            # The ith waypoint x and y
            p_world = self.waypoints[i, :2]
            # Euclidean distance from car to waypoint
            dist = np.linalg.norm(p_world - np.array([car_x, car_y]))
            
            # Transform the world-frame point to the car's local frame
            p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, p_world)
            
            # CRITICAL CONDITIONS:
            # 1. dist <= lookahead_dist: Within the lookahead circle.
            # 2. dist >= longest_dist: Pick the furthest point in the circle for smoothness.
            # 3. p_car[0] > 0: The point MUST be in front of the car (Forward Fix).
            if dist <= lookahead_dist and dist >= longest_dist and p_car[0] > 0:
                longest_dist = dist
                final_i = i

        # Fallback Logic: If no point found in the local window, return the last known index
        if final_i != -1:
            self.current_idx = final_i
        else:
            distances = np.linalg.norm(self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1)
            # Find indices where point is in front
            for i in np.argsort(distances): # Check points starting from the closest
                p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, self.waypoints[i, :2])
                if p_car[0] > 0: # If it's in front, snap to it
                    final_i = i
                    break

        target_pt_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, self.waypoints[final_i, :2])
        return target_pt_car, longest_dist, final_i

    def calculate_steering(self, target_point, lookahead_dist, k_p):
        y = target_point[1]
        # Calculated from https://docs.google.com/presentation/d/1jpnlQ7ysygTPCi8dmyZjooqzxNXWqMgO31ZhcOlKVOE/edit#slide=id.g63d5f5680f_0_33
        safe_la = max(lookahead_dist, 0.1)
        steering_angle = k_p * (2.0 * y) / (safe_la**2)
        if np.isnan(steering_angle) or np.isinf(steering_angle):
            steering_angle = 0.0
        return steering_angle