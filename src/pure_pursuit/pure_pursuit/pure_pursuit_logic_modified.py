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
        self.blind_trailing_speed = 3.0

        self.i_gap = 0.0
        self.loop_rate = 20.0 

    def trailing_controller(self, ego_s, ego_vel, opp_s, opp_vel, global_speed, track_length):
        # 1. Calculate the raw difference
        raw_gap = opp_s - ego_s
        # 2. Normalize the gap to be between -track_length/2 and +track_length/2
        # This handles the start/finish line wrap AND prevents the overtake bug
        gap_actual = (raw_gap + (track_length / 2)) % track_length - (track_length / 2)

        EMERGENCY_GAP = 0.8
        if gap_actual < EMERGENCY_GAP:
            return opp_vel * 0.8

        gap_error = self.trailing_gap - gap_actual
        v_diff = ego_vel - opp_vel
    
        # Only integrate if the error is relatively small (e.g., within 2 meters)
        # This prevents massive windup during cornering noise
        if abs(gap_error) < 2.0:
            self.i_gap = np.clip(self.i_gap + gap_error / self.loop_rate, -5.0, 5.0)
        else:
            # If the error is huge, decay the integral so it stops holding the brakes
            self.i_gap *= 0.9
    
        p_value = gap_error * self.trailing_p_gain
        i_value = self.i_gap * self.trailing_i_gain
        d_value = v_diff * self.trailing_d_gain
    
        trailing_speed = np.clip(
            opp_vel - p_value - i_value - d_value,
            0, global_speed
        )
        # Only apply blind speed floor when gap is larger than desired
        # Never override PID when we're already too close
        if gap_actual > self.trailing_gap:
            trailing_speed = max(self.blind_trailing_speed, trailing_speed)

        '''
        BLIND TRAILING: FOR ACTUAL IMPLEMENTATION
        time_since_opp = (self.get_clock().now() - self.last_opp_time).nanoseconds / 1e9
        if time_since_opp > 0.5:  
            trailing_speed = max(self.blind_trailing_speed, trailing_speed)
        '''
        return trailing_speed

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
        3. Find exact geometric intersection of path segments with lookahead circle.
        4. Enforce forward-half-plane (target must be in front of the car).
        """
        start = self.current_idx
        # Use modulo to create a circular buffer effect
        end = (start + 100) % self.num_waypoints 
        
        final_i = -1
        target_pt_world = None

        # Define the search range based on whether it loops around the array end
        if end < start:
            # Case: Window crosses the finish line (e.g., from index 950 to 50)
            search_range = list(range(start, self.num_waypoints)) + list(range(0, end))
        else:
            # Case: Normal sequential search
            search_range = range(start, end)
            
        car_pos = np.array([car_x, car_y])

        for i in search_range:
            # Define the line segment between current waypoint and the next
            p1 = self.waypoints[i, :2]
            p2 = self.waypoints[(i + 1) % self.num_waypoints, :2]
            
            # Line-circle intersection math
            d = p2 - p1 # Vector of the line segment
            f = p1 - car_pos # Vector from circle center (car) to segment start
            
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - lookahead_dist**2
            
            if a == 0:
                continue # Edge case: waypoints are identical, avoid division by zero
                
            discriminant = b**2 - 4 * a * c
            
            # If discriminant >= 0, the line intersects the circle
            if discriminant >= 0:
                # Solve for t (where t=0 is p1, and t=1 is p2)
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                
                # Check if the intersection(s) lie strictly ON the segment (between 0 and 1)
                for t in [t1, t2]:
                    if 0 <= t <= 1:
                        # Calculate the exact world coordinate of the intersection
                        intersect_pt_world = p1 + t * d
                        
                        # Transform to car frame to enforce forward-half-plane constraint
                        p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, intersect_pt_world)
                        
                        # Forward Fix: The point MUST be in front of the car
                        if p_car[0] > 0:
                            target_pt_world = intersect_pt_world
                            final_i = i

        # If a valid exact intersection was found
        if target_pt_world is not None:
            self.current_idx = final_i
            target_pt_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, target_pt_world)
            # The distance is mathematically exactly the lookahead_dist
            return target_pt_car, lookahead_dist, final_i
            
        else:
            # print("FALLBACK TO FIND WAYPOINT IN FRONT (NO INTERSECTION)")
            # 1. Calculate distances to all waypoints (O(N) operation)
            distances = np.linalg.norm(self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1)
            
            # 2. Find the index of the absolute closest waypoint (O(N) operation)
            closest_idx = np.argmin(distances)
            final_i = closest_idx
            
            # 3. Iterate forward from the closest point to find the first one in front of the car
            for offset in range(self.num_waypoints):
                check_idx = (closest_idx + offset) % self.num_waypoints
                p_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, self.waypoints[check_idx, :2])
                
                if p_car[0] > 0 and distances[check_idx] >= lookahead_dist: 
                    final_i = check_idx
                    break

            self.current_idx = final_i
            target_pt_car = self.transform_point_to_car_frame(car_x, car_y, car_yaw, self.waypoints[final_i, :2])
            longest_dist = distances[final_i] # Return the actual discrete distance

            # print(f"FALLBACK TRIGGERED at idx={self.current_idx}")
            return target_pt_car, longest_dist, final_i

    def calculate_steering(self, target_point, lookahead_dist):
        y = target_point[1]
        safe_la = max(lookahead_dist, 0.1)
        # Ackermann-adjusted pure pursuit formula from Eq. 2
        # theta = arctan(2 * L * y / l_d^2)
        steering_angle = np.arctan((2.0 * self.L * y) / (safe_la**2))
        if np.isnan(steering_angle) or np.isinf(steering_angle):
            steering_angle = 0.0
        steering_angle *= self.steering_gain
        
        return steering_angle

        # y = target_point[1]
        # # Calculated from https://docs.google.com/presentation/d/1jpnlQ7ysygTPCi8dmyZjooqzxNXWqMgO31ZhcOlKVOE/edit#slide=id.g63d5f5680f_0_33
        # safe_la = max(lookahead_dist, 0.1)
        # steering_angle = k_p * (2.0 * y) / (safe_la**2)
        # if np.isnan(steering_angle) or np.isinf(steering_angle):
        #     steering_angle = 0.0
        # return steering_angle