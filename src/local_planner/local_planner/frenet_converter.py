import numpy as np

class FrenetConverter:
    def __init__(self, waypoints):
        """
        Initializes the FrenetConverter with a reference path.
        :param waypoints: Nx2 or Nx3 numpy array of waypoints [x, y, ...].
        """
        self.waypoints = np.array(waypoints)[:, :2]
        self.s = self._calculate_s(self.waypoints)
        self.total_length = self.s[-1]

    def _calculate_s(self, waypoints):
        """ Pre-calculate cumulative distance along the path. """
        diffs = np.diff(waypoints, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        s = np.zeros(len(waypoints))
        s[1:] = np.cumsum(dists)
        return s

    def get_frenet(self, x, y):
        """
        Converts Cartesian (x, y) to Frenet (s, d).
        """
        # 1. Find nearest waypoint index
        dists = np.linalg.norm(self.waypoints - np.array([x, y]), axis=1)
        idx = np.argmin(dists)

        # 2. Find the two closest waypoints to form a segment
        # Check adjacent segments and pick the one where the projection is within bounds
        if idx == 0:
            idx_start, idx_end = 0, 1
        elif idx == len(self.waypoints) - 1:
            idx_start, idx_end = len(self.waypoints) - 2, len(self.waypoints) - 1
        else:
            # Check segment [idx-1, idx] and [idx, idx+1]
            # For simplicity, we just pick the one that results in a projection between 0 and 1
            # Or just use the one with the smallest perpendicular distance
            idx_start, idx_end = self._find_best_segment(x, y, idx)

        p1 = self.waypoints[idx_start]
        p2 = self.waypoints[idx_end]
        
        # Vector from p1 to p2
        v = p2 - p1
        # Vector from p1 to point (x, y)
        w = np.array([x, y]) - p1
        
        # Projection of w onto v
        v_norm_sq = np.dot(v, v)
        if v_norm_sq == 0:
            return self.s[idx_start], 0.0
            
        t = np.dot(w, v) / v_norm_sq
        # Clamp t to [0, 1] if we want to stay on the segment, 
        # but for Frenet we usually allow projection outside for continuity if needed.
        # However, for a racecar, we usually expect to be near a segment.
        t_clamped = np.clip(t, 0.0, 1.0)
        
        # Longitudinal distance s
        s = self.s[idx_start] + t_clamped * np.linalg.norm(v)
        
        # Lateral distance d (signed)
        # Normal vector to v (rotate v by 90 degrees CCW)
        normal = np.array([-v[1], v[0]])
        normal = normal / np.linalg.norm(normal)
        
        projection_point = p1 + t_clamped * v
        offset_vec = np.array([x, y]) - projection_point
        d = np.dot(offset_vec, normal)
        
        return s, d

    def _find_best_segment(self, x, y, idx):
        """ Helper to find the best segment among the two neighbors of idx. """
        # Try [idx-1, idx]
        p_prev = self.waypoints[idx-1]
        p_curr = self.waypoints[idx]
        p_next = self.waypoints[idx+1]
        
        def dist_to_segment(p1, p2, px, py):
            v = p2 - p1
            w = np.array([px, py]) - p1
            v_norm_sq = np.dot(v, v)
            if v_norm_sq == 0: return np.linalg.norm(w)
            t = np.clip(np.dot(w, v) / v_norm_sq, 0.0, 1.0)
            proj = p1 + t * v
            return np.linalg.norm(np.array([px, py]) - proj)

        d1 = dist_to_segment(p_prev, p_curr, x, y)
        d2 = dist_to_segment(p_curr, p_next, x, y)
        
        if d1 < d2:
            return idx - 1, idx
        else:
            return idx, idx + 1

    def get_cartesian(self, s, d):
        """
        Converts Frenet (s, d) to Cartesian (x, y).
        """
        # Use modulo to find position on the track (looping)
        s = s % self.total_length
        idx = np.searchsorted(self.s, s)
        
        if idx == 0:
            idx_start, idx_end = 0, 1
        elif idx >= len(self.s):
            idx_start, idx_end = len(self.s) - 2, len(self.s) - 1
        else:
            idx_start, idx_end = idx - 1, idx
            
        s_start = self.s[idx_start]
        s_end = self.s[idx_end]
        
        p1 = self.waypoints[idx_start]
        p2 = self.waypoints[idx_end]
        
        # Linear interpolation factor
        if s_end == s_start:
            t = 0.0
        else:
            t = (s - s_start) / (s_end - s_start)
            
        # Point on the reference path
        p_ref = p1 + t * (p2 - p1)
        
        # Normal vector to the segment
        v = p2 - p1
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return p_ref[0], p_ref[1]
            
        normal = np.array([-v[1], v[0]]) / v_norm
        
        # Add offset d
        p_cartesian = p_ref + d * normal
        
        return p_cartesian[0], p_cartesian[1]
