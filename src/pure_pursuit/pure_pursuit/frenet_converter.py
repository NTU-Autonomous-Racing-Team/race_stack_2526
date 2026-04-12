import numpy as np


class FrenetConverter:
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)[:, :2]
        self.s = self._calculate_s(self.waypoints)
        self.total_length = self.s[-1]

    def _calculate_s(self, waypoints):
        diffs = np.diff(waypoints, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        s = np.zeros(len(waypoints))
        s[1:] = np.cumsum(dists)
        return s

    def get_frenet(self, x, y):
        dists = np.linalg.norm(self.waypoints - np.array([x, y]), axis=1)
        idx = np.argmin(dists)

        if idx == 0:
            idx_start, idx_end = 0, 1
        elif idx == len(self.waypoints) - 1:
            idx_start, idx_end = len(self.waypoints) - 2, len(self.waypoints) - 1
        else:
            idx_start, idx_end = self._find_best_segment(x, y, idx)

        p1 = self.waypoints[idx_start]
        p2 = self.waypoints[idx_end]

        v = p2 - p1
        w = np.array([x, y]) - p1

        v_norm_sq = np.dot(v, v)
        if v_norm_sq == 0:
            return self.s[idx_start], 0.0

        t = np.dot(w, v) / v_norm_sq
        t_clamped = np.clip(t, 0.0, 1.0)

        s = self.s[idx_start] + t_clamped * np.linalg.norm(v)

        normal = np.array([-v[1], v[0]])
        normal = normal / np.linalg.norm(normal)

        projection_point = p1 + t_clamped * v
        offset_vec = np.array([x, y]) - projection_point
        d = np.dot(offset_vec, normal)

        return s, d

    def _find_best_segment(self, x, y, idx):
        p_prev = self.waypoints[idx - 1]
        p_curr = self.waypoints[idx]
        p_next = self.waypoints[idx + 1]

        def dist_to_segment(p1, p2, px, py):
            v = p2 - p1
            w = np.array([px, py]) - p1
            v_norm_sq = np.dot(v, v)
            if v_norm_sq == 0:
                return np.linalg.norm(w)
            t = np.clip(np.dot(w, v) / v_norm_sq, 0.0, 1.0)
            proj = p1 + t * v
            return np.linalg.norm(np.array([px, py]) - proj)

        d1 = dist_to_segment(p_prev, p_curr, x, y)
        d2 = dist_to_segment(p_curr, p_next, x, y)

        if d1 < d2:
            return idx - 1, idx
        return idx, idx + 1

    def get_cartesian(self, s, d):
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

        if s_end == s_start:
            t = 0.0
        else:
            t = (s - s_start) / (s_end - s_start)

        p_ref = p1 + t * (p2 - p1)

        v = p2 - p1
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return p_ref[0], p_ref[1]

        normal = np.array([-v[1], v[0]]) / v_norm
        p_cartesian = p_ref + d * normal

        return p_cartesian[0], p_cartesian[1]
