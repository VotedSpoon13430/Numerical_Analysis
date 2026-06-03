import time
import math
import numpy as np
from scipy.spatial import Delaunay
from functionUtils import AbstractShape


class Assignment5:
    def __init__(self):
        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """Compute area from a contour function using adaptive sampling."""

        def shoelace(pts):
            pts = np.asarray(pts)
            if pts.shape[0] < 3:
                return 0.0
            nxt = np.roll(pts, -1, axis=0)
            return 0.5 * abs(np.sum(pts[:, 0] * nxt[:, 1] - pts[:, 1] * nxt[:, 0]))

        last = None
        for n in [128, 512, 2048, 8192]:
            a = shoelace(contour(n))
            if last is not None and abs(a - last) / max(a, 1e-12) < maxerr:
                return np.float32(a)
            last = a
        return np.float32(a)

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """Fit a shape to sampled points using adaptive algorithms."""
        start = time.time()
        points = []

        # Optimized sampling - aim for 10,000-15,000 points max
        target_samples = 12000
        time_limit = min(maxtime - 1.5, 18.5)  # Reserve 1.5s for calculations

        # Fast sampling with periodic checks
        batch_size = 100
        while len(points) < target_samples:
            if time.time() - start > time_limit:
                break

            # Sample in batches for efficiency
            for _ in range(batch_size):
                try:
                    points.append(sample())
                except:
                    break

            # Early exit if we have enough points and time is running out
            if len(points) >= 5000 and time.time() - start > time_limit * 0.8:
                break

        if len(points) < 5:
            return MyPolygon(points)

        pts = np.asarray(points, dtype=np.float32)  # Use float32 for speed

        # Compute centroid and convert to polar coordinates
        center = np.mean(pts, axis=0)
        dx = pts[:, 0] - center[0]
        dy = pts[:, 1] - center[1]
        radii = np.sqrt(dx * dx + dy * dy)
        angles = np.arctan2(dy, dx)

        mean_r = np.mean(radii)
        std_r = np.std(radii)
        cv = std_r / (mean_r + 1e-12)

        # Quick bounding box
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        aspect = max(x_range, y_range) / max(min(x_range, y_range), 1e-12)

        # Fast interior fraction check
        interior_frac = self._compute_interior_fraction_fast(pts, x_min, x_max, y_min, y_max)

        # OPTIMIZED CLASSIFICATION

        # 1. CIRCLES - fastest path
        if cv < 0.12:
            area_val = np.pi * max(0, mean_r ** 2 - std_r ** 2)
            return MyCircle(center[0], center[1], mean_r, area_val)

        # 2. THIN SHAPES
        elif interior_frac < 0.1:
            area_val = self._delaunay_area_fast(pts)
            return MyShapeWithArea(area_val, pts[:min(1000, len(pts))].tolist())

        # 3. ELONGATED SHAPES
        elif aspect > 100:
            area_val = self._spatial_binning_area_fast(pts, x_range, y_range, center)
            return MyShapeWithArea(area_val, pts[:min(1000, len(pts))].tolist())

        # 4. VERY COMPLEX SHAPES - use fewer bins
        elif cv > 0.7:
            area_val = self._angular_binning_area_fast(pts, center, angles, n_bins=8)
            return MyShapeWithArea(area_val, pts[:min(1000, len(pts))].tolist())

        # 5. COMPLEX SHAPES
        elif cv > 0.5:
            area_val = self._angular_binning_area_fast(pts, center, angles, n_bins=24)
            area_val = self._apply_noise_correction_fast(area_val, radii, angles, mean_r, n_bins=24)
            return MyShapeWithArea(area_val, pts[:min(1000, len(pts))].tolist())

        # 6. GENERAL POLYGONS - most common case
        else:
            # Adaptive bin count based on number of points
            if len(pts) > 8000:
                n_bins = 120
            elif len(pts) > 4000:
                n_bins = 90
            else:
                n_bins = 60

            area_val = self._angular_binning_area_fast(pts, center, angles, n_bins)
            area_val = self._apply_noise_correction_fast(area_val, radii, angles, mean_r, n_bins)
            return MyShapeWithArea(area_val, pts[:min(1000, len(pts))].tolist())

    def _compute_interior_fraction_fast(self, pts, x_min, x_max, y_min, y_max):
        """Fast interior fraction calculation."""
        margin = 0.15
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin

        interior = (
                (pts[:, 0] > x_min + x_margin) &
                (pts[:, 0] < x_max - x_margin) &
                (pts[:, 1] > y_min + y_margin) &
                (pts[:, 1] < y_max - y_margin)
        )

        return np.sum(interior) / len(pts)

    def _trimmed_mean_fast(self, arr, trim_pct=0.1):
        """Fast trimmed mean using percentiles."""
        if len(arr) < 5:
            return np.mean(arr)
        lower = np.percentile(arr, trim_pct * 100)
        upper = np.percentile(arr, (1 - trim_pct) * 100)
        mask = (arr >= lower) & (arr <= upper)
        return np.mean(arr[mask]) if np.any(mask) else np.mean(arr)

    def _angular_binning_area_fast(self, pts, center, angles, n_bins=72):
        """Optimized angular binning with vectorized operations."""
        # Digitize angles into bins
        bin_indices = ((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins

        # Store boundary points with their angles for proper ordering
        boundary_points = []
        boundary_angles = []

        # Vectorized binning
        for i in range(n_bins):
            mask = bin_indices == i
            count = np.sum(mask)
            if count >= 3:
                # Use percentile based trimmed mean for speed
                x_vals = pts[mask, 0]
                y_vals = pts[mask, 1]

                if count > 10:
                    # Trim outliers
                    x_sorted = np.sort(x_vals)
                    y_sorted = np.sort(y_vals)
                    trim = max(1, int(count * 0.1))
                    bx = np.mean(x_sorted[trim:-trim])
                    by = np.mean(y_sorted[trim:-trim])
                else:
                    bx = np.mean(x_vals)
                    by = np.mean(y_vals)

                boundary_points.append([bx, by])
                # Store the actual angle of this boundary point
                boundary_angles.append(np.arctan2(by - center[1], bx - center[0]))

        if len(boundary_points) < 3:
            return 0.0

        # Sort boundary points by angle to ensure correct ordering
        boundary_points = np.array(boundary_points)
        boundary_angles = np.array(boundary_angles)
        sort_idx = np.argsort(boundary_angles)
        boundary_points = boundary_points[sort_idx]

        x = boundary_points[:, 0]
        y = boundary_points[:, 1]

        # Vectorized Shoelace formula - always returns positive area
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area

    def _delaunay_area_fast(self, pts):
        """Fast Delaunay with sampling for large point sets."""
        # Sample points if too many
        if len(pts) > 3000:
            indices = np.random.choice(len(pts), 3000, replace=False)
            pts = pts[indices]

        if len(pts) < 3:
            return 0.0

        try:
            tri = Delaunay(pts)
            triangles = pts[tri.simplices]

            # Vectorized edge length computation
            edges = np.concatenate([
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
            ])

            median_edge = np.median(edges)
            threshold = median_edge * 1.3

            # Compute all edge lengths per triangle
            e1 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
            e2 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
            e3 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)

            # Filter triangles
            valid = (e1 < threshold) & (e2 < threshold) & (e3 < threshold)

            # Vectorized area computation using cross product
            v1 = triangles[valid, 1] - triangles[valid, 0]
            v2 = triangles[valid, 2] - triangles[valid, 0]
            areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

            return np.sum(areas)
        except:
            return 0.0

    def _spatial_binning_area_fast(self, pts, x_range, y_range, center):
        """Fast spatial binning for elongated shapes."""
        n_bins = min(80, len(pts) // 50)  # Adaptive bin count

        if x_range > y_range:
            # Bin along X
            bins = np.linspace(pts[:, 0].min(), pts[:, 0].max(), n_bins + 1)
            bin_indices = np.digitize(pts[:, 0], bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            median_y = np.median(pts[:, 1])
            upper_mask = pts[:, 1] >= median_y

            boundary = []
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask & upper_mask) > 0:
                    upper_pt = pts[mask & upper_mask].mean(axis=0)
                    boundary.append(upper_pt)

            for i in range(n_bins - 1, -1, -1):
                mask = bin_indices == i
                if np.sum(mask & ~upper_mask) > 0:
                    lower_pt = pts[mask & ~upper_mask].mean(axis=0)
                    boundary.append(lower_pt)
        else:
            # Bin along Y
            bins = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n_bins + 1)
            bin_indices = np.digitize(pts[:, 1], bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            median_x = np.median(pts[:, 0])
            left_mask = pts[:, 0] < median_x

            boundary = []
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask & left_mask) > 0:
                    left_pt = pts[mask & left_mask].mean(axis=0)
                    boundary.append(left_pt)

            for i in range(n_bins - 1, -1, -1):
                mask = bin_indices == i
                if np.sum(mask & ~left_mask) > 0:
                    right_pt = pts[mask & ~left_mask].mean(axis=0)
                    boundary.append(right_pt)

        if len(boundary) < 3:
            return 0.0

        boundary = np.array(boundary)
        x, y = boundary[:, 0], boundary[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area

    def _apply_noise_correction_fast(self, area, radii, angles, mean_r, n_bins):
        """Fast noise correction."""
        # Sample-based variance estimation
        if len(radii) > 5000:
            sample_indices = np.random.choice(len(radii), 5000, replace=False)
            radii = radii[sample_indices]
            angles = angles[sample_indices]

        bin_indices = ((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins

        variances = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) >= 5:
                variances.append(np.var(radii[mask]))

        if len(variances) > n_bins // 3:
            est_noise = np.sqrt(np.median(variances))  # Use median for robustness
            noise_ratio = est_noise / (mean_r + 1e-12)

            if noise_ratio > 0.03:
                # More conservative correction
                correction_factor = 1 + 1.2 * noise_ratio ** 2
                area = area / correction_factor

        return area


class MyPolygon(AbstractShape):
    def __init__(self, points):
        self.points = points

    def contour(self, n: int):
        if len(self.points) < 3:
            return self.points
        pts = np.asarray(self.points)
        idx = np.linspace(0, len(pts), n, endpoint=False)
        i0 = np.floor(idx).astype(int) % len(pts)
        i1 = (i0 + 1) % len(pts)
        t = idx - np.floor(idx)
        out = pts[i0] + (pts[i1] - pts[i0]) * t[:, None]
        return [tuple(p) for p in out]

    def area(self):
        pts = np.asarray(self.points)
        if pts.shape[0] < 3:
            return 0.0
        nxt = np.roll(pts, -1, axis=0)
        return 0.5 * abs(np.sum(pts[:, 0] * nxt[:, 1] - pts[:, 1] * nxt[:, 0]))


class MyCircle(AbstractShape):
    def __init__(self, cx, cy, r, area_val=None):
        self.cx = cx
        self.cy = cy
        self.r = r
        self._area_val = area_val if area_val is not None else math.pi * r * r

    def contour(self, n: int):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return [(self.cx + self.r * math.cos(a),
                 self.cy + self.r * math.sin(a)) for a in t]

    def area(self):
        return self._area_val


class MyShapeWithArea(AbstractShape):
    """Generic shape with precomputed area."""

    def __init__(self, area_val, boundary_points):
        self._area_val = area_val
        self.points = boundary_points

    def contour(self, n: int):
        if len(self.points) < 3:
            return self.points
        pts = np.asarray(self.points)
        idx = np.linspace(0, len(pts), n, endpoint=False)
        i0 = np.floor(idx).astype(int) % len(pts)
        i1 = (i0 + 1) % len(pts)
        t = idx - np.floor(idx)
        out = pts[i0] + (pts[i1] - pts[i0]) * t[:, None]
        return [tuple(p) for p in out]

    def area(self):
        return np.float32(self._area_val)
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
