"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import math
from functools import partial
import numpy as np
from scipy import interpolate
from typing import Optional
from evobankmpc.utils.utils import nearest_point_on_trajectory, nearest_point_on_trajectory_jax
from numba import njit
import jax.numpy as jnp
import jax


class CubicSpline2D:
    """
    Cubic Spline2D class for trajectory representation.
    """

    def __init__(self, x, y,
                 psis: Optional[np.ndarray] = None,
                 ks: Optional[np.ndarray] = None,
                 vxs: Optional[np.ndarray] = None,
                 axs: Optional[np.ndarray] = None,
                 ss: Optional[np.ndarray] = None):
        psis = psis if psis is not None else self._calc_yaw_from_xy(x, y)
        ks = ks if ks is not None else self._calc_kappa_from_xy(x, y)
        vxs = vxs if vxs is not None else np.ones_like(x)
        axs = axs if axs is not None else np.zeros_like(x)

        self.points = np.c_[x, y,
                            np.cos(psis), np.sin(psis),
                            ks, vxs, axs]

        # Ensure periodic closure
        if np.any(self.points[-1, :2] != self.points[0, :2]):
            self.points = np.vstack((self.points, self.points[0]))
        else:
            self.points[-1] = self.points[0]

        self.points_jax = jnp.array(self.points)
        self.s = ss if ss is not None else self.__calc_s(self.points[:, 0], self.points[:, 1])
        self.ss, self.psis, self.ks = self.s, psis, ks
        self.s_interval = (self.s[-1] - self.s[0]) / len(self.s)
        self.s_frame_max = self.s[-1]

        # Use scipy CubicSpline with periodic boundary conditions
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")
        self.spline_x = np.array(self.spline.x)
        self.spline_c = np.array(self.spline.c)
        self.s_jax = jnp.array(self.s)
        self.spline_x_jax = jnp.array(self.spline.x)
        self.spline_c_jax = jnp.array(self.spline.c)
        self.num_segments = len(self.spline_x)

    def find_segment_for_s(self, x):
        return (x / (self.spline.x[-1]) * (len(self.spline_x) - 2)).astype(int)

    @partial(jax.jit, static_argnums=(0))
    def find_segment_for_s_jax(self, x):
        return (x / self.spline_x_jax[-1] * (len(self.spline_x_jax) - 2)).astype(int)

    def predict_with_spline(self, point, segment, state_index=0):
        exp_x = ((point - self.spline.x[segment]) ** np.arange(4)[::-1])[:, None]
        vec = self.spline.c[:, segment, state_index]
        point = vec.dot(exp_x)
        return point

    @partial(jax.jit, static_argnums=(0))
    def predict_with_spline_jax(self, point, segment, state_index=0):
        exp_x = ((point - self.spline_x_jax[segment]) ** jnp.arange(4)[::-1])[:, None]
        vec = self.spline_c_jax[:, segment, state_index]
        point = vec.dot(exp_x)
        return point

    def __calc_s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        return np.concatenate([np.array([0]), np.cumsum(self.ds)])

    def _calc_yaw_from_xy(self, x, y):
        dx_dt = np.gradient(x, edge_order=2)
        dy_dt = np.gradient(y, edge_order=2)
        heading = np.arctan2(dy_dt, dx_dt)
        return heading

    def _calc_kappa_from_xy(self, x, y):
        # Extend for stable gradients
        x_extended = np.concatenate((x[-2:], x, x[:2]))
        y_extended = np.concatenate((y[-2:], y, y[:2]))
        dx_dt = np.gradient(x_extended, edge_order=2)
        dy_dt = np.gradient(y_extended, edge_order=2)
        d2x_dt2 = np.gradient(dx_dt, edge_order=2)
        d2y_dt2 = np.gradient(dy_dt, edge_order=2)
        curvature = (dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        return curvature[2:-2]

    def calc_position(self, s: float, segment=None) -> np.ndarray:
        segment = segment or self.find_segment_for_s(s)
        x = self.predict_with_spline(s, segment, 0)[0]
        y = self.predict_with_spline(s, segment, 1)[0]
        return x, y

    @partial(jax.jit, static_argnums=(0))
    def calc_position_jax(self, s: float) -> np.ndarray:
        segment = self.find_segment_for_s_jax(s)
        x = self.predict_with_spline_jax(s, segment, 0)[0]
        y = self.predict_with_spline_jax(s, segment, 1)[0]
        return x, y

    def calc_curvature(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_s(s)
        k = self.predict_with_spline(s, segment, 4)[0]
        return k

    @partial(jax.jit, static_argnums=(0))
    def calc_curvature_jax(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_s_jax(s)
        k = self.predict_with_spline_jax(s, segment, 4)[0]
        return k

    def find_curvature(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_s(s)
        k = self.points[segment, 4]
        return k

    @partial(jax.jit, static_argnums=(0))
    def find_curvature_jax(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_s_jax(s)
        k = self.points_jax[segment, 4]
        return k

    def calc_yaw(self, s: float, segment=None) -> Optional[float]:
        segment = segment or self.find_segment_for_s(s)
        cos = self.predict_with_spline(s, segment, 2)[0]
        sin = self.predict_with_spline(s, segment, 3)[0]
        yaw = np.arctan2(sin, cos)
        return yaw

    def calc_yaw_jax(self, s: float) -> Optional[float]:
        segment = self.find_segment_for_s_jax(s)
        cos = self.predict_with_spline_jax(s, segment, 2)[0]
        sin = self.predict_with_spline_jax(s, segment, 3)[0]
        yaw = jnp.arctan2(sin, cos)
        return yaw

    def calc_arclength_inaccurate(self, x: float, y: float, s_inds=None) -> tuple[float, float]:
        """
        Fast calculation of arclength for a given point (x, y) on the trajectory.
        """
        if s_inds is None:
            ey, t, min_dist_segment = nearest_point_on_trajectory(
                np.asarray([x, y]).astype(np.float32), self.points[:, :2]
            )
        else:
            ey, t, min_dist_segment = nearest_point_on_trajectory(
                np.asarray([x, y]).astype(np.float32), self.points[s_inds, :2]
            )
            min_dist_segment = s_inds[min_dist_segment]
        s = float(
            self.s[min_dist_segment]
            + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])
        )
        return s, ey

    @partial(jax.jit, static_argnums=(0))
    def calc_arclength_jax(self, x, y, s_inds):
        ey, t, min_dist_segment = nearest_point_on_trajectory_jax(
            jnp.array([x, y]), self.points_jax[s_inds, :2]
        )
        min_dist_segment_s_ind = s_inds[min_dist_segment]
        s = self.s_jax[min_dist_segment_s_ind] + \
            t * (self.s_jax[min_dist_segment_s_ind + 1] - self.s_jax[min_dist_segment_s_ind]).astype(jnp.float32)
        return s, ey

    def _calc_tangent(self, s: float) -> np.ndarray:
        dx, dy = self.spline(s, 1)[:2]
        tangent = np.array([dx, dy])
        return tangent

    def _calc_normal(self, s: float) -> np.ndarray:
        dx, dy = self.spline(s, 1)[:2]
        normal = np.array([-dy, dx])
        return normal