"""Piecewise angle definition for CrackDet (Sec. 3.1, Fig. 2 of the paper).

The standard 5-parameter oriented-box definition (x, y, h, w, theta) with
theta in [0, 180) suffers from the "boundary discontinuity" problem: a box
at theta=179 deg and one at theta=1 deg are visually near-identical but
178 degrees apart numerically, so plain angle regression has a loss spike
at the wrap-around boundary.

CrackDet's fix (Fig. 2b): split [0, 180) into 4 branches of 45 degrees
each. Every branch redefines (theta, h, w) into a *local* (theta_i, h_i, w_i)
using a branch-specific trig reprojection, so no branch ever straddles a
wrap-around point. At inference the branch-local prediction is mapped back
to the global frame with the inverse maps Gamma (angle) and Delta (size
factor) from Eq. 6-7.

Branch assignment (half-open, degrees), matching Eq. 1 / Fig. 3b:
    branch 0: theta in [  0,  45)
    branch 1: theta in [ 45,  90)
    branch 2: theta in [ 90, 135)
    branch 3: theta in [135, 180)

Forward reprojection (Fig. 2b, using 0-indexed branches for branch i+1 in
the paper's 1-indexed notation):
    branch 0: theta_0 = theta            h_0 =  h * cos(theta)   w_0 =  w * cos(theta)
    branch 1: theta_1 = 90 - theta       h_1 =  h * sin(theta)   w_1 =  w * sin(theta)
    branch 2: theta_2 = theta - 90       h_2 =  h * sin(theta)   w_2 =  w * sin(theta)
    branch 3: theta_3 = 180 - theta      h_3 = -h * cos(theta)   w_3 = -w * cos(theta)

Inverse maps used at inference (Eq. 6-7), applied to the winning branch i*
with estimated local angle theta_i_hat:
    Gamma(theta_i_hat): branch 0 -> theta_i_hat
                         branch 1 -> 90 - theta_i_hat
                         branch 2 -> 90 + theta_i_hat
                         branch 3 -> 180 - theta_i_hat
    Delta(theta_i_hat): branch 0 -> cos(theta_i_hat)
                         branch 1 -> sin(90 - theta_i_hat)
                         branch 2 -> sin(90 + theta_i_hat)
                         branch 3 -> -cos(180 - theta_i_hat)
    ho = h_i_hat / Delta(theta_i_hat),  wo = w_i_hat / Delta(theta_i_hat)

Round-trip check (done by hand before writing this code, not assumed): for
each branch, substituting the branch's forward theta_i back into its own
Gamma recovers the original theta exactly, and substituting the branch's
h_i/w_i factor back into Delta(theta_i) recovers the same trig factor used
in the forward pass (so h_i / Delta(theta_i) == h exactly, algebraically,
for all four branches). Delta's magnitude stays in the range [cos(45deg),
1] = [~0.7071, 1] on every branch's valid local-angle domain, so the
division is never near-singular for angles that are actually inside their
branch's domain.
"""

from __future__ import annotations

import torch

NUM_BRANCHES = 4
BRANCH_WIDTH_DEG = 45.0
# Half-open [lo, hi) boundaries per branch, in degrees.
BRANCH_BOUNDS_DEG = [(0.0, 45.0), (45.0, 90.0), (90.0, 135.0), (135.0, 180.0)]

_DEG2RAD = torch.pi / 180.0
_EPS = 1e-3


def _deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * _DEG2RAD


def _rad2deg(x: torch.Tensor) -> torch.Tensor:
    return x / _DEG2RAD


def get_branch_index(theta_deg: torch.Tensor) -> torch.Tensor:
    """Map angle(s) in [0, 180) degrees to a branch index in {0,1,2,3}.

    Values are clamped into [0, 180) first so that boundary/floating-point
    edge cases (e.g. exactly 180.0 from an annotation tool) land in branch 3
    rather than overflowing to a nonexistent branch 4.
    """
    theta = theta_deg.clamp(min=0.0, max=180.0 - 1e-4)
    branch = torch.div(theta, BRANCH_WIDTH_DEG, rounding_mode="floor").long()
    return branch.clamp(min=0, max=NUM_BRANCHES - 1)


def forward_transform(theta_deg: torch.Tensor, h: torch.Tensor, w: torch.Tensor):
    """Redefine global (theta, h, w) into branch-local (theta_i, h_i, w_i).

    All inputs are elementwise-aligned tensors of the same shape. theta_deg
    must already lie in [0, 180). Returns (theta_i, h_i, w_i, branch_idx),
    same shape as the inputs plus branch_idx (long tensor).
    """
    theta_deg = theta_deg.clamp(min=0.0, max=180.0 - 1e-4)
    branch_idx = get_branch_index(theta_deg)
    theta_rad = _deg2rad(theta_deg)
    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

    theta_i = torch.where(
        branch_idx == 0,
        theta_deg,
        torch.where(
            branch_idx == 1,
            90.0 - theta_deg,
            torch.where(branch_idx == 2, theta_deg - 90.0, 180.0 - theta_deg),
        ),
    )
    # Trig factor is identical for h and w, and identical for branches 1&2
    # (sin) vs branches 0&3 (cos, with a sign flip on branch 3).
    factor = torch.where(
        branch_idx == 0,
        cos_t,
        torch.where(branch_idx == 1, sin_t, torch.where(branch_idx == 2, sin_t, -cos_t)),
    )
    h_i = h * factor
    w_i = w * factor
    return theta_i, h_i, w_i, branch_idx


def gamma_inverse(theta_i_hat: torch.Tensor, branch_idx: torch.Tensor) -> torch.Tensor:
    """Eq. 6: map a branch-local predicted angle back to the global frame."""
    return torch.where(
        branch_idx == 0,
        theta_i_hat,
        torch.where(
            branch_idx == 1,
            90.0 - theta_i_hat,
            torch.where(branch_idx == 2, 90.0 + theta_i_hat, 180.0 - theta_i_hat),
        ),
    )


def delta_inverse(theta_i_hat: torch.Tensor, branch_idx: torch.Tensor) -> torch.Tensor:
    """Eq. 7: the size-reprojection factor used to undo the forward h_i/w_i scaling."""
    t_rad = _deg2rad(theta_i_hat)
    return torch.where(
        branch_idx == 0,
        torch.cos(t_rad),
        torch.where(
            branch_idx == 1,
            torch.sin(_deg2rad(90.0 - theta_i_hat)),
            torch.where(
                branch_idx == 2,
                torch.sin(_deg2rad(90.0 + theta_i_hat)),
                -torch.cos(_deg2rad(180.0 - theta_i_hat)),
            ),
        ),
    )


def decode_box(h_i_hat: torch.Tensor, w_i_hat: torch.Tensor, theta_i_hat: torch.Tensor,
                branch_idx: torch.Tensor):
    """Invert the piecewise definition: branch-local (h_i, w_i, theta_i) -> global (h, w, theta).

    Matches the inference step in Sec. 3.4. `delta` is clamped away from 0
    purely as a defensive guard against a not-yet-converged network
    predicting a local angle outside its branch's [0, 45]-ish domain; on
    the true domain delta never drops below ~0.707 (see module docstring).
    """
    delta = delta_inverse(theta_i_hat, branch_idx)
    delta_safe = torch.where(delta.abs() < _EPS, torch.full_like(delta, _EPS), delta)
    h = h_i_hat / delta_safe
    w = w_i_hat / delta_safe
    theta = gamma_inverse(theta_i_hat, branch_idx)
    return h, w, theta
