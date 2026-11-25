"""Audio schedule derivative for the pinned ComfyUI MiniMax-H3 shift.

ComfyUI 2fcb6cef exports ``time_shift_sigma`` and not ``time_shift_slope``.
The forward scales the audio velocity by d(sigma_audio)/d(sigma_video).
"""

from __future__ import annotations

import torch


def time_shift_slope(sigma, from_shift, to_shift):
    """Return d(time_shift_sigma(sigma, from_shift, to_shift))/d(sigma).

    ``time_shift_sigma`` inverts ``sigma = s*b/(1+(s-1)*b)`` back to the base
    grid and applies the other shift. Differentiating that closed form gives
    ``to * from / ((1+(to-1)*base)^2 * (from+(1-from)*sigma)^2)``.
    Equal shifts therefore have slope 1.
    """

    sigma_t = torch.as_tensor(sigma)
    if not sigma_t.is_floating_point():
        sigma_t = sigma_t.float()
    source = torch.as_tensor(from_shift, dtype=sigma_t.dtype, device=sigma_t.device)
    target = torch.as_tensor(to_shift, dtype=sigma_t.dtype, device=sigma_t.device)
    base = sigma_t / (source + sigma_t * (1.0 - source))
    denom_sigma = source + sigma_t * (1.0 - source)
    denom_shift = 1.0 + (target - 1.0) * base
    return target * source / (denom_shift.square() * denom_sigma.square())
