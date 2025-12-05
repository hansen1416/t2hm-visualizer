import numpy as np


def sample_smpl_betas(
    batch_size: int,
    num_betas: int = 16,
    per_dim_clip: float = 2.0,
    max_norm: float = 5.0,
    diversity_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample SMPL betas ~ N(0, I), truncated to
        |beta_i| <= per_dim_clip  for all i
        ||beta||_2 <= max_norm
    using rejection sampling.

    Args:
        batch_size: number of beta vectors.
        num_betas: dimensionality of shape space (e.g. 10 or 16).
        per_dim_clip: per-coordinate bound; use None to disable.
        max_norm: L2 norm bound; use None to disable.
        diversity_scale: scales the base Gaussian (1.0 = N(0, I)).
        rng: optional np.random.Generator for reproducibility.

    Returns:
        betas: (batch_size, num_betas) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    collected = []

    while sum(b.shape[0] for b in collected) < batch_size:
        # oversample to reduce loop iterations
        n_try = max(batch_size - sum(b.shape[0] for b in collected), 1) * 4

        cand = rng.standard_normal(size=(n_try, num_betas)) * diversity_scale

        # per-dimension clip / rejection
        if per_dim_clip is not None:
            mask = np.all(np.abs(cand) <= per_dim_clip, axis=1)
            cand = cand[mask]

        # global norm constraint
        if max_norm is not None and cand.size > 0:
            norms = np.linalg.norm(cand, axis=1)
            mask = norms <= max_norm
            cand = cand[mask]

        if cand.size > 0:
            collected.append(cand)

    betas = np.concatenate(collected, axis=0)[:batch_size]
    return betas


# example:
if __name__ == "__main__":
    betas_10 = sample_smpl_betas(batch_size=1024, num_betas=10)
    betas_16 = sample_smpl_betas(batch_size=1024, num_betas=16)
    print(betas_10.shape, betas_16.shape)
