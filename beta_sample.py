import numpy as np


def sample_betas_energy_uniform(
    batch_size: int,
    num_betas: int = 16,
    energy_max: float = 20.25,  # E_max = 4.5^2
    energy_min: float = 0.0,  # set = energy_max for fixed energy
    per_dim_clip: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample SMPL betas with:
        - beta in R^{num_betas}
        - each component in [-per_dim_clip, per_dim_clip]
        - energy E = ||beta||^2 in [energy_min, energy_max]
        - directions uniform on the sphere
        - energy approximately uniform in [energy_min, energy_max]

    Args:
        batch_size: number of vectors to sample.
        num_betas: dimensionality (e.g. 10 or 16).
        energy_max: maximum energy (e.g. 20.25).
        energy_min: minimum energy (0 for full range; set equal to energy_max
                    if you want fixed energy).
        per_dim_clip: per-component bound (e.g. 3.0).
        rng: optional np.random.Generator.

    Returns:
        betas: (batch_size, num_betas) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    collected = []

    def enough():
        return sum(x.shape[0] for x in collected) >= batch_size

    while not enough():
        # oversample to reduce the number of while-loop iterations
        n_try = max(batch_size - sum(x.shape[0] for x in collected), 1) * 8

        # 1) directions ~ uniform on sphere
        z = rng.standard_normal(size=(n_try, num_betas))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0.0] = 1.0
        dirs = z / norms

        # 2) energies: uniform in [energy_min, energy_max]
        E = rng.uniform(energy_min, energy_max, size=(n_try, 1))
        r = np.sqrt(E)

        # 3) construct candidates
        cand = dirs * r  # shape (n_try, num_betas)

        # 4) enforce per-dim constraint via rejection
        mask = np.all(np.abs(cand) <= per_dim_clip, axis=1)
        cand = cand[mask]

        if cand.size > 0:
            collected.append(cand)

    betas = np.concatenate(collected, axis=0)[:batch_size]
    return betas


# examples
if __name__ == "__main__":

    num_betas = 10  # SMPL standard shape code used in your thesis
    per_dim_clip = 3.0  # matches β ∈ [-3,3] in the project
    energy_max = 20.25  # = 4.5^2, i.e. ‖β‖ ≤ 4.5 (your chosen max_norm)
    energy_min = 0.0  # uniform energy from 0 up to the boundary

    betas_10 = sample_betas_energy_uniform(
        batch_size=1024,
        num_betas=num_betas,
        per_dim_clip=per_dim_clip,
        energy_max=energy_max,
        energy_min=energy_min,
    )

    print(betas_10.shape)
