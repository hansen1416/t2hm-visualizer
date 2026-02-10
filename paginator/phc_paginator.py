import os
import math
import random
from glob import glob
import joblib
from typing import List, Dict, Any, Iterable, Optional, Union

import numpy as np
import torch


def _to_tensor(v: Any, device) -> torch.Tensor:
    if torch.is_tensor(v):
        return v.to(device, dtype=torch.float32)
    if isinstance(v, np.ndarray):
        t = torch.from_numpy(v)
        return t.to(device, dtype=torch.float32)
    t = torch.tensor(v)
    return t.to(device, dtype=torch.float32)


class PHCPager:
    """
    Paginate PHC .pkl files with optional (lazy/eager) loading.

    Parameters
    ----------
    root : str
        Root directory that contains AMASS .pkl files (searched recursively).
    page_size : int
        Number of items per page.
    shuffle : bool
        If True, shuffle file order deterministically using `seed`.
    seed : int
        RNG seed used when `shuffle=True`.
    device : torch.device or str
        Target device when converting arrays to torch tensors (only if to_torch=True).
    """

    def __init__(
        self,
        dataset_root: str,
        page_size: int = 20,
        shuffle: bool = False,
        seed: int = 46,
    ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.root = os.path.abspath(os.path.expanduser(dataset_root))
        self.page_size = int(page_size)

        pattern = os.path.join(self.root, "**", "*.pkl")
        files = glob(pattern, recursive=True)

        files.sort()  # deterministic order
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(files)
        self.files: List[str] = files

    # --------- basic pagination API ---------
    @property
    def num_items(self) -> int:
        return len(self.files)

    @property
    def total_pages(self) -> int:
        if self.page_size <= 0:
            return 0
        return math.ceil(self.num_items / self.page_size)

    def _slice_indices(self, page_index: int) -> slice:
        if page_index < 0 or page_index >= self.total_pages:
            raise IndexError(
                f"page_index {page_index} out of range [0, {self.total_pages-1}]"
            )
        start = page_index * self.page_size
        end = min(start + self.page_size, self.num_items)
        return slice(start, end)

    def get_paths_by_page(self, page_index: int) -> List[str]:
        """Return only file paths for the page (no I/O)."""
        s = self._slice_indices(page_index)
        return self.files[s]

    def get_names_by_page(self, page_index: int) -> List[str]:
        paths = self.get_paths_by_page(page_index)
        # exclude the dataset root and the .pkl extension
        return [os.path.splitext(os.path.relpath(p, self.root))[0] for p in paths]

    def get_by_page(
        self,
        page_index: int,
        *,
        keys: Optional[List[str]] = None,
        to_torch: bool = True,
        mmap: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load and return a list of dicts (one per file on the page).

        Parameters
        ----------
        keys : list[str] or None
            If provided, only these keys are returned per file (missing keys ->
            omitted unless `strict_keys=True`).
        to_torch : bool
            Convert numpy arrays to torch tensors on `self.device`.
        mmap : bool
            Use memory mapping (np.load(..., mmap_mode='r')) to reduce peak memory.
        strict_keys : bool
            If True and any requested key is absent, raise KeyError.
        skip_errors : bool
            If True, skip files that fail to load; otherwise raise.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict includes at least {"__path__": str}. Other fields are arrays.
        """
        s = self._slice_indices(page_index)
        batch_paths = self.files[s]
        out: List[Dict[str, Any]] = []

        for fp in batch_paths:
            try:
                item = self.load_single(
                    fp,
                    keys=keys,
                    to_torch=to_torch,
                    mmap=mmap,
                )
                out.append(item)
            except Exception as e:
                raise e
        return out

    def iter_pages(
        self,
        *,
        keys: Optional[List[str]] = None,
        to_torch: bool = False,
        mmap: bool = True,
    ) -> Iterable[List[Dict[str, Any]]]:
        """Iterator over all pages (loaded)."""
        for p in range(self.total_pages):
            yield self.get_by_page(
                p,
                keys=keys,
                to_torch=to_torch,
                mmap=mmap,
            )

    # --------- helpers ---------
    def load_single(
        self,
        path: str,
    ) -> Dict[str, Any]:
        load_kwargs = {"allow_pickle": False}

        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        raw_motion_data = joblib.load(path)

        for motion_name, motion_data in raw_motion_data.items():
            # dict_keys(['pose_quat_global', 'pose_quat', 'trans_orig', 'root_trans_offset', 'beta', 'gender', 'pose_aa', 'fps'])
            # pose_quat_global: (106, 24, 4)
            # pose_quat: (106, 24, 4)
            # trans_orig: (106, 3)
            # root_trans_offset: torch.Size([106, 3])
            # beta: (16,)
            # gender: neutral
            # pose_aa: (106, 72)
            # fps: 30

            root_trans = _to_tensor(motion_data["trans_orig"], self.device)

            pose_aa = motion_data["pose_aa"].reshape(-1, 24, 3)

            root_aa = _to_tensor(pose_aa[:, 0, :], self.device)
            body_aa = _to_tensor(pose_aa[:, 1:, :], self.device)

            # most of the time, betas are zeros anyway
            betas = _to_tensor(motion_data["beta"][:10], self.device)

            betas = betas.unsqueeze(0).expand(root_trans.shape[0], 10)

            # print(type(motion_name))
            # print(root_trans.shape)
            # print(root_aa.shape)
            # print(body_aa.shape)

            # there is juts one motion, so we return in the loop
            return {
                "motion_name": motion_name,
                "gender": motion_data["gender"],
                "root_trans": root_trans,
                "root_aa": root_aa,
                "body_aa": body_aa,
                "betas": betas,
            }

        # print(raw_motion_data.keys())


# ---------------- example usage ----------------
if __name__ == "__main__":
    # Your existing root
    folder = os.path.join("/home/hlz/repos/ASE/ase/data/motions")

    pager = PHCPager(
        dataset_root=folder,
    )

    print(f"Found {pager.num_items} .pkl files across {pager.total_pages} pages.")

    # Get just paths for page 0
    page0_paths = pager.get_names_by_page(0)
    print(f"First page has {len(page0_paths)} files. Example: {page0_paths[0]}")

    # Load a page (lazy, memory-mapped), typical AMASS keys include:
    # 'poses', 'betas', 'trans', 'gender', 'mocap_framerate', etc.
    motion_data = pager.load_single(page0_paths[0])

    for k, v in motion_data.items():
        print(f"{k}:")
        if hasattr(v, "shape"):
            print(v.shape)
        else:
            print(v)
