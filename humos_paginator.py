import os
import math
import random
from glob import glob
from typing import List, Dict, Any, Iterable, Optional, Union

import numpy as np
import torch


class HumosPager:
    """
    Paginate Humos results .pt files with optional (lazy/eager) loading.

    Parameters
    ----------
    root : str
        Root directory that contains AMASS .npz files (searched recursively).
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
        device: Union[str, torch.device] = "cpu",
    ):
        self.root = os.path.abspath(os.path.expanduser(dataset_root))
        self.page_size = int(page_size)
        self.device = torch.device(device)

        self.ext = "pt"

        pattern = os.path.join(self.root, "**", f"*.{self.ext}")
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
        # exclude the dataset root and the .npz extension
        return [os.path.splitext(os.path.relpath(p, self.root))[0] for p in paths]

    def get_by_page(
        self,
        page_index: int,
        *,
        keys: Optional[List[str]] = None,
        to_torch: bool = True,
        mmap: bool = True,
        strict_keys: bool = False,
        skip_errors: bool = True,
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
                    strict_keys=strict_keys,
                )
                out.append(item)
            except Exception as e:
                if skip_errors:
                    # You may log this if desired
                    continue
                raise e
        return out

    def iter_pages(
        self,
        *,
        keys: Optional[List[str]] = None,
        to_torch: bool = False,
        mmap: bool = True,
        strict_keys: bool = False,
        skip_errors: bool = True,
    ) -> Iterable[List[Dict[str, Any]]]:
        """Iterator over all pages (loaded)."""
        for p in range(self.total_pages):
            yield self.get_by_page(
                p,
                keys=keys,
                to_torch=to_torch,
                mmap=mmap,
                strict_keys=strict_keys,
                skip_errors=skip_errors,
            )

    # --------- helpers ---------
    def load_single(
        self,
        path: str,
        *,
        keys: Optional[List[str]] = [
            "betas",
            "gender",
            "root_orient",
            "pose_body",
            "trans",
        ],
    ) -> Dict[str, Any]:

        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        if not path.endswith(f".{self.ext}"):
            path = f"{path}.{self.ext}"

        result = torch.load(path, map_location=self.device)

        return result


# ---------------- example usage ----------------
if __name__ == "__main__":
    # Your existing root
    folder = os.path.join(
        os.path.expanduser("~"),
        "repos",
        "humos",
        "debug_pred",
    )

    pager = HumosPager(
        dataset_root=folder,
    )

    print(f"Found {pager.num_items} .npz files across {pager.total_pages} pages.")

    # Get just paths for page 0
    page0_paths = pager.get_names_by_page(0)
    print(f"First page has {len(page0_paths)} files. Example: {page0_paths[0]}")

    # Load a page (lazy, memory-mapped), typical AMASS keys include:
    # 'poses', 'betas', 'trans', 'gender', 'mocap_framerate', etc.
    motion_data = pager.load_single(page0_paths[0])

    print(motion_data)
