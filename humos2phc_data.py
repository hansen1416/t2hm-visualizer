import os
import math
import random
import re
import unicodedata
from glob import glob
from typing import List, Dict, Any, Iterable, Optional, Union, Tuple

import joblib
import numpy as np
import torch


def safe_prefix_filename(text: str, n: int = 16) -> str:
    """
    Take the first n characters of `text`, replace spaces with underscores,
    and sanitize to a filesystem-safe ASCII-ish token.
    """
    if not isinstance(text, str):
        text = str(text)

    # take first n chars, replace whitespace runs with single underscore
    s = text[:n]
    s = re.sub(r"\s+", "_", s.strip())

    # normalize to ASCII (drop accents), then keep only safe chars
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)  # replace unsafe chars with _
    s = re.sub(r"_+", "_", s).strip("._-")  # collapse underscores, trim edges

    return s or "untitled"


def data_format_humos2phc(humos_path):

    phc_data_folder = os.path.join(".", "phc_test")

    os.makedirs(phc_data_folder, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    humos_result = torch.load(humos_path, map_location=device)

    text = humos_result["text"][0]
    motion_name = safe_prefix_filename(text)

    for gender in ["male", "female", "neutral"]:

        for beta_key, humos_motion_data in humos_result[gender].items():

            n_frame = humos_motion_data["trans"].shape[0]

            phc_motion = {}

            phc_motion["gender"] = gender
            phc_motion["beta"] = humos_motion_data["betas"][0]

            # [n, 3]
            phc_motion["trans_orig"] = humos_motion_data["trans"]
            # [n, 24, 3]
            phc_motion["pose_aa"] = torch.zeros(n_frame, 24, 3)

            phc_motion["pose_aa"][:, 0, :] = humos_motion_data["root_orient"]
            phc_motion["pose_aa"][:, 1:, :] = humos_motion_data["pose_body"]

            phc_motion["pose_aa"] = phc_motion["pose_aa"].reshape(n_frame, -1)

            file_name = f"{motion_name}_{gender}.pkl"

            joblib.dump(
                {motion_name: phc_motion},
                os.path.join(phc_data_folder, file_name),
            )


if __name__ == "__main__":

    folder = os.path.join(
        os.path.expanduser("~"),
        "repos",
        "humos",
        "output",
    )

    pattern = os.path.join(folder, "**", f"*.pt")
    files = glob(pattern, recursive=True)

    i = 0

    for file in files:

        data_format_humos2phc(file)

        i += 1

        if i > 9:
            break
