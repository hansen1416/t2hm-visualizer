import os.path
from glob import glob
from itertools import islice


def get_nth_npy_file(folder, n):
    with os.scandir(folder) as entries:
        npy_files = (entry.path for entry in entries if entry.name.endswith(".npy"))
        return next(islice(npy_files, n, None), None)


class MotionDataLoader:

    def __init__(self):
        self.root_path = os.path.join(os.path.expanduser("~"), "Downloads", "motion-x")

        self.categories = [
            "animation",
            # "fitness",
            "haa500",
            "humman",
            "idea400",
            "kungfu",
            "music",
            "perform",
        ]

    def get(self, idx=0, category="animation"):

        motion_folder_path = os.path.join(
            self.root_path, "motion", "motion_generation", "smplx322", category
        )

        # motion_files = glob(os.path.join(motion_folder_path, "*.npy"))
        # motion_file = motion_files[idx]
        motion_file = get_nth_npy_file(motion_folder_path, idx)

        video_name = os.path.splitext(os.path.basename(motion_file))[0]

        # get the base filename

        video_file = os.path.join(
            self.root_path, "video", category, f"{video_name}.mp4"
        )
        # check if video file exists
        assert os.path.exists(video_file), f"Video file {video_file} does not exist."

        return motion_file, video_file
