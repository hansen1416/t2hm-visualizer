import os
import time
import threading
import sys
from pathlib import Path

module_dir = Path("/home/hlz/repos/humos/aitviewer_humos").as_posix()
sys.path.insert(0, module_dir)

import torch
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from smplx import SMPLX

from utils.utils import get_checkerboard_plane
from humos_paginator import HumosPager
from third_party.aitviewer_humos.aitviewer.models.smpl import SMPLLayer

from typing import Iterable, Optional

# SMPL/SMPL-H foot landmarks (0-indexed) used in the HUMOS grounding note
DEFAULT_FOOT_VIDS = (3387, 6787, 3216, 6617, 3226, 6624)


def apply_static_ground_offset(
    verts: torch.Tensor,
    up_axis: int = 2,  # 2 for Z-up, 1 for Y-up
    safety_margin: float = 0.0,  # e.g., 0.002 for 2mm
) -> torch.Tensor:
    """
    Static grounding: compute offset from first frame only and shift verts for all frames.

    Args:
        verts: (T, V, 3) vertices in world coordinates (i.e., after applying translation).
        up_axis: which axis is "height" (Z-up=2, Y-up=1).
        safety_margin: small positive margin to avoid initial penetration.

    Returns:
        offset: scalar tensor (same dtype/device as verts), >= 0.
    """
    # Ensure torch tensor (torch.as_tensor shares memory with numpy on CPU)
    if not torch.is_tensor(verts):
        verts = torch.as_tensor(verts)

    if up_axis == "y":
        up_axis_idx = 1
    elif up_axis == "z":
        up_axis_idx = 2
    elif up_axis == "x":
        up_axis_idx = 0

    assert (
        verts.ndim == 3 and verts.shape[-1] == 3
    ), f"Expected (T,V,3), got {verts.shape}"

    h0_min = verts[0, :, up_axis_idx].amin()  # tensor scalar
    margin = verts.new_tensor(safety_margin)  # tensor scalar, same dtype/device
    offset = (-h0_min + margin).clamp_min(0.0)  # tensor scalar >= 0

    # adjust the humanoid height
    verts[..., up_axis_idx].add_(offset)
    return offset


def gender_from_motion_name(motion_name: str) -> str:
    """
    motion_name examples:
      "000002_-1" -> female
      "000002_0"  -> neutral
      "000002_1"  -> male
    Also accepts suffixes like "_male", "_female", "_neutral".
    """
    base = os.path.basename(motion_name)
    base = os.path.splitext(base)[0]
    tag = base.rsplit("_", 1)[-1] if "_" in base else ""

    m = {
        "-1": "female",
        "0": "neutral",
        "1": "male",
        "f": "female",
        "female": "female",
        "n": "neutral",
        "neutral": "neutral",
        "m": "male",
        "male": "male",
    }
    return m.get(tag.lower(), "male")  # safe default


class AnimPlayer:

    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # We need to initalize the application, which finds the necessary shaders
        # for rendering and prepares the cross-platform window abstraction.
        gui.Application.instance.initialize()

        width, height = 1920, 1080

        self.window = gui.Application.instance.create_window("Open3D", width, height)

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._scene)

        self.window.set_on_layout(self._on_layout)
        # for AMASS use z-up, for motion-x use y-up
        self.up_axis = "z"

        # load the motion data before add ui and after init smpl
        dataset_folder = os.path.join(
            os.path.expanduser("~"),
            "repos",
            "humos",
            "output",
        )
        # load all the motion data paths
        self.pager = HumosPager(
            dataset_root=dataset_folder,
            device=self.device,
        )

        self.batch_size = 64
        self.verts_glob = [None] * self.batch_size
        self.offsets = [None] * self.batch_size

        cols = 8
        rows = 8
        spacing = 2.5
        x_offset = (cols - 1) * spacing / 2
        z_offset = (rows - 1) * spacing / 2

        for mesh_idx in range(self.batch_size):

            row, col = divmod(mesh_idx, cols)
            offset = np.array(
                [
                    col * spacing - x_offset,
                    row * spacing - z_offset,
                    0,
                ]
            )

            self.offsets[mesh_idx] = offset

        self._setup_lighting()
        self._add_ground()
        self._set_camera()
        self._add_ui()

        # load smpl model
        self._init_smpl()

        # load first page
        fisrt_path = self._load_batch(0)

        # # load first motion
        self._load_data(fisrt_path)

        # # thread animation testing
        threading.Thread(target=self.animate_mesh, daemon=True).start()

    @property
    def play_animation(self):
        return self._play_animation

    @play_animation.setter
    def play_animation(self, value):
        self._play_animation = value
        self.play_button.text = "Pause" if value else "Play"

    def _setup_lighting(self):

        # self._scene.scene.set_background([0.96, 0.94, 0.91, 1])
        self._scene.scene.set_background([0.46, 0.44, 0.41, 1])
        # self._scene.scene.show_skybox(True)
        self._scene.scene.show_axes(True)

        # # Add indirect light from an environment map  # Open3D includes a default environment
        # self._scene.scene.scene.enable_indirect_light(True)
        # self._scene.scene.scene.set_indirect_light_intensity(10000)

        self._scene.scene.scene.set_sun_light(
            [-0.577, -0.577, -0.577],  # direction
            [1.0, 1.0, 1.0],  # color
            50000,  # intensity
        )
        self._scene.scene.scene.enable_sun_light(True)

    def _add_ground(self):
        gp = get_checkerboard_plane(
            plane_width=20, num_boxes=20, ground_level=0.0, up_axis=self.up_axis
        )

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()
            self._scene.scene.add_geometry(
                f"__ground_{idx:04d}__", g, rendering.MaterialRecord()
            )

    def _set_camera(self):

        center = [0, 0, 0]  # center of the ground plane

        height = 2.0
        far = 14.0

        if self.up_axis == "y":
            eye = [0, height, far]  # slightly above and behind
            up = [0, 1, 0]
        elif self.up_axis == "z":
            eye = [0, -far, height]  # slightly above and behind
            up = [0, 0, 1]
        elif self.up_axis == "x":
            eye = [far, 0, height]  # slightly above and behind
            up = [1, 0, 0]
        else:
            raise ValueError("up_axis must be one of {'x','y','z'}")

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-1, -1, -1], max_bound=[1, 1, 1]
        )
        self._scene.setup_camera(60.0, bbox, center)

        # Move camera position manually
        self._scene.scene.camera.look_at(center, eye, up)

    def _init_smpl(self, gender: str = "male"):
        self._scene.scene.remove_geometry("__body_model__")

        # SMPLLayer internally looks up SMPL/SMPLH assets via its configured paths.
        # HUMOS uses: SMPLLayer(model_type="smplh", gender=..., device=...)
        self.smpl_model = SMPLLayer(
            model_type="smplh",
            gender=gender,
            device=self.device,
        )

        # Faces for SMPLH topology (F,3)
        faces = np.asarray(self.smpl_model.faces, dtype=np.int32)

        # Create a neutral T-pose mesh (all-zero pose, trans, betas)
        # SMPLH body joints count in HUMOS is 21 => 21*3 = 63 axis-angle dims
        # Root orient is (3,), trans is (3,), betas is (10,)
        zeros_body = np.zeros((1, 63), dtype=np.float32)
        zeros_root = np.zeros((1, 3), dtype=np.float32)
        zeros_trans = np.zeros((1, 3), dtype=np.float32)
        zeros_betas = np.zeros((1, 10), dtype=np.float32)

        # SMPLLayer forward returns (verts, joints)
        # verts: (1, 6890, 3)
        verts_torch, _ = self.smpl_model(
            poses_body=torch.from_numpy(zeros_body).to(self.device),
            poses_root=torch.from_numpy(zeros_root).to(self.device),
            trans=torch.from_numpy(zeros_trans).to(self.device),
            betas=torch.from_numpy(zeros_betas).to(self.device),
        )
        verts = verts_torch[0].detach().cpu().numpy()

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        self.body_meshes = []

        for mesh_idx in range(self.batch_size):

            mesh_verts = verts + self.offsets[mesh_idx]

            body_mesh = o3d.geometry.TriangleMesh()
            body_mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
            body_mesh.triangles = o3d.utility.Vector3iVector(faces)
            body_mesh.compute_vertex_normals()
            body_mesh.paint_uniform_color([0.5, 0.5, 0.5])

            # Floor alignment (same as your original)
            # min_y = -body_mesh.get_min_bound()[1]
            # body_mesh.translate([0, min_y, 0])

            name = f"__body_model_{mesh_idx}__"
            self._scene.scene.add_geometry(name, body_mesh, self.material)
            self.body_meshes.append(body_mesh)

    def _add_ui(self):
        """
        Add page selection, motion selection and text description
        """
        em = self.window.theme.font_size

        self._widget_layout = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # Category Selector
        self.category_combo = gui.Combobox()

        # Index Selector (1–500)
        self.index_combo = gui.Combobox()

        for i in range(1, self.pager.total_pages + 1):
            self.index_combo.add_item(str(i))

        self.index_combo.set_on_selection_changed(self._on_page_changed)
        self._widget_layout.add_child(gui.Label("Select Page"))
        self._widget_layout.add_child(self.index_combo)

        self.category_combo.set_on_selection_changed(self._on_motion_changed)
        self._widget_layout.add_child(gui.Label("Select Motion"))
        self._widget_layout.add_child(self.category_combo)

        self.play_button = gui.Button("Pause")
        self.play_button.enabled = False
        self.play_button.set_on_clicked(self._on_run_button_click)
        self._widget_layout.add_child(self.play_button)

        self.window.add_child(self._widget_layout)

        # Create a horizontal layout to align the label to the left
        self.label_layout = gui.Horiz()
        self.label = gui.Label("Text description")
        self.label_layout.add_child(self.label)

        self.window.add_child(self.label_layout)

    def _on_layout(self, layout_context):
        """adjust the layout position"""

        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._widget_layout.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._widget_layout.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        # place the label at the bottom left
        self.label_layout.frame = gui.Rect(0, r.height - 80, 700, 80)

    def _load_batch(self, page_index: int):
        """load motion path for give page"""

        self.play_animation = False
        self.frame_idx = 0
        self.play_button.enabled = False

        self.motion_batch = self.pager.get_names_by_page(page_index)

        # do not forget to update motion selection
        self.category_combo.clear_items()

        # Add new motion names
        for motion_name in self.motion_batch:
            self.category_combo.add_item(motion_name)

        return self.motion_batch[0]

    def _load_data(self, motion_name: str):
        """load motion data"""

        self.play_animation = False

        # 'betas', 'gender', 'root_orient', 'pose_body', 'trans'
        # [64, 200, x]
        # motion_name are like # 000002_-1; # 000002_0; # 000002_1
        self.motion_data, text = self.pager.load_single(motion_name)

        # decide gender from motion name
        gender = gender_from_motion_name(motion_name)

        # cache SMPLLayer per gender (avoid re-creating it 64x per load)
        if not hasattr(self, "_smplh_cache"):
            self._smplh_cache = {}

        if gender not in self._smplh_cache:
            self._smplh_cache[gender] = SMPLLayer(
                model_type="smplh", gender=gender, device=self.device
            )

        bm = self._smplh_cache[gender]

        print(f"load SMPLLayer with gender {gender}")

        self.label.text = text[0]

        self.num_frames = self.motion_data["betas"].shape[1]

        self.frame_idx = 0

        for mesh_idx in range(self.batch_size):

            motion_params = {
                "betas": self.motion_data["betas"][mesh_idx],
                "transl": self.motion_data["trans"][mesh_idx],
                "global_orient": self.motion_data["root_orient"][mesh_idx],
                "body_pose": self.motion_data["pose_body"][mesh_idx],
                "jaw_pose": torch.zeros(
                    (self.num_frames, 3), dtype=torch.float32, device=self.device
                ),
                "leye_pose": torch.zeros(
                    (self.num_frames, 3), dtype=torch.float32, device=self.device
                ),
                "reye_pose": torch.zeros(
                    (self.num_frames, 3), dtype=torch.float32, device=self.device
                ),
                "left_hand_pose": torch.zeros(
                    (self.num_frames, 45), dtype=torch.float32, device=self.device
                ),
                "right_hand_pose": torch.zeros(
                    (self.num_frames, 45), dtype=torch.float32, device=self.device
                ),
                "expression": torch.zeros(
                    (self.num_frames, 10), dtype=torch.float32, device=self.device
                ),
            }

            m_verts, m_joints = bm(
                poses_body=motion_params["body_pose"],
                betas=motion_params["betas"],
                poses_root=motion_params["global_orient"],
                trans=motion_params["transl"],
            )

            self.verts_glob[mesh_idx] = m_verts.cpu().numpy() + self.offsets[mesh_idx]

            apply_static_ground_offset(
                self.verts_glob[mesh_idx],
                up_axis=self.up_axis,
                safety_margin=0.002,
            )

            self.body_meshes[mesh_idx].vertices = o3d.utility.Vector3dVector(
                self.verts_glob[mesh_idx][self.frame_idx].copy()
            )

            name = f"__body_model_{mesh_idx}__"

            self._scene.scene.remove_geometry(name)
            self._scene.scene.add_geometry(
                name, self.body_meshes[mesh_idx], self.material
            )

        self.play_button.enabled = True

    def _on_motion_changed(self, value, _):

        try:

            self._load_data(value)

        except Exception as e:
            msg = gui.Dialog("Error")
            msg_layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))
            msg_layout.add_child(gui.Label(f"Invalid folder selected."))
            ok_button = gui.Button("OK")
            ok_button.set_on_clicked(lambda: self.window.close_dialog())
            msg_layout.add_child(ok_button)
            msg.add_child(msg_layout)
            self.window.show_dialog(msg)

            print(e)
            return

    def _on_page_changed(self, value, _):
        try:

            motion_path = self._load_batch(int(value) - 1)
            self._load_data(motion_path)

        except Exception as e:
            self.selected_index = None

            msg = gui.Dialog("Error")
            msg_layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))
            msg_layout.add_child(gui.Label(f"Invalid index selected."))
            ok_button = gui.Button("OK")
            ok_button.set_on_clicked(lambda: self.window.close_dialog())
            msg_layout.add_child(ok_button)
            msg.add_child(msg_layout)
            self.window.show_dialog(msg)

            print(e)
            return

    def _on_run_button_click(self):

        self.play_animation = not self.play_animation

    def animate_mesh(self):

        step = 1 / 60

        while True:

            if not self.play_animation:
                time.sleep(0.1)
                continue

            # snapshot frame index in the worker thread
            frame_idx = self.frame_idx

            # freeze per-mesh vertex arrays for this frame (no Open3D calls here)
            verts_frame = [
                self.verts_glob[i][frame_idx].copy() for i in range(self.batch_size)
            ]

            def update(verts_frame=verts_frame):
                # UI thread ONLY: touch Open3D objects here
                for i in range(self.batch_size):
                    self.body_meshes[i].vertices = o3d.utility.Vector3dVector(
                        verts_frame[i]
                    )

                    name = f"__body_model_{i}__"
                    self._scene.scene.remove_geometry(name)
                    self._scene.scene.add_geometry(
                        name, self.body_meshes[i], self.material
                    )

            gui.Application.instance.post_to_main_thread(self.window, update)

            # advance frame index (worker thread state only)
            self.frame_idx = (frame_idx + 1) % self.num_frames
            time.sleep(step)

            # while self.frame_idx < self.num_frames and self.play_animation:

            #     for mesh_idx in range(self.batch_size):

            #         verts = self.verts_glob[mesh_idx][self.frame_idx].copy()

            #         self.body_meshes[mesh_idx].vertices = o3d.utility.Vector3dVector(
            #             verts
            #         )

            #         # Schedule image update in the GUI thread
            #         def update():
            #             name = f"__body_model_{mesh_idx}__"
            #             self._scene.scene.remove_geometry(name)
            #             self._scene.scene.add_geometry(
            #                 name, self.body_meshes[mesh_idx], self.material
            #             )

            #         gui.Application.instance.post_to_main_thread(self.window, update)

            #     self.frame_idx += 1
            #     time.sleep(step)

            #     if self.frame_idx >= self.num_frames:
            #         self.frame_idx = 0

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":

    animPlayer = AnimPlayer()

    animPlayer.run()
