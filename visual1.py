import os
import time
import threading

import cv2
import torch
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from smplx import SMPLX

from utils.utils import get_checkerboard_plane
from motion_paginator import AmassPager


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

        self.up_axis = "z"

        # load the motion data before add ui and after init smpl
        dataset_folder = os.path.join(
            os.path.expanduser("~"), "datasets", "AMASS", "datasets"
        )

        self.pager = AmassPager(
            dataset_root=dataset_folder,
            device=self.device,
        )

        self._setup_lighting()
        self._add_ground()
        self._set_camera()
        self._add_ui()

        self._init_smpl()

        fisrt_path = self._load_batch(0)

        self._load_data(fisrt_path)

        # thread animation testing
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
            plane_width=10, num_boxes=10, ground_level=0.0, up_axis=self.up_axis
        )

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()
            self._scene.scene.add_geometry(
                f"__ground_{idx:04d}__", g, rendering.MaterialRecord()
            )

    def _set_camera(self):

        center = [0, 0, 0]  # center of the ground plane

        if self.up_axis == "y":
            eye = [0, 1.0, 4.0]  # slightly above and behind
            up = [0, 1, 0]
        elif self.up_axis == "z":
            eye = [0, 4.0, 1.0]  # slightly above and behind
            up = [0, 0, 1]
        elif self.up_axis == "x":
            eye = [4.0, 0, 1.0]  # slightly above and behind
            up = [1, 0, 0]
        else:
            raise ValueError("up_axis must be one of {'x','y','z'}")

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-1, -1, -1], max_bound=[1, 1, 1]
        )
        self._scene.setup_camera(60.0, bbox, center)

        # Move camera position manually
        self._scene.scene.camera.look_at(center, eye, up)

    def _init_smpl(self):
        self._scene.scene.remove_geometry("__body_model__")

        model_path = os.path.join("data", "body_models", "smplx")

        self.smpl_model = SMPLX(
            model_path,
            gender="neutral",
            use_pca=False,
            # num_expression_coeffs=50,  # for motion_generation *.npy files and global motion
            num_expression_coeffs=10,  # for local motion json files
        ).to(self.device)

        faces = self.smpl_model.faces

        model_output = self.smpl_model()
        verts = model_output.vertices[0].detach().cpu().numpy()

        self.body_mesh = o3d.geometry.TriangleMesh()

        self.body_mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.body_mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.body_mesh.compute_vertex_normals()
        self.body_mesh.paint_uniform_color([0.5, 0.5, 0.5])

        min_y = -self.body_mesh.get_min_bound()[1]
        self.body_mesh.translate([0, min_y, 0])

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        self._scene.scene.add_geometry("__body_model__", self.body_mesh, self.material)

    def _add_ui(self):
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

        self.label_layout.frame = gui.Rect(1000, 900, 700, 80)

    def _load_batch(self, page_index: int):

        self.play_animation = False
        self.frame_idx = 0
        self.play_button.enabled = False

        self.motion_batch = self.pager.get_names_by_page(page_index)

        self.category_combo.clear_items()

        # Add new motion names
        for motion_name in self.motion_batch:
            self.category_combo.add_item(motion_name)

        return self.motion_batch[0]

    def _load_data(self, motion_name: str):

        self.play_animation = False

        self.motion_data = self.pager.load_single(motion_name)

        self.label.text = motion_name

        motion_params = {
            # "betas": self.motion_data["betas"],
            "betas": torch.zeros(
                (self.motion_data["poses"].shape[0], 10), dtype=torch.float32
            ).to(self.device),
            "transl": self.motion_data["trans"],
            "global_orient": self.motion_data["poses"][:, :3],
            "body_pose": self.motion_data["poses"][:, 3 : 63 + 3],
            "jaw_pose": torch.zeros(
                (self.motion_data["poses"].shape[0], 3), dtype=torch.float32
            ).to(self.device),
            "leye_pose": torch.zeros(
                (self.motion_data["poses"].shape[0], 3), dtype=torch.float32
            ).to(self.device),
            "reye_pose": torch.zeros(
                (self.motion_data["poses"].shape[0], 3), dtype=torch.float32
            ).to(self.device),
            "left_hand_pose": self.motion_data["poses"][:, 66 : 66 + 45],
            "right_hand_pose": self.motion_data["poses"][:, 66 + 45 : 66 + 45 + 45],
            "expression": torch.zeros(
                (self.motion_data["poses"].shape[0], 10), dtype=torch.float32
            ).to(self.device),
        }

        output = self.smpl_model.forward(
            return_verts=True,
            **motion_params,
        )

        self.verts_glob = output.vertices.cpu().numpy()
        self.frame_idx = 0

        self.body_mesh.vertices = o3d.utility.Vector3dVector(
            self.verts_glob[self.frame_idx].copy()
        )

        self._scene.scene.remove_geometry("__body_model__")
        self._scene.scene.add_geometry("__body_model__", self.body_mesh, self.material)

        self.frame_idx = 0
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

        while True:

            if not self.play_animation:
                time.sleep(0.1)
                continue

            step = 1 / 60

            while self.frame_idx < self.verts_glob.shape[0] and self.play_animation:

                verts = self.verts_glob[self.frame_idx].copy()

                self.body_mesh.vertices = o3d.utility.Vector3dVector(verts)

                # Schedule image update in the GUI thread
                def update():
                    self._scene.scene.remove_geometry("__body_model__")
                    self._scene.scene.add_geometry(
                        "__body_model__", self.body_mesh, self.material
                    )

                gui.Application.instance.post_to_main_thread(self.window, update)

                self.frame_idx += 1
                time.sleep(step)

            if self.frame_idx >= self.verts_glob.shape[0]:
                self.frame_idx = 0

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":

    animPlayer = AnimPlayer()

    animPlayer.run()
