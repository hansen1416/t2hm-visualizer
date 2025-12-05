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
        self.num_betas = 16
        self.betas = torch.zeros(1, self.num_betas, device=self.device)

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
        self.up_axis = "y"

        self._setup_lighting()
        self._add_ground()
        self._set_camera()
        self._add_ui()

        # load smpl model
        self._init_smpl()

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

        height = 2.0
        far = 6.0

        if self.up_axis == "y":
            eye = [0, height, far]  # slightly above and behind
            up = [0, 1, 0]
        elif self.up_axis == "z":
            eye = [0, far, height]  # slightly above and behind
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

    def _init_smpl(self):
        self._scene.scene.remove_geometry("__body_model__")

        model_path = os.path.join("data", "body_models", "smplx")

        self.smpl_model = SMPLX(
            model_path,
            gender="neutral",
            use_pca=False,
            # num_expression_coeffs=50,  # for motion_generation *.npy files and global motion
            num_expression_coeffs=10,  # for local motion json files
            # num_betas=16,
            num_betas=self.num_betas,
        ).to(self.device)

        faces = self.smpl_model.faces

        # # e.g. one shape vector for a single body
        # betas = torch.zeros(1, 16, device=self.device)  # (B, num_betas)
        # betas[0, 0] = 5.0  # change first shape component
        # betas[0, 1] = -0.5  # change second component, etc.

        # model_output = self.smpl_model(betas=betas)
        # verts = model_output.vertices[0].detach().cpu().numpy()

        self.body_mesh = o3d.geometry.TriangleMesh()

        # self.body_mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.body_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # self.body_mesh.compute_vertex_normals()
        self.body_mesh.paint_uniform_color([0.5, 0.5, 0.5])

        # min_y = -self.body_mesh.get_min_bound()[1]
        # self.body_mesh.translate([0, min_y, 0])

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        # self._scene.scene.add_geometry("__body_model__", self.body_mesh, self.material)
        self._update_body_mesh_from_betas()

    def _update_body_mesh_from_betas(self):
        with torch.no_grad():
            model_output = self.smpl_model(betas=self.betas)

        verts = model_output.vertices[0].detach().cpu().numpy()

        min_y = -np.min(verts[:, 1])
        verts[:, 1] += min_y

        self.body_mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.body_mesh.compute_vertex_normals()

        self._scene.scene.remove_geometry("__body_model__")
        self._scene.scene.add_geometry("__body_model__", self.body_mesh, self.material)

    def _add_ui(self):
        """
        Add page selection, motion selection and text description
        """
        em = self.window.theme.font_size

        self._widget_layout = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        self._widget_layout.add_child(gui.Label("SMPL Betas"))
        self.beta_sliders = []
        for i in range(self.num_betas):
            slider_layout = gui.Horiz(0.25 * em)
            slider_label = gui.Label(f"Beta {i}")
            # slider_label.horizontal_padding_em = 0
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-5.0, 5.0)
            slider.double_value = float(self.betas[0, i].item())
            slider.set_on_value_changed(
                lambda value, idx=i: self._on_beta_changed(idx, value)
            )

            slider_layout.add_child(slider_label)
            slider_layout.add_child(slider)
            self.beta_sliders.append(slider)
            self._widget_layout.add_child(slider_layout)

        self.window.add_child(self._widget_layout)

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

    def _on_beta_changed(self, index: int, value: float):
        self.betas[0, index] = value
        self._update_body_mesh_from_betas()

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":

    animPlayer = AnimPlayer()

    animPlayer.run()
