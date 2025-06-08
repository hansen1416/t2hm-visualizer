import os
import time
import threading

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from smplx import SMPL, SMPLX, MANO, FLAME
import torch

from utils.utils import (
    get_checkerboard_plane,
)


class AnimPlayer:

    def __init__(self):

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

        self._setup_lighting()
        self._add_ground()
        self._set_camera()

        self._init_smpl()

        self._add_ui()

        self.step = 0

        self.frame_idx = 0

        self.total_frame_count = 0

        self.verts_glob = None

        # thread animation testing
        threading.Thread(target=self.animate_mesh, daemon=True).start()

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
        gp = get_checkerboard_plane(plane_width=10, num_boxes=10)

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()
            self._scene.scene.add_geometry(
                f"__ground_{idx:04d}__", g, rendering.MaterialRecord()
            )

    def _set_camera(self):
        center = [0, 0, 0]  # center of the ground plane
        eye = [0, 1.0, 4.0]  # slightly above and behind
        up = [0, 1, 0]

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-1, -1, -1], max_bound=[1, 1, 1]
        )
        self._scene.setup_camera(60.0, bbox, center)

        # Move camera position manually
        self._scene.scene.camera.look_at(center, eye, up)

    def _init_smpl(self):
        self._scene.scene.remove_geometry("__body_model__")

        # load smpl models
        self.smpl_model = SMPL(
            os.path.join("data", "body_models", "smpl", "SMPL_NEUTRAL.pkl")
        )

        faces = self.smpl_model.faces

        model_output = self.smpl_model()
        verts = model_output.vertices[0].detach().numpy()

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

        button = gui.Button("Run Function")
        button.set_on_clicked(self._on_run_button_click)
        self._widget_layout.add_child(button)

        self.window.add_child(self._widget_layout)

        # panel.frame = gui.Rect(10, 10, 1, 50)

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

    def _on_run_button_click(self):

        results_folder = os.path.join(
            os.path.expanduser("~"), "repos", "t2hm-dataset", "outputs", "demo"
        )

        # iterate over results folder
        for video_name in os.listdir(results_folder):

            joints_glob = torch.load(
                os.path.join(
                    results_folder,
                    video_name,
                    "joints_glob.pt",
                )
            )

            verts_glob = torch.load(
                os.path.join(
                    results_folder,
                    video_name,
                    "verts_glob.pt",
                )
            )

            joints_glob = joints_glob.cpu().numpy()
            verts_glob = verts_glob.cpu().numpy()

            # data: dict = torch.load(hmr_result)
            # print(data.keys())
            # dict_keys(['smpl_params_global', 'smpl_params_incam', 'K_fullimg', 'net_outputs'])

            # print(data["smpl_params_global"].keys())
            # dict_keys(['body_pose', 'betas', 'global_orient', 'transl'])

            # for k, v in data["smpl_params_global"].items():
            # print(f"{k}: {v.shape}")
            # body_pose: torch.Size([336, 63])
            # betas: torch.Size([336, 10])
            # global_orient: torch.Size([336, 3])
            # transl: torch.Size([336, 3])

            # print(data["smpl_params_incam"].keys())
            # dict_keys(['body_pose', 'betas', 'global_orient', 'transl'])

            # for k, v in data["smpl_params_incam"].items():
            #     print(f"{k}: {v.shape}")
            # body_pose: torch.Size([336, 63])
            # betas: torch.Size([336, 10])
            # global_orient: torch.Size([336, 3])
            # transl: torch.Size([336, 3])

            # print(data["K_fullimg"].shape)
            # torch.Size([336, 3, 3])

            # print(data["net_outputs"].keys())
            # dict_keys(['model_output', 'decode_dict', 'pred_smpl_params_incam', 'pred_smpl_params_global', 'static_conf_logits'])
            # this the full output of the network, including both 'smpl_params_global' and 'smpl_params_incam'
            # for more information, refer to hmr4d/model/gvhmr/gvhmr_pl_demo.py

            video_path = os.path.join(
                os.path.expanduser("~"),
                "Downloads",
                "videos",
                f"{video_name}.mp4",
            )

            # check if video file exists
            if not os.path.exists(video_path):
                print(f"Video file does not exist: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()

            self.step = 1 / fps

            self.frame_idx = 0

            self.total_frame_count = total_frame_count

            self.verts_glob = verts_glob

            break

    def animate_mesh(self):

        while True:

            if self.verts_glob is not None:

                while self.frame_idx < self.total_frame_count:

                    self.body_mesh.vertices = o3d.utility.Vector3dVector(
                        self.verts_glob[self.frame_idx]
                    )

                    def update_scene():
                        self._scene.scene.remove_geometry("__body_model__")
                        self._scene.scene.add_geometry(
                            "__body_model__", self.body_mesh, self.material
                        )

                    gui.Application.instance.post_to_main_thread(
                        self.window, update_scene
                    )
                    self.frame_idx += 1
                    time.sleep(self.step)  # ~30 FPS

    def run(self):
        gui.Application.instance.run()

    # def start_animation(self):
    #     self.window.set_on_tick_event(lambda: print(time.time()))


if __name__ == "__main__":

    animPlayer = AnimPlayer()

    animPlayer.run()
