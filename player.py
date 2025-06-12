import os
import time
import threading

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from smplx import SMPL, SMPLX, MANO, FLAME

from utils.utils import get_checkerboard_plane, gvhmr_result_loader


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

        self.step = 1000

        self.frame_idx = 0

        self.total_frame_count = 0

        self.verts_glob = None

        self._play_animation = False

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

    def _add_ui(self):
        em = self.window.theme.font_size

        self._widget_layout = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        select_button = gui.Button("Browse")
        select_button.set_on_clicked(self._on_browse)
        self._widget_layout.add_child(select_button)

        self.play_button = gui.Button("Pause")
        self.play_button.enabled = False
        self.play_button.set_on_clicked(self._on_run_button_click)
        self._widget_layout.add_child(self.play_button)

        self.window.add_child(self._widget_layout)

    def _on_browse(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR, "Select Folder", self.window.theme
        )
        dlg.set_on_cancel(self._on_browse_cancel)
        dlg.set_on_done(self._on_browse_done)

        dlg.set_path(os.path.join(os.path.expanduser("~"), "Downloads"))

        self.window.show_dialog(dlg)

    def _on_browse_cancel(self):
        self.window.close_dialog()

    def _on_browse_done(self, folder_path):
        self.window.close_dialog()

        joints_glob = os.path.join(
            folder_path,
            "joints_glob.pt",
        )

        verts_glob = os.path.join(
            folder_path,
            "verts_glob.pt",
        )

        if os.path.exists(joints_glob) and os.path.exists(verts_glob):
            # then it's a wham result folder, visualize the mesh

            try:

                self.total_frame_count, self.step, self.verts_glob, _ = (
                    gvhmr_result_loader(joints_glob, verts_glob)
                )

            except Exception as e:
                msg = gui.Dialog("Error")
                msg_layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))
                msg_layout.add_child(gui.Label(f"Invalid folder selected."))
                ok_button = gui.Button("OK")
                ok_button.set_on_clicked(lambda: self.window.close_dialog())
                msg_layout.add_child(ok_button)
                msg.add_child(msg_layout)
                self.window.show_dialog(msg)
                return

        self.frame_idx = 0
        self.play_animation = True
        self.play_button.enabled = True  # disables

    def _on_run_button_click(self):

        self.play_animation = not self.play_animation

    def animate_mesh(self):

        while True:

            while self.play_animation and self.frame_idx < self.total_frame_count:

                self.body_mesh.vertices = o3d.utility.Vector3dVector(
                    self.verts_glob[self.frame_idx]
                )

                def update_scene():
                    self._scene.scene.remove_geometry("__body_model__")
                    self._scene.scene.add_geometry(
                        "__body_model__", self.body_mesh, self.material
                    )

                gui.Application.instance.post_to_main_thread(self.window, update_scene)
                self.frame_idx += 1
                time.sleep(self.step)  # ~30 FPS

                # last frame, reset everything
                if self.frame_idx == self.total_frame_count:

                    self.frame_idx = 0
                    self.play_animation = False

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":

    animPlayer = AnimPlayer()

    animPlayer.run()
