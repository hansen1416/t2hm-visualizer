import os
import math

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from smplx import SMPLX

from beta_sample import sample_betas_energy_uniform
from utils.utils import get_checkerboard_plane


class BodyShapeGallery:
    def __init__(
        self,
        num_betas: int = 10,
        batch_size: int = 64,
        per_dim_clip: float = 3.0,
        energy_max: float = 20.25,
        energy_min: float = 0.0,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_betas = num_betas

        rng = np.random.default_rng(46)
        sampled = sample_betas_energy_uniform(
            batch_size=batch_size,
            num_betas=num_betas,
            per_dim_clip=per_dim_clip,
            energy_max=energy_max,
            energy_min=energy_min,
            rng=rng,
        )
        self.betas = torch.from_numpy(sampled).to(self.device)

        self.page_size = 32
        self.total_pages = int(batch_size / self.page_size)

        gui.Application.instance.initialize()

        width, height = 1920, 1080
        self.window = gui.Application.instance.create_window(
            "SMPL Body Shape Gallery", width, height
        )

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._scene)

        self.window.set_on_layout(self._on_layout)
        self.up_axis = "y"

        self._setup_lighting()
        self._add_ground()
        self._set_camera()

        self._init_smpl()

        self._add_ui()

    def _setup_lighting(self):

        self._scene.scene.set_background([0.46, 0.44, 0.41, 1])
        self._scene.scene.show_axes(True)

        self._scene.scene.scene.set_sun_light(
            [-0.577, -0.577, -0.577],  # direction
            [1.0, 1.0, 1.0],  # color
            50000,  # intensity
        )
        self._scene.scene.scene.enable_sun_light(True)

    def _add_ground(self):
        gp = get_checkerboard_plane(
            plane_width=40, num_boxes=10, ground_level=0.0, up_axis=self.up_axis
        )

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()
            self._scene.scene.add_geometry(
                f"__ground_{idx:04d}__", g, rendering.MaterialRecord()
            )

    def _set_camera(self):

        center = [0, 0, 0]  # center of the ground plane

        height = 8.0
        far = 14.0

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

        # self.body_mesh = o3d.geometry.TriangleMesh()

        # # self.body_mesh.vertices = o3d.utility.Vector3dVector(verts)
        # self.body_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # # self.body_mesh.compute_vertex_normals()
        # self.body_mesh.paint_uniform_color([0.5, 0.5, 0.5])

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        self.body_meshes = []

        for mesh_idx in range(self.page_size):
            body_mesh = o3d.geometry.TriangleMesh()
            body_mesh.triangles = o3d.utility.Vector3iVector(faces)
            body_mesh.paint_uniform_color([0.5, 0.5, 0.5])

            name = f"__body_model_{mesh_idx}__"
            self._scene.scene.add_geometry(name, body_mesh, self.material)
            self.body_meshes.append(body_mesh)

        # self._scene.scene.add_geometry("__body_model__", self.body_mesh, self.material)
        self._update_body_mesh_from_betas(0)

    def _update_body_mesh_from_betas(self, page: int) -> None:
        current_betas = self.betas[
            int(page * self.page_size) : int((page + 1) * self.page_size),
            :,
        ]

        betas = torch.as_tensor(current_betas, dtype=torch.float32, device=self.device)
        B = betas.shape[0]

        # Neutral pose / expression for the whole batch
        body_pose = torch.zeros(
            B, self.smpl_model.NUM_BODY_JOINTS * 3, device=self.device
        )
        global_orient = torch.zeros(B, 3, device=self.device)
        left_hand_pose = torch.zeros(
            B, self.smpl_model.NUM_HAND_JOINTS * 3, device=self.device
        )
        right_hand_pose = torch.zeros(
            B, self.smpl_model.NUM_HAND_JOINTS * 3, device=self.device
        )
        expression = torch.zeros(
            B, self.smpl_model.num_expression_coeffs, device=self.device
        )

        # Head and eye joints: (B, 3) so that .reshape(-1, 1, 3) works
        jaw_pose = torch.zeros(B, 3, device=self.device)
        leye_pose = torch.zeros(B, 3, device=self.device)
        reye_pose = torch.zeros(B, 3, device=self.device)

        with torch.no_grad():
            model_output = self.smpl_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                expression=expression,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
            )

        verts = model_output.vertices.detach().cpu().numpy()

        min_y = -np.min(verts[:, :, 1])
        verts[:, :, 1] += min_y

        cols = 8
        rows = math.ceil(B / cols)
        spacing = 2.5
        x_offset = (cols - 1) * spacing / 2
        z_offset = (rows - 1) * spacing / 2

        for mesh_idx in range(self.page_size):
            name = f"__body_model_{mesh_idx}__"

            if mesh_idx >= B:
                self._scene.scene.remove_geometry(name)
                continue

            row, col = divmod(mesh_idx, cols)
            offset = np.array(
                [
                    col * spacing - x_offset,
                    0,
                    row * spacing - z_offset,
                ]
            )

            mesh_verts = verts[mesh_idx] + offset

            body_mesh = self.body_meshes[mesh_idx]
            body_mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
            body_mesh.compute_vertex_normals()

            self._scene.scene.remove_geometry(name)
            self._scene.scene.add_geometry(name, body_mesh, self.material)

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

        for i in range(1, self.total_pages + 1):
            self.index_combo.add_item(str(i))

        self.index_combo.set_on_selection_changed(self._on_page_changed)
        self._widget_layout.add_child(gui.Label("Select Page"))
        self._widget_layout.add_child(self.index_combo)

        self.window.add_child(self._widget_layout)

    def _on_page_changed(self, value, _):
        self._update_body_mesh_from_betas(int(value) - 1)

    def _on_layout(self, layout_context) -> None:
        r = self.window.content_rect
        self._scene.frame = r
        width = 20 * layout_context.theme.font_size
        height = min(
            r.height,
            self._widget_layout.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )

        self._widget_layout.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def run(self) -> None:
        gui.Application.instance.run()


if __name__ == "__main__":
    gallery = BodyShapeGallery()
    gallery.run()
