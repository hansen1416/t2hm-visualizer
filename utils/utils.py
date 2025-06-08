import numpy as np
import open3d as o3d


def get_checkerboard_plane(plane_width=20, num_boxes=15, center=True):

    pw = plane_width / num_boxes
    # white = [0.8, 0.8, 0.8]
    # black = [0.2, 0.2, 0.2]
    white = [230.0 / 255.0, 244.0 / 255.0, 244.0 / 255.0]
    black = [int(150 / 1.3) / 255.0, int(217 / 1.3) / 255.0, int(217 / 1.3) / 255.0]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            # ground = trimesh.primitives.Box(
            #     center=[0, 0, -0.0001],
            #     extents=[pw, pw, 0.0002]
            # )
            ground = o3d.geometry.TriangleMesh.create_box(
                width=pw, height=0.0002, depth=pw
            )

            if center:
                c = c[0] + (pw / 2) - (plane_width / 2), c[1] + (pw / 2) - (
                    plane_width / 2
                )
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.translate([c[0], 0, c[1]])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.paint_uniform_color(black if ((i + j) % 2) == 0 else white)
            meshes.append(ground)

    return meshes
