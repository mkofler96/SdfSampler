import numpy as np
import trimesh
from SdfSampler.sdf_sampler import SDFfromMesh


# SDFfromMesh Test
def test_sdf_from_mesh():
    # Prepare a simple mesh
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],  # Vertex 0
            [0.5, -0.5, -0.5],  # Vertex 1
            [0.5, 0.5, -0.5],  # Vertex 2
            [-0.5, 0.5, -0.5],  # Vertex 3
            [-0.5, -0.5, 0.5],  # Vertex 4
            [0.5, -0.5, 0.5],  # Vertex 5
            [0.5, 0.5, 0.5],  # Vertex 6
            [-0.5, 0.5, 0.5],  # Vertex 7
        ]
    )

    # Define the faces of the cube (each face is a triangle made from 3 vertices)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Bottom face
            [4, 5, 6],
            [4, 6, 7],  # Top face
            [0, 1, 5],
            [0, 5, 4],  # Front face
            [1, 2, 6],
            [1, 6, 5],  # Right face
            [2, 3, 7],
            [2, 7, 6],  # Back face
            [3, 0, 4],
            [3, 4, 7],  # Left face
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf = SDFfromMesh(mesh)

    queries = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    distances = sdf(queries)

    # We expect the distances to be close to the true signed distances for a tetrahedron.
    expected_distances = np.array(
        [-0.5, 0.5, 1.5]
    )  # Just an example; actual values may differ
    np.testing.assert_allclose(distances.flatten(), expected_distances, rtol=1e-5)
