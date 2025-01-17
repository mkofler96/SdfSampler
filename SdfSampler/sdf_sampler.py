import os
import numpy as np
import time
import datetime
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typing
import gustaf as gus
import skimage.measure as sc
import igl
import trimesh

from abc import ABC, abstractmethod
import numpy.typing as npt


class SDFBase(ABC):
    @abstractmethod
    def __call__(self, queries: npt.ArrayLike) -> npt.ArrayLike:
        """
        This method must be implemented by subclasses.

        Args:
            queries: numpy arraylike coordinates for which the signed distance
            will be determined

        Returns:
            The return type and description of what this method should return.
        """
        pass

    def __add__(self, other):
        return SummedSDF(self, other)

    def __neg__(self):
        return NegatedCallable(self)


class SummedSDF(ABC):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

    def __call__(self, input_param):
        result1 = self.obj1(input_param)
        result2 = self.obj2(input_param)
        return -np.maximum(-result1, -result2)


class NegatedCallable(SDFBase):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, input_param):
        result = self.obj(input_param)
        return -result


class BoxSDF(SDFBase):
    def __init__(
        self, box_size: float = 1, center: npt.ArrayLike = np.array([0, 0, 0])
    ):
        self.box_size = box_size
        self.center = center

    def __call__(self, queries: npt.ArrayLike) -> npt.ArrayLike:
        output = (
            np.linalg.norm(queries - self.center, axis=1, ord=np.inf)
            - self.box_size / 2
        )
        return output.reshape(-1, 1)


class DataSetInfo(typing.TypedDict):
    dataset_name: str
    class_name: str


class SphereParameters(typing.TypedDict):
    cx: float
    cy: float
    cz: float
    r: float


class RandomSampleSDF:
    samples: npt.ArrayLike
    distances: npt.ArrayLike

    def split_pos_neg(self):
        pos_mask = np.where(self.distances >= 0.0)[0]
        neg_mask = np.where(self.distances < 0.0)[0]
        pos = RandomSampleSDF(
            samples=self.samples[pos_mask], distances=self.distances[pos_mask]
        )
        neg = RandomSampleSDF(
            samples=self.samples[neg_mask], distances=self.distances[neg_mask]
        )
        return pos, neg

    def create_gus_plottable(self):
        vp = gus.Vertices(vertices=self.samples)
        vp.vertex_data["distance"] = self.distances
        return vp

    @property
    def stacked(self):
        return np.hstack((self.samples, self.distances))

    def __init__(self, samples, distances):
        self.samples = samples
        self.distances = distances

    def __add__(self, other):
        return RandomSampleSDF(
            samples=np.vstack((self.samples, other.samples)),
            distances=np.vstack((self.distances, other.distances)),
        )


class SDFSampler:
    def __init__(self, outdir, splitdir) -> None:
        self.outdir = outdir
        self.splitdir = splitdir

    def sample_sdfs(
        self,
        sdfs,
        data_set_info: DataSetInfo,
        show=False,
        n_samples: int = 1e5,
        sampling_strategy="uniform",
        clamp_distance=0.1,
        box_size=None,
        stds=[0.0025, 0.00025],
    ) -> list[str]:

        start_tot = time.time()

        split = []
        for i, current_sdf in enumerate(sdfs):

            file_name = f"{data_set_info['class_name']}_{10000 + i}.npz"

            folder_name = pathlib.Path(
                f"{self.outdir}/{data_set_info['dataset_name']}/{data_set_info['class_name']}"
            )
            fname = folder_name / file_name
            split.append(fname.stem)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            if os.path.isfile(fname) and show == False:
                continue

            sampled_sdf = random_sample_sdf(
                current_sdf,
                bounds=(-1, 1),
                n_samples=int(n_samples),
                type=sampling_strategy,
            )

            pos, neg = sampled_sdf.split_pos_neg()

            if show:
                vp_pos = pos.create_gus_plottable()
                vp_neg = neg.create_gus_plottable()
                vp_pos.show_options["cmap"] = "coolwarm"
                vp_neg.show_options["cmap"] = "coolwarm"
                vp_pos.show_options["vmin"] = -0.1
                vp_pos.show_options["vmax"] = 0.1
                vp_neg.show_options["vmin"] = -0.1
                vp_neg.show_options["vmax"] = 0.1
                gus.show(vp_neg, vp_pos)
            np.savez(fname, neg=neg.stacked, pos=pos.stacked)
            tot_time = time.time() - start_tot
            avg_time_per_reconstruction = tot_time / (i + 1)
            estimated_remaining_time = avg_time_per_reconstruction * (
                len(sdfs) - (i + 1)
            )
            time_string = str(
                datetime.timedelta(seconds=round(estimated_remaining_time))
            )
            print(
                f"Sampling {fname} ({(i+1)}/{len(sdfs)}) [{(i+1)/len(sdfs)*100:.2f}%] in {time_string} ({avg_time_per_reconstruction:.2f}s/file)"
            )
        return split

    def write_json(self, json_fname, data_info, split_files):
        json_content = {
            data_info["dataset_name"]: {data_info["class_name"]: split_files}
        }
        json_fname = pathlib.Path(f"{self.splitdir}/{json_fname}")
        print("saving json")
        json.dump(json_content, open(json_fname, "w"), indent=4)


def move(t_mesh, new_center):
    t_mesh.vertices += new_center - t_mesh.bounding_box.centroid


def noisy_sample(t_mesh, std, count):
    return t_mesh.sample(int(count)) + np.random.normal(scale=std, size=(int(count), 3))


def random_points(count):
    """random points in a unit sphere centered at (0, 0, 0)"""
    points = np.random.uniform(-1, 1, (int(count * 3), 3))
    points = points[np.linalg.norm(points, axis=1) <= 1]
    if points.shape[0] < count:
        print("Too little random sampling points. Resampling.......")
        random_points(count=count, boundary="unit_sphere")
    elif points.shape[0] > count:
        return points[np.random.choice(points.shape[0], count)]
    else:
        return points


def random_points_cube(count, box_size):
    """random points in a cube with size box_size centered at (0, 0, 0)"""
    points = np.random.uniform(-box_size / 2, box_size / 2, (int(count), 3))
    return points


def random_sample_sdf(sdf, bounds, n_samples, type="uniform"):
    if type == "plane":
        samples = np.random.uniform(bounds[0], bounds[1], (n_samples, 2))
        samples = np.hstack((samples, np.zeros((n_samples, 1))))
    elif type == "spherical_gaussian":
        samples = np.random.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1).reshape(-1, 1)
        # samples += np.random.uniform(bounds[0], bounds[1], (n_samples, 3))
        samples = samples + np.random.normal(0, 0.01, (n_samples, 3))
    elif type == "uniform":
        samples = np.random.uniform(bounds[0], bounds[1], (n_samples, 3))
    distances = sdf(samples)
    return RandomSampleSDF(samples=samples, distances=distances)


class SDFfromMesh(SDFBase):
    def __init__(self, mesh, dtype=np.float32, flip_sign=False):
        """
        Computes signed distance for 3D meshes.

        Parameters
        -----------
        mesh: trimesh.Trimesh
        queries: (n, 3) np.ndarray
        dtype: type
        (Optional) Default is "np.float32". Any numpy compatible dtypes.

        Returns
        --------
        signed_distances: (n,) np.ndarray
        """
        self.mesh = mesh
        self.dtype = dtype
        self.flip_sign = flip_sign

    def __call__(self, queries):
        # Get squared distance
        (
            squared_distance,
            hit_index,
            hit_coordinates,
        ) = igl.point_mesh_squared_distance(
            np.array(queries),
            self.mesh.vertices,
            np.array(self.mesh.faces, np.int32),
        )

        distances = np.sqrt(squared_distance, dtype=self.dtype)

        # Determine sign with unnecessarily long "one line"
        distances[
            trimesh.ray.ray_pyembree.RayMeshIntersector(
                self.mesh, scale_to_box=False
            ).contains_points(queries)
        ] *= -1.00

        return distances.reshape(-1, 1)
