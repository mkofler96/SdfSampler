# SdfSampler
Tool to sample SDFs for DeepSDF
## Example Use
```
from SdfSampler.sdf_sampler import SDFSampler, BoxSDF
sampler = SDFSampler(outdir="test_dir", splitdir="test_split")
data_set_info = {"dataset_name": "test", "class_name": "box"}
sdfs = [
    BoxSDF(box_size=1, center=np.array([0, 0, 0])),
    BoxSDF(box_size=1, center=np.array([1, 0, 0])),
]
split_files = sampler.sample_sdfs(sdfs, data_set_info)
```
## Installation
To install SdfSampler run
```
pip install -e .
```
To install additional developer dependencies run
```
pip install -e ".[develop]"
```