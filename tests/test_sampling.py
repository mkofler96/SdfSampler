import pytest
import numpy as np
from unittest.mock import MagicMock
from SdfSampler.sdf_sampler import SDFSampler, BoxSDF
import os


def test_uniform_sample():
    sampler = SDFSampler(outdir="test_dir", splitdir="test_split")
    data_set_info = {"dataset_name": "test", "class_name": "box"}

    # Mocking file I/O
    with pytest.MonkeyPatch.context() as m:
        mock_savez = MagicMock()
        m.setattr(np, "savez", mock_savez)
        m.setattr(os, "makedirs", MagicMock())
        sdfs = [
            BoxSDF(box_size=1, center=np.array([0, 0, 0])),
            BoxSDF(box_size=1, center=np.array([1, 0, 0])),
        ]
        n_samples = 1000
        split_files = sampler.sample_sdfs(
            sdfs, data_set_info, sampling_strategy="uniform", n_samples=n_samples
        )
        # You can also check the data passed as the 'neg' and 'pos' arguments
        args, kwargs = mock_savez.call_args
        neg_data = kwargs["neg"]
        pos_data = kwargs["pos"]

        assert isinstance(neg_data, np.ndarray)
        assert isinstance(pos_data, np.ndarray)
        assert (neg_data.shape[0] + pos_data.shape[0]) == n_samples
        assert neg_data.shape[1] == 4
        assert pos_data.shape[1] == 4
