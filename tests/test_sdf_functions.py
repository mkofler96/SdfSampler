import pytest
import numpy as np
from unittest.mock import MagicMock
from SdfSampler.sdf_sampler import SDFSampler, BoxSDF


# BoxSDF Test
def test_box_sdf():
    box_sdf = BoxSDF(box_size=1, center=np.array([0, 0, 0]))
    queries = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0]])

    distances = box_sdf(queries)
    expected_distances = np.array([[0.5], [1.5], [-0.5]])

    np.testing.assert_allclose(distances, expected_distances, rtol=1e-5)


# SummedSDF Test
def test_summed_sdf():
    box_sdf1 = BoxSDF(box_size=1, center=np.array([0, 0, 0]))
    box_sdf2 = BoxSDF(box_size=1, center=np.array([1, 0, 0]))
    summed_sdf = box_sdf1 + box_sdf2

    queries = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0]])
    distances = summed_sdf(queries)

    # The expected result would be the maximum of the two distances
    expected_distances = np.array([[-0.5], [0.5], [-0.5]])
    np.testing.assert_allclose(distances, expected_distances, rtol=1e-5)


# NegatedCallable Test
def test_negated_callable():
    box_sdf = BoxSDF(box_size=2, center=np.array([0, 0, 0]))
    neg_sdf = -box_sdf

    queries = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0]])
    distances = neg_sdf(queries)
    expected_distances = np.array([[0], [-1], [1]])

    np.testing.assert_allclose(distances, expected_distances, rtol=1e-5)


# Write JSON Test
def test_write_json():
    sampler = SDFSampler(outdir="test_dir", splitdir="test_split")
    data_info = {"dataset_name": "test", "class_name": "box"}
    split_files = ["box_10001"]

    # Mocking file I/O for json writing
    with pytest.MonkeyPatch.context() as m:
        mock_file = MagicMock()
        m.setattr("builtins.open", mock_file)

        sampler.write_json("split.json", data_info, split_files)

        # mock_file.assert_called_once_with(pathlib.Path('test_split/split.json'), 'w')
        written_data = "".join(
            [
                call_args[0][0]
                for call_args in mock_file.return_value.write.call_args_list
            ]
        ).strip()

        # Strip all whitespaces (spaces, newlines, tabs, etc.)
        written_data_stripped = "".join(written_data.split())

        # The expected JSON content stripped of all whitespace
        expected_data = '{"test":{"box":["box_10001"]}}'

        # Assert that the stripped written data matches the expected stripped data
        assert written_data_stripped == expected_data
