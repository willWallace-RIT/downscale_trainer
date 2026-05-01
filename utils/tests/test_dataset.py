import cv2
import numpy as np
from dataset import PairedContourDataset

def test_dataset_shapes(tmp_path):
    img = (np.random.rand(128,128,3)*255).astype("uint8")
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), img)

    ds = PairedContourDataset(tmp_path, 128)
    sample = ds[0]

    assert sample["lr"].shape == (3,128,128)
    assert sample["contour"].shape == (1,128,128)
    assert sample["hr"].shape == (3,128,128)
