import sys
from pathlib import Path
import pytest
from PIL import Image

# Ensure repository root is on sys.path so we can import local modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.crop_faces import process_image

IMG_SIZE = (128, 128)


def test_process_image_normal():
    img = Image.new('RGB', (200, 200), (255, 0, 0))
    out = process_image(img, (50, 50, 60, 60), IMG_SIZE)
    assert out.size == IMG_SIZE


def test_process_image_float_box():
    img = Image.new('RGB', (200, 200), (255, 0, 0))
    out = process_image(img, (50.4, 50.6, 60.8, 60.2), IMG_SIZE)
    assert out.size == IMG_SIZE


def test_process_image_negative_coords_clamped():
    img = Image.new('RGB', (200, 200), (255, 0, 0))
    out = process_image(img, (-10, -10, 50, 50), IMG_SIZE)
    assert out.size == IMG_SIZE


@pytest.mark.parametrize("box", [(10, 10, 0, 50), (10, 10, 50, 0), (10, 10, -5, 20)])
def test_process_image_invalid_box_raises(box):
    img = Image.new('RGB', (200, 200), (255, 0, 0))
    with pytest.raises(ValueError):
        process_image(img, box, IMG_SIZE)
