import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.data_loader import StreamDataLoader, YoloDataLoader, create_dataloader


def test_create_dataloader_cache(train_cfg: Config):
    train_cfg.task.data.shuffle = False
    train_cfg.task.data.batch_size = 2

    cache_file = Path("tests/data/train.cache")
    cache_file.unlink(missing_ok=True)

    make_cache_loader = create_dataloader(train_cfg.task.data, train_cfg.dataset)
    load_cache_loader = create_dataloader(train_cfg.task.data, train_cfg.dataset)
    m_batch_size, m_images, _, m_reverse_tensors, m_image_paths = next(iter(make_cache_loader))
    l_batch_size, l_images, _, l_reverse_tensors, l_image_paths = next(iter(load_cache_loader))
    assert m_batch_size == l_batch_size
    assert m_images.shape == l_images.shape
    assert m_reverse_tensors.shape == l_reverse_tensors.shape
    assert m_image_paths == l_image_paths


def test_training_data_loader_correctness(train_dataloader: YoloDataLoader):
    """Test that the training data loader produces correctly shaped data and metadata."""
    batch_size, images, _, reverse_tensors, image_paths = next(iter(train_dataloader))
    assert batch_size == 2
    assert images.shape == (2, 3, 640, 640)
    assert reverse_tensors.shape == (2, 5)
    expected_paths = [
        Path("tests/data/images/train/000000050725.jpg"),
        Path("tests/data/images/train/000000167848.jpg"),
    ]
    assert list(image_paths) == list(expected_paths)


def test_validation_data_loader_correctness(validation_dataloader: YoloDataLoader):
    batch_size, images, targets, reverse_tensors, image_paths = next(iter(validation_dataloader))
    assert batch_size == 4
    assert images.shape == (4, 3, 640, 640)
    assert targets.shape == (4, 18, 5)
    assert reverse_tensors.shape == (4, 5)
    expected_paths = [
        Path("tests/data/images/val/000000151480.jpg"),
        Path("tests/data/images/val/000000284106.jpg"),
        Path("tests/data/images/val/000000323571.jpg"),
        Path("tests/data/images/val/000000570456.jpg"),
    ]
    assert list(image_paths) == list(expected_paths)


def test_file_stream_data_loader_frame(file_stream_data_loader: StreamDataLoader):
    """Test the frame output from the file stream data loader."""
    frame, rev_tensor, origin_frame = next(iter(file_stream_data_loader))
    assert frame.shape == (1, 3, 640, 640)
    assert rev_tensor.shape == (1, 5)
    assert origin_frame.size == (1024, 768)


def test_directory_stream_data_loader_frame(directory_stream_data_loader: StreamDataLoader):
    """Test the frame output from the directory stream data loader."""
    frame, rev_tensor, origin_frame = next(iter(directory_stream_data_loader))
    assert frame.shape == (1, 3, 640, 640)
    assert rev_tensor.shape == (1, 5)
    assert origin_frame.size != (640, 640)
