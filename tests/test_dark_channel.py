import numpy as np


def test_dark_channel_returns_channelwise_minimum_for_unit_block() -> None:
    from video_smoke_detection.dark_channel import get_dark_channel

    image = np.array([[[0.2, 0.4, 0.1]]], dtype=np.float64)

    dark_channel = get_dark_channel(image, 1)

    assert np.isclose(dark_channel[0, 0], 0.1)


def test_transmission_clamps_low_values_to_floor() -> None:
    from video_smoke_detection.dark_channel import transmission

    image = np.array(
        [
            [[0.9, 0.8, 0.7], [0.2, 0.5, 0.4]],
            [[0.6, 0.4, 0.3], [0.1, 0.2, 0.9]],
        ],
        dtype=np.float64,
    )
    atmosphere = np.zeros((1, 3), dtype=np.float64)

    result = transmission(image, atmosphere, 1, use_guided_filter=False)

    assert result.tolist() == [[0.1, 0.1], [0.1, 0.1]]
