def test_package_exposes_version() -> None:
    import video_smoke_detection

    assert hasattr(video_smoke_detection, "__version__")
