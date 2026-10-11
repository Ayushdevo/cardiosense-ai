from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ecg_plot import render_ecg_plot


def test_plot_does_not_write_files_and_image_survives_buffer_close(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = render_ecg_plot(np.sin(np.arange(600) / 15), [20, 110, 510], 72)
    image.load()
    assert image.width > image.height > 0
    assert len(set(image.tobytes())) > 10
    assert list(tmp_path.iterdir()) == []


def test_concurrent_plots_return_independent_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    signals = [np.sin(np.arange(600) / 15), np.cos(np.arange(600) / 30)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        images = list(pool.map(lambda signal: render_ecg_plot(signal, [], None), signals))
    assert images[0] is not images[1]
    first = images[0].tobytes()
    assert first != images[1].tobytes()
    render_ecg_plot(signals[1], [10], 80)
    assert images[0].tobytes() == first
    assert list(tmp_path.iterdir()) == []
