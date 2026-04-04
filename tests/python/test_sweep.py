import numpy as np
import highuvlith as huv


class TestBatchSimulator:
    def test_batch_defocus(self, source, optics, mask, grid):
        batch = huv.BatchSimulator(source, optics, mask, grid)
        results = batch.batch_defocus(focuses=[-100.0, 0.0, 100.0])
        assert len(results) == 3
        for focus, img in results:
            arr = np.asarray(img)
            assert arr.shape == (grid.size, grid.size)
            assert arr.min() >= -1e-10

    def test_process_window(self, source, optics, mask, grid):
        batch = huv.BatchSimulator(source, optics, mask, grid)
        pw = batch.process_window(
            doses=[20.0, 30.0, 40.0],
            focuses=[-200.0, 0.0, 200.0],
        )
        cd = np.asarray(pw.cd_matrix)
        assert cd.shape == (3, 3)
        doses = np.asarray(pw.doses)
        focuses = np.asarray(pw.focuses)
        assert len(doses) == 3
        assert len(focuses) == 3

    def test_single_focus_batch(self, source, optics, mask, grid):
        """batch_defocus with a single focus value should return exactly 1 result."""
        batch = huv.BatchSimulator(source, optics, mask, grid)
        results = batch.batch_defocus(focuses=[0.0])
        assert len(results) == 1
        focus_val, img = results[0]
        assert abs(focus_val - 0.0) < 1e-10
        arr = np.asarray(img)
        assert arr.shape == (grid.size, grid.size)
