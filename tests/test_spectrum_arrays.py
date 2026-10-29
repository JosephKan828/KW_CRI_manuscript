"""Regression checks for the notebook's pairwise ndarray spectral workflow."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import cross_spectrum_ana as workflow
import spectrum


class ArraySpectraTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.source = self.rng.normal(size=(80, 3, 16))
        self.reference = self.rng.normal(size=self.source.shape)
        self.window = 32
        self.overlap = 16

    def chunks(self, field):
        return workflow.segment_fields(field, self.window, self.overlap)

    def test_array_pipeline_matches_named_pipeline(self):
        workflow.validate_aligned_fields(self.source, self.reference)
        source_before = self.source.copy()
        source_chunks = self.chunks(self.source)
        reference_chunks = self.chunks(self.reference)
        self.assertEqual(source_chunks.shape, (4, 32, 3, 16))
        assert_allclose(self.source, source_before)
        named_chunks = workflow.segment_fields(
            {"QR": self.source}, self.window, self.overlap,
        )
        assert_allclose(source_chunks, named_chunks["QR"])
        arrays = workflow.calculate_segment_spectra(
            source_chunks, reference_chunks, self.window,
        )
        named = workflow.calculate_segment_spectra(
            named_chunks, reference_chunks, self.window,
        )
        for actual, expected in zip(arrays, (named[0]["QR"], named[1]["QR"], named[2])):
            self.assertIsInstance(actual, np.ndarray)
            self.assertEqual(actual.shape, (4, 32, 16))
            assert_allclose(actual, expected)
        means = workflow.average_segment_spectra(*arrays)
        named_means = workflow.average_segment_spectra(*named)
        for actual, expected in zip(
            means, (named_means[0]["QR"], named_means[1]["QR"], named_means[2]),
        ):
            self.assertEqual(actual.shape, (32, 16))
            assert_allclose(actual, expected)

    def test_scaled_identical_fields_have_unit_coherence(self):
        spectra = workflow.calculate_segment_spectra(
            self.chunks(-3 * self.source), self.chunks(self.source), self.window,
        )
        means = workflow.average_segment_spectra(*spectra)
        coherence = spectrum.calculate_squared_coherence(*means)
        assert_allclose(coherence, 1.0, atol=1e-14)

    def test_opposite_segment_phases_cancel_before_coherence(self):
        segment = self.rng.normal(size=(1, 32, 3, 16)) * np.hanning(32)[None, :, None, None]
        source = np.concatenate([segment, -segment])
        reference = np.concatenate([segment, segment])
        spectra = workflow.calculate_segment_spectra(source, reference, self.window)
        # Each segment is perfectly coherent, but their cross-spectra cancel.
        assert_allclose(spectrum.calculate_squared_coherence(*spectra), 1.0)
        means = workflow.average_segment_spectra(*spectra)
        assert_allclose(spectrum.calculate_squared_coherence(*means), 0.0, atol=1e-14)

    def test_wave_frequency_wavenumber_and_normalization(self):
        time = np.arange(32)[:, None, None]
        longitude = np.arange(16)[None, None, :]
        wave = np.cos(2 * np.pi * (3 * longitude / 16 - 5 * time / 32))
        cross, power, _ = spectrum.calculate_segment_spectra(
            wave[None], wave[None], window_energy=1.0,
        )
        expected = np.zeros((1, 32, 16))
        expected[0, 5, 3] = expected[0, -5, -3] = 0.25
        assert_allclose(power, expected, atol=1e-14)
        assert_allclose(cross, expected, atol=1e-14)

    def test_zero_power_is_undefined(self):
        chunks = np.zeros((2, 32, 3, 16))
        spectra = workflow.calculate_segment_spectra(chunks, chunks, self.window)
        coherence = spectrum.calculate_squared_coherence(
            *workflow.average_segment_spectra(*spectra)
        )
        self.assertTrue(np.isnan(coherence).all())

    def test_invalid_dimensions_are_rejected(self):
        chunks = self.chunks(self.source)
        with self.assertRaises(ValueError):
            workflow.calculate_segment_spectra(chunks, chunks, self.window + 1)
        with self.assertRaises(ValueError):
            workflow.calculate_segment_spectra(chunks[:1], chunks, self.window)
        with self.assertRaises(ValueError):
            workflow.calculate_segment_spectra(chunks[:0], chunks[:0], self.window)
        with self.assertRaises(ValueError):
            workflow.average_segment_spectra(
                np.zeros((4, 32, 16)), np.zeros((4, 32, 1)), np.zeros((4, 32, 16)),
            )
        with self.assertRaises(TypeError):
            workflow.average_segment_spectra(
                np.zeros((4, 32, 16)), {"QR": np.zeros((4, 32, 16))},
                np.zeros((4, 32, 16)),
            )

    def test_notebook_retains_every_source_reference_pair(self):
        from itertools import product

        notebook = json.loads((PROJECT / "Code/ERA5/coherence_square_stat.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        context = {
            "np": np, "workflow": workflow, "spectrum": spectrum, "product": product,
            "tqdm": lambda values: values,
            "CHUNK_SIZE": self.window, "OVERLAP_SIZE": self.overlap,
            "source_anom": {"QR": self.source, "LW": 2 * self.source, "SW": -self.source},
            "reference_components": {
                "CWV": self.reference, "W1": self.source, "W2": self.reference + self.source,
            },
        }
        exec(compile(cells["b4af6832"], "notebook:calc_mcs", "exec"), context)
        exec(compile(cells["1c8047e5"], "notebook:pair_loop", "exec"), context)
        results = context["msc_coherence_squared"]
        self.assertEqual(set(results), {"CWV", "W1", "W2"})
        for ref, by_source in results.items():
            self.assertEqual(set(by_source), {"QR", "LW", "SW"})
            for src, coherence in by_source.items():
                self.assertEqual(coherence.shape, (32, 16))
                self.assertTrue(np.all((0 <= coherence) & (coherence <= 1)))
                helper_coherence, helper_spectra = context["calc_mcs"](
                    context["source_chunks"][src], context["reference_chunks"][ref], self.window,
                )
                assert_allclose(coherence, helper_coherence)
                for values in helper_spectra.values():
                    self.assertEqual(values.shape, coherence.shape)


if __name__ == "__main__":
    unittest.main()
