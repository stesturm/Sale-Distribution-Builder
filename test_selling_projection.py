"""Regression tests: run `python -m unittest test_selling_projection.py`."""

import unittest

import numpy as np

from SellingProjection import ProjectionError, ShiftedArithmeticBrownian, project_distribution


class SellingProjectionTests(unittest.TestCase):
    def setUp(self):
        self.brownian = ShiftedArithmeticBrownian(2, 0, 1)

    def test_hellinger_reported_uniform_case_is_ordered(self):
        process = ShiftedArithmeticBrownian(100, 0, 1)
        support = np.linspace(0, 200, 50)[22:33]
        result = project_distribution(support, np.ones(len(support)), process, "Squared Hellinger")
        ratio = result.projected_mass[1:] / result.target_mass[1:]
        self.assertTrue(np.all(np.diff(ratio) < 0))
        self.assertGreater(result.projected_mass[0], 0)
        self.assertLess(abs(result.projected_scaled_mean), 1e-7)
        self.assertGreater(result.objective, 0)

    def test_nonzero_mean_never_returns_unchanged_zero_distance(self):
        for name, alpha in [("Squared Hellinger", 2), ("Total variation", 2),
                            ("Rényi", .5), ("Rényi", 2), ("Kullback–Leibler", 2)]:
            result = project_distribution([1, 3, 5], [1, 1, 1], self.brownian, name, alpha=alpha)
            self.assertGreater(result.target_scaled_mean, 0)
            self.assertGreater(result.objective, 0)
            self.assertFalse(np.allclose(result.target_mass, result.projected_mass))
            self.assertLess(abs(result.projected_scaled_mean), 1e-7)

    def test_tv_unique_paper_example(self):
        b = project_distribution([1, 3, 5], [1, 1, 1], self.brownian, "Total variation")
        c = project_distribution([0, 1, 2], [1, 1, 1], self.brownian, "Total variation", cap=5)
        np.testing.assert_allclose(b.projected_mass, [1/5, 1/3, 1/3, 2/15], atol=1e-7)
        np.testing.assert_allclose(c.projected_mass, [2/15, 1/3, 1/3, 1/5], atol=1e-7)
        again = project_distribution([1, 3, 5], [1, 1, 1], self.brownian, "Total variation")
        np.testing.assert_allclose(b.projected_mass, again.projected_mass, atol=1e-8)

    def test_renyi_orders_and_conditional_endpoint(self):
        for alpha in [.5, 1, 1.5, 2, 4]:
            # Here the target already has mass on both sides of r0=2, so
            # adding a new atom at zero is permitted for alpha<1, NOT required.
            b = project_distribution([1, 3, 5], [1, 1, 1], self.brownian, "Rényi", alpha=alpha)
            self.assertLess(abs(b.projected_scaled_mean), 1e-7)
            self.assertGreater(b.objective, 0)
            if alpha >= 1:
                self.assertEqual(b.support[0], 1)

        # All target payoffs now have strictly positive scale. A zero-scaled-
        # mean law therefore needs a new zero-payoff atom. Only alpha<1 allows
        # that atom at finite divergence.
        endpoint = project_distribution([3, 4, 5], [1, 1, 1], self.brownian,
                                        "Rényi", alpha=.5)
        self.assertEqual(endpoint.support[0], 0)
        self.assertGreater(endpoint.projected_mass[0], .1)
        self.assertLess(abs(endpoint.projected_scaled_mean), 1e-7)
        for alpha in (1, 2):
            with self.assertRaises(ProjectionError):
                project_distribution([3, 4, 5], [1, 1, 1], self.brownian,
                                     "Rényi", alpha=alpha)
        with self.assertRaises(ProjectionError):
            project_distribution([1, 3], [1, 1], self.brownian, "Rényi", alpha=0)
        with self.assertRaises(ProjectionError):
            project_distribution([1, 3], [1, 1], self.brownian, "Rényi", alpha=1.001)

    def test_moderate_drift_stiff_scale_all_distances(self):
        support = np.linspace(0, 200, 50)[22:33]
        for drift in (-2, -1, 1, 2):
            process = ShiftedArithmeticBrownian(100, drift, 1)
            for distance, alpha in (("Squared Hellinger", 2), ("Total variation", 2),
                                    ("Kullback–Leibler", 2), ("Rényi", .5), ("Rényi", 2)):
                result = project_distribution(support, np.ones(support.size), process,
                                              distance, alpha=alpha, cap=200)
                self.assertGreater(result.objective, 0, (drift, distance, alpha))
                self.assertAlmostEqual(result.projected_mass.sum(), 1, places=7)
                self.assertLess(abs(result.projected_scaled_mean), 1e-6)
                if result.problem == "CM′":
                    self.assertTrue(np.all(np.cumsum(result.projected_mass)[:-1] <=
                                           np.cumsum(result.target_mass)[:-1] + 1e-7))

    def test_subpixel_endpoint_mass_not_rejected_as_unchanged(self):
        process = ShiftedArithmeticBrownian(100, .1, 1)
        support = np.linspace(0, 200, 50)[22:33]
        masses = [.07403, .04179, .13808, .03713, .11263, .04816,
                  .11691, .17969, .05459, .06535, .13164]
        result = project_distribution(support, masses, process, "Total variation")
        self.assertGreater(result.projected_mass[0], 0)
        self.assertLess(result.projected_mass[0], 1e-8)
        self.assertLess(abs(result.projected_scaled_mean), 1e-6)
        self.assertIn("below chart resolution", result.message)

    def test_scale_overflow_is_not_silently_clipped(self):
        process = ShiftedArithmeticBrownian(100, 5, 1)
        with self.assertRaises(ProjectionError):
            process.scale([0])

    def test_fosd_and_infeasibility(self):
        c = project_distribution([0, 1, 3], [.5, .3, .2], self.brownian,
                                 "Squared Hellinger", cap=5)
        self.assertTrue(np.all(np.cumsum(c.projected_mass)[:-1] <=
                               np.cumsum(c.target_mass)[:-1] + 1e-8))
        with self.assertRaises(ProjectionError):
            project_distribution([4, 5], [1, 1], self.brownian, "Kullback–Leibler")


if __name__ == "__main__":
    unittest.main()
