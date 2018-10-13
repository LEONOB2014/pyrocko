from __future__ import division, print_function, absolute_import

import logging
import unittest
import numpy as num

from pyrocko.gf import eikonal_ext

from .common import Benchmark

assert_ae = num.testing.assert_almost_equal


logger = logging.getLogger('pyrocko.test.test_gf_eikonal')

benchmark = Benchmark()

km = 1000.


class GFEikonalTestCase(unittest.TestCase):

    def test_empty(self):

        speeds = num.zeros((0, 0))
        times = num.zeros((0, 0))
        deltas = num.array([1., 1.])

        eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, deltas)


if __name__ == '__main__':
    util.setup_logging('test_gf_eikonal', 'warning')
    unittest.main()
