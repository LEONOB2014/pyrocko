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
        delta = 1.0

        eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

    def test_mini(self):

        nx, ny = 500, 500
        delta = 10. / float(nx)
        x = num.arange(nx) * delta
        y = num.arange(ny) * delta

        x2 = x[num.newaxis, :]
        y2 = y[:, num.newaxis]
        speeds = num.ones((ny, nx)) # + num.sin(x2)*0.1
        #speeds = num.random.random((ny, nx)) + 1.0
        times = num.zeros((ny, nx)) - 1.0
        #times[:, 0] = y**2 / 10.0
        #times[:, 0] = y*10 
        times[0, 0] = 0.0
        times[0, -1] = 0.0
        #times[ny//2, nx//2] = 0.0

        @benchmark.labeled('mini')
        def run():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

        run()
        print()
        print(times)

        from matplotlib import pyplot as plt

        plt.gcf().add_subplot(1,1,1, aspect=1.0)
        plt.contourf(x, y, times)
        plt.show()
        print(benchmark)

    def test_earthmodel(self):
        from pyrocko import cake
        mod = cake.load_model()

        nx, ny = 5000, 5000
        delta = (cake.earthradius * 2.0) / (nx-1)

        x = num.arange(nx) * delta - cake.earthradius
        y = num.arange(ny) * delta - cake.earthradius
        print(x.min(), x.max())

        x2 = x[num.newaxis, :]
        y2 = y[:, num.newaxis]

        z = cake.earthradius - num.sqrt(x2**2 + y2**2)

        vp_pro = mod.profile('vp')
        z_pro = mod.profile('z')

        vp = num.interp(z, z_pro, vp_pro)

        inside = z > 0.0

        speeds = num.ones((ny, nx))        
        speeds[:,:] = 300.
        speeds[inside] = vp[inside]

        times = num.zeros((ny, nx)) - 1.0

        iy = ny - int(round(600*km / delta))
        ix = nx//2

        times[iy, ix] = 0.0

        @benchmark.labeled('test_earthmodel')
        def run():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

        run()

        times[num.logical_not(inside)] = num.nan

        from matplotlib import pyplot as plt

        plt.gcf().add_subplot(1,1,1, aspect=1.0)
        plt.pcolormesh(x, y, speeds, cmap='gray', edgecolor='none')
        plt.contour(x, y, times, levels=num.linspace(0., 1200, 20))
        plt.gca().axis('off')
        plt.show()

        print(benchmark)


if __name__ == '__main__':
    util.setup_logging('test_gf_eikonal', 'warning')
    unittest.main()
