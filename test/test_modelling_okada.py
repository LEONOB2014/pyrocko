from __future__ import division, print_function, absolute_import
import os
import numpy as num
import unittest

from pyrocko import util
from pyrocko.orthodrome import latlon_to_ne_numpy
from pyrocko.modelling import okada_ext, OkadaSource, DislocProcessor


d2r = num.pi / 180.
m2km = 1000.


show_plot = int(os.environ.get('MPL_SHOW', 0))


class OkadaTestCase(unittest.TestCase):

    def test_okada(self):
        n = 10

        north = num.linspace(0., 5., n)
        east = num.linspace(-5., 0., n)
        down = num.linspace(15., 5., n)

        strike = 100.
        dip = 50.
        rake = 90.
        slip = 1.0
        opening = 0.

        al1 = 0.
        al2 = 0.5
        aw1 = 0.
        aw2 = 0.25
        poisson = 0.25

        nthreads = 0

        source_patches = num.zeros((n, 9))
        source_patches[:, 0] = north
        source_patches[:, 1] = east
        source_patches[:, 2] = down
        source_patches[:, 3] = strike
        source_patches[:, 4] = dip
        source_patches[:, 5] = al1
        source_patches[:, 6] = al2
        source_patches[:, 7] = aw1
        source_patches[:, 8] = aw2

        source_disl = num.zeros((n, 3))
        source_disl[:, 0] = num.cos(rake * d2r) * slip
        source_disl[:, 1] = num.sin(rake * d2r) * slip
        source_disl[:, 2] = opening

        receiver_coords = source_patches[:, :3].copy()

        results = okada_ext.okada(
            source_patches, source_disl, receiver_coords, poisson, nthreads)

        assert results.shape == tuple((n, 12))

        source_list2 = [OkadaSource(
            lat=0., lon=0.,
            north_shift=north[i], east_shift=east[i],
            depth=down[i], al1=al1, al2=al2, aw1=aw1, aw2=aw2,
            strike=strike, dip=dip,
            rake=rake, slip=slip, opening=opening, nu=poisson)
            for i in range(n)]
        source_patches2 = num.array([
            source.source_patch() for source in source_list2])
        assert (source_patches == source_patches2).all()

        source_disl2 = num.array([
            patch.source_disloc() for patch in source_list2])
        assert (source_disl == source_disl2).all()

        results2 = okada_ext.okada(
            source_patches2, source_disl2, receiver_coords, poisson, nthreads)
        assert (results == results2).all()


    def test_okada_vs_disloc_single_Source(self):
        north = 0.
        east = 0.
        depth = 10. * m2km
        length = 50. * m2km
        width = 10. * m2km

        strike = 45.
        dip = 89.
        rake = 90.
        slip = 1.0
        opening = 0.
        poisson = 0.25

        nthreads = 0

        al1 = -length / 2.
        al2 = length / 2.
        aw1 = -width
        aw2 = 0.

        nrec_north = 100
        nrec_east = 200
        rec_north = num.linspace(
            -2. * length + north, 2. * length + north, nrec_north)
        rec_east = num.linspace(
            -2. * length + east, 2. * length + east, nrec_east)
        nrec = nrec_north * nrec_east
        receiver_coords = num.zeros((nrec, 3))
        receiver_coords[:, 0] = num.tile(rec_north, nrec_east)
        receiver_coords[:, 1] = num.repeat(rec_east, nrec_north)

        segments = [OkadaSource(
            lat=0., lon=0.,
            north_shift=north, east_shift=east,
            depth=depth, al1=al1, al2=al2, aw1=aw1, aw2=aw2,
            strike=strike, dip=dip,
            rake=rake, slip=slip, opening=opening, nu=poisson)]

        res_ok2d = DislocProcessor.process(
            segments, num.array(receiver_coords[:, ::-1][:, 1:]))

        source_patch = num.array([patch.source_patch() for patch in segments])
        source_disl = num.array([patch.source_disloc() for patch in segments])
        res_ok3d = okada_ext.okada(
            source_patch, source_disl, receiver_coords, poisson, nthreads)

        def compare_plot(param1, param2):
            import matplotlib.pyplot as plt

            valmin = num.min([param1, param2])
            valmax = num.max([param1, param2])

            def add_subplot(
                fig, param, ntot, n, sharedaxis=None, title='',
                    vmin=None, vmax=None):

                ax = fig.add_subplot(
                    ntot, 1, n, sharex=sharedaxis, sharey=sharedaxis)
                scat = ax.scatter(
                    receiver_coords[:, 1], receiver_coords[:, 0], s=20,
                    c=param, vmin=vmin, vmax=vmax, cmap='seismic',
                    edgecolor='none')
                fig.colorbar(scat, shrink=0.8, aspect=5)
                rect = plt.Rectangle((
                    -num.sin(strike * d2r) * length / 2.,
                    -num.cos(strike * d2r) * length / 2.),
                    num.cos(dip * d2r) * width, length,
                    angle=-strike, edgecolor='green', facecolor='None')
                ax.add_patch(rect)
                ax.set_title(title)
                plt.axis('equal')
                return ax

            fig = plt.figure()
            ax = add_subplot(
                fig, 100. * (param1 - param2) / num.max(num.abs([
                    valmin, valmax])), 3, 1,
                title='Okada Surface minus Okada Halfspace [%]')
            add_subplot(
                fig, param1, 3, 2, sharedaxis=ax,
                title='Okada Surface', vmin=valmin, vmax=valmax)
            add_subplot(
                fig, param2, 3, 3, sharedaxis=ax,
                title='Okada Halfspace', vmin=valmin, vmax=valmax)

            plt.show()

        if show_plot:
            compare_plot(res_ok2d['displacement.e'], res_ok3d[:, 1])

    def test_okada_GF_fill(self):
        ref_north = 0.
        ref_east = 0.
        ref_depth = 10.

        nlength = 10
        nwidth = 8

        strike = 0.
        dip = 50.
        length = 0.5
        width = 0.25

        al1 = -length / 2.
        al2 = length / 2.
        aw1 = -width / 2.
        aw2 = width / 2.
        poisson = 0.25
        mu = 32. * 1e9
        lamb = (2 * poisson * mu) / (1 - 2 * poisson)

        nthreads = 0

        npoints = nlength * nwidth
        source_coords = num.zeros((npoints, 3))

        for il in range(nlength):
            for iw in range(nwidth):
                idx = il * nwidth + iw
                source_coords[idx, 0] = \
                    num.cos(strike * d2r) * (
                        il * (num.abs(al1) + num.abs(al2)) + num.abs(al1)) - \
                    num.sin(strike * d2r) * num.cos(dip * d2r) * (
                        iw * (num.abs(aw1) + num.abs(aw2)) + num.abs(aw1)) + \
                    ref_north
                source_coords[idx, 1] = \
                    num.sin(strike * d2r) * (
                        il * (num.abs(al1) + num.abs(al2)) + num.abs(al1)) - \
                    num.cos(strike * d2r) * num.cos(dip * d2r) * (
                        iw * (num.abs(aw1) + num.abs(aw2)) + num.abs(aw1)) + \
                    ref_east
                source_coords[idx, 2] = \
                    ref_depth + num.sin(dip * d2r) * iw * (
                        num.abs(aw1) + num.abs(aw2)) + num.abs(aw1)

        receiver_coords = source_coords.copy()
        slip = 1.0
        opening = 1.0
        disl_cases = {
            "strike": {
                "slip": slip,
                "rake": 0.,
                "opening": 0.},
            "dip": {
                "slip": slip,
                "rake": 90.,
                "opening": 0.},
            "tensile": {
                "slip": 0.,
                "rake": 0.,
                "opening": opening}}

        gf = num.zeros((npoints * 6, npoints * 3))

        rotmat = num.zeros((3, 3))
        rotmat[0, 0] = num.cos(strike * d2r)
        rotmat[0, 1] = num.sin(strike * d2r)
        rotmat[0, 2] = 0.
        rotmat[1, 0] = -num.sin(strike * d2r) * num.cos(dip * d2r)
        rotmat[1, 1] = num.cos(strike * d2r) * num.cos(dip * d2r)
        rotmat[1, 2] = num.sin(dip * d2r)
        rotmat[2, 0] = num.sin(strike * d2r) * num.sin(dip * d2r)
        rotmat[2, 1] = num.cos(strike * d2r) * num.sin(dip * d2r)
        rotmat[2, 2] = num.cos(dip * d2r)

        def rot_tens33(tensor, rotmat):
            tensor_out = num.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    tensor_out[i, j] = num.sum([[
                        rotmat[i, m] * rotmat[j, n] * tensor[m, n]
                        for n in range(3)] for m in range(3)])
            return tensor_out

        for idisl, disl_type in enumerate(['strike', 'dip', 'tensile']):
            disl = disl_cases[disl_type]
            source_list = [OkadaSource(
                lat=0., lon=0.,
                north_shift=coords[0], east_shift=coords[1],
                depth=coords[2], al1=al1, al2=al2, aw1=aw1, aw2=aw2,
                strike=strike, dip=dip, rake=disl[1]['rake'],
                slip=disl[1]['slip'], opening=disl[1]['opening'],
                nu=poisson)
                for coords in source_coords]

            source_patches = [src.source_patch() for src in source_list]
            source_disl = [src.source_disloc() for src in source_list]

            for isource, (source, disl) in enumerate(zip(
                    source_patches, source_disl)):

                results = okada_ext.okada(
                    source[num.newaxis, :],
                    disl[num.newaxis, :],
                    receiver_coords,
                    poisson,
                    nthreads)

                for irec in range(receiver_coords.shape[0]):
                    eps = num.zeros((3, 3))

                    for m in range(3):
                        for n in range(3):
                            eps[m, n] = 0.5 * (
                                results[irec][m * 3 + n + 3] +
                                results[irec][n * 3 + m + 3])

                    eps_rot = rot_tens33(eps, rotmat)
                    assert num.abs(eps_rot[0, 1] - eps_rot[1, 0]) < 1e-6
                    assert num.abs(eps_rot[0, 2] - eps_rot[2, 0]) < 1e-6
                    assert num.abs(eps_rot[1, 2] - eps_rot[2, 1]) < 1e-6

                    for isig, (m, n) in enumerate(zip([
                            0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])):

                        sig = \
                            lamb * num.kron(m, n) * eps_rot[m, n] + \
                            2. * mu * eps_rot[m, n]
                        gf[irec * 6 + isig, isource * 3 + idisl] = \
                            sig / disl[disl.nonzero()][0]

        gf = num.matrix(gf)
        assert num.linalg.det(gf.T * gf) != 0.

        # Function to test the computed GF
        dstress_xz = -1.5e9

        stress = num.zeros((npoints * 6, 1))
        for il in range(nlength):
            for iw in range(nwidth):
                idx = il * nwidth + iw

                if il > 1 and il < 8 and iw > 1 and iw < 7:
                    stress[idx * 6 + 2] = dstress_xz

        disloc_est = num.linalg.inv(gf.T * gf) * gf.T * stress

        if show_plot:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            scat = ax.scatter(
                receiver_coords[:, 1], receiver_coords[:, 0],
                zs=-receiver_coords[:, 2], zdir='z', s=20,
                c=num.array([i for i in disloc_est[::3]]),
                edgecolor='None')
            fig.colorbar(scat, shrink=0.5, aspect=5)

            plt.show()



if __name__ == '__main__':
    util.setup_logging('test_okada', 'warning')
    unittest.main()