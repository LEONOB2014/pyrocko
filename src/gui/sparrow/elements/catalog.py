# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

import calendar
import numpy as num
import copy

from pyrocko.guts import \
    Object, Bool, Float, StringChoice, String, List

from pyrocko.himesh import HiMesh
from pyrocko import cake, table, model
from pyrocko.client import fdsn
from pyrocko.gui.qt_compat import qw, qc, fnpatch
from pyrocko import automap

from pyrocko.gui.vtk_util import ScatterPipe, TrimeshPipe
from .. import common
from pyrocko import geometry

from .base import Element, ElementState

guts_prefix = 'sparrow'


attribute_names = [
    'time', 'lat', 'lon', 'northing', 'easting', 'depth', 'magnitude']

attribute_dtypes = [
    'f16', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

name_to_icol = dict(
    (name, icol) for (icol, name) in enumerate(attribute_names))

event_dtype = num.dtype(list(zip(attribute_names, attribute_dtypes)))

t_time = num.float


def binned_statistic(values, ibins, function):
    order = num.argsort(ibins)
    values_sorted = values[order]
    ibins_sorted = ibins[order]
    parts = num.concatenate((
        [0],
        num.where(num.diff(ibins_sorted) != 0)[0] + 1,
        [ibins.size]))

    results = []
    ibins_result = []
    for ilow, ihigh in zip(parts[:-1], parts[1:]):
        values_part = values_sorted[ilow:ihigh]
        results.append(function(values_part))
        ibins_result.append(ibins_sorted[ilow])

    return ibins_result, results


def load_text(
        filepath,
        column_names=('time', 'lat', 'lon', 'depth', 'magnitude'),
        time_format='seconds'):

    with open(filepath, 'r') as f:
        if column_names == 'from_header':
            line = f.readline()
            column_names = line.split()

        name_to_icol_in = dict(
            (name, icol) for (icol, name) in enumerate(column_names)
            if name in attribute_names)

    data_in = num.loadtxt(filepath, skiprows=1)

    nevents = data_in.shape[0]
    data = num.zeros(nevents, dtype=event_dtype)

    for icol, name in enumerate(column_names):
        icol_in = name_to_icol_in.get(name, None)
        if icol_in is not None:
            if name == 'time':
                if time_format == 'seconds':
                    data[name] = data_in[:, icol_in]
                elif time_format == 'year':
                    data[name] = decimal_year_to_time(data_in[:, icol_in])
                else:
                    assert False, 'invalid time_format'
            else:
                data[name] = data_in[:, icol_in]

    return data


def decimal_year_to_time(year):
    iyear_start = num.floor(year).astype(num.int)
    iyear_end = iyear_start + 1

    iyear_min = num.min(iyear_start)
    iyear_max = num.max(iyear_end)

    iyear_to_time = num.zeros(iyear_max - iyear_min + 1, dtype=t_time)
    for iyear in range(iyear_min, iyear_max+1):
        iyear_to_time[iyear-iyear_min] = calendar.timegm(
            (iyear, 1, 1, 0, 0, 0))

    tyear_start = iyear_to_time[iyear_start - iyear_min]
    tyear_end = iyear_to_time[iyear_end - iyear_min]

    t = tyear_start + (year - iyear_start) * (tyear_end - tyear_start)

    return t


def events_to_points(events):
    coords = num.zeros((len(events), 3))

    for i, ev in enumerate(events):
        coords[i, :] = ev.lat, ev.lon, ev.depth

    station_table = table.Table()

    station_table.add_cols(
        [table.Header(name=name) for name in
            ['lat', 'lon', 'depth']],
        [coords],
        [table.Header(name=name) for name in['coords']])

    return geometry.latlondepth2xyz(
        station_table.get_col_group('coords'),
        planetradius=cake.earthradius)


class LoadingChoice(StringChoice):
    choices = [choice.upper() for choice in [
        'file',
        'fdsn']]


class FDSNSiteChoice(StringChoice):
    choices = [key.upper() for key in fdsn.g_site_abbr.keys()]


class CatalogSelection(Object):
    pass


class FileCatalogSelection(CatalogSelection):
    paths = List.T(String.T())

    def get_events(self):
        from pyrocko.io import quakeml

        events = []
        for path in self.paths:
            if path.split('.')[-1].lower() in ['xml', 'qml', 'quakeml']:
                qml = quakeml.QuakeML.load_xml(filename=path)
                events.extend(qml.get_pyrocko_events())

            else:
                events.extend(model.load_events(path))

        return events


class FileCatalogSelection2(CatalogSelection):
    paths = List.T(String.T())

    def get_points(self):
        for path in self.paths:
            data = load_text(
                path, column_names='from_header', time_format='year')

            latlondepth = num.zeros((data.shape[0], 3))
            latlondepth[:, 0] = data['lat']
            latlondepth[:, 1] = data['lon']
            latlondepth[:, 2] = data['depth'] * 1000.

            return geometry.latlondepth2xyz(
                latlondepth,
                planetradius=cake.earthradius)


class CatalogState(ElementState):
    visible = Bool.T(default=True)
    size = Float.T(default=5.0)
    catalog_selection = CatalogSelection.T(optional=True)

    @classmethod
    def get_name(self):
        return 'Catalog'

    def create(self):
        element = CatalogElement()
        element.bind_state(self)
        return element


class CatalogElement(Element):

    def __init__(self):
        Element.__init__(self)
        self._parent = None
        self._pipe = None
        self._mesh = None
        self._controls = None
        self._points = num.array([])

    def bind_state(self, state):
        upd = self.update
        self._listeners.append(upd)
        state.add_listener(upd, 'visible')
        state.add_listener(upd, 'size')
        state.add_listener(upd, 'catalog_selection')
        self._state = state
        self._current_selection = None
        self._himesh = HiMesh(order=9)
        cpt_data = [
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.3, 0.3),
            (1.0, 0.5, 0.5, 0.3)]

        self.cpt = automap.CPT(
            levels=[
                automap.CPTLevel(
                    vmin=a[0],
                    vmax=b[0],
                    color_min=[255*x for x in a[1:]],
                    color_max=[255*x for x in b[1:]])
                for (a, b) in zip(cpt_data[:-1], cpt_data[1:])])

    def unbind_state(self):
        self._listeners = []

    def get_name(self):
        return 'Catalog'

    def set_parent(self, parent):
        self._parent = parent
        self._parent.add_panel(
            self.get_name(), self._get_controls(), visible=True)
        self.update()

    def unset_parent(self):
        self.unbind_state()
        if self._parent:
            if self._pipe:
                self._parent.remove_actor(self._pipe.actor)
                self._pipe = None

            if self._mesh:
                self._parent.remove_actor(self._mesh.actor)
                self._mesh = None

            if self._controls:
                self._parent.remove_panel(self._controls)
                self._controls = None

            self._parent.update_view()
            self._parent = None

    def update(self, *args):
        state = self._state
        if self._pipe and \
                self._current_selection is not state.catalog_selection:

            self._current_selection = None
            self._parent.remove_actor(self._pipe.actor)
            self._parent.remove_actor(self._mesh.actor)
            self._pipe = None
            self._mesh = None

        if not state.visible:
            if self._pipe:
                self._parent.remove_actor(self._pipe.actor)

        else:
            if self._current_selection is not state.catalog_selection:
                points = state.catalog_selection.get_points()

                print(num.sum(points.shape[0]))
                ifaces = self._himesh.points_to_faces(points)
                ifaces_max = num.max(ifaces)
                colors = num.random.random((ifaces_max+1, 3))

                ifaces_x, sizes = binned_statistic(
                    ifaces, ifaces, lambda part: part.shape[0])

                print(num.sum(sizes))
                print(ifaces_x, sizes)

                vertices = self._himesh.get_vertices()
                vertices *= 0.95
                faces = self._himesh.get_faces()

                values = num.zeros(faces.shape[0])
                values[ifaces_x] = sizes

                self._mesh = TrimeshPipe(vertices, faces, values=values)
                cpt = copy.deepcopy(self.cpt)
                cpt.scale(0., num.max(sizes))
                self._mesh.set_cpt(cpt)

                colors2 = colors[ifaces, :]

                self._pipe = ScatterPipe(points)
                self._pipe.set_colors(colors2)
                self._current_selection = state.catalog_selection

            if self._pipe:
                self._parent.add_actor(self._pipe.actor)
                self._pipe.set_size(state.size)

            if self._mesh:
                self._parent.add_actor(self._mesh.actor)

        self._parent.update_view()

    def open_file_load_dialog(self):
        caption = 'Select one or more files to open'

        fns, _ = fnpatch(qw.QFileDialog.getOpenFileNames(
            self._parent, caption, options=common.qfiledialog_options))

        self._state.catalog_selection = FileCatalogSelection2(
            paths=[str(fn) for fn in fns])

    def _get_controls(self):
        if not self._controls:
            from ..state import state_bind_checkbox, state_bind_slider

            frame = qw.QFrame()
            layout = qw.QGridLayout()
            frame.setLayout(layout)

            layout.addWidget(qw.QLabel('Size'), 0, 0)

            slider = qw.QSlider(qc.Qt.Horizontal)
            slider.setSizePolicy(
                qw.QSizePolicy(
                    qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed))
            slider.setMinimum(0)
            slider.setMaximum(100)
            layout.addWidget(slider, 0, 1)
            state_bind_slider(self, self._state, 'size', slider, factor=0.1)

            lab = qw.QLabel('Load from:')
            pb_file = qw.QPushButton('File')

            layout.addWidget(lab, 1, 0)
            layout.addWidget(pb_file, 1, 1)

            pb_file.clicked.connect(self.open_file_load_dialog)

            cb = qw.QCheckBox('Show')
            layout.addWidget(cb, 2, 0)
            state_bind_checkbox(self, self._state, 'visible', cb)

            pb = qw.QPushButton('Remove')
            layout.addWidget(pb, 2, 1)
            pb.clicked.connect(self.unset_parent)

            layout.addWidget(qw.QFrame(), 3, 0, 1, 3)

        self._controls = frame

        return self._controls


__all__ = [
    'CatalogElement',
    'CatalogState',
]
