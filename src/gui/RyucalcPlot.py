import sys
from enum import Enum

import numpy as np
from PyQt5.QtWidgets import QLabel
from pyqtgraph import PlotWidget, InfiniteLine, Point, mkPen, TargetItem, TextItem
from pyqtgraph.Qt import QtCore

from src import renderer
from src.lib import loglib
from src.party import maths
from src.renderer import rv, RendererState, RenderMode

to_seconds = rv.to_seconds

mprint = loglib.make_log('RyucalcWindow')


class SignalEntry:
    def __init__(self, index, name, signal, line, color):
        self.index = index
        self.name = name
        self.array = signal
        self.line_item = line
        self.min = np.min(signal)
        self.max = np.max(signal)


class TimeMode(Enum):
    Seconds = 0
    Frames = 1


time_mode = TimeMode.Seconds


def x_to_seconds(x):
    if time_mode == TimeMode.Seconds:
        return x
    elif time_mode == TimeMode.Frames:
        return x / rv.fps


def x_to_frame(x):
    f = 0
    if time_mode == TimeMode.Seconds:
        f = int(x * rv.fps)
    elif time_mode == TimeMode.Frames:
        f = int(x)

    return maths.clamp(f, 0, rv.n - 1)


def x_from_seconds(t):
    if time_mode == TimeMode.Seconds:
        return t
    elif time_mode == TimeMode.Frames:
        return int(t * rv.fps)


def x_from_frame(t):
    if time_mode == TimeMode.Seconds:
        return t / rv.fps
    elif time_mode == TimeMode.Frames:
        return int(t)


def snap_x_to_frame(x):
    if time_mode == TimeMode.Seconds:
        f = int(x * rv.fps)
        return to_seconds(f)
    else:
        return int(x)


def find_nearest_signal(signals, x, y) -> SignalEntry | None:
    f = x_to_frame(x)
    minentry = None
    mindist = sys.maxsize
    for i, entry in enumerate(reversed(signals)):
        array = entry.array
        f_clamped = np.clip(f, 0, len(array) - 1)
        f_value = array[f_clamped]

        d = abs(f_value - y)
        if d < mindist:
            minentry = entry
            mindist = d

    if minentry == -1:
        return None

    return minentry


def make_vline_pen():
    if renderer.audio.is_playing():
        return mkPen('green', width=2, style=QtCore.Qt.PenStyle.SolidLine)
    else:
        return mkPen('white', width=2, style=QtCore.Qt.PenStyle.DotLine)


class RyucalcPlot(PlotWidget):
    sigKeyPress = QtCore.Signal(object)

    def __init__(self):
        super(RyucalcPlot, self).__init__()

        self.time_array = None
        self.mousepos = Point(0, 0)
        self.last_xrange = 0
        self.last_yrange = 0
        self.infopos_x = 0.0
        self.infopos_y = 0.0
        self.infopos_signal = None

        # self.vb = self.plotItem.vb
        # self.plot = self.plotItem.plot
        # self.setRange = self.plotItem.setRange
        # self.setAutorange = self.plotItem.vb.autoRange
        # self.setXRange = self.plotItem.vb.autoRange
        # self.setYRange = self.plotItem.vb.autoRange
        # self.showGrid = self.plotItem.showGrid
        # self.removeItem = self.vb.removeItem
        # self.addItem = self.vb.addItem
        # self.clear = self.vb.clear
        # self.state = self.vb.state

        # t-line
        self.vline = InfiniteLine(20, pen=make_vline_pen())
        self.plotItem.vb.addItem(self.vline, ignoreBounds=True)

        # Main coord
        self.coord_label = QLabel()
        self.coord_label.setStyleSheet("background-color: none; color: white")
        self.coord_label.setParent(self)
        self.coord_label.setText("x=0\ny=0")
        self.coord_label.setGeometry(50, 0, 300, 200)
        self.coord_label.adjustSize()
        self.coord_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.coord_label.show()
        self.coord_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Plot coord
        self.coord_textitem = TextItem('y=0')
        self.coord_textitem.hide()
        self.target_item = TargetItem(movable=False, size=7)
        self.target_item.hide()
        self.plotItem.vb.addItem(self.target_item)
        self.plotItem.vb.addItem(self.coord_textitem)

        # Events
        self.plotItem.scene().sigMouseMoved.connect(self.on_plot_move)
        self.plotItem.scene().sigMouseClicked.connect(self.on_plot_click)

        # self.scene = self.plotItem.scene()

    @property
    def xmin(self):
        return self.get_view_range()[0]

    @property
    def xmax(self):
        return self.get_view_range()[1]

    @property
    def ymin(self):
        return self.get_view_range()[2]

    @property
    def ymax(self):
        return self.get_view_range()[3]

    @property
    def x_range(self):
        return self.view_range()[0]

    @property
    def y_range(self):
        return self.view_range()[1]

    @property
    def view_range(self):
        return self.state['viewRange']

    def update_yrange(self):
        xr, yr = self.get_spans()

        if len(self.signals) > 0 and (xr != self.last_xrange or yr != self.last_yrange):
            self.plot.set_y_span_auto()

        self.last_xrange = xr
        self.last_yrange = yr

    def get_spans(self):
        # viewrange = self.get_viewrange()
        xmax, xmin, ymax, ymin = self.get_range()
        xspan = xmax - xmin
        yspan = ymax - ymin
        return xspan, yspan

    def get_range(self):
        viewrange = self.plotItem.vb.viewRange()
        left = viewrange[0][0]
        right = viewrange[0][1]
        top = viewrange[1][0]
        bottom = viewrange[1][1]
        return left, right, top, bottom

    def tick(self):
        if renderer.state == RendererState.READY and renderer.mode == RenderMode.PLAY:
            x = x_from_frame(rv.session.f)
            self.set_infopos_xy(x, 0, self.infopos_signal)

    def on_plot_move(self, ev):
        if ev is None: return
        mp = self.plotItem.vb.mapSceneToView(ev)
        x = mp.x()
        y = mp.y()
        self.mousepos = mp

        if renderer.state == RendererState.READY and renderer.mode == RenderMode.PAUSE:
            self.set_infopos_xy(x, y)
            renderer.seek(x_to_frame(x), clamp=False)

    def on_plot_click(self, ev):
        if ev.button() == 4:
            self.plot.autoRange()

    def rebuild_time(self):
        n = rv.n
        n = maths.clamp(n, 1, 999999999)
        if time_mode == TimeMode.Seconds:
            self.time_array = np.linspace(0, n / rv.fps, n)
        else:
            self.time_array = np.linspace(0, n, n)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)

    def find_nearest_signal(self, x=None, y=None) -> SignalEntry | None:
        entry = find_nearest_signal(self.signals, x, y) or self.infopos_signal
        if entry is not None:
            return entry

        return None

    def set_infopos_xy(self, x, y, signal=None):
        x = self.clamp_time(x)
        x = snap_x_to_frame(x)
        f = x_to_frame(x)
        t = x_to_seconds(x)
        is_set_signal = signal is not None

        self.infopos_x = t

        # Update the label and target
        if not is_set_signal:
            signal = self.find_nearest_signal(x, y)
            self.infopos_signal = signal

        if signal is not None:
            f = np.clip(f, 0, signal.array.shape[0] - 1)

            # if is_set_signal:
            self.set_playline_t(t)
            self.infopos_y = signal.array[f]

            if signal is not None:
                v = signal.array[f]

                # Top-left label (anchored to window corner)
                self.coord_label.show()
                if time_mode == TimeMode.Seconds:
                    self.coord_label.setText(f"x={x:.2f}\ny={v:.2f}")
                elif time_mode == TimeMode.Frames:
                    self.coord_label.setText(f"x={x:.0f}\ny={v:.2f}")


                # self.target_item.setPen(mkPen(color=plot_colors[inear]))
                # self.target_item.setPos(t, v)
                # self.target_item.show()
                # self.wtarget.setPen(mkPen(color='white', brush=mkBrush(color='black')))
                self.target_item.setPos(x, v)
                self.target_item.show()

                # self.coord_textitem.setText(f"y={v:.2f}")
                self.coord_textitem.setPos(x, v)
                self.coord_textitem.show()

    def clamp_time(self, x):
        return np.clip(x, 0, self.time_array[-1])

    def set_y_span_auto(self, only_viewed=True):
        xrange, yrange = self.get_spans()

        if only_viewed:
            viewed_xrange_f_min = x_to_frame(self.xmin())
            viewed_xrange_f_max = x_to_frame(self.xmax())
            viewed_xrange_f_min = np.clip(viewed_xrange_f_min, 0, len(self.time_array) - 1)
            viewed_xrange_f_max = np.clip(viewed_xrange_f_max, 0, len(self.time_array) - 1)
        else:
            viewed_xrange_f_min = 0
            viewed_xrange_f_max = len(self.time_array) - 1

        ymax = np.max([np.max(s[viewed_xrange_f_min:viewed_xrange_f_max]) for s in self.signals])
        ymin = np.min([np.min(s[viewed_xrange_f_min:viewed_xrange_f_max]) for s in self.signals])

        # self.wplot.setLimits(yMax=ymax, yMin=ymin)
        padding = 0.1
        self.setRange(yRange=(ymin - yrange * padding, ymax + yrange * padding), padding=0)

    def set_x_center(self, x, newrange=None):
        xrange, yrange = self.get_spans()
        self.setRange(xRange=(x - xrange / 2, x + xrange / 2), padding=0)
        if newrange:
            self.set_x_span(newrange)
        self.set_infopos_xy(x_from_seconds(x), self.infopos_y)

    def set_x_span(self, x_span):
        xmin, xmax, ymin, ymax = self.get_view_range()
        xcenter = (xmin + xmax) / 2
        self.setRange(xRange=(xcenter - x_span / 2, xcenter + x_span / 2), padding=0)

    def set_playline_x(self, x):
        self.vline.setPos(x)
        self.vline.setPen(make_vline_pen())

    def set_playline_t(self, t):
        self.set_playline_x(x_from_seconds(t))
