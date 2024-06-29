import pyqtgraph as pg
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QPointF, QPoint
from PyQt5.QtGui import QColor, QPainterPath, QPainter, QIntValidator
from PyQt5.QtWidgets import QLabel, QSpacerItem, QSizePolicy, QLineEdit, QMenu, QPushButton
from pyqtgraph import LayoutWidget, mkPen
from typing_extensions import override

from .algorithms import de_casteljau
from .algorithms import degree_elevation

from .utils import delete_content
from .utils import compute_bbox_of_points

from .color import JChooseColor
from .color import setup_color

from .arrow import JArrowDock

from .remove_item import JRemoveItem

class BezierCurveItem(pg.ROI):
    """
    A bezier curve that can be added to a plot.
    The points are normalized on the X axis and optionally so on the Y axis.
    The structure for the points goes [point, point_handle, ...]
    """
    def __init__(self, positions, norm_x=True, norm_y=False, clamp_x=False, clamp_y=False):
        """
        :param positions: The positions of the control points
        :param norm_x: Normalize the x axis when baking the points to array.
        :param norm_y: Normalize the y axis when baking the points to array.
        :param clamp_x: Clamp the x axis to the X axis of the plot (or 0-1 if norm_x is True)
        :param clamp_y: Clamp the y axis to the Y axis of the plot (or 0-1 if norm_y is True)
        """
        pg.ROI.__init__(self, pos=[0, 0], size=[1, 1])

        self.array = None
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.clamp_x = clamp_x
        self.clamp_y = clamp_y

        self.linePen = mkPen(color=(200, 200, 255), style=QtCore.Qt.PenStyle.DotLine)
        self.handlePen.setColor(QColor(255, 255, 255))
        for p in positions:
            self.addFreeHandle(p)


    def clear_points(self):
        while 0 < len(self.handles):
            self.removeHandle(self.handles[0]["item"])

    def get_first_point(self):
        return self.handles[0]["pos"]

    def get_point(self, i):
        return self.handles[i]["pos"]

    def get_point_count(self):
        return len(self.handles)

    def iterate_points(self):
        for hnd in self.handles:
            hnd_x, hnd_y = hnd["pos"].x(), hnd["pos"].y()
            yield hnd_x, hnd_y

    def iterate_point_handle(self):
        """
        Iterate the point pairs (point, handle)
        Meaning if we have 4 points, the indices should go
        0, 2, 4, 6
        """
        for i in range(0, len(self.handles), 2):
            pt = self.handles[i]
            hnd = self.handles[i + 1]
            px, py = pt["pos"].x(), pt["pos"].y()
            hx, hy = hnd["pos"].x(), hnd["pos"].y()

            yield QPointF(px, py), QPointF(hx, hy)

    def get_raster_points(self):
        points = []
        for pair in self.iterate_point_handle():
            points.append(pair[0])
            points.append(pair[1])
        parameters = np.linspace(0.0, 1.0, 100)
        return [de_casteljau(points, t) for t in parameters]

    def to_numpy_array(self, n):
        if self.array is None or self.array.shape[0] != n:
            self.array = np.zeros(n)

        points = self.get_raster_points()
        for i in range(n):
            self.array[i] = points[i][1]

        return self.array

    def update_shape(self):
        self.prepareGeometryChange()
        self.stateChanged(finish=True)

    @override
    def shape(self):
        if self.shape_invalidated:
            _ = QPainterPath()
            for pair in self.iterate_point_handle():
                _.moveTo(pair[0])
                _.lineTo(pair[1])

            start = self.get_first_point()
        else:
            return self._shape

    @override
    def boundingRect(self):
        return self.shape().boundingRect()

    @override
    def paint(self, p, *args):
        pts = self.get_raster_points()
        points = [QPointF(pt[0], pt[1]) for pt in pts]
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self.linePen)
        for i in range(len(points) - 1):
            p.drawLine(points[i], points[i + 1])

    def mouseClickEvent(self, ev):
        # self._display_info_dock()
        # if ev.button() == QtCore.RightButton:
        #     self.raise_menu(ev)
        pass


    # def _changed_resolution(self):
    #     try:
    #         self.resolution = float(self._resolution_edit.text())
    #     except ValueError:
    #         pass
    #     self.update()

    # def get_resolution_dock(self):
    #
    #     layout = LayoutWidget()
    #
    #     label = QLabel("Resolution")
    #     layout.addWidget(label, row=0, col=0)
    #
    #     line_edit = QLineEdit(str(self.resolution))
    #     validator = QIntValidator(20, 1000)
    #     line_edit.setValidator(validator)
    #     line_edit.textChanged.connect(self._changed_resolution)
    #     layout.addWidget(line_edit, row=0, col=1)
    #     self._resolution_edit = line_edit
    #
    #     layout.layout.setContentsMargins(0, 0, 0, 5)
    #
    #     return layout
    #
    # def get_degree_elevate_dock(self):
    #     layout = LayoutWidget()
    #     button = QPushButton("Elevate Degree")
    #     button.clicked.connect(self.elevate_degree)
    #     layout.addWidget(button, row=0, col=0)
    #     layout.layout.setContentsMargins(0, 0, 0, 0)
    #
    #     return layout

    # def save(self, points=None):
    #     if points is None: points = self.get_save_control_points()
    #     data = dict(
    #         control_points=points,
    #         resolution=self.resolution,
    #         color=self.color,
    #         arrow_start=self._arrow_start,
    #         arrow_width=self._arrow_width)
    #
    # @classmethod
    # def load(cls, s, viewbox=None):
    #     if "*JBezierCurve" not in s:
    #         print("Error reading a Bezier curve from string %s" % s)
    #
    #     s = s.replace("*JBezierCurve", "")
    #
    #     if s[0] != "{" or s[-1] != "}":
    #         print("Error the string is in the wrong format")
    #
    #     data = eval(s)
    #     curve = cls(data["control points"], data["resolution"],
    #                 viewbox=viewbox, arrow=data["arrow"],
    #                 arrow_start=data["arrow start"],
    #                 arrow_width=data["arrow width"])
    #     setup_color(curve, data["color"])
    #
    #     # if viewbox is not None:
    #     #     viewbox.label.setText("Bezier Curve loaded.")
    #
    #     return curve

    # def _display_info_dock(self):
    #     if self.info_dock is None:
    #         return
    #
    #     delete_content(self.info_dock)
    #
    #     container = LayoutWidget()
    #     label = QLabel("Curve")
    #     container.addWidget(label, row=0, col=0)
    #
    #     degree_dock_widget = self.get_degree_elevate_dock()
    #     container.addWidget(degree_dock_widget, row=1, col=0)
    #
    #     resolution_dock_widget = self.get_resolution_dock()
    #     container.addWidget(resolution_dock_widget, row=2, col=0)
    #
    #     arrow_dock_widget = self.get_arrow_dock_widget()
    #     container.addWidget(arrow_dock_widget, row=3, col=0)
    #
    #     color_dock_widget = self.get_color_dock_widget()
    #     container.addWidget(color_dock_widget, row=4, col=0)
    #
    #     remove_item_widget = self.get_remove_item_dock_widget()
    #     container.addWidget(remove_item_widget, row=5, col=0)
    #
    #     vertical_spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
    #     container.layout.addItem(vertical_spacer, 6, 0)
    #
    #     self.info_dock.addWidget(container)
