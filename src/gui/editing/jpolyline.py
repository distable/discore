import numpy as np
from pyqtgraph import ROI
from pyqtgraph import QtGui
from pyqtgraph import QtCore
from pyqtgraph import LayoutWidget

from .utils import perpendicular
from .utils import normalize
from .utils import construct_arrow
from .utils import delete_content
from .utils import compute_bbox_of_points

from .color import JChooseColor
from .color import setup_color

from .arrow import JArrowDock

from .remove_item import JRemoveItem


class JpolyLine(ROI, JChooseColor, JArrowDock, JRemoveItem):

    def __init__(self, positions, viewbox=None, arrow=False,
                 arrow_start=0.9, arrow_width=0.5):
        pos = [0, 0]
        ROI.__init__(self, pos, size=[1, 1])

        self.handlePen.setColor(QtGui.QColor(0, 0, 0))

        for p in positions:
            self.addFreeHandle(p)

        self.info_dock = viewbox.info_dock
        self._menu = self._build_menu()
        self._arrow = False

        JChooseColor.__init__(self)
        self.set_black_color()
        JArrowDock.__init__(self, arrow, start=arrow_start, width=arrow_width)
        JRemoveItem.__init__(self, viewbox)

        self._display_info_dock()

    @classmethod
    def load(cls, s, viewbox=None):
        if "*JPolyline" not in s:
            print("Error reading a Bezier curve from string %s" % s)

        s = s.replace("*JPolyline", "")

        if s[0] != "{" or s[-1] != "}":
            print("Error the string is in the wrong format.")

        data = eval(s)
        polyline = cls(data["control points"], viewbox=viewbox,
                       arrow=data["arrow"],
                       arrow_start=data["arrow start"],
                       arrow_width=data["arrow width"])
        setup_color(polyline, data["color"])

        if viewbox is not None:
            viewbox.label.setText("Polyline loaded.")

        return polyline

    def get_save_control_points(self):
        points = self.get_points()
        dx = self.pos().x()
        dy = self.pos().y()
        return [[p[0] + dx, p[1] + dy] for p in points]

    def save(self, file, points=None):
        data = {}
        if points is None:
            points = self.get_save_control_points()
        data["control points"] = points
        data["color"] = self.color
        data["arrow"] = self._arrow
        data["arrow start"] = self._arrow_start
        data["arrow width"] = self._arrow_width

        file.write("*JPolyline\n")
        file.write(str(data) + "\n")

    def get_points(self):
        points = []
        for p in self.handles:
            vector = np.array([p["pos"].x(), p["pos"].y()])
            points.append(vector)
        return points

    def shape(self):
        p = QtGui.QPainterPath()
        points = self._get_drawing_points()

        if not points:
            return p

        if len(points) == 2:
            vector = points[1] - points[0]
            per = perpendicular(vector)
            per = normalize(per)

            norm_vector = np.linalg.norm(vector)
            dper = per * norm_vector * 0.1

            p0 = points[0] + dper
            p1 = points[1] + dper

            p2 = points[1] - dper
            p3 = points[0] - dper
            points = [p0, p1, p2, p3]

        start = points[0]
        p.moveTo(start[0], start[1])
        for point in points[1:]:
            p.lineTo(point[0], point[1])

        p.lineTo(start[0], start[1])
        return p

    def boundingRect(self):
        return self.shape().boundingRect()

    def get_drawing_points(self):
        points = self._get_drawing_points()
        dx = self.pos().x()
        dy = self.pos().y()
        return [[x + dx, y + dy] for x, y in points]

    def _get_drawing_points(self):

        if not self._arrow:
            return self.get_points()
        else:
            pts = self.get_points()

            if len(pts) < 2:
                return pts

            last_2 = pts[-2]
            last = pts[-1]

            t = self._arrow_start
            mid = last_2 * (1 - t) + last * t
            arrow_points = construct_arrow(mid, last, self._arrow_width)

            new_pts = pts[:-1]
            new_pts.extend(arrow_points)
            return new_pts

    def compute_bbox(self):
        points = self._get_drawing_points()
        points = [[x[0], x[1]] for x in points]
        return compute_bbox_of_points(points)

    def paint(self, p, *args):

        pts = self._get_drawing_points()
        points = [QtCore.QPointF(pt[0], pt[1]) for pt in pts]

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.linePen)
        for i in range(len(points) - 1):
            p.drawLine(points[i], points[i + 1])

    def _build_menu(self):
        menu = QtGui.QMenu()
        menu.setTitle("JpolyLine")
        menu.addAction("Remove", self.remove_item)
        return menu

    def _toggle_arrow(self):
        self._arrow = not self._arrow
        self.update()

    def mouseClickEvent(self, ev):
        self._display_info_dock()

        if ev.button() == QtCore.Qt.RightButton:
            self._raise_menu(ev)
        else:
            super().mouseClickEvent(ev)

    def _raise_menu(self, event):
        pos = event.screenPos()
        self._menu.popup(QtCore.QPoint(pos.x(), pos.y()))

    def _display_info_dock(self):

        if self.info_dock is None:
            return

        delete_content(self.info_dock)

        container = LayoutWidget()
        label = QtGui.QLabel("Polyline")
        container.addWidget(label, row=0, col=0)

        arrow_dock_widget = self.get_arrow_dock_widget()
        container.addWidget(arrow_dock_widget, row=1, col=0)

        color_dock_widget = self.get_color_dock_widget()
        container.addWidget(color_dock_widget, row=2, col=0)

        remove_item_widget = self.get_remove_item_dock_widget()
        container.addWidget(remove_item_widget, row=3, col=0)

        vertical_spacer = QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum,
                                            QtGui.QSizePolicy.Expanding)
        container.layout.addItem(vertical_spacer, 4, 0)

        self.info_dock.addWidget(container)
