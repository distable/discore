from pyqtgraph import ROI
from pyqtgraph import QtGui
from pyqtgraph import QtCore
from pyqtgraph import LayoutWidget

from .utils import compute_bbox
from .utils import compute_weights
from .utils import compute_points
from .utils import delete_content

from .remove_item import JRemoveItem

from .bezier_curve import BezierCurveItem
from .jpolyline import JpolyLine
from .jrectangle import Jrectangle


class Jcomposition(ROI, JRemoveItem):

    def __init__(self, objects, viewbox=None):

        pos = [0, 0]
        ROI.__init__(self, pos, size=[1, 1])

        self.objects = objects

        self.setPen(200, 200, 200)

        self.info_dock = viewbox.info_dock
        self.viewbox = viewbox
        bbox = self.compute_bbox()

        self.handlePen.setColor(QtGui.QColor(0, 0, 0))
        for corner in bbox:
            self.addFreeHandle(corner)

        self.weights = None
        self.weight_of_control_points = None
        self.objects_control_points = None
        self.set_weights()

        JRemoveItem.__init__(self, viewbox)
        self._display_info_dock()

    def save(self, file):
        bbox = self.get_save_bbox()
        for obj, w in zip(self.objects, self.weight_of_control_points):
            if type(obj) is BezierCurveItem or type(obj) is JpolyLine:
                points = compute_points(bbox, w)
                obj.save(file, points=points)
            if type(obj) is Jrectangle:
                points = compute_points(bbox, w)
                pos = [points[0][0], points[0][1]]
                high = [points[2][0], points[2][1]]
                size = [high[0] - pos[0], high[1] - pos[1]]
                obj.save(file, pos=pos, size=size)

    def set_weights(self):
        bbox = self.get_bbox()
        self.weights = []
        self.weight_of_control_points = []
        for obj in self.objects:
            points = obj.get_raster_points()
            weights_of_points = compute_weights(bbox, points)
            self.weights.append(weights_of_points)
            control_points = obj.get_save_control_points()
            weights_of_cp = compute_weights(bbox, control_points)
            self.weight_of_control_points.append(weights_of_cp)

    def compute_bbox(self):
        return compute_bbox(self.objects)

    def get_bbox(self):
        handle0 = self.handles[0]["pos"]
        handle1 = self.handles[1]["pos"]
        return [[handle0.x(), handle0.y()], [handle1.x(), handle1.y()]]

    def get_save_bbox(self):
        bbox = self.get_bbox()
        dx = self.pos().x()
        dy = self.pos().y()
        return [[x + dx, y + dy] for x, y in bbox]

    def shape(self):
        bbox = self.get_bbox()
        x_1, y_1 = bbox[0]
        x_2, y_2 = bbox[1]

        p = QtGui.QPainterPath()
        p.moveTo(x_1, y_1)
        p.lineTo(x_1, y_2)
        p.lineTo(x_2, y_2)
        p.lineTo(x_2, y_1)
        p.lineTo(x_1, y_1)
        return p

    def boundingRect(self):
        return self.shape().boundingRect()

    def _get_drawing_points(self):
        drawing_points = []
        bbox = self.get_bbox()
        for w in self.weights:
            pts = compute_points(bbox, w)
            drawing_points.append(pts)
        return drawing_points

    def get_drawing_points(self):
        drawing_points = self._get_drawing_points()
        dx = self.pos().x()
        dy = self.pos().y()
        export_drawing_points = []
        for points in drawing_points:
            export_points = [[x + dx, y + dy] for x, y in points]
            export_drawing_points.append(export_points)
        return export_drawing_points

    def paint(self, p, *args):

        p.setRenderHint(QtGui.QPainter.Antialiasing)

        drawing_points = self._get_drawing_points()
        for pts, obj in zip(drawing_points, self.objects):
            p.setPen(obj.linePen)
            points = [QtCore.QPointF(pt[0], pt[1]) for pt in pts]
            for i in range(len(points) - 1):
                p.drawLine(points[i], points[i + 1])

    def mouseClickEvent(self, ev):
        self._display_info_dock()

    def _display_info_dock(self):

        if self.info_dock is None:
            return

        delete_content(self.info_dock)

        container = LayoutWidget()
        label = QtGui.QLabel("Composition")
        container.addWidget(label, row=0, col=0)

        remove_item_widget = self.get_remove_item_dock_widget()
        container.addWidget(remove_item_widget, row=1, col=0)

        vertical_spacer = QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum,
                                            QtGui.QSizePolicy.Expanding)
        container.layout.addItem(vertical_spacer, 2, 0)
        self.info_dock.addWidget(container)
