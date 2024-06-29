from pyqtgraph import ViewBox
from pyqtgraph import QtCore
from pyqtgraph import QtGui

from .bezier_curve import BezierCurveItem
from .jrectangle import Jrectangle
from .jpolyline import JpolyLine
from .jcomposition import Jcomposition
from .jtext import Jtext
from .jtext import JtextROI
from .io import split_strings
from .io import get_open_filename
from .io import construct_object
from .utils import delete_content


class JviewBox(ViewBox):

    def __init__(self, label, info_dock, **kargs):
        ViewBox.__init__(self, **kargs)
        self.label = label
        self.info_dock = info_dock

        self._creating_curve = False
        self._creating_polyline = False
        self._creating_square = False
        self._creating_text = False
        self._curve_control_points = []
        self._polyline_points = []
        self._square_points = []

        self.setBackgroundColor((255, 255, 255))
        self.border.setColor(QtGui.QColor(255, 0, 0))
        self._menu = self._build_menu()

    def delete_info_dock(self):
        delete_content(self.info_dock)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self._raise_menu(ev)
        else:
            super().mouseClickEvent(ev)

    def removeItem(self, item):
        self.label.setText("Item removed.")
        if type(item) is Jtext:
            try:
                self.addedItems.remove(item)
            except:
                pass
        elif type(item) is JtextROI:
            super().removeItem(item.text)
            super().removeItem(item)
        else:
            super().removeItem(item)

    def mousePressEvent(self, event):
        if self._process_event():
            if self._creating_curve: self._creating_curve_press_event(event)
            if self._creating_polyline: self._creating_polyline_press_event(event)
            if self._creating_square: self._creating_square_press_event(event)
            if self._creating_text: self._creating_text_press_event(event)

        else:
            super().mousePressEvent(event)

    def add_curve(self):
        self._reset_private_variables()
        self._creating_curve = True

    def add_line(self):
        self._reset_private_variables()
        self._creating_polyline = True

    def add_square(self):
        self._reset_private_variables()
        self._creating_square = True

    def add_text(self):
        self._reset_private_variables()
        self._creating_text = True

    def _process_event(self):
        return self._creating_polyline or self._creating_square or \
            self._creating_curve or self._creating_text

    def _creating_square_press_event(self, event):
        p = self.mapSceneToView(event.scenePos())
        point = [p.x(), p.y()]
        self._square_points.append(point)

        if len(self._square_points) < 2:
            self.label.setText("Creating square. Click to create upper right point...")
        else:
            self.label.setText("New square created.")

            p0 = self._square_points[0]
            p1 = self._square_points[1]

            dx = abs(p1[0] - p0[0])
            dy = abs(p1[1] - p0[1])
            size = [dx, dy]

            rectangle = Jrectangle(p0, size, viewbox=self)
            self.addItem(rectangle)
            self._reset_private_variables()

    def _creating_polyline_press_event(self, event):
        p = self.mapSceneToView(event.scenePos())
        point = [p.x(), p.y()]
        self._polyline_points.append(point)

        if event.button() == QtCore.Qt.LeftButton:
            self.label.setText("Creating polyline, left click to create points, right click to stop.")
        elif event.button() == QtCore.Qt.RightButton:
            self.label.setText("New polyline created.")
            polyline = JpolyLine(self._polyline_points, viewbox=self)
            self.addItem(polyline)
            self._reset_private_variables()

    def _creating_curve_press_event(self, event):
        p = self.mapSceneToView(event.scenePos())
        point = [p.x(), p.y()]
        self._curve_control_points.append(point)

        if event.button() == QtCore.Qt.LeftButton:
            text = "Creating Bezier curve, left click to "
            text += "create control points, right click to stop."
            self.label.setText(text);
        elif event.button() == QtCore.Qt.RightButton:
            self.label.setText("New Bezier curve created")
            curve = BezierCurveItem(self._curve_control_points, viewbox=self)
            self.addItem(curve)
            self._reset_private_variables()

    def _creating_text_press_event(self, event):
        p = self.mapSceneToView(event.scenePos())
        point = [p.x(), p.y()]

        self.label.setText("New text created")

        text = Jtext("Text")
        self.addItem(text)
        text_roi = JtextROI(point, text, viewbox=self, screen_bbox=self.viewRange())
        self.addItem(text_roi)
        self._reset_private_variables()

    def _reset_private_variables(self):
        self._creating_curve = False
        self._curve_control_points = []

        self._creating_polyline = False
        self._polyline_points = []

        self._creating_square = False
        self._square_points = []

        self._creating_text = False

    def save_as(self):
        file_name = QtGui.QFileDialog.getSaveFileName(None,
                                                      "Save File",
                                                      "",
                                                      "Files (*.jdes)")
        if file_name[-5:] != ".jdes":
            file_name += ".jdes"

        self.label.setText("Saving to %s." % file_name)

        with open(file_name, "w") as file:
            for item in self.addedItems:
                if type(item) is Jtext:
                    continue
                else:
                    item.save(file)

    def load(self):
        file_name = get_open_filename()
        if file_name is None:
            return
        strings = split_strings(file_name)

        for s in strings:

            obj = construct_object(s, self)
            if obj is not None:
                self.addItem(obj)

    def load_as_composition(self):
        file_name = get_open_filename()
        if file_name is None:
            return
        strings = split_strings(file_name)

        objects = []
        for s in strings:
            obj = construct_object(s, self)
            if type(obj) is JtextROI:
                # hack, otherwise pyqtgraph would like to use object.text
                self.addItem(obj)
                self.removeItem(obj)
                print("Skipping text, can not have text in composition.")
            else:
                objects.append(obj)

        composition = Jcomposition(objects, viewbox=self)
        self.addItem(composition)
        self.label.setText("Composition loaded...")

    def _build_menu(self):
        menu = QtGui.QMenu()
        menu.setTitle("JviewBox")
        menu.addAction("Save As", self.save_as)
        return menu

    def _raise_menu(self, event):
        pos = event.screenPos()
        self._menu.popup(QtCore.QPoint(pos.x(), pos.y()))
