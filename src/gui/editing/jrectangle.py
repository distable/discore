from pyqtgraph import RectROI

from pyqtgraph import QtCore
from pyqtgraph import QtGui
from pyqtgraph import LayoutWidget

from .utils import delete_content
from .utils import compute_bbox_of_points

from .color import JChooseColor
from .color import setup_color

from .remove_item import JRemoveItem


class Jrectangle(RectROI, JChooseColor, JRemoveItem):

    def __init__(self, pos, size, viewbox=None):
        RectROI.__init__(self, pos, size)

        JChooseColor.__init__(self)
        self.set_black_color()
        JRemoveItem.__init__(self, viewbox)

        self.info_dock = viewbox.info_dock

        self._menu = self._build_menu()
        self._display_info_dock()

        for h in self.handles:
            handle = h["item"]
            handle.linePen.setColor(QtGui.QColor(0, 0, 0))

    @classmethod
    def load(cls, s, viewbox=None):
        if "*JRectangle" not in s:
            print("Error reading a rectangle from a string %s" % s)

        s = s.replace("*JRectangle", "")

        if s[0] != "{" or s[-1] != "}":
            print("Error the string is in the wrong format")

        data = eval(s)
        rectangle = cls(data["pos"], data["size"], viewbox=viewbox)
        setup_color(rectangle, data["color"])

        if viewbox is not None:
            viewbox.label.setText("Rectangle loaded.")

        return rectangle

    def get_save_control_points(self):
        return self.get_drawing_points()

    def save(self, file, pos=None, size=None):

        if pos is None and size is None:
            pos = [self.pos().x(), self.pos().y()]
            size = [self.size().x(), self.size().y()]

        data = {
            "color": self.color,
            "pos": pos,
            "size": size
        }

        file.write("*JRectangle\n")
        file.write(str(data) + "\n")

    def _build_menu(self):
        menu = QtGui.QMenu()
        menu.setTitle("Rectangle")
        menu.addAction("Remove", self.remove_item)
        return menu

    def mouseClickEvent(self, ev):
        self._display_info_dock()
        if ev.button() == QtCore.Qt.RightButton:
            self._raise_menu(ev)

    def get_drawing_points(self):
        low = [self.pos().x(), self.pos().y()]
        disp = [self.size().x(), self.size().y()]

        low_x = low[0]
        low_y = low[1]
        dx = disp[0]
        dy = disp[1]

        pts = [low, [low_x, low_y + dy], [low_x + dx, low_y + dy],
               [low_x + dx, low_y], low]
        return pts

    def compute_bbox(self):
        return compute_bbox_of_points(self.get_drawing_points())

    def paint(self, p, *args):

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.linePen)
        super().paint(p, *args)

    def _raise_menu(self, event):
        pos = event.screenPos()
        self._menu.popup(QtCore.QPoint(pos.x(), pos.y()))

    def _display_info_dock(self):
        if self.info_dock is None:
            return

        delete_content(self.info_dock)

        container = LayoutWidget()
        label1 = QtGui.QLabel("Rectangle")
        container.addWidget(label1, row=0, col=0)

        color_dock_widget = self.get_color_dock_widget()
        container.addWidget(color_dock_widget, row=1, col=0)

        remove_item_widget = self.get_remove_item_dock_widget()
        container.addWidget(remove_item_widget, row=2, col=0)

        vertical_spacer = QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum,
                                            QtGui.QSizePolicy.Expanding)
        container.layout.addItem(vertical_spacer, 3, 0)

        self.info_dock.addWidget(container)
