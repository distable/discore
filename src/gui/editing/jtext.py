from pyqtgraph import ROI
from pyqtgraph import QtGui
from pyqtgraph import QtCore
from pyqtgraph import TextItem
from pyqtgraph import LayoutWidget

from .utils import delete_content
from .utils import get_bigger_bbox

from .color import JChooseColor
from .color import setup_color

from .remove_item import JRemoveItem


class Jtext(TextItem):

    def __init__(self, text):

        TextItem.__init__(self, text, angle=0)
        self.setPos(0, 0)


class JtextROI(ROI, JRemoveItem, JChooseColor):

    def __init__(self, position, text, viewbox=None, screen_bbox=None,
                 size=10, transpose=False):

        ROI.__init__(self, position, size=[1, 1])
        self.handlePen.setColor(QtGui.QColor(0, 0, 0))

        dx = screen_bbox[0][1] - screen_bbox[0][0]
        dy = screen_bbox[1][1] - screen_bbox[1][0]
        self._character_width = dx * 0.01
        self._character_height = dy * 0.04

        self.text = text

        self.width = None
        self.height = None

        self.size = size
        self._bbox = None
        self._bigger_bbox = None
        self.text.setParentItem(self)

        self.info_dock = viewbox.info_dock

        self._menu = self._build_menu()
        self.transpose = False

        JRemoveItem.__init__(self, viewbox)
        self._viewbox = viewbox

        self._first_transform = None
        self._last_transform = None

        self._transpose_check_box = None
        self._text_line_edit = None
        self._text_spin_box = None

        JChooseColor.__init__(self)
        self._set_width_and_height()
        self._display_info_dock()
        self.set_black_color()

        self._changed_size()
        if transpose:
            self._toggle_transpose()
            self._transpose_check_box.setChecked(transpose)

    @classmethod
    def load(cls, s, viewbox=None):
        if "*JText" not in s:
            print("Error reading a Text from string %s" % s)

        s = s.replace("*JText", "")

        if s[0] != "{" or s[-1] != "}":
            print("Error the string is in the wrong format")

        data = eval(s)
        text = Jtext(data["text"])
        viewbox.addItem(text)
        text_roi = JtextROI(data["position"], text, viewbox=viewbox,
                            screen_bbox=viewbox.viewRange(),
                            size=data["size"],
                            transpose=data["transpose"])
        setup_color(text_roi, data["color"])

        if viewbox is not None:
            viewbox.label.setText("Text loaded.")

        return text_roi

    def save(self, file):

        data = {
            "position": [self.pos().x(), self.pos().y()],
            "text": self.text.textItem.toPlainText(),
            "size": self.size,
            "transpose": self.transpose,
            "color": self.color
        }

        file.write("*JText\n")
        file.write(str(data) + "\n")

    # does not work properly
    # def compute_bbox(self):
    #    return compute_bbox_of_points(self._get_bbox())

    def _build_menu(self):
        menu = QtGui.QMenu()
        menu.setTitle("Text")
        menu.addAction("Transpose", self._click_transpose)
        menu.addAction("Remove", self.remove_item)
        return menu

    def _click_transpose(self):
        b = self._transpose_check_box.isChecked()
        self._transpose_check_box.setChecked(not b)

    def _toggle_transpose(self):
        self.transpose = not self.transpose
        if self.transpose:
            self.text.textItem.setRotation(270)
        else:
            self.text.textItem.setRotation(0)

        self._build_bbox()
        view_box = self.viewbox
        view_box.update()

    def _set_width_and_height(self):

        text = self.text.textItem.toPlainText()
        n = len(text)
        factor = self.size / 10

        self.width = factor * n * self._character_width
        self.height = factor * self._character_height
        self._build_bbox()

    def _build_bbox(self):
        x, y = self._get_position()
        bbox = []
        width = self.width
        height = self.height
        if self.transpose:
            bbox.append([x + height, y])
            bbox.append([x + height, y + width])
            bbox.append([x, y + width])
            bbox.append([x, y])
            bbox.append([x + height, y])
        else:
            bbox.append([x,         y - height])
            bbox.append([x + width, y - height])
            bbox.append([x + width, y])
            bbox.append([x,         y])
            bbox.append([x,         y - height])
        self._bbox = bbox

    def _get_bbox(self):
        if self._bbox is None:
            self._build_bbox()
        return self._bbox

    def _change_bbox(self, f1, f2):
        self._build_bbox()
        bbox = [[x * f1, y * f2] for x, y, in self._bbox]
        self._bbox = bbox

    def _get_position(self):
        pt = self.pos()
        x, y = pt.x(), pt.y()
        pt = self.mapFromParent(x, y)
        return pt.x(), pt.y()

    def shape(self):
        p = QtGui.QPainterPath()
        bbox = self._get_bbox()
        big_bbox = self._bigger_bbox
        if big_bbox is None:
            big_bbox = bbox
        else:
            big_bbox = get_bigger_bbox(big_bbox, bbox)

        p.moveTo(big_bbox[0][0], big_bbox[0][1])
        for pt in big_bbox[1:]:
            p.lineTo(pt[0], pt[1])

        self._bigger_bbox = bbox
        return p

    def boundingRect(self):

        return self.shape().boundingRect()

    def paint(self, p, *args):
        t = p.transform()
        if self._first_transform is None:
            self._first_transform = t

        if self._first_transform != t:
            f1 = self._first_transform.m11() / t.m11()
            f2 = self._first_transform.m22() / t.m22()
            self._change_bbox(f1, f2)
        bbox = self._get_bbox()
        points = [QtCore.QPointF(pt[0], pt[1]) for pt in bbox]
        self.linePen.setWidth(2)
        p.setPen(self.linePen)
        for i in range(len(points) - 1):
            p.drawLine(points[i], points[i + 1])

    def mouseClickEvent(self, ev):

        self._display_info_dock()

        if ev.button() == QtCore.Qt.RightButton:
            self._raise_menu(ev)

    def _raise_menu(self, event):
        pos = event.screenPos()
        self._menu.popup(QtCore.QPoint(pos.x(), pos.y()))

    def _dock_toggle_transpose(self):
        b = self._transpose_check_box.isChecked()
        if self.transpose != b:
            self._toggle_transpose()

    def _transpose_dock_widget(self):

        layout = LayoutWidget()

        label = QtGui.QLabel("Transpose")
        layout.addWidget(label, row=0, col=0)

        check_box = QtGui.QCheckBox()
        layout.addWidget(check_box, row=0, col=1)
        check_box.setChecked(self.transpose)
        check_box.toggled.connect(self._dock_toggle_transpose)
        self._transpose_check_box = check_box

        layout.layout.setContentsMargins(0, 0, 0, 5)

        return layout

    def _changed_text(self):
        text = self._text_line_edit.text()
        if text == "":
            return
        else:
            self.text.setText(text, self._color_rgb)
            self._set_width_and_height()
        self.update()

    def _changed_size(self):
        self.size = self._text_spin_box.value()
        self.text.textItem.setScale(self.size / 10)
        self._set_width_and_height()
        self.update()

    def _text_dock_widget(self):

        layout = LayoutWidget()

        label = QtGui.QLabel("Text:")
        layout.addWidget(label, row=0, col=0)

        line_edit = QtGui.QLineEdit(self.text.textItem.toPlainText())
        layout.addWidget(line_edit, row=0, col=1)
        line_edit.textChanged.connect(self._changed_text)
        self._text_line_edit = line_edit

        label1 = QtGui.QLabel("Size:")
        layout.addWidget(label1, row=1, col=0)

        spin_box = QtGui.QSpinBox()
        layout.addWidget(spin_box, row=1, col=1)
        spin_box.setMinimum(6)
        spin_box.setMaximum(20)
        spin_box.setValue(self.size)
        spin_box.valueChanged.connect(self._changed_size)
        self._text_spin_box = spin_box

        layout.layout.setContentsMargins(0, 0, 0, 5)

        return layout

    def set_red_color(self):
        super().set_red_color()
        self._changed_text()

    def set_blue_color(self):
        super().set_blue_color()
        self._changed_text()

    def set_green_color(self):
        super().set_green_color()
        self._changed_text()

    def set_white_color(self):
        super().set_white_color()
        self._changed_text()

    def set_black_color(self):
        super().set_black_color()
        self._changed_text()

    def _display_info_dock(self):
        if self.info_dock is None:
            return

        delete_content(self.info_dock)

        container = LayoutWidget()
        label = QtGui.QLabel("Text")
        container.addWidget(label, row=0, col=0)

        text_dock_widget = self._text_dock_widget()
        container.addWidget(text_dock_widget, row=1, col=0)

        transpose_dock_widget = self._transpose_dock_widget()
        container.addWidget(transpose_dock_widget, row=2, col=0)

        choose_color_widget = self.get_color_dock_widget()
        container.addWidget(choose_color_widget, row=3, col=0)

        remove_item_widget = self.get_remove_item_dock_widget()
        container.addWidget(remove_item_widget, row=4, col=0)

        vertical_spacer = QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum,
                                            QtGui.QSizePolicy.Expanding)
        container.layout.addItem(vertical_spacer, 4, 0)

        self.info_dock.addWidget(container)
