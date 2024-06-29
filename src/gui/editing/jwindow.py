import pyqtgraph as pg
import pyqtgraph.dockarea
from pyqtgraph import QtGui

from .jviewbox import JviewBox
from .plotting import JPlotting
from .utils import mm_to_inch


class Jwindow:
    def __init__(self, title="JDesigner"):
        self.win = QtGui.QMainWindow()
        self.set_file_menu()

        self.area = pg.dockarea.DockArea()

        self.win.setCentralWidget(self.area)
        self.win.resize(1000, 720)
        self.win.setWindowTitle(title)

        left_dock = pg.dockarea.Dock("", size=(200, 350))
        left_bottom_dock = pg.dockarea.Dock("Selected Item", size=(200, 350))
        left_bottom_dock.label.setDim(False)
        self.rightDock = pg.dockarea.Dock("", size=(800, 700))
        bottom_dock = pg.dockarea.Dock("", size=(1000, 20))

        self.area.addDock(left_dock, "left")
        self.area.addDock(left_bottom_dock, "bottom", left_dock)
        self.area.addDock(self.rightDock, "right")
        self.area.addDock(bottom_dock, "bottom")

        self.label = QtGui.QLabel("My new Label")

        glayout = pg.GraphicsLayoutWidget()
        glayout.setBackground(QtGui.QColor(255, 255, 255))
        self.viewBox = JviewBox(self.label, left_bottom_dock, lockAspect=True)
        glayout.addItem(self.viewBox, None, None, 1, 1)

        button_add_curve = QtGui.QPushButton("Add Bezier curve")
        button_add_line = QtGui.QPushButton("Add polyline")
        button_add_square = QtGui.QPushButton("Add rectangle")
        button_add_text = QtGui.QPushButton("Add text")
        button_clear = QtGui.QPushButton("Clear")
        vertical_spacer = QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum,
                                            QtGui.QSizePolicy.Expanding)

        w1 = pg.LayoutWidget()
        w1.addWidget(button_add_curve, row=0, col=0)
        w1.addWidget(button_add_line, row=1, col=0)
        w1.addWidget(button_add_square, row=2, col=0)
        w1.addWidget(button_add_text, row=3, col=0)
        w1.addWidget(button_clear, row=4, col=0)
        w1.layout.addItem(vertical_spacer)

        left_dock.addWidget(w1)
        self.rightDock.addWidget(glayout)
        bottom_dock.addWidget(self.label)

        button_add_curve.clicked.connect(self.add_curve)
        button_add_line.clicked.connect(self.add_polyline)
        button_add_square.clicked.connect(self.add_rectangle)
        button_add_text.clicked.connect(self.add_text)
        button_clear.clicked.connect(self.clear)

        self.label.setText("JDesigner started...")

    def set_file_menu(self):
        bar = self.win.menuBar()
        file_menu = bar.addMenu("&File")

        save_action = QtGui.QAction("Save as", self.win)
        save_action.triggered.connect(self.save_as)
        file_menu.addAction(save_action)

        load_action = QtGui.QAction("Load", self.win)
        load_action.triggered.connect(self.load)
        file_menu.addAction(load_action)

        load_composition_action = QtGui.QAction("Load Composition", self.win)
        load_composition_action.triggered.connect(self.load_composition)
        file_menu.addAction(load_composition_action)

        file_menu.addSeparator()

        export_action = QtGui.QAction("Export", self.win)
        export_action.triggered.connect(self.export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QtGui.QAction("Quit", self.win)
        quit_action.triggered.connect(self.quit)
        file_menu.addAction(quit_action)

    def add_curve(self):
        self.label.setText("Adding a Bezier curve... Click to create control points...")
        self.viewBox.add_curve()

    def add_polyline(self):
        self.label.setText("Adding a polyline... Click to create points... ")
        self.viewBox.add_line()

    def add_rectangle(self):
        self.label.setText("Adding a square... Click to creare lower left corner and upper right corner")
        self.viewBox.add_square()

    def add_text(self):
        self.label.setText("Adding a text... Click to add a text...")
        self.viewBox.add_text()

    def clear(self):
        self.viewBox.clear()
        self.viewBox.delete_info_dock()

    def show(self):
        self.win.show()
