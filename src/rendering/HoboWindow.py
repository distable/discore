from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QStackedLayout, QGridLayout, QVBoxLayout, QPushButton, QWidget

from src.rendering.SurfaceWidget import SurfaceWidget
from src.gui.QtUtil import keyevent_to_args


class HoboWindow(QMainWindow):
    def __init__(self, surf, parent=None):
        super(HoboWindow, self).__init__(parent)
        self.current_surface = None
        self.surface_widget = SurfaceWidget(surf)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_timeout)
        self.timer.start(int(1000 / 60))
        self.timeout_handlers = []
        self.key_handlers = []
        self.key_release_handlers = []
        self.is_shift_down = False
        self.dropenter_handlers = []
        self.dropleave_handlers = []
        self.dropfile_handlers = []
        self.focusgain_handlers = []
        self.focuslose_handlers = []
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        root_layout_widget = QWidget()
        root_layout = QStackedLayout(root_layout_widget)
        root_layout.setStackingMode(QStackedLayout.StackAll)
        self.setCentralWidget(root_layout_widget)

        inner_layout_container = QWidget()
        # inner_layout_container.setStyleSheet("")
        inner_layout = QVBoxLayout(inner_layout_container)
        inner_layout.setContentsMargins(20,20,20,20)
        inner_layout.setSpacing(20)
        inner_layout.setAlignment(Qt.AlignHCenter)
        # inner_layout.setStyleSheet("")
        root_layout.addWidget(inner_layout_container)
        root_layout.addWidget(self.surface_widget)

        self.vbox_layout = inner_layout

    def set_buttons(self, string_func_pairs=None):
        # Clear the vbox
        # ----------------------------------------
        while self.vbox_layout.count():
            child = self.vbox_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add the buttons
        # ----------------------------------------
        if string_func_pairs is not None:
            self.vbox_layout.addStretch(1)
            for label, func in string_func_pairs:
                btn = QPushButton(label)
                btn.setFixedSize(150, 30)
                btn.clicked.connect(func)
                self.vbox_layout.addWidget(btn, alignment=Qt.AlignHCenter)
            self.vbox_layout.addStretch(1)

    def set_surface(self, surf):
        self.current_surface = surf
        self.surface_widget.surface = surf


    def on_timer_timeout(self):
        for hnd in self.timeout_handlers:
            hnd()
        self.update()
        # self.centralWidget().repaint()


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Shift:
            self.is_shift_down = True

        for hnd in self.key_handlers:
            hnd(*keyevent_to_args(event, shift=self.is_shift_down))

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Shift:
            self.is_shift_down = False
        for hnd in self.key_release_handlers:
            hnd(*keyevent_to_args(event, shift=self.is_shift_down))

    def dropEvent(self, event):
        qurls = event.mimeData().urls()
        strs = [qurl.toLocalFile() for qurl in qurls]
        for hnd in self.dropfile_handlers:
            hnd(strs)
        event.accept()

    def dragEnterEvent(self, event):
        qurls = event.mimeData().urls()
        strs = [qurl.toLocalFile() for qurl in qurls]
        for hnd in self.dropenter_handlers:
            hnd(strs)
        event.accept()

    def dragLeaveEvent(self, event):
        for hnd in self.dropleave_handlers:
            hnd()
        event.accept()

    def focusInEvent(self, event):
        for hnd in self.focusgain_handlers:
            hnd()
        event.accept()

    def focusOutEvent(self, event):
        for hnd in self.focuslose_handlers:
            hnd()
        event.accept()
