import pyqtgraph
from PyQt5.QtCore import QEvent
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import *


class RyuPlotWindow(QMainWindow):
    sigResize = pyqtgraph.Qt.QtCore.Signal(object)
    sigFocusIn = pyqtgraph.Qt.QtCore.Signal(object)
    sigFocusOut = pyqtgraph.Qt.QtCore.Signal(object)
    sigKeyPress = pyqtgraph.Qt.QtCore.Signal(object)

    def __init__(self):
        super(RyuPlotWindow, self).__init__()
        self.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        evtype = event.type()
        if evtype == QtCore.QEvent.Type.Resize:
            self.sigResize.emit(event)
        elif evtype == QtCore.QEvent.Type.WindowActivate:
            self.sigFocusIn.emit(event)
        elif evtype == QtCore.QEvent.Type.WindowDeactivate:
            self.sigFocusOut.emit(event)
        return super().eventFilter(obj, event)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)