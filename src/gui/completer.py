from PyQt6.QtCore import QStringListModel
from PyQt6.QtWidgets import QCompleter
from PyQt6 import QtCore


class MyCompleter(QCompleter):
    insertText = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        self.qlist = QStringListModel()

        QCompleter.__init__(self, self.qlist, parent)

        self.lastSelected = None

        self.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.highlighted.connect(self.setHighlighted)

    def setHighlighted(self, text):
        self.lastSelected = text

    def getSelected(self):
        return self.lastSelected
