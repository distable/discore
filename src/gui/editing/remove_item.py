from pyqtgraph import LayoutWidget
from pyqtgraph import QtGui

from .utils import delete_content


class JRemoveItem:
    def __init__(self, viewbox):
        self.viewbox = viewbox

    def remove_item(self):
        delete_content(self.info_dock)
        self.viewbox.removeItem(self)

    def get_remove_item_dock_widget(self):
        layout = LayoutWidget()
        button = QtGui.QPushButton("Remove Item")
        button.clicked.connect(self.remove_item)
        layout.addWidget(button, row=0, col=0)

        layout.layout.setContentsMargins(0, 0, 0, 5)

        return layout
