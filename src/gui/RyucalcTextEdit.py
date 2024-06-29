import PyQt5
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QTextEdit, QCompleter, QApplication
from PyQt5.QtCore import Qt

from .completer import MyCompleter


class RyucalcTextEdit(QTextEdit):
    def __init__(self, parent=None):
        QTextEdit.__init__(self, parent)

        QTextEdFontMetrics = QtGui.QFontMetrics(self.font())
        self.QTextEdRowHeight = QTextEdFontMetrics.lineSpacing()
        self.setFixedHeight(2 * self.QTextEdRowHeight)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # CONNECT WIDGET SIGNAL
        self.textChanged.connect(self.validateCharacters)

        self.completer_enabled = True
        self.completer = MyCompleter()
        self.completer.setWidget(self)
        self.completer.insertText.connect(self.insertCompletion)

    def insertCompletion(self, completion):
        tc = self.textCursor()
        extra = (len(completion) - len(self.completer.completionPrefix()))
        tc.movePosition(QTextCursor.MoveOperation.Left)
        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:])
        self.setTextCursor(tc)

        if self.completer.completionMode() == QCompleter.CompletionMode.PopupCompletion:
            self.completer.popup().hide()

    def focusInEvent(self, event):
        if self.completer:
            self.completer.setWidget(self)
        QTextEdit.focusInEvent(self, event)

    def keyPressEvent(self, ev):
        # Selection-less ctrl shortcuts
        if not self.textCursor().hasSelection():
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier) == Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_W:
                ev = PyQt5.QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.ControlModifier)
                super(RyucalcTextEdit, self).keyPressEvent(ev)
                return
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier) == Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_C:
                txt = self.toPlainText()
                QApplication.clipboard().setText(txt)
                return
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier) == Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_X:
                txt = self.toPlainText()
                self.setPlainText("")
                QApplication.clipboard().setText(txt)
                return

        # Completion
        tc = self.textCursor()

        if self.completer_enabled and ev.key() == QtCore.Qt.Key.Key_Tab and (self.completer.completionMode() != QCompleter.CompletionMode.PopupCompletion or self.completer.popup().isVisible()):
            self.completer.insertText.emit(self.completer.getSelected())
            # self.completer.setCompletionMode(QCompleter.PopupCompletion)
            return

        super(RyucalcTextEdit, self).keyPressEvent(ev)

        if self.completer_enabled:
            tc.select(QTextCursor.SelectionType.WordUnderCursor)
            cr = self.cursorRect()

            if len(tc.selectedText()) > 0:
                self.completer.setCompletionPrefix(tc.selectedText())

                if self.completer.completionMode() == QCompleter.CompletionMode.PopupCompletion:
                    popup = self.completer.popup()
                    popup.setCurrentIndex(self.completer.completionModel().index(0, 0))
                    cr.setWidth(self.completer.popup().sizeHintForColumn(0) + self.completer.popup().verticalScrollBar().sizeHint().width())

                self.completer.complete(cr)
            else:
                if self.completer.completionMode() == QCompleter.CompletionMode.PopupCompletion:
                    self.completer.popup().hide()

    def validateCharacters(self):
        badChars = ['\n']
        cursor = self.textCursor()
        curPos = cursor.position()
        for badChar in badChars:
            origText = self.toPlainText()
            for char in origText:
                if char in badChars:
                    cleanText = origText.replace(char, '')
                    self.blockSignals(True)
                    self.setText(cleanText)
                    self.blockSignals(False)
                    cursor.setPosition(curPos - 1)
        self.setTextCursor(cursor)