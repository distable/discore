import re

import numpy as np
from PyQt5.QtCore import QObject, QEvent, pyqtSignal

from src import renderer
from . import QtUtil, RyucalcLib
from .Highlighter import Highlighter
from .QtUtil import keyevent_to_args
from .RyucalcPlot import RyucalcPlot, TimeMode, x_from_seconds, SignalEntry
from .RyucalcTextEdit import RyucalcTextEdit
from .editing.bezier_curve import BezierCurveItem
from ..rendering import hobo
from ..rendering.rendervars import Signal

pyexec = exec  # this gets replaced when you import * from pyqt

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from pyqtgraph import *
import pyqtgraph as pg

rv = renderer.rv
C_INIT_EVAL = 'chg drum bass other'

plot_colors = RyucalcLib.generate_colors(20)




def find_signal(word):
    envdic = RyucalcLib.get_evalenv()
    found_value = None
    if word in envdic:
        found_value = envdic[word]
    elif word in envdic['rv'].__dict__:
        found_value = envdic['rv'].__dict__[word]

    if isinstance(found_value, np.ndarray):
        return found_value
    elif isinstance(found_value, Signal):
        return found_value.data
    elif isinstance(found_value, (int, float)):
        return np.full(envdic['rv'].n, found_value)
    else:
        return None




class RyucalcWindow(QMainWindow):
    sigResize = pyqtSignal(object)
    sigFocusIn = pyqtSignal(object)
    sigFocusOut = pyqtSignal(object)
    sigKeyPress = pyqtSignal(object)
    sigToggleWindow = pyqtSignal()

    def __init__(self, parent, fps=24, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.installEventFilter(self)


        # Control
        self.request_reload = False

        # State
        self.signals = []
        self.norm_mode = False

        # Set window to the bottom half of the screen
        root_layout = QVBoxLayout()
        root_layout.setSpacing(0)
        root_widget = QWidget()
        root_widget.setLayout(root_layout)

        self.setCentralWidget(root_widget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet('background-color: black')
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Evaluation Input
        # self.winput = QLineEdit()
        self.textedit_widget = RyucalcTextEdit()
        self.textedit_widget.setStyleSheet("background-color: #1b1818; color: #8a8585; font: 9pt 'Input'")
        self.textedit_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.textedit_widget.completer_enabled = False
        self.textedit_widget.completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
        self.textedit_highlighter = Highlighter()  # theme: https://atelierbram.github.io/syntax-highlighting/atelier-schemes/plateau/
        self.textedit_highlighter.addForeground(r'[\*\#\+\-\\/]', '#8a8585')  # Operators
        self.textedit_highlighter.addForeground(r"[\(\)\[\]]", '#8a8585')  # Symbols
        self.textedit_highlighter.addForeground(r"[+-]?[0-9]+[.][0-9]*([e][+-]?[0-9]+)?", '#b45a3c')  # Numbers
        self.textedit_highlighter.addForeground(";", '#7e7777')
        self.textedit_highlighter.functions = '#5485b6'
        self.textedit_highlighter.ndarrays = '#b45a3c'
        self.textedit_highlighter.addToDocument(self.textedit_widget.document())

        self.plot = RyucalcPlot()
        self.plot.showGrid(True, True, 0.75)
        self.plot.signals = self.signals
        root_layout.addWidget(self.textedit_widget)
        root_layout.addWidget(self.plot)

        QApplication.clipboard().dataChanged.connect(self.on_clipboard_change)
        QApplication.clipboard().selectionChanged.connect(self.on_selection_change)

        def on_win_resize(ev):
            # text_size = self.plot.coord_indicator_label.fontMetrics().boundingRect(self.plot.coord_indicator_label.text())
            # self.coord_indicator_label.setGeometry(
            #     int(self.size().width() / 2 - text_size.width() / 2),
            #     24,
            #     text_size.width() + 4,
            #     text_size.height() * 2 + 4)
            # # self.wcoord_main.adjustSize()
            pass

        # Events
        self.plot.sigKeyPress.connect(self.on_plot_keypress)
        self.textedit_widget.textChanged.connect(self.on_text_changed)
        self.sigResize.connect(on_win_resize)
        self.sigKeyPress.connect(self.on_win_keypress)
        self.sigFocusIn.connect(self.on_win_focus)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / 60))  # 60 fps timer

        self.set_signals([
            np.linspace(0, 1, 2000),
            np.linspace(1, 0, 2000),
            np.linspace(0.5, 0.5, 2000)
        ])

        self.textedit_widget.setText(C_INIT_EVAL)

    def focus(self):
        self.plot.scene().setFocus()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        match event.type():
            case QtCore.QEvent.Type.Resize:
                self.sigResize.emit(event)
            case QtCore.QEvent.Type.WindowActivate:
                self.sigFocusIn.emit(event)
            case QtCore.QEvent.Type.WindowDeactivate:
                self.sigFocusOut.emit(event)

        return super().eventFilter(obj, event)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)

    def reload_signals(self):
        self.set_text(self.textedit_widget.toPlainText())

    def reload(self):
        self.reload_signals()
        envdic = RyucalcLib.get_evalenv()
        self.textedit_widget.completer.qlist.setStringList(envdic.keys())
        self.textedit_highlighter.datadict = envdic

    def tick(self):
        if not self.isVisible(): return

        self.update()

        if self.textedit_widget.alignment() != QtCore.Qt.AlignmentFlag.AlignCenter:
            self.textedit_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        if not renderer.is_paused():
            self.centralWidget().repaint()

        self.plot.tick()

    def set_signals(self,
                    new_signals: np.array,
                    new_names: list[str] = None):
        """
        :param new_signals: list of signals to display
        :param new_names: list of names to display
        :return:
        """
        self.plot.clear()
        self.signals.clear()

        # Keep only  (N,) arrays
        new_signals = [x for x in new_signals if isinstance(x, np.ndarray) and len(x.shape) == 1]

        # Ensure rv.n large enough
        for signal in new_signals:
            if signal.shape[0] > rv.n:
                rv.n = signal.shape[0]

        self.plot.rebuild_time()

        if self.norm_mode:
            ymax = np.max([np.max(np.abs(x)) for x in new_signals])
            new_signals = [x / ymax for x in new_signals]

        for i, signal in enumerate(new_signals):
            signal = np.nan_to_num(signal)
            signal = np.pad(signal, (0, self.plot.time_array.shape[0] - signal.shape[0]), 'edge')
            color = plot_colors[i % len(plot_colors)]
            item = self.plot.plot(self.plot.time_array, signal, label=f"Signal {i}", pen=pg.mkPen(color, width=1.5))

            name = f"signal_{i}"
            if new_names is not None and i < len(new_names):
                name = new_names[i]

            self.signals.append(SignalEntry(i, name, signal, item, color))

        self.plot.removeItem(self.plot.target_item)
        self.plot.addItem(self.plot.target_item, ignoreBounds=True)

        # TODO remove this, its just for testing
        # self.plot.addItem(BezierCurveItem([
        #     (0, 0),
        #     (1, 1),
        #     (2, 0),
        #     (3, 2)
        # ]))

    def set_text(self, text:str):
        """
        Change the displayed text, and update the signals.
        """
        text = text.replace(",\n", ";")

        strings = []
        found_signals = []
        formulas = text.split(';')

        envdic = RyucalcLib.get_evalenv()

        # Regex to check if the string is a list of words separated by space
        word_list_regex = re.compile(r'^[a-zA-Z_]\w*(\s+[a-zA-Z_]\w*)*$')
        if word_list_regex.match(text):
            # LIST MODE
            text = text.replace(',', ' ')
            words = text.split(' ')
            found_signals = []
            for word in words:
                found_signal = find_signal(word)
                if found_signal is not None:
                    found_signals.append(found_signal)

            strings.extend(words)
        else:
            # EVAL MODE
            try:
                for f in formulas:
                    envdic['t'] = self.plot.time_array
                    v = eval(f, envdic)

                    if isinstance(v, Signal):
                        strings.append(v.name)
                        found_signals.append(v.data)
                    elif isinstance(v, np.ndarray):
                        strings.append(f)
                        found_signals.append(v)

            except Exception as e:
                print(e)
                pass

        if len(found_signals) >= 1:
            self.set_signals(found_signals, strings)
            return True
        else:
            self.set_signals([], [])

        return False

    # region Event Handlers

    def on_text_changed(self):
        text = self.textedit_widget.toPlainText()
        self.set_text(text)

    def on_clipboard_change(self):
        if QApplication.focusWindow() is not None: return
        txt = QApplication.clipboard().text(QtGui.QClipboard.Mode.Clipboard)
        try:
            eval(txt, RyucalcLib.get_evalenv())
            self.textedit_widget.setText(txt)
        except:
            pass

    def on_selection_change(self):
        if QApplication.focusWindow() is not None: return
        txt = QApplication.clipboard().text(QtGui.QClipboard.Mode.Selection)
        try:
            eval(txt, RyucalcLib.get_evalenv())
            self.textedit_widget.setText(txt)
        except:
            # Match all words with \w+
            symbols = ""
            import re
            for match in re.findall(r'\w+', txt):
                if find_signal(match) is not None:
                    symbols += match + " "

            if len(symbols) > 0:
                # Note: this will also dispatch text changed event
                self.textedit_widget.setText(symbols)


    def on_win_keypress(self, ev):
        key, ctrl, shift, alt = keyevent_to_args(ev)
        scale = 0.33

        if key == QtCore.Qt.Key.Key_F1:
            from src.gui import ryucalc
            ryucalc.toggle()
        elif key == QtCore.Qt.Key.Key_Escape:
            if self.plot.hasFocus():
                self.wplot.clearFocus()
                self.winput.setFocus()
            if self.textedit_widget.hasFocus():
                from src.gui import ryucalc
                ryucalc.toggle()
        # ctrl-shift-minus and ctrl-shift-plus to zoom in/out vertically
        elif ctrl and shift and key == QtCore.Qt.Key.Key_Underscore:
            self.plot.scaleBy(y=1 - scale)
        elif ctrl and shift and key == QtCore.Qt.Key.Key_Plus:
            self.plot.scaleBy(y=1 + scale)
        # ctrl-minus and ctrl-plus to zoom in/out horizontally
        elif ctrl and key == QtCore.Qt.Key.Key_Minus:
            self.plot.scaleBy(x=1 - scale)
        elif ctrl and key == QtCore.Qt.Key.Key_Equal:
            self.plot.scaleBy(x=1 - scale)

    def on_win_focus(self):
        # hobo.raise_()
        pass

    def on_plot_keypress(self, ev):
        global time_mode
        key, ctrl, shift, alt = QtUtil.keyevent_to_args(ev)

        if ev.key() == QtCore.Qt.Key.Key_Control or ev.key() == QtCore.Qt.Key.Key_Shift:
            return

        if hobo.on_keydown(*keyevent_to_args(ev), action_whitelist=['key_seek']):
            return

        if ev.key() == QtCore.Qt.Key.Key_F2:
            time_mode = TimeMode((time_mode.value + 1) % len(TimeMode))
            self.set_signals([*self.signals])
            return

        if ev.key() == QtCore.Qt.Key.Key_Space:
            # Play at mouse position
            # ----------------------------------------
            hobo.audio.desired_name = self.signals[0].name
            # x = x_from_seconds(self.plot.mouse_x_seconds)
            # self.plot.change_info_pos(x, self.mouse_y)
            renderer.seek(rv.to_frame(self.plot.infopos_x), clamp=False)
            renderer.toggle_pause()
            return

        if ev.key() == QtCore.Qt.Key.Key_Backslash:
            self.plot.set_y_span_auto()
            self.plot.set_x_center(x_from_seconds(rv.session.t))
            self.plot.set_x_span(x_from_seconds(6))
            return

        if ev.key() == QtCore.Qt.Key.Key_Tab:
            self.norm_mode = True
            self.reload_signals()
            return

        if ev.key() == QtCore.Qt.Key.Key_Escape:
            self.sigToggleWindow.emit()

        if ev.key() == QtCore.Qt.Key.Key_M and ctrl and shift:
            s = rv.get_timestamp_string('chapters')

            # Copy to clipboard
            QApplication.clipboard().setText(s)
            return

        else:
            self.textedit_widget.setFocus()
            self.textedit_widget.keyPressEvent(ev)

    # endregion

