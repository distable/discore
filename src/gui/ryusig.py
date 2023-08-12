import math
import re
import types
from colorsys import hsv_to_rgb
from enum import Enum

import numpy as np
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QScreen

from src import renderer
from src.party import maths
from .Highlighter import Highlighter
from .QtUtils import get_keypress_args
from .RyuPlotWidget import RyuPlotWidget
from .RyuPlotWindow import RyuPlotWindow
from .RyuTextEdit import RyuTextEdit
from ..renderer import RendererState, RenderMode
from ..rendering import hobo

pyexec = exec  # this gets replaced when you import * from pyqt

rv = renderer.rv
session = renderer.session

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import *
from pyqtgraph import *
import pyqtgraph as pg

pg.setConfigOptions(antialias=False, useOpenGL=False, useNumba=True, segmentedLineMode='on', mouseRateLimit=30)

rgb_to_hex = lambda tuple: f"#{int(tuple[0] * 255):02x}{int(tuple[1] * 255):02x}{int(tuple[2] * 255):02x}"


def generate_colors(n):
    golden_ratio_conjugate = 0.618033988749895
    h = 0
    ret = []
    for i in range(n):
        h += golden_ratio_conjugate
        ret.append(rgb_to_hex(hsv_to_rgb(h, 0.825, 0.915)))

    return ret


plot_colors = generate_colors(20)

C_INIT_EVAL = 'chg drum bass other'


# Plot -------------------------------------
# TODO text popup to jump/center to a specific X value
# Textbox ----------------------------------
# TODO support plotting static numbers by wrapping in ndarray and padding to framecount
# TODO select text to display subsection
# TODO grow selection to parentheses
# TODO jump through numbers
# TODO cycle numbers with ctrl-up/down


def on_t_selected_ryusig(t):
    from src import renderer
    renderer.seek_t(t, pause=False)


def on_t_selected_renderer(t):
    app.set_playback_t(t)


def on_script_loaded():
    app.request_reload = True

def on_start_playback():
    app.request_reload = True


def init():
    from src import renderer
    global initialized
    global app

    app = RyusigApp()

    app.init_qapp()
    app.init_qwindow()

    # Set position on the screen to bottom half (use setGeometry)
    app.win.hide()  # hidden by default on startup
    app.win.update()

    renderer.on_t_changed.append(on_t_selected_renderer)
    renderer.on_script_loaded.append(on_script_loaded)
    renderer.on_start_playback.append(on_start_playback)
    on_script_loaded()

    initialized = True


class TimeMode(Enum):
    Seconds = 0
    Frames = 1

time_mode = TimeMode.Seconds

class RyusigApp:
    def __init__(self, fps=24):
        self.mousepos = Point(0, 0)
        self.time_array = None
        rv.n = None

        self.keypress_handlers = []

        # Control
        self.request_quit = False
        self.request_reload = False

        # State
        self.mouse_signal_index = -1
        self.mouse_x_seconds = 0.0
        self.mouse_y = 0.0
        self.clines = []
        self.cnames = []
        self.csignals = []
        self.csignals_min = []
        self.csignals_max = []

        # UI
        # ----------------------------------------
        self.win = None
        self.winput = None

        # Features
        # ----------------------------------------
        self.norm_mode = False
        self.last_xrange = 0
        self.last_yrange = 0


    def init_qapp(self):
        pg.mkQApp("Ryusig")

    def init_qwindow(self):
        self.win = RyuPlotWindow()
        self.win.resize(1280, 720)
        # Set window to the bottom half of the screen
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        from src.rendering import hobo
        size = 1, 1 - hobo.window_separation
        anchor = 0, 1
        self.win.setGeometry(int(screen.width() * anchor[0]),
                             int(screen.height() * anchor[1]),
                             int(screen.width() * size[0]),
                             int(screen.height() * size[1]))

        vstack = QVBoxLayout()
        vstack.setSpacing(0)

        wmain = QWidget()
        wmain.setLayout(vstack)
        self.win.setCentralWidget(wmain)
        self.win.setContentsMargins(0, 0, 0, 0)
        self.win.setStyleSheet('background-color: black')
        vstack.setContentsMargins(0, 0, 0, 0)

        # QShortcut(QtGui.QKeySequence("Ctrl+s"), wmain, self.on_shortcut_save)
        # QShortcut(QtGui.QKeySequence("Ctrl+r"), wmain, self.on_shortcut_reload)

        # Evaluation Input
        # self.winput = QLineEdit()
        self.winput = RyuTextEdit()
        self.winput.setStyleSheet("background-color: #1b1818; color: #8a8585; font: 9pt 'Input'")
        self.winput.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.winput.completer_enabled = False
        self.winput.completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)

        # https://atelierbram.github.io/syntax-highlighting/atelier-schemes/plateau/
        self.highlighter = Highlighter()
        self.highlighter.addForeground(r'[\*\#\+\-\\/]', '#8a8585')  # Operators
        self.highlighter.addForeground(r"[\(\)\[\]]", '#8a8585')  # Symbols
        self.highlighter.addForeground(r"[+-]?[0-9]+[.][0-9]*([e][+-]?[0-9]+)?", '#b45a3c')  # Numbers
        self.highlighter.addForeground(";", '#7e7777')
        self.highlighter.functions = '#5485b6'
        self.highlighter.ndarrays = '#b45a3c'
        self.highlighter.addToDocument(self.winput.document())
        self.vplot = RyuPlotWidget()
        self.wplot = self.vplot.getPlotItem()
        self.wplot.showGrid(True, True, 0.75)
        self.vbplot = self.wplot.vb
        vstack.addWidget(self.winput)
        vstack.addWidget(self.vplot)

        # Main coord
        self.wcoord_main = QLabel()
        self.wcoord_main.setStyleSheet("background-color: none; color: white")
        self.wcoord_main.setParent(self.vplot)
        self.wcoord_main.setText("x=0\ny=0")
        self.wcoord_main.setGeometry(0, 0, 300, 200)
        self.wcoord_main.adjustSize()
        self.wcoord_main.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.wcoord_main.show()
        self.wcoord_main.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        QApplication.clipboard().dataChanged.connect(self.on_clipboard_change)
        QApplication.clipboard().selectionChanged.connect(self.on_selection_change)

        def on_win_resize(ev):
            text_size = self.wcoord_main.fontMetrics().boundingRect(self.wcoord_main.text())
            self.wcoord_main.setGeometry(
                    int(self.win.size().width() / 2 - text_size.width() / 2),
                    24,
                    text_size.width() + 4,
                    text_size.height() * 2 + 4)
            # self.wcoord_main.adjustSize()

        # t-line
        self.tline = InfiniteLine(5, pen=mkTlinePen())
        self.wplot.vb.addItem(self.tline, ignoreBounds=True)
        # Ploot coord
        self.wcoord = TextItem('y=0')
        # self.wtargetu = TargetItem(movable=False, size=10)
        # self.wtargetu.setPen(mkPen(width=2, color='black'))
        # self.wtargetu.hide()
        self.wtarget = TargetItem(movable=False, size=7)
        self.wtarget.hide()
        self.wcoord.hide()
        self.vbplot.addItem(self.wcoord, ignoreBounds=True)
        # self.vbplot.addItem(self.wtargetu, ignoreBounds=True)
        self.vbplot.addItem(self.wtarget, ignoreBounds=True)
        # Events
        self.winput.textChanged.connect(self.on_text_changed)
        self.vplot.sigKeyPress.connect(self.on_plot_keypress)
        self.vplot.sceneObj.sigMouseMoved.connect(self.on_plot_move)
        self.vplot.sceneObj.sigMouseClicked.connect(self.on_plot_click)
        self.win.sigResize.connect(on_win_resize)
        self.win.sigKeyPress.connect(self.on_win_keypress)
        self.win.sigFocusIn.connect(self.on_win_focus)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_update)
        self.timer.start(int(1000 / 60))  # 60 fps timer

        self.set_signals([
            np.linspace(0, 1, 2000),
            np.linspace(1, 0, 2000),
            np.linspace(0.5, 0.5, 2000)
        ])
        self.winput.setText(C_INIT_EVAL)

        # Setup the main window
        # self.win.show()


    def set_signals(self, new_signals, new_names=None):
        self.wplot.clear()
        self.csignals.clear()
        self.csignals_min.clear()
        self.csignals_max.clear()
        self.clines.clear()
        self.cnames.clear()

        # Update Max
        # --------------------------------------------------
        rv.n = max(np.array([x.shape[0] for x in new_signals]))

        # Update time
        # --------------------------------------------------
        n = rv.n - 1
        if time_mode == TimeMode.Seconds:
            n /= rv.fps
        self.time_array = np.linspace(0, n, rv.n)

        if self.norm_mode:
            ymax = np.max([np.max(np.abs(x)) for x in new_signals])
            new_signals = [x / ymax for x in new_signals]

        for i, signal in enumerate(new_signals):
            signal = np.nan_to_num(signal)
            signal = np.pad(signal, (0, self.time_array.shape[0] - signal.shape[0]), 'edge')

            col = plot_colors[i % len(plot_colors)]
            line = self.wplot.plot(self.time_array, signal, label=f"Signal {i}", pen=pg.mkPen(col, width=1.5))

            self.csignals.append(signal)
            self.csignals_min.append(np.min(signal))
            self.csignals_max.append(np.max(signal))
            self.clines.append(line)

            if new_names is not None and i < len(new_names):
                self.cnames.append(new_names[i])

        self.vbplot.removeItem(self.wtarget)
        self.vbplot.addItem(self.wtarget, ignoreBounds=True)

        # self.update_yrange()

        # renderer.audio.update(self.csignals, self.cnames)

    def eval(self, text):
        text = text.replace(",\n", ";")

        strings = []
        found_signals = []
        formulas = text.split(';')

        envdic = get_env()

        # Regex to check if the string is a list of words separated by space
        word_list_regex = re.compile(r'^[a-zA-Z]+(\s[a-zA-Z]+)+\s*$')
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
                    envdic['t'] = self.time_array
                    v = eval(f, envdic)
                    if isinstance(v, np.ndarray):
                        strings.append(f)
                        found_signals.append(v)

            except Exception as e:
                print(e)
                pass

        if len(found_signals):
            self.set_signals(found_signals, strings)
            return True
        return False

    def set_mouse_pos(self, x, y):
        x = self.clip_x(x)
        x = snap_x_to_frame(x)
        f = x_to_frame(x)
        t = x_to_seconds(x)

        self.mouse_y = y
        self.mouse_x_seconds = t
        self.set_playback_t(t)

        # Update the label and target
        i_signal, signal = self.find_nearest_signal(x, y)
        if signal is not None:
            f = np.clip(f, 0, signal.shape[0] - 1)
            v = signal[f]

            self.wcoord.show()
            self.wcoord.setText(f"y={v:.2f}")
            self.wcoord.setPos(x, v)

            self.wcoord_main.show()
            if time_mode == TimeMode.Seconds:
                self.wcoord_main.setText(f"x={x:.2f}\ny={v:.2f}")
            elif time_mode == TimeMode.Frames:
                self.wcoord_main.setText(f"x={x:.0f}\ny={v:.2f}")

            # self.wtargetu.setPen(mkPen(color=plot_colors[inear]))
            # self.wtargetu.setPos(t, v)
            # self.wtargetu.show()
            # self.wtarget.setPen(mkPen(color='white', brush=mkBrush(color='black')))
            self.wtarget.setPos(x, v)
            self.wtarget.show()

            # renderer.audio.select(signal_name)
            self.mouse_signal_index = i_signal

    def get_mouse_signal(self):
        if self.mouse_signal_index is not None:
            return self.csignals[self.mouse_signal_index]
        else:
            return None

    def find_nearest_signal(self, x=None, y=None):
        index = find_nearest_signal_index(self.csignals, x, y)
        if index == -1:
            index = self.mouse_signal_index
        if index is not None and index < len(self.csignals):
            index = clamp(index, 0, len(self.csignals) - 1)
            signal = self.csignals[index]
            return index, signal
        else:
            return None, None

    def refresh_signals(self):
        self.eval(self.winput.toPlainText())

    def set_playback_t(self, t_start):
        self.tline.setPos(seconds_to_x(t_start))
        self.tline.setPen(mkTlinePen())

    def on_text_changed(self):
        if self.eval(self.winput.toPlainText()):
            pass

    def on_clipboard_change(self):
        if QApplication.focusWindow() is not None: return
        txt = QApplication.clipboard().text(QtGui.QClipboard.Mode.Clipboard)
        try:
            eval(txt, get_env())
            self.winput.setText(txt)
        except:
            pass

    def on_selection_change(self):
        if QApplication.focusWindow() is not None: return
        txt = QApplication.clipboard().text(QtGui.QClipboard.Mode.Selection)
        try:
            eval(txt, get_env())
            self.winput.setText(txt)
        except:
            # Match all words with \w+
            symbols = ""
            import re
            for match in re.findall(r'\w+', txt):
                if find_signal(match) is not None:
                    symbols += match + " "

            if len(symbols) > 0:
                # Note: this will also dispatch text changed event
                self.winput.setText(symbols)


    def on_plot_move(self, ev):
        mp = self.vbplot.mapSceneToView(ev)
        x = mp.x()
        y = mp.y()
        self.mousepos = mp

        if ev is not None and renderer.state == RendererState.READY and renderer.mode == RenderMode.PAUSE:
            self.set_mouse_pos(x, y)
            renderer.seek(x_to_frame(x), clamp=False)

    def on_plot_click(self, ev):
        if ev.button() == 4:
            self.wplot.autoRange()

    def on_win_keypress(self, ev):
        key, ctrl, shift, alt = get_keypress_args(ev)
        scale = 0.33

        if key == QtCore.Qt.Key.Key_F1:
            toggle()
        elif key == QtCore.Qt.Key.Key_Escape:
            if self.winput.hasFocus():
                self.winput.clearFocus()
                self.vplot.setFocus()
            else:
                toggle()
                # self.wplot.clearFocus()
                # self.winput.setFocus()
        # ctrl-shift-minus and ctrl-shift-plus to zoom in/out vertically
        elif ctrl and shift and key == QtCore.Qt.Key.Key_Underscore:
            self.vbplot.scaleBy(y=1-scale)
        elif ctrl and shift and key == QtCore.Qt.Key.Key_Plus:
            self.vbplot.scaleBy(y=1+scale)
        # ctrl-minus and ctrl-plus to zoom in/out horizontally
        elif ctrl and key == QtCore.Qt.Key.Key_Minus:
            self.vbplot.scaleBy(x=1-scale)
        elif ctrl and key == QtCore.Qt.Key.Key_Equal:
            self.vbplot.scaleBy(x=1-scale)

    def on_win_focus(self):
        # self.center_on_x(seconds_to_x(rv.t), 6)
        hobo.win.raise_()
        pass

    def on_plot_keypress(self, ev):
        # Proxy to text input
        # if ev.key() == QtCore.Qt.Key.Key_Left:
        #     init_iwav()
        #     renderer.audio.play_marker()
        # elif ev.key() == QtCore.Qt.Key.Key_Right:
        #     init_iwav()
        #     renderer.audio.play_marker()
        global time_mode
        audio = renderer.audio

        ctrl = ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        shift = ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier

        if ev.key() == QtCore.Qt.Key.Key_Control or ev.key() == QtCore.Qt.Key.Key_Shift:
            return

        # if ev.key() == QtCore.Qt.Key.Key_Up:
        #     audio.set_wav(-1)
        # elif ev.key() == QtCore.Qt.Key.Key_Down:
        #     audio.set_wav(1)
        if hobo.handle_seek(ev.key()):
            pass
        elif ev.key() == QtCore.Qt.Key.Key_F2:
            time_mode = TimeMode((time_mode.value + 1) % len(TimeMode))
            self.set_signals([*self.csignals])
        elif ev.key() == QtCore.Qt.Key.Key_Space:
            audio.desired_name = self.cnames[0]
            self.set_mouse_pos(seconds_to_x(self.mouse_x_seconds), self.mouse_y)
            renderer.seek(to_frame(self.mouse_x_seconds), clamp=False)
            renderer.toggle_pause()
        elif ev.key() == QtCore.Qt.Key.Key_Backslash:
            self.encompass_on_y()
            self.center_on_x(seconds_to_x(session.t))
            self.range_on_x(seconds_to_x(6))
        elif ev.key() == QtCore.Qt.Key.Key_Tab:
            self.norm_mode = True
            self.refresh_signals()
        elif ev.key() == QtCore.Qt.Key.Key_Escape:
            toggle()
        elif ev.key() == QtCore.Qt.Key.Key_M and ctrl and shift:
            s = rv.get_timestamp_string('chapters')

            # Copy to clipboard
            QApplication.clipboard().setText(s)

        else:
            self.winput.setFocus()
            self.winput.keyPressEvent(ev)

    def clip_x(self, x):
        return np.clip(x, 0, self.time_array[-1])

    def on_timer_update(self):
        if self.request_quit:
            self.win.close()
            self.request_quit = False
            # QApplication.instance().quit()
            return

        if self.request_reload:
            self.request_reload = False
            self.refresh_signals()
            envdic = get_env()
            self.winput.completer.qlist.setStringList(envdic.keys())
            self.highlighter.datadict = envdic

        self.win.update()

        if self.winput.alignment() != QtCore.Qt.AlignmentFlag.AlignCenter:
            self.winput.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        if not renderer.is_paused():
            f = renderer.session.f
            t = to_seconds(f)
            x = frame_to_x(f)
            # self.set_mouse_pos(x, self.mouse_y)
            # self.set_playback_t(t)

            # signal = self.get_mouse_signal()
            # if signal is not None:
            #     self.center_on_x(x)
            self.win.centralWidget().repaint()
        # else:
        #     self.update_yrange()

    def update_yrange(self):
        xrange, yrange = self.get_window_range()

        if len(self.csignals) > 0 and (xrange != self.last_xrange or yrange != self.last_yrange):
            self.encompass_on_y()

        self.last_xrange = xrange
        self.last_yrange = yrange

    def get_window_range(self):
        viewrange = self.get_viewrange()
        xmax, xmin, ymax, ymin = self.get_window_bounds()
        xrange = xmax - xmin
        yrange = ymax - ymin
        return xrange, yrange

    def get_window_bounds(self):
        xmin = self.get_vxrange()[0]
        xmax = self.get_vxrange()[1]
        ymin = self.get_vyrange()[0]
        ymax = self.get_vyrange()[1]
        return xmin, xmax, ymin, ymax

    def get_xmin(self):
        return self.get_window_bounds()[0]

    def get_xmax(self):
        return self.get_window_bounds()[1]

    def get_ymin(self):
        return self.get_window_bounds()[2]

    def get_ymax(self):
        return self.get_window_bounds()[3]

    def get_vxrange(self):
        return self.get_viewrange()[0]

    def get_vyrange(self):
        return self.get_viewrange()[1]

    def get_viewrange(self):
        viewrange = self.wplot.getViewBox().state['viewRange']
        return viewrange

    def encompass_on_y(self, only_viewed=True):
        xrange, yrange = self.get_window_range()

        if only_viewed:
            viewed_xrange_f_min = x_to_frame(self.get_xmin())
            viewed_xrange_f_max = x_to_frame(self.get_xmax())
            viewed_xrange_f_min = np.clip(viewed_xrange_f_min, 0, len(self.time_array) - 1)
            viewed_xrange_f_max = np.clip(viewed_xrange_f_max, 0, len(self.time_array) - 1)
        else:
            viewed_xrange_f_min = 0
            viewed_xrange_f_max = len(self.time_array) - 1

        ymax = np.max([np.max(s[viewed_xrange_f_min:viewed_xrange_f_max]) for s in self.csignals])
        ymin = np.min([np.min(s[viewed_xrange_f_min:viewed_xrange_f_max]) for s in self.csignals])

        # self.wplot.setLimits(yMax=ymax, yMin=ymin)
        padding = 0.1
        self.wplot.setRange(yRange=(ymin - yrange * padding, ymax + yrange * padding), padding=0)

    def center_on_x(self, x, newrange=None):
        xrange, yrange = self.get_window_range()
        self.wplot.setRange(xRange=(x - xrange / 2, x + xrange / 2), padding=0)
        if newrange:
            self.range_on_x(newrange)
        app.set_mouse_pos(seconds_to_x(x), app.mouse_y)

    def range_on_x(self, newrange):
        xmin, xmax, ymin, ymax = self.get_window_bounds()
        xcenter = (xmin + xmax) / 2
        self.wplot.setRange(xRange=(xcenter - newrange / 2, xcenter + newrange / 2), padding=0)

def get_env():
    envdic = {}

    def add_object(obj):
        envdic.update(obj.__dict__)

    def add_module(mod):
        envdic.update(mod2dic(mod))

    def add_dict(dic):
        envdic.update(dic)

    add_module(math)
    add_module(np)
    add_module(maths)
    add_object(renderer.script)
    add_object(rv)
    add_dict(rv.signals)
    envdic['rv'] = rv

    return envdic

def x_to_seconds(x):
    if time_mode == TimeMode.Seconds:
        return x
    elif time_mode == TimeMode.Frames:
        return x / rv.fps

def x_to_frame(x):
    f = 0
    if time_mode == TimeMode.Seconds:
        f = int(x * rv.fps)
    elif time_mode == TimeMode.Frames:
        f = int(x)

    return clamp(f, 0, rv.n - 1)

def seconds_to_x(t):
    if time_mode == TimeMode.Seconds:
        return t
    elif time_mode == TimeMode.Frames:
        return int(t * rv.fps)

def frame_to_x(t):
    if time_mode == TimeMode.Seconds:
        return t / rv.fps
    elif time_mode == TimeMode.Frames:
        return int(t)


def snap_x_to_frame(x):
    if time_mode == TimeMode.Seconds:
        f = int(x * rv.fps)
        return to_seconds(f)
    else:
        return int(x)

def find_nearest_signal_index(signals, x, y):
    f = x_to_frame(x)
    idx = -1
    mindist = sys.maxsize
    for i, signal_arr in enumerate(reversed(signals)):
        clamped_f = np.clip(f, 0, len(signal_arr) - 1)
        v = signal_arr[clamped_f]

        d = abs(v - y)
        if d < mindist:
            idx = i
            mindist = d

    if idx == -1:
        return -1

    return len(signals) - idx - 1

def find_signal(word):
    envdic = get_env()
    found_value = None
    if word in envdic:
        found_value = envdic[word]
    elif word in envdic['rv'].__dict__:
        found_value = envdic['rv'].__dict__[word]

    if isinstance(found_value, np.ndarray):
        return found_value
    elif isinstance(found_value, (int, float)):
        return np.full(envdic['rv'].n, found_value)
    else:
        return None

def mkTlinePen():
    if renderer.audio.is_playing():
        return mkPen('green', width=2, style=QtCore.Qt.PenStyle.SolidLine)
    else:
        return mkPen('white', width=2, style=QtCore.Qt.PenStyle.DotLine)


def mod2dic(module):
    if isinstance(module, dict):
        return module
    elif isinstance(module, types.ModuleType):
        return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    elif isinstance(module, list):
        dics = [mod2dic(m) for m in module]
        return {k: v for d in dics for k, v in d.items()}
    else:
        return module.__dict__


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def clamp01(v):
    return min(max(v, 0), 1)


def ilerp(lo, hi, v):
    ret = clamp01((v - lo) / (hi - lo))
    if math.isnan(ret):
        return 0
    return ret

initialized = False
app: RyusigApp | None = None

def toggle():
    if not initialized:
        init()

    from src.rendering import hobo

    w = app.win
    if w.isVisible():
        app.win.hide()
        hobo.win.showNormal()
        hobo.win.raise_()
        hobo.win.setWindowState(QtCore.Qt.WindowState.WindowActive)
    else:
        app.win.show()
        app.win.raise_()
        app.win.setWindowState(QtCore.Qt.WindowState.WindowActive)
        app.vplot.setFocus()
        app.center_on_x(frame_to_x(session.f))
        # hobo.win.hide()

def refresh():
    if not initialized: return
    app.refresh_signals()

def on_hobo_seek(frame):
    if not initialized: return
    app.center_on_x(frame_to_x(frame))

to_frame = rv.to_frame
to_seconds = rv.to_seconds
