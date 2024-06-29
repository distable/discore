import math
from colorsys import hsv_to_rgb

from src import renderer
from . import QtUtil, RyucalcPlot
from .RyucalcWindow import RyucalcWindow

pyexec = exec  # this gets replaced when you import * from pyqt

from PyQt5 import QtCore
from pyqtgraph import *
import pyqtgraph as pg

rv = renderer.rv

pg.setConfigOptions(antialias=False,
                    useOpenGL=False,
                    useCupy=True,
                    # useNumba=True,
                    segmentedLineMode='auto',
                    # mouseRateLimit=30
                    )



initialized = False
calc:RyucalcWindow = None


def init():
    from src import renderer
    global initialized
    global calc

    # Build the Qt instance
    # ----------------------------------------
    pg.mkQApp("Ryusig")
    calc = RyucalcWindow(None)
    QtUtil.resize_window(calc, QtUtil.get_screen_width(), 720)
    QtUtil.move_window(calc, (0.5, 1), (0.5, 1))
    calc.hide()  # hidden by default on startup
    calc.update()

    renderer.on_t_changed.append(on_t_selected_renderer)
    renderer.on_script_loaded.append(on_script_loaded)
    renderer.on_start_playback.append(on_start_playback)
    on_script_loaded()

    initialized = True


def toggle():
    if not initialized:
        init()
        set_invisible()
        return

    if calc.isVisible():
        set_invisible()
    else:
        set_visible()
        # hobo.win.hide()


def set_invisible():
    from src.rendering import hobo
    calc.hide()
    hobo.window.showNormal()
    hobo.window.raise_()
    hobo.window.setWindowState(QtCore.Qt.WindowState.WindowActive)


def set_visible():
    calc.show()
    calc.raise_()
    calc.setWindowState(QtCore.Qt.WindowState.WindowActive)
    calc.focus()
    if rv.session is not None:
        calc.plot.set_x_center(RyucalcPlot.x_from_frame(rv.session.f))


def refresh():
    if not initialized: return
    calc.reload_signals()


def on_hobo_seek(frame):
    if not initialized: return
    if not calc.isVisible(): return
    calc.plot.set_x_center(RyucalcPlot.x_from_frame(frame))




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
    renderer.seek_t(t)  # TODO we used to request a pause seek


def on_t_selected_renderer(t):
    calc.plot.set_playline_t(t)


def on_script_loaded():
    calc.request_reload = True


def on_start_playback():
    calc.request_reload = True




def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def clamp01(v):
    return min(max(v, 0), 1)


def ilerp(lo, hi, v):
    ret = clamp01((v - lo) / (hi - lo))
    if math.isnan(ret):
        return 0
    return ret

