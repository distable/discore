import os

from PyQt5 import QtCore
from PyQt5.QtCore import Qt

keymap = {}
for key, value in vars(Qt).items():
    if isinstance(value, Qt.Key):
        keymap[value] = key.partition('_')[2]

modmap = {
    Qt.KeyboardModifier.ControlModifier: Qt.Key.Key_Control,
    Qt.KeyboardModifier.AltModifier: Qt.Key.Key_Alt,
    Qt.KeyboardModifier.ShiftModifier: Qt.Key.Key_Shift,
    Qt.KeyboardModifier.MetaModifier: Qt.Key.Key_Meta,
    Qt.KeyboardModifier.GroupSwitchModifier: Qt.Key.Key_AltGr,
    Qt.KeyboardModifier.KeypadModifier: Qt.Key.Key_NumLock,
}


def move_window(win, window_anchor, screen_anchor, window_padding=0):
    from PyQt5 import QtGui
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    sx = screen.width() * screen_anchor[0]
    sy = screen.height() * screen_anchor[1]
    wx = win.width() * window_anchor[0]
    wy = win.height() * window_anchor[1]
    if os.name == 'posix':
        win.move(int(sx - wx),
                 int(sy - wy - window_padding * screen.height()))
    elif os.name == 'nt':
        win.move(int(sx - wx),
                 int(sy - wy))

def center_window(win):
    move_window(win, (0.5, 0.5), (0.5, 0.5))

def resize_window(win, width, height):
    cx = win.x() + win.width() / 2
    cy = win.y() + win.height() / 2
    win.resize(width, height)
    win.move(int(cx - width / 2), int(cy - height / 2))

def keyevent_to_args(event, *, ctrl=False, shift=False, alt=False):
    has_ctrl = (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) != QtCore.Qt.KeyboardModifier.NoModifier
    has_shift = (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier) != QtCore.Qt.KeyboardModifier.NoModifier
    has_alt = (event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) != QtCore.Qt.KeyboardModifier.NoModifier

    return event.key(), has_ctrl or ctrl, has_shift or shift, has_alt or alt


def keyevent_to_string(event):
    sequence = []
    for modifier, text in modmap.items():
        if event.modifiers() & modifier:
            sequence.append(text)
    key = keymap.get(event.key(), event.text())
    if key not in sequence:
        sequence.append(key)
    return '-'.join(sequence)


def key_to_string(key):
    # Qt::Key
    if key in keymap:
        return keymap[key].lower()

    if key < 0x01000000:
        return chr(key).lower()

    raise ValueError('Unknown key: %s' % key)


def get_screen_width():
    from PyQt5 import QtGui
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    return screen.width()

def get_screen_height():
    from PyQt5 import QtGui
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    return screen.height()