from PyQt6 import QtCore


def get_keypress_args(event):
    has_ctrl = event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
    has_shift = event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
    has_alt = event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier

    return event.key(), has_ctrl, has_shift, has_alt
