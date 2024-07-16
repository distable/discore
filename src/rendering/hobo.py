"""
This is DreamStudio hobo!
A fun retro crappy PyGame interface to do
you work in.
"""
import os
import time
from os import environ

from src.gui.QtUtil import move_window, resize_window

environ['PYQTGRAPH_QT_LIB'] = "PyQt5"

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.util import random_noise

from jargs import args

import jargs
import uiconf
from src.classes.paths import get_script_module_path
from src.gui.SelectorWindow import show_string_input, show_popup_selector
from src.party import tricks
from src.renderer import RendererState, RenderingRepetition, RenderMode, SeekImageMode
from src.rendering import hud
from src.classes import paths
from src.classes.Session import Session
from src.lib import corelib, loglib
from src.gui import ryucalc
from src import renderer
from src.gui.AudioPlayback import AudioPlayback
from src.rendering.HoboWindow import HoboWindow
from src.rendering.rendervars import RenderVars

QAPP_NAME = 'Discore'
QWIN_TITLE = 'Discore'

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
from pygame import draw

is_running = False

enable_hud = False
fps_stops = [1, 4, 6, 8, 10, 12, 24, 30, 50, 60]

discovered_actions = []
f_update = 0
f_displayed = None

sel_action_page = 0
sel_snapshot = -1

last_vram_reported = 0

surface: pygame.Surface | None = None
font: pygame.font.Font | None = None
copied_frame = 0
current_segment = -1
invalidated = True
colors = corelib.generate_colors(8, v=1, s=0.765)
rv: RenderVars | None = None
audio = AudioPlayback()
window: HoboWindow | None = None

window_separation = 0.55
window_padding = 0.025

keychord_map = {}  # chord|[chord, ...] -> name
keyfunc_map = {}  # name -> func

current_dragndrop_file = None
current_dragndrop_session = None
current_dragndrop_img = None

log = loglib.make_log('hobo')


def discover_actions():
    global discovered_actions
    discovered_actions = list(paths.iter_scripts())


def start():
    import pyqtgraph
    from PyQt5 import QtCore, QtGui
    from PyQt5.QtWidgets import QApplication

    global enable_hud
    global font
    global discovered_actions
    global surface
    global rv

    global window
    global is_running

    if is_running:
        return

    pyqtgraph.mkQApp(QAPP_NAME)
    discover_actions()
    discover_keymap()

    # Setup pygame renderer
    ryucalc.rv = rv

    pygame.init()
    audio.init()
    font = pygame.font.Font(paths.gui_font.as_posix(), 15)
    update_surface()

    # Setup Qt window
    window = HoboWindow(surface)
    window.resize(rv.w, rv.h)
    window.setWindowTitle(QWIN_TITLE)
    window.setWindowIcon(QtGui.QIcon(paths.hobo_icon.as_posix()))
    window.show()
    window.timeout_handlers.append(lambda: update())
    window.key_handlers.append(lambda k, ctrl, shift, alt: on_keydown(k, ctrl, shift, alt))
    window.dropenter_handlers.append(lambda f: on_dragndrop_enter(f))
    window.dropfile_handlers.append(lambda f: on_dragndrop(f))
    window.dropleave_handlers.append(lambda: on_dragndrop_leave())
    window.focusgain_handlers.append(lambda: on_focus_gained())
    window.focuslose_handlers.append(lambda: on_focus_lost())

    # Load the latest session if it's in the last 12 hours
    if renderer.session is None:
        latest_session, latest_session_mtime = paths.get_latest_session()
        if latest_session_mtime > time.time() - 60 * 60 * 12:
            renderer.change_session(Session(latest_session))
            log(f'Loaded latest session: {latest_session}')

    on_session_changed(renderer.session)

    # Center window on the screen
    move_window(window,
                window_anchor=(0.5, 1),
                screen_anchor=(0.5, window_separation))

    ryucalc.toggle()

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        return_code = QApplication.instance().exec()
        sys.exit(return_code)


def quit():
    # Quit
    from PyQt5.QtWidgets import QApplication
    QApplication.instance().quit()


def show_startup_menu():
    """
    The startup routine when launching the app without any argument.
    """

    if rv.session is not None:
        return

    def show_session_create_or_open():
        items = ['New session', 'Open session']

        if paths.has_last_session():
            f'Continue last session ({paths.fetch_last_session_name()})'

        index = show_popup_selector(items, window)
        if index == 0:
            return show_string_input()
        elif index == 1:
            return action_change_session()
        elif index is None:
            quit()

    name = show_session_create_or_open()
    while name is None:
        name = show_session_create_or_open()

    session = Session(name)
    renderer.change_session(session)


def update_surface():
    global surface
    global invalidated
    global f_displayed
    global frame_surface

    session = rv.session
    has_session = session is not None
    has_window = window is not None
    size_changed = False
    w = rv.w
    h = rv.h
    f = rv.f
    if has_window:
        if not has_session:
            # Window size is free
            w = window.width()
            h = window.height()
            if w != rv.w or h != rv.h:
                rv.w = w
                rv.h = h
                size_changed = True
        else:
            # Window size depends on session
            w = session.w
            h = session.h
            if w != window.width() or h != window.height():
                resize_window(window, w, h)
                size_changed = True

    # Create surface if it doesn't exist
    if surface is None or w != surface.get_width() or h != surface.get_height():
        surface = pygame.Surface((w, h))
        if has_window:
            window.set_surface(surface)

    def get_display_img():
        if has_session:
            im = rv.session.img
            if im is None:
                im = np.zeros((h, w, 3), dtype=np.uint8)
                im[:, :, 0] = 255
        else:
            im = np.zeros((h, w, 3), dtype=np.uint8)

        if has_session and not session.f_exists and renderer.mode == RenderMode.PAUSE:
            # Dim
            im = im * 0.75

        if -1 < sel_snapshot < len(hud.snaps):
            im = hud.snaps[sel_snapshot][1]

        # This can happen when we get bullshit in
        if not (im.shape >= (1, 1, 1)):
            return None

        return im

    changed = (renderer.invalidated or
               invalidated or
               f_displayed != f or
               size_changed or
               renderer.state == RendererState.RENDERING)
    if changed:
        img = get_display_img()
        try:
            img = np.swapaxes(img, 0, 1)  # cv2=HWC pygame=WHC
            # TODO if the surface is already the exact same size we should just blit for performance (array_to_surface)
            frame_surface = pygame.surfarray.make_surface(img)
            renderer.invalidated = False
            invalidated = False
            f_displayed = f
        except Exception as e:
            print(e)
            frame_surface = pygame.Surface((w, h))
            frame_surface.fill((255, 0, 255))

    surface.fill((0, 0, 255))
    surface.blit(frame_surface, (0, 0))


def update():
    global enable_hud
    global current_segment
    global copied_frame
    global sel_action_page, discovered_actions
    global f_displayed, last_vram_reported
    global f_update
    global surface

    session = rv.session

    if renderer.is_running():
        # TODO idk wtf this is for
        if renderer.enable_readonly and f_update % 120 == 0 and renderer.is_paused():
            refresh_session()

        update_surface()
        f_update += 1
        if renderer.is_paused() and f_update % 60 == 0:
            # Update VRAM using nvidia-smi
            try:
                last_vram_reported = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip()
                last_vram_reported = int(last_vram_reported)
            except:
                last_vram_reported = 0

        audio.flush_requests()

        # changed = renderer.invalidated or invalidated
        # if changed or f_displayed != session.f or renderer.state == RendererState.RENDERING:
        draw()

        if renderer.is_gui_dispatch():  # renderer.enable_dev and
            renderer.run_iter()
    else:
        log('Quitting ...')
        quit()


def draw():
    global f_update
    global enable_hud
    global current_segment
    global copied_frame
    global sel_action_page, discovered_actions
    global f_displayed, pilsurface, last_vram_reported
    global invalidated

    session = rv.session
    w = window.width()
    h = window.height()
    w2 = w / 2
    h2 = h / 2
    ht = font.get_height() + 2

    fps = 1 / max(renderer.loop.last_frame_dt, 1 / rv.fps)
    pad = 12
    right = w - pad
    left = pad
    top = pad
    bottom = h - pad
    playback_color = (255, 255, 255)
    if renderer.is_paused():
        playback_color = (0, 255, 255)

    if rv.session is None:
        draw_text("<no session>", w2, h2, (255, 255, 255), anchor=(0.5, 0.5))
        return

    f_last = session.f_last
    f_first = session.f_first
    f = session.f
    base_ul_offset = 5

    def draw_upper_right():
        # Paused / FPS
        if not renderer.is_paused() and fps > 1:
            draw_text(f"{int(fps)} FPS", right, top, playback_color, anchor=(1, 0))
        elif renderer.state == RendererState.RENDERING:
            draw_text(f"{int(fps)} FPS", right, top, playback_color, anchor=(1, 0))
        else:
            draw_text(f"Paused", right, top, playback_color, anchor=(1, 0))

        # VRAM
        draw_text(f"{last_vram_reported} MB", right, top + ht, (255, 255, 255), anchor=(1, 0))
        draw_text("Dev" if renderer.enable_dev else "", right, top + ht * 2, (255, 255, 255), anchor=(1, 0))
        draw_text("Trace" if loglib.print_trace else "", right, top + ht * 3, (255, 255, 255), anchor=(1, 0))
        draw_text("Bake" if renderer.enable_bake_on_script_reload else "", right, top + ht * 4, (255, 255, 255), anchor=(1, 0))

    def draw_top_center():
        if renderer.state == RendererState.RENDERING:
            color = (255, 255, 255)
            if renderer.render_repetition == RenderingRepetition.FOREVER:
                color = (255, 0, 255)
                draw_text("-- Rendering --", w2, top2, color, anchor=(0.5, 0.5))
            elif renderer.render_repetition == RenderingRepetition.ONCE:
                draw_text("-- Busy --", w2, top2, color, anchor=(0.5, 0.5))

            draw_text(f"{renderer.loop.n_rendered} frames", w2, top2 + ht, color, anchor=(0.5, 0.5))
            draw_text(f"{renderer.loop.n_rendered / rv.fps:.02f}s", w2, top2 + ht + ht, color, anchor=(0.5, 0.5))

            render_progressbar_y = top2 + ht + ht

            # Draw progress as a tiny bar under the rendering text
            if renderer.state == RendererState.RENDERING:
                render_progress = renderer.render_progress
                if render_progress >= 0:
                    bw = 64
                    bw2 = bw / 2
                    bh = 3
                    yoff = -2
                    pygame.draw.rect(surface, (0, 0, 0), (w2 - bw2 - 1, render_progressbar_y + ht + yoff, bw + 2, bh))
                    pygame.draw.rect(surface, (0, 255, 255), (w2 - bw2, render_progressbar_y + ht + yoff, bw * render_progress, bh))

    def draw_upper_left():
        if f <= f_last:
            draw_text(f"{f} / {f_last}", left, top, playback_color)
        else:
            draw_text(f"{f}", left, top, playback_color)

        total_seconds = f / rv.fps
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        draw_text(f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}", left, top + ht * 1, playback_color)

    def draw_lower_left():
        draw_text(session.name, left, bottom - ht * 1)
        draw_text(f"{session.w}x{session.h}", left, bottom - ht * 2)
        draw_text(f"{rv.fps} fps", left, bottom - ht * 3)
        # draw_text(renderer.script_name, left, bottom - ht * 2, col=(0, 255, 0))

    def draw_lower_right():
        if -1 < sel_snapshot < len(hud.snaps):
            snap_str = f"Snapshot {sel_snapshot}"
            snapshot = hud.snaps[sel_snapshot]
            snap_str += f" ({snapshot[0]})"

            draw_text(snap_str, right, bottom - ht * 1, anchor=(1, 0))

    def draw_dragndrop_overlay():
        if current_dragndrop_file is not None:
            if current_dragndrop_session is not None:
                draw_text(f"Session: {current_dragndrop_session.name}", w2, h2, (255, 255, 255), anchor=(0.5, 0.5), bg=(0, 0, 0))
            elif current_dragndrop_img is not None:
                draw_text(f"Image: {current_dragndrop_img}", w2, h2, (255, 255, 255), anchor=(0.5, 0.5), bg=(0, 0, 0))
            return

    def draw_lagindicator():
        if renderer.state == RendererState.RENDERING:
            # lagindicator = np.random.randint(0, 255, (8, 8), dtype=np.uint8)

            lagindicator = random_noise(np.zeros((8, 8)), 's&p', amount=0.1)
            lagindicator = (lagindicator * 255).astype(np.uint8)
            lagindicator = np.stack([lagindicator, lagindicator, lagindicator], axis=2)
            lagindicator_pil = Image.fromarray(lagindicator)
            lagindicator_surface = pygame.image.frombuffer(lagindicator_pil.tobytes(), lagindicator_pil.size, 'RGB')
            surface.blit(lagindicator_surface, (w - 8 - 2, h - 8 - 2))

    def draw_progress_bar():
        playback_thickness = 3
        segment_thickness = 3
        segment_offset = 2
        pygame.draw.rect(surface, (0, 0, 0), (0, 0, w, playback_thickness + 2))

        if f_last > 0:
            # Draw segment bars on top of the progress bar
            for i, t in enumerate(get_segments()):
                lo, hi = t
                progress_lo = lo / f_last
                progress_hi = hi / f_last
                color = colors[i % len(colors)]
                yo = 0
                if i == current_segment:
                    yo = playback_thickness

                x = w * progress_lo
                y = segment_offset + yo
                ww = w * (progress_hi - progress_lo)
                hh = segment_thickness

                pygame.draw.rect(surface, (0, 0, 0), (x + 1, y + 1, ww, hh))
                pygame.draw.rect(surface, (0, 0, 0), (x - 1, y + 1, ww, hh))
                pygame.draw.rect(surface, color, (x, y, ww, hh))

            # Draw a progress bar above the frame number
            progress = f / f_last

            pygame.draw.rect(surface, (0, 0, 0), (0, 0, w, playback_thickness))
            pygame.draw.rect(surface, (255, 255, 255) if not renderer.mode == RenderMode.PAUSE else (0, 255, 255), (0, 0, w * progress, playback_thickness))

            # Draw ticks
            major_ticks = 60 * rv.fps
            minor_ticks = 15 * rv.fps
            major_tick_height = playback_thickness - 1
            minor_tick_height = playback_thickness - 1
            major_tick_color = (193, 193, 193)
            minor_tick_color = (72, 72, 72)
            x = 0
            ppf = w / f_last
            while x < w:
                y = 0
                height = minor_tick_height
                color = minor_tick_color
                # print(session.w, session.f_last, ppf, x, minor_ticks * ppf, session.w // major_ticks, major_ticks, rv.fps, int(major_ticks*ppf))
                if int(x) % max(1, int(major_ticks * ppf)) == 0:
                    height = major_tick_height
                    color = major_tick_color

                pygame.draw.line(surface, color, (int(x), y), (int(x), y + height))
                pygame.draw.line(surface, color, (int(x) + 1, y), (int(x) + 1, y + height))
                x += minor_ticks * ppf

        ht = font.size("a")[1]

    def draw_hud():
        x = pad
        y = ht * base_ul_offset
        dhud = session.get_frame_data('hud', rv.f, clamp=True)
        if dhud:
            for i, row in enumerate(dhud):
                value = row[0]
                color = row[1]
                changed = False
                if i > 0:
                    last_value = dhud[i - 1][0]
                    changed = value != last_value

                # This is a hack, the color is stored in file and i want to change it for existing files..
                # if changed:
                #     color = [000, 255, 000]

                fragments = value.split('\n')
                for frag in fragments:
                    draw_text(frag, x, y, color)
                    y += ht

        # Draw signal bars
        s_bar_x = left
        s_bar_height = 32
        s_bar_width = 6
        s_bar_spacing = 8
        s_border_width = 2
        for s in hud.draw_signals:
            s_name = s.name
            s_min = s.min
            s_max = s.max
            s_values = rv._signals.get(s_name)
            if s.valid:
                v = s_values[rv.f]
                v_norm = (v - s_min) / (s_max - s_min)
                is_11 = s_min < 0
                if not is_11:
                    bar_h = s_bar_height * v_norm
                    bar_w = s_bar_width
                    bar_x = s_bar_x
                    bar_y = bottom - ht * 4 - s_bar_height
                    pygame.draw.rect(surface, (0, 0, 0), (bar_x - s_border_width,
                                                          bar_y - s_bar_height - s_border_width,
                                                          bar_w + s_border_width * 2,
                                                          s_bar_height + s_border_width * 2))
                    pygame.draw.rect(surface, (255, 255, 255), (bar_x, bar_y - bar_h, bar_w, bar_h))
                else:
                    vnorm_11 = (v_norm - 0.5) * 2
                    bar_x = s_bar_x
                    bar_w = s_bar_width
                    bar_h = s_bar_height * abs(vnorm_11)
                    bar_y_center = bottom - ht * 4 - s_bar_height
                    bar_y = bar_y_center

                    if vnorm_11 > 0:
                        bar_y -= s_bar_height * vnorm_11

                    pygame.draw.rect(surface, (0, 0, 0), (bar_x - s_border_width,
                                                          bar_y_center - s_bar_height - s_border_width,
                                                          bar_w + s_border_width * 2,
                                                          s_bar_height * 2 + s_border_width * 2))
                    pygame.draw.rect(surface, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h))
                    # Draw black tick line at halfway point
                    pygame.draw.rect(surface, (0, 0, 0), (bar_x, bar_y_center - 1, bar_w, 2))

                s_bar_x += s_bar_width
                s_bar_x += s_bar_spacing

        # draw_text(f"{rv.fps} fps", left, bottom - ht * 3)

    top2 = top + 6
    draw_progress_bar()
    draw_top_center()
    draw_upper_right()
    draw_upper_left()
    draw_lower_left()
    draw_lower_right()
    draw_dragndrop_overlay()
    if enable_hud:
        draw_hud()


def draw_text(s, x, y, col=(255, 255, 255), anchor=(0, 0), bg=None):
    if s is None: return
    size = font.size(s)
    x -= size[0] * anchor[0]
    y -= size[1] * anchor[1]

    if bg is not None:
        pygame.draw.rect(surface, bg, (x, y, size[0], size[1]))

    # Shadow
    text = font.render(s, False, (0, 0, 0))
    surface.blit(text, (x + -1, y + 0))
    surface.blit(text, (x + -1, y + -1))
    surface.blit(text, (x + 1, y + 1))
    surface.blit(text, (x + 0, y + -1))

    # Main
    text = font.render(s, False, col)
    surface.blit(text, (x, y))


def upfont(param):
    global font, fontsize
    fontsize += param
    font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)


def on_keydown(key, ctrl, shift, alt, action_whitelist=None):
    global invalidated
    from src.gui.QtUtil import key_to_string
    from PyQt5 import QtCore
    qkeys = QtCore.Qt.Key
    session = rv.session

    # print(f'keydown: {str_key}, {ctrl}, {shift}, {alt}')

    chord = ''
    if ctrl: chord += 'c-'
    if shift: chord += 's-'
    if alt: chord += 'a-'
    chord += key_to_string(key)

    if chord in keychord_map:
        actions = keychord_map[chord]
        for action_name in actions:
            if action_whitelist is not None and any([action_name.startswith(whitelist) for whitelist in action_whitelist]):
                continue

            if action_name in keyfunc_map:
                func = keyfunc_map[action_name]
                func()
            elif action_name in discovered_actions:
                func = discovered_actions[action_name]
                func()
            else:
                print(f'The keychord {chord}:{action_name} is defined by the user but does not exist in hobo!')

    # if key_mode == 'main':
    #     pass
    # elif key_mode == 'action':
    #     if key == qkeys.Key_Escape or key == qkeys.Key_W:
    #         key_mode = 'main'
    #     else:
    #         i = key - qkeys.Key_1
    #         if i in range(1, 9):
    #             action_slice = discovered_actions[sel_action_page:sel_action_page + 10]
    #             if 1 <= i <= qkeys.Key_9:
    #                 name, path = action_slice[i]
    #                 session = f"discore {session.dirpath} {name} "
    #
    #                 if shift:
    #                     session += f"--frames {segments_to_frames()}"
    #
    #                 os.popen(session)
    #                 key_mode = 'main'


def on_session_changed(session):
    if session is None:
        entries = [
            ('New session', action_new_session),
            ('Open session', action_change_session),
        ]
        if paths.has_last_session():
            entries.append(('Continue last session', action_open_last_session))
        window.set_buttons(entries)
    else:
        window.set_buttons()
        audio.discover_audio(root=session.dirpath)


def on_focus_lost():
    renderer.detect_script_every = 1


def on_focus_gained():
    renderer.requests.script_check = True
    renderer.detect_script_every = -1

    if renderer.enable_readonly:
        refresh_session()


def on_dragndrop_enter(files):
    global current_dragndrop_file, current_dragndrop_session, current_dragndrop_img
    file = Path(files[0])
    if file.is_dir():
        current_dragndrop_file = file
        current_dragndrop_session = Session(file, fixpad=True)
        current_dragndrop_img = None
    elif file.suffix in paths.image_exts:
        current_dragndrop_file = file
        current_dragndrop_session = None
        current_dragndrop_img = None


def on_dragndrop_leave():
    global current_dragndrop_file, current_dragndrop_session, current_dragndrop_img
    current_dragndrop_file = None
    current_dragndrop_session = None
    current_dragndrop_img = None


def on_dragndrop(files):
    file = Path(files[0])
    if file.is_dir():
        session = Session(file, fixpad=True)
        session.seek_min()

        # TODO on_session_changed
        if session.w and session.h:
            pygame.display.set_mode((session.w, session.h))

        renderer.change_session(session)
    elif file.suffix in paths.image_exts:
        renderer.change_image(file)


def show_session_selector(update_cache=False):
    items = []
    load_paths = paths.iter_session_paths(update_cache=update_cache)
    for file in load_paths:
        file = Path(file)
        items.append(f'{file.stem} (<i>{file.as_posix()}</i>)')
    index = show_popup_selector(items, window)
    if index is not None:
        selected = load_paths[index]
        return selected


def show_action_selector():
    # Show all functions in this file (globals) that start by 'action_'
    g = globals()

    items = {string: g[string] for string in g if string.startswith('action_')}

    # Add all discovered actions
    for name, path in discovered_actions:
        label = f'{name} (<i>{path}</i>)'
        items[label] = (name, path)

    item = show_popup_selector(items, window)
    if not item:
        return

    # if is callable
    if callable(item):
        item()
    elif isinstance(item, tuple):
        name, path = item
        import importlib
        actmod = importlib.import_module(get_script_module_path(name), package=name.split('/')[-1])
        f_getargs = getattr(actmod, 'get_args', None)
        f_action = getattr(actmod, 'action')

        argdefs = f_getargs() if f_getargs else []
        if isinstance(argdefs, list):
            for adef in argdefs:
                if isinstance(adef, (tuple, list)):
                    argname = adef[0]
                    argtype = adef[1]
                    argdefault = adef[2]
                    argdesc = adef[3]
                    argitems = []
                    if len(adef) >= 5:
                        argitems = adef[4]
                    it = show_popup_selector(argitems, window, title=argname, desc=argdesc)
                elif isinstance(adef, str):
                    # Get details from jargs
                    argp = jargs.argp  # an ArgumentParser with all the arguments configured
                    arg = argp._option_string_actions.get(adef, None) or \
                          argp._option_string_actions.get('--' + adef, None) or \
                          argp._option_string_actions.get('-' + adef, None)

                    argname = arg.dest
                    argtype = arg.type
                    argdefault = arg.default
                    argdesc = arg.help
                    it = show_string_input(window, title=argname, desc=argdesc, default=argdefault)
                else:
                    raise Exception(f'Invalid argdef: {adef}')

                if it is not None:
                    setattr(jargs.args, argname, it)

        setattr(jargs.args, 'session', rv.session.dirpath)

        result = None
        try:
            result = f_action(jargs.args)
        except:
            pass

        if isinstance(result, Session):
            renderer.change_session(result)


# region API

def segments_to_frames():
    # example:
    # return '30:88,100:200,3323:4000'

    return '-'.join([f'{s[0]}:{s[1]}' for s in get_segments()])


def get_segments():
    dat = rv.session.data
    if not 'segments' in dat:
        dat['segments'] = []

    return dat['segments']


def create_segment(off):
    global current_segment
    get_segments().append((rv.session.f, rv.session.f + off))
    current_segment = len(get_segments()) - 1
    rv.session.save_data()


def get_fps_stop(current, offset):
    stops = fps_stops
    pairs = list(zip(stops, stops[1:]))

    idx = stops.index(current)
    if idx >= 0:
        idx = max(0, min(idx + offset, len(stops) - 1))
        return stops[idx]

    for i, p in enumerate(pairs):
        a, b = p
        if a <= current <= b:
            return a if offset < 0 else b

    return current


def invalidate():
    global invalidated
    invalidated = True


def seek(frame):
    if renderer.seek(frame):
        ryucalc.on_hobo_seek(frame)


# endregion

# region Keymap

def discover_keymap():
    global keyfunc_map
    from PyQt5 import QtCore
    qkeys = QtCore.Qt.Key
    keyglobals = uiconf.__dict__
    for keyname, keyvalue in keyglobals.items():
        if not keyname.startswith('key_'): continue
        if isinstance(keyvalue, str):
            keyvalue = (keyvalue,)
        for chord in keyvalue:
            if chord not in keychord_map:
                keychord_map[chord] = []
            keychord_map[chord].append(keyname)

    funcglobals = globals()
    for funcname, func in funcglobals.items():
        if not funcname.startswith('key_'): continue
        keyfunc_map[funcname] = func


def key_snapshot_prev():
    # if key == qkeys.Key_Left and shift:
    global sel_snapshot
    n_snapshots = len(hud.snaps)
    if n_snapshots > 1:
        sel_snapshot = max(sel_snapshot - 1, -1)
        invalidate()


def key_snapshot_next():
    # if key == qkeys.Key_Right and shift:
    global sel_snapshot
    n_snapshots = len(hud.snaps)
    if n_snapshots > 1:
        sel_snapshot = min(sel_snapshot + 1, n_snapshots - 1)
        invalidate()


def key_set_last_frame():
    # if key == qkeys.Key_ParenRight:
    rv.session.f_last = rv.session.f
    rv.session.f_last_path = rv.session.det_frame_path(rv.session.f)
    rv.session.save_data()


def key_action_prev():
    global sel_action_page
    sel_action_page = max(0, sel_action_page - 10)


def key_action_next():
    global sel_action_page
    max_page = len(discovered_actions) // 10
    sel_action_page += 10
    sel_action_page = min(sel_action_page, max_page)


def do_keymap_dbg(key, ctrl, shift, alt):
    from PyQt5 import QtCore
    from src.rendering import dbg
    qkeys = QtCore.Qt.Key

    dbg_increment_offset = 0
    if ctrl: dbg_increment_offset = 1
    if shift: dbg_increment_offset = 2

    if key == uiconf.key_dbg_up:
        dbg.up(dbg_increment_offset)
    if key == uiconf.key_dbg_down:
        dbg.down(dbg_increment_offset)
    if key == uiconf.key_dbg_cycle_increment:
        dbg.cycle_increment()
    if key == qkeys.Key_1:
        dbg.select(0)
    if key == qkeys.Key_2:
        dbg.select(1)
    if key == qkeys.Key_3:
        dbg.select(2)
    if key == qkeys.Key_4:
        dbg.select(3)
    if key == qkeys.Key_5:
        dbg.select(4)
    if key == qkeys.Key_6:
        dbg.select(5)
    if key == qkeys.Key_7:
        dbg.select(6)
    if key == qkeys.Key_8:
        dbg.select(7)
    if key == qkeys.Key_9:
        dbg.select(8)
    if key == qkeys.Key_0:
        dbg.select(9)


def key_switch_session_with_cache_reload():
    action_change_session(True)


def key_switch_session():
    action_change_session(False)


def key_run_action():
    show_action_selector()


def key_toggle_dev_mode():
    if not renderer.state == RendererState.RENDERING:
        renderer.enable_dev = not renderer.enable_dev


def key_toggle_trace():
    loglib.print_trace = not loglib.print_trace
    loglib.print_gputrace = not loglib.print_gputrace

def key_toggle_bake_enabled():
    renderer.enable_bake_on_script_reload = not renderer.enable_bake_on_script_reload

def key_open_script_in_editor():
    """
    Open in the file system's script editor.
    """
    if rv.session is None: return False
    os.startfile(rv.session.res_script())


def key_open__in_terminal():
    """
    Open in the file system's script editor.
    """
    if rv.session is None: return False
    os.system(f'start "{paths.root.as_posix()}"')


def key_show_session_in_explorer():
    if rv.session is None: return False
    os.system(f'explorer.exe {rv.session.dirpath}')


def key_reload_session():
    if rv.session is None: return False
    s = Session(rv.session.dirpath)
    s.f = rv.session.f
    s.load_f()
    s.load_f_img()

    renderer.change_session(s)


def key_pause():
    if rv.session is None: return False
    if renderer.mode == RenderMode.PAUSE and rv.session.f >= rv.session.f_last:
        audio.desired_name = None
        renderer.seek(renderer.play.last_play_start_frame)
        renderer.set_play()
    else:
        renderer.toggle_pause()


def key_cancel():
    if rv.session is None: return False
    if renderer.mode != RenderMode.PAUSE:
        renderer.seek(renderer.play.last_play_start_frame)
        renderer.set_pause()


def key_seek_prev():
    if rv.session is None: return
    seek(rv.session.f - 1)


def key_seek_next():
    if rv.session is None: return
    seek(rv.session.f + 1)


def key_seek_prev_second():
    if rv.session is None: return
    seek(rv.session.f - rv.fps)


def key_seek_next_second():
    if rv.session is None: return
    seek(rv.session.f + rv.fps)


def key_seek_prev_percent():
    if rv.session is None: return
    seek(rv.session.f - int(rv.session.f_last * uiconf.hobo_seek_percent))


def key_seek_next_percent():
    if rv.session is None: return
    seek(rv.session.f + int(rv.session.f_last * uiconf.hobo_seek_percent))


def key_seek_last():
    if rv.session is None: return
    # Search for the next last frame (real file on disk)
    f = rv.session.f
    session = rv.session
    validity_state = session.det_f_exists(f + 1)
    while f < session.f_last:
        f += 1
        if session.det_f_exists(f) != validity_state:
            f -= 1
            break

    seek(f)


def key_seek_first():
    if rv.session is None: return
    # Search for the next first frame (real file on disk)
    seek(rv.session.f_first)
    f = rv.session.f
    session = rv.session
    validity_state = session.det_f_exists(f - 1)
    while f > max(0, session.f_first):
        f -= 1
        if session.det_f_exists(f) != validity_state:
            f += 1
            break
    seek(f)


def key_toggle_ryusig():
    if not window.hasFocus(): return
    ryucalc.toggle()


def key_fps_down():
    if rv.session is None: return
    rv.fps = get_fps_stop(rv.fps, -1)
    ryucalc.refresh()


def key_fps_up():
    if rv.session is None: return
    rv.fps = get_fps_stop(rv.fps, 1)
    ryucalc.refresh()


def key_copy_prompt():
    """
    Copy the current frame prompt to the clipboard.
    """
    if rv.session is None: return
    global copied_frame
    copied_frame = rv.session.img


def key_copy_frame():
    if rv.session is None: return
    global copied_frame
    copied_frame = rv.session.img


def key_paste_frame():
    if rv.session is None: return
    if copied_frame is not None:
        rv.session.img = copied_frame
        rv.session.save()
        rv.session.save_data()
        renderer.invalidated = True


def key_delete_frame():
    if rv.session is None: return
    if rv.session.delete_f():
        renderer.invalidated = True


def key_render_forever():
    if rv.session is None: return
    renderer.toggle_render(RenderingRepetition.FOREVER)


def key_render_once():
    if rv.session is None: return
    renderer.toggle_render(RenderingRepetition.ONCE)


def key_reload_script():
    if rv.session is None: return
    renderer.reload_script()

def key_reload_script_hard():
    if rv.session is None: return
    renderer.reload_script(hard=True)

def key_toggle_hud():
    if rv.session is None: return
    global enable_hud
    enable_hud = not enable_hud


def key_segment_set_start():
    if rv.session is None: return

    if len(get_segments()) and not rv.f > get_segments()[current_segment][1]:
        get_segments()[current_segment] = (rv.f, get_segments()[current_segment][1])
        rv.session.save_data()
    else:
        create_segment(50)


def key_segment_set_end():
    if rv.session is None: return
    if len(get_segments()) and not rv.f < get_segments()[current_segment][0]:
        get_segments()[current_segment] = (get_segments()[current_segment][0], rv.f)
        rv.session.save_data()
    else:
        create_segment(-50)


def key_seek_prev_segment():
    if rv.session is None: return
    indices = [i for s in get_segments() for i in s]
    indices.sort()
    # Find next value in indices that is less than session.f
    for i in range(len(indices) - 1, -1, -1):
        if indices[i] < rv.f:
            renderer.seek(indices[i])
            break


def key_seek_next_segment():
    if rv.session is None: return
    indices = [i for s in get_segments() for i in s]
    indices.sort()
    # Find next value in indices that is greater than session.f
    for i in range(len(indices)):
        if indices[i] > rv.f:
            renderer.seek(indices[i])
            break


def key_segment_select_next():
    if rv.session is None: return
    if not len(get_segments()): return
    global current_segment
    current_segment = min(len(get_segments()) - 1, current_segment + 1)
    renderer.seek(get_segments()[current_segment][0])


def key_segment_select_prev():
    if rv.session is None: return
    if not len(get_segments()): return
    global current_segment
    current_segment = max(0, current_segment - 1)
    renderer.seek(get_segments()[current_segment][0])


def key_play_segment():
    if rv.session is None: return
    if current_segment < len(get_segments()): return

    lo, hi = get_segments()[current_segment]
    rv.session.seek(lo)
    renderer.play_until = hi
    renderer.request_pause = False
    renderer.looping = True
    renderer.looping_start = lo


# endregion

# region Actions

def refresh_session():
    ses = rv.session
    nextfile = ses.det_frame_path(ses.f_last + 1)
    if os.path.exists(nextfile):
        tmp_f = ses.f
        was_last = ses.f >= ses.f_last
        rv.session.load()
        rv.session.f = tmp_f
        if was_last:
            ses.seek(ses.f_last)


def action_new_session():
    name = show_string_input()
    if name is None:
        return None

    session = Session(name)
    renderer.change_session(session)
    return session


def action_open_last_session():
    if not paths.has_last_session():
        return
    name = paths.fetch_last_session_name()
    ses = Session(name)
    renderer.change_session(ses)


def action_change_session(update_cache=False):
    selected = show_session_selector(update_cache=update_cache)
    if selected is not None:
        renderer.change_session(Session(selected))


def action_quit():
    quit()


def action_rediscover_sessions():
    pass


def action_render_from_time():
    it = show_string_input(window, title="Timed Render", desc="Enter a time to start rendering at. This will set a new 'f_first' for the render.")
    if it:
        t = paths.parse_time_to_seconds(it)
        if t is not None:
            t_frame = rv.to_frame(t)
            rv.session.f_first = t_frame
            rv.session.f_first_path = rv.session.det_f_first_path()
            renderer.seek(t_frame, clamp=False, with_pause=True)
            renderer.toggle_render(RenderingRepetition.FOREVER)


# endregion

def check_keychord(chord, inputs):
    """
    key_chord is a string in the vim format eg. 'c-r', 'c-s', 'c-s-r'
    where c is ctrl, s is shift, a is alt
    and r is the key
    """
    from src.gui.QtUtil import key_to_string

    key = inputs[0]
    ctrl = inputs[1]
    shift = inputs[2]
    alt = inputs[3]
    chord_tokens = chord.split('-')
    str_key = key_to_string(key)
    for i, t in enumerate(chord_tokens):
        t = t.lower()
        if t == str_key and i == len(chord_tokens) - 1: return True
        if t == 'c' and not ctrl: return False
        if t == 's' and not shift: return False
        if t == 'a' and not alt: return False


def on_script_baked():
    ryucalc.refresh()