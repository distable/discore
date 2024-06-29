from PIL import Image, ImageDraw, ImageFont
from yachalk import chalk

from src.lib import loglib
from src.lib.loglib import printkw
from src.classes import paths
from src.classes.convert import save_png
from src.rendering.rendervars import RenderVars

log = loglib.make_log('tricks')
logerr = loglib.make_logerr('tricks')

snaps = []
rows = []  # list[tuple[str, tuple[int, int, int]]]
rows_tmp = []
draw_signals = []
rv:RenderVars = None

def is_printing_enabled():
    from src import renderer
    return renderer.is_cli()


class DrawSignal:
    def __init__(self, name):
        self.name = name
        self.min = 0
        self.max = 0
        self.valid = False

def update_draw_signals():
    """
    Update the min and max of all signals
    """
    for s in draw_signals:
        signal = rv._signals.get(s.name)
        if signal is not None:
            s.min = signal.min()
            s.max = signal.max()
            s.valid = True
        else:
            s.valid = False

def set_draw_signals(*names):
    """
    Set the signals to draw
    """
    draw_signals.clear()
    for name in names:
        draw_signals.append(DrawSignal(name))

def snap(name, img=None):
    if img is None:
        img = rv.session.img.copy()
    snaps.append((name, img))

def hud(*args, tcolor=(255, 255, 255), **kwargs):
    # Turn args and kwargs into a string like 'a1 a2 x=1 y=2'
    # Format numbers to 3 decimal places (if they are number)
    s = ''
    for a in args:
        s += loglib.value_to_print_str(a)
        s += ' '

    # TODO auto-snap if kwargs is ndarray hwc

    for k, v in kwargs.items():
        s += f'{loglib.value_to_print_str(k)}='
        s += loglib.value_to_print_str(v)
        s += ' '

    maxlen = 80
    s = '\n'.join([s[i:i + maxlen] for i in range(0, len(s), maxlen)])

    if is_printing_enabled():
        printkw(**kwargs, chalk=chalk.magenta, print_func=log)
    rows.append((s, tcolor))


def clear():
    """
    Clear the HUD
    """
    snaps.clear()
    rows.clear()

def save(session, hud):
    save_png(hud,
             session.det_current_frame_path('prompt_hud').with_suffix('.png'),
             with_async=True)


def to_pil(session):
    """
    Add a HUD and save/edit current in hud folder for this frame
    """
    # Create a new black pil extended vertically to fit an arbitrary string
    rows_tmp.clear()
    rows_tmp.extend(rows)
    rows.clear()

    lines = len(rows_tmp)
    # count the number of \n in the work_rows (list of tuple[str,_])
    for row in rows_tmp:
        lines += row[0].count('\n')

    w = session.w
    h = session.h
    padding = 12
    font = ImageFont.truetype(str(paths.plug_res / 'vt323.ttf'), 15)
    tw, ht = font.getsize_multiline("foo")

    new_pil = Image.new('RGB', (w + padding * 2, h + ht * lines + padding * 2), color=(0, 0, 0))

    # Draw the old pil on the new pil at the top
    if session.image:
        new_pil.paste(session.image, (padding, padding))

    # Draw the arbitrary string on the new pil at the bottom
    draw = ImageDraw.Draw(new_pil)
    x = padding
    y = h + padding * 1.25
    for i, row in enumerate(rows_tmp):
        s = row[0]
        color = row[1]
        fragments = s.split('\n')
        for frag in fragments:
            draw.text((x, y), frag, font=font, fill=color)
            y += ht

    return new_pil




# ----------------------

def hud_base():
    hud(chg=rv.chg, cfg=rv.cfg, seed=rv.seed)
    hud_ccg()
    hud(prompt=rv.prompt)


def hud_ccg():
    ccgs = []
    iccgs = []
    ccgas = []
    ccgbs = []
    i = 1
    while f'ccg{i}' in rv:
        ccg = rv[f'ccg{i}']
        iccg = rv[f'iccg{i}']
        ccga = rv[f'ccg{i}_a']
        ccgb = rv[f'ccg{i}_b']
        ccgs.append(ccg)
        iccgs.append(iccg)
        ccgas.append(ccga)
        ccgbs.append(ccgb)
        i += 1
    hud(ccgs=tuple(ccgs))
    hud(iccgs=tuple(iccgs))
    hud(ccgas=tuple(ccgas))
    hud(ccgbs=tuple(ccgbs))


def snap_ccg_imgs():
    print("SNAP_CCG_IMGS")
    # std.save_guidance_imgs((rv.ccg3_img, rv.ccg2_img, rv.ccg1_img))
    i = 1
    while f'ccg{i}_img' in rv:
        img = rv[f'ccg{i}_img']
        snap(f'ccg{i}_img', img)
        i += 1
