import asyncio
import logging
from enum import Enum
from typing import TypeVar, Optional, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.application.application import _CombinedRegistry
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding.key_processor import KeyProcessor
from prompt_toolkit.layout import HSplit, VSplit, Float, Dimension, FormattedTextControl
from prompt_toolkit.layout.containers import FloatContainer, HorizontalAlign, ConditionalContainer, WindowAlign
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.widgets import Frame, TextArea, Box, Label

import jargs
from src.deploy.AppButton import AppButton
from src.deploy.ButtonList import ButtonList
from src.deploy.DiscoRemote import DiscoRemote, DiscoRemoteView, SSHConnectionState
from . import deploy_constants
from .Spacer import Spacer
from .VastInstance import VastInstance
from .deploy_utils import forget, invalidate, make_header_text, scroll_to_end, is_scrolled_to_bottom
from .ui_vars import *

log = logging.getLogger("DeployUI")

_T = TypeVar("_T")
just_started = True


async def main():
    with patch_stdout(app):
        # FIX: this is a patch for prompt_toolkit which doesn't allow changing global keybindings after the global app has been instantiated
        app.key_processor = KeyProcessor(_CombinedRegistry(app))

        change_state(UIState.MAIN)
        await app.run_async()


class UIState(Enum):
    NONE = 0
    DIALOG = 1
    MAIN = 2
    OFFERS = 3
    MANAGE_INSTANCE = 4


class RemoteList(ButtonList):
    """
    Displays DiscoRemoteViews
    """

    def __init__(self):
        super().__init__(data=["Empty!"],
                         hidden_headers=['_remote'])
        self.instances = []
        # self.headers = ['index', 'cuda', 'model', 'price']

    @property
    def has_data(self):
        return len(self.data) > 0 and isinstance(self.data[0], DiscoRemoteView)

    async def add_instance(self, remote: DiscoRemote):
        self.data.append(await DiscoRemoteView.from_remote(remote))
        self.update()
        self.enable_confirm = True

    async def fetch_instances(self):
        self.enable_confirm = False
        self.data = [(None, "Loading...")]
        get_app().invalidate()

        vdatas = await vast.fetch_instances()

        self.data = [await (await rman.get_remote(vdata)).to_view() for vdata in vdatas]
        self.update()

        self.enable_confirm = True
        get_app().invalidate()

        global just_started
        if just_started and self.has_data:
            just_started = False
            it = self.data[0]
            await manage_layout.change(it._remote)
            change_state(UIState.MANAGE_INSTANCE)

    def update(self):
        if not self.data:
            self.data = ["No instances found. Press enter to create !"]
            self.enable_confirm = False


class OfferList(ButtonList):
    def __init__(self, handler):
        super().__init__(data=[(None, "Empty!")], handler=handler)
        self.offers = []

    async def fetch_offers(self):
        self.data = [(None, "Loading...")]
        get_app().invalidate()

        self.offers = await vast.fetch_offers()
        self.data = self.offers
        self._process_data()

        self.sort_column(self._cached_headers.index('price'))  # hehe

        get_app().invalidate()


# region Key bindings
def k_c_c(event):
    app.exit()


def k_tab(event):
    event.app.layout.focus_next()


def k_s_tab(event):
    event.app.layout.focus_previous()


# Again as separate functions
def k_up(event):
    event.app.layout.focus_previous()


def k_down(event):
    event.app.layout.focus_next()


# endregion

async def on_offer_confirm(offer):
    """
    Create new instance when an offer is selected
    """
    if not offer:
        return

    # await task and 3 seconds, both at once
    change_state_dialog("Create", "Creating ...")
    forget(vast.create_instance(offer.id))  # TODO There is a bug, for some reason this never ends.
    await asyncio.sleep(2)

    # change_state_dialog("Create", f"Instance created successfully.")
    # await asyncio.sleep(2)

    change_state(UIState.MAIN)


async def on_remote_confirm(view):
    """
    Update the status display when an instance is selected
    """
    if not isinstance(view, DiscoRemoteView):
        change_state(UIState.OFFERS)
        return

    vdata = view.vdata
    if vdata:
        status_display.text = f"Selected Instance:\nID: {vdata.id}\nStatus: {vdata.status}\nIP: {vdata.ip}"
    else:
        status_display.text = "No instance selected"

    change_state_dialog("Connecting ...", f"Connecting to instance root@{view.ip}:{view.port} (instance #{view.id})...")

    await manage_layout.change(vdata)

    change_state(UIState.MANAGE_INSTANCE)


def change_state_dialog(title, text):
    change_state(UIState.DIALOG)
    app.layout = create_dialog_layout(title, text)


def change_state(new_state):
    global state
    if state == new_state:
        return

    if state == UIState.MANAGE_INSTANCE:
        manage_layout.stop_refresh_task()

    state = new_state

    if state == UIState.MAIN:
        forget(remote_list.fetch_instances())
        app.layout = create_main_layout()
        app.layout.focus(remote_list)
    elif state == UIState.OFFERS:
        forget(offer_list.fetch_offers())
        app.layout = create_offers_dialog()
        app.layout.focus(offer_list)
    elif state == UIState.MANAGE_INSTANCE:
        manage_layout.session = jargs.get_discore_session().name
        app.layout = manage_layout.window
        app.layout.focus(app.layout.current_window)

        manage_layout.start_refresh_task()


def create_dialog_layout(title, text):
    return Layout(
        FloatContainer(
            content=Window(),
            floats=[
                Float(
                    Frame(
                        title=title,
                        body=Box(
                            HSplit([
                                Window(FormattedTextControl(text), align=WindowAlign.CENTER)
                                # AppButton("Close", handler=lambda: change_state(UIState.MAIN), key='escape')
                            ], width=Dimension(preferred=99999999999)),
                            padding=1
                        )
                    ),
                    top=2, left=2, right=2, bottom=2
                )
            ]
        )
    )


def create_main_layout():
    return Layout(
        HSplit([
            VSplit([
                Frame(remote_list, title="Instances"),
                Frame(status_display, title="Status")
            ]),
            VSplit([
                new_instance_button,
                refresh_button,
                quit_button,
            ], align=HorizontalAlign.CENTER, padding=3)
        ])
    )


def create_offers_dialog():
    return Layout(
        FloatContainer(
            content=Window(),
            floats=[
                Float(
                    Frame(HSplit([
                        offer_list,
                        back_button
                    ], width=Dimension(preferred=99999999999)), title="Select an offer"),
                    top=2, left=2, right=2, bottom=2
                )
            ]
        )
    )


class ManageLayout(Layout):
    def __init__(self):
        super().__init__(Window())

        self.txt_logs = None
        self.txt_session = None
        self.txt_toolbar = None
        self.remote: Optional[DiscoRemote] = None
        self.window = None
        self._last_status = None
        self._last_balance = None
        self._refresh_task = None

        self.txt_toolbar = TextArea(multiline=False,
                                    height=1,
                                    style="bg:#CCCCCC #000000",
                                    read_only=True)
        self.txt_session = TextArea(multiline=False, height=1)
        self.txt_logs = TextArea(scrollbar=True, wrap_lines=True, read_only=True)

        self.update_window()

    @property
    def vdata(self):
        return self.remote.vdata if self.remote else None

    @property
    def id(self):
        return self.vdata.id if self.vdata else None

    def update_content(self):
        if self._last_balance is not None:
            self.txt_toolbar.text = f'Balance: {self._last_balance:.02f}$'
        else:
            self.txt_toolbar.text = "Balance: Loading..."
        if self.remote:
            self.txt_session.text = self._last_status.work_session
            if self.remote.ssh:
                old = self.txt_logs.text
                new = self.remote.ssh.logs.text
                if old != new:
                    self.txt_logs.text = self.remote.ssh.logs.text
                    if not is_scrolled_to_bottom(self.txt_logs):
                        scroll_to_end(self.txt_logs)
        invalidate()

    def update_window(self):
        @Condition
        def is_connected():
            return self.remote and self.remote.connection_state == SSHConnectionState.CONNECTED

        @Condition
        def can_deploy():
            return self.remote and self.remote.can_deploy(self.txt_session.text)

        @Condition
        def is_running():
            return self.remote and self._last_status and self.vdata.status == "running"

        @Condition
        def is_comfy_running():
            return self.remote and self._last_status and self._last_status.is_comfy_running

        @Condition
        def is_discore_running():
            return self.remote and self._last_status and self._last_status.is_discore_running

        log.debug(f"Updating window for instance {self.id} (balance: {self._last_balance}, status: {self._last_status}")
        self.update_content()

        left_col = [
            Label("Commands"),
            # AppButton("Refresh1", self._refresh, key='r'),
            Spacer(2),
            AppButton("Refresh", self._refresh, key='r'),
            Spacer(2),
            ConditionalContainer(AppButton("Start", self._start, key='s'), ~is_running),
            ConditionalContainer(AppButton("Stop", self._stop, key='t'), is_running),
            ConditionalContainer(AppButton("Connect", self._connect, key='c'), ~is_connected),
            ConditionalContainer(AppButton("Disconnect", self._disconnect, key='C'), is_connected),
            ConditionalContainer(HSplit([
                ConditionalContainer(AppButton("Shell", self._shell, key='h'), is_running),
                ConditionalContainer(AppButton("Mount", self._mount, key='m'), is_running),
            ]), is_running),
            Spacer(1),
            ConditionalContainer(HSplit([
                self.txt_session,
                AppButton("Deploy", self._deploy, key='d'),
                AppButton("Send fast", self._send_fast, key='1'),
                AppButton("Send slow", self._send_slow, key='2'),
                AppButton("Pip Upgrades", self._pipupgrades, key='u'),
                ConditionalContainer(AppButton("Start Comfy", self._start_comfy, key='3'), ~is_comfy_running),
                ConditionalContainer(AppButton("Stop Comfy", self._start_comfy, key='3'), is_comfy_running),
                ConditionalContainer(AppButton("Start Discore", self._start_discore, key='4'), ~is_discore_running),
                ConditionalContainer(AppButton("Stop Discore", self._stop_discore(), key='4'), is_discore_running),
                Spacer(),
            ]), is_connected),
            AppButton("Destroy", self._destroy, key='D'),
            Spacer(1),
            AppButton("Back", handler=lambda: change_state(UIState.MAIN), key='escape'),
        ]

        kb = KeyBindings()

        @kb.add("pagedown", is_global=True)
        def scroll_page_down(event) -> None:
            """
            Scroll page down. (Prefer the cursor at the top of the page, after scrolling.)
            """
            w = self.txt_logs.window
            b = self.txt_logs.buffer

            if w and w.render_info:
                # Scroll down one page.
                line_index = max(w.render_info.last_visible_line(), w.vertical_scroll + 1)
                w.vertical_scroll = line_index

                b.cursor_position = b.document.translate_row_col_to_index(line_index, 0)
                b.cursor_position += b.document.get_start_of_line_position(
                    after_whitespace=True
                )

        @kb.add("pageup", is_global=True)
        def scroll_page_up(event) -> None:
            """
            Scroll page up. (Prefer the cursor at the bottom of the page, after scrolling.)
            """
            w = self.txt_logs.window
            b = self.txt_logs.buffer

            if w and w.render_info:
                # Put cursor at the first visible line. (But make sure that the cursor
                # moves at least one line up.)
                line_index = max(
                    0,
                    min(w.render_info.first_visible_line(), b.document.cursor_position_row - 1),
                )

                b.cursor_position = b.document.translate_row_col_to_index(line_index, 0)
                b.cursor_position += b.document.get_start_of_line_position(
                    after_whitespace=True
                )

                # Set the scroll offset. We can safely set it to zero; the Window will
                # make sure that it scrolls at least until the cursor becomes visible.
                w.vertical_scroll = 0

        main_container = Box(
            padding_left=4,
            padding_right=4,
            padding_top=2,
            padding_bottom=2,
            body=HSplit([
                self.txt_toolbar,
                Spacer(),
                VSplit([
                    HSplit(left_col, padding=0),
                    Frame(self.txt_logs, title="Logs", height=Dimension(preferred=9999999999999)),
                ], padding=4),
            ]),
            key_bindings=kb
        )

        app.key_bindings.add
        self.window = Layout(FloatContainer(
            content=Window(),
            floats=[
                Float(
                    left=2, right=2, top=2, bottom=2,
                    content=Frame(
                        title="Manage Instance",
                        body=main_container),
                )
            ]
        ))

        invalidate()

    async def change(self, vdata_or_remote: Union[VastInstance, DiscoRemote, int, None]):
        if self.remote is not None:
            self.remote.ssh.logs.handlers.remove(self._on_ssh_log_line)

        match vdata_or_remote:
            case id if isinstance(id, int) and not self.id == id:
                self.remote = await rman.get_remote(id)
            case vdata if isinstance(vdata, VastInstance) and not self.id == vdata.id:
                self.remote = await rman.get_remote(vdata)
            case remote if isinstance(remote, DiscoRemote):
                self.remote = remote
            case None:
                self.remote = None

        if self.remote:
            change_state_dialog("Connecting ...", f"Waiting for instance #{self.remote.vdata.id} to load...")
            await self.remote.wait_for_ready()

            # Auto-connect
            if deploy_constants.enable_auto_connect and await self.remote.is_ready():
                change_state_dialog("Connecting ...", f"Connecting to root@{self.remote.ip}:{self.remote.port} (instance #{self.remote.vdata.id})...")
                await self.remote.connect()

            change_state_dialog("Connecting ...", f"Probing instance #{self.remote.vdata.id}...")
            self._last_status = await self.remote.probe_deployment_status()
            if self.remote.ssh:
                self.remote.ssh.logs.handlers.append(self._on_ssh_log_line)

            change_state_dialog("Connecting ...", f"Refreshing instance #{self.remote.vdata.id}...")
            # await self.remote.refresh_data()

        self.update_window()
        invalidate()

    async def _connect(self):
        contask = self.remote.connect()
        change_state_dialog("Connecting ...", f"Connecting to root@{self.remote.ip}:{self.remote.port} (instance #{self.remote.vdata.id})...")
        await contask
        change_state(UIState.MANAGE_INSTANCE)
        self.update_window()

    async def refresh_balance(self):
        if self.remote:
            self._last_balance = await vast.fetch_balance()
            log.info(f"Retrieved balance: {self._last_balance}")
            self.update_window()

    def start_refresh_task(self):
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_balance_loop())

    def stop_refresh_task(self):
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

    async def _refresh_balance_loop(self):
        while True:
            log.debug("Refreshing balance from _refresh_balance_loop...")
            await self.refresh_balance()
            await asyncio.sleep(30)

    async def _refresh(self):
        self.remote.refresh_data()

        status = await self.remote.probe_deployment_status()
        self._last_status = status

        text = f""
        for k, v in status.__dict__.items():
            text += f"{k}: {v}\n"

        self.update_window()

    async def _start(self):
        forget(vast.start_instance(self.vdata.id))

    async def _stop(self):
        forget(vast.stop_instance(self.vdata.id))

    async def _reboot(self):
        forget(vast.reboot_instance(self.vdata.id))
        await self._refresh()

    async def _destroy(self):
        forget(vast.destroy_instance(self.vdata.id))
        change_state(UIState.MAIN)
        # TODO show a blocking popup for like 1 second (if you refresh too soon after destroying, the instance will still be there)

    async def _disconnect(self):
        self.remote.info(make_header_text("Disconnecting ..."))
        await self.remote.disconnect()
        self.update_window()

    async def _deploy(self):
        await self.remote.deploy(self.txt_session.text)
        self.remote.info("== DONE ==")

    async def _mount(self):
        await self.remote.mount()

    async def _shell(self):
        await self.remote.shell()

    async def _send_fast(self):
        self.remote.info(make_header_text("Sending fast uploads ..."))
        await self.remote.send_fast_uploads()
        self.remote.info("== DONE ==")

    async def _send_slow(self):
        self.remote.info(make_header_text("Sending slow uploads ..."))
        await self.remote.send_slow_uploads()
        self.remote.info("== DONE ==")

    async def _pipupgrades(self):
        self.remote.info(make_header_text("Running pip upgrade ..."))
        await self.remote.pip_upgrade()
        self.remote.info("== DONE ==")

    async def _start_comfy(self):
        self.remote.info(make_header_text("Starting Comfy ..."))
        await self.remote.start_comfy()
        self.remote.info("== DONE ==")

    async def _stop_comfy(self):
        self.remote.info(make_header_text("Stopping Comfy ..."))
        await self.remote.stop_comfy()
        self.remote.info("== DONE ==")

    async def _start_discore(self):
        self.remote.info(make_header_text("Starting Discore ..."))
        await self.remote.start_discore()
        self.remote.info("== DONE ==")

    async def _stop_discore(self):
        self.remote.info(make_header_text("Stopping Discore ..."))
        await self.remote.stop_discore()
        self.remote.info("== DONE ==")

    def _on_ssh_log_line(self, _):
        self.update_content()

    def __pt_container__(self):
        return self.window


state = UIState.NONE

kb.add("c-c")(k_c_c)
kb.add("tab")(k_tab)
kb.add("s-tab")(k_s_tab)
kb.add("up")(k_up)
kb.add("down")(k_down)


def on_new_instance_btn():
    change_state(UIState.OFFERS)


def on_back_btn():
    change_state(UIState.MAIN)


def on_refresh_btn():
    forget(remote_list.fetch_instances())


def on_quit_btn():
    app.exit()


status_display = TextArea(text="No instance selected", read_only=True)
remote_list = RemoteList()
offer_list = OfferList(on_offer_confirm)
new_instance_button = AppButton("  New Instance", handler=on_new_instance_btn, key='n')
refresh_button = AppButton("  Refresh2", handler=on_refresh_btn, key='r')
quit_button = AppButton("󰩈  Quit", handler=on_quit_btn, key='q')
back_button = AppButton("󰁮  Back", handler=on_back_btn, key=('escape', 'backspace'))
manage_layout = ManageLayout()

remote_list.handler = on_remote_confirm
offer_list.handler = on_offer_confirm

kb.add("q")(lambda _: app.exit())
