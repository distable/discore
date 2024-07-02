import logging
from enum import Enum
from typing import TypeVar

from prompt_toolkit import Application, HTML
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, HSplit, VSplit, Float, Window, Dimension
from prompt_toolkit.layout.containers import FloatContainer, HorizontalAlign, to_container, ConditionalContainer, Container
from prompt_toolkit.mouse_events import MouseEventType, MouseEvent
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea, Button

from src.deploy import VastAIManager, VastInstance
from src.deploy.ButtonList import ButtonList
from src.deploy.deploy_utils import fire_and_forget

log = logging.getLogger(__name__)

E = KeyPressEvent
vast_manager = VastAIManager.instance

_T = TypeVar("_T")


class State(Enum):
    MAIN = 1
    DIALOG = 2


kb = KeyBindings()


def is_control_visible(app: Application, control_to_find) -> bool:
    def traverse(container):
        if isinstance(container, Window):
            if container.content == control_to_find:
                return True
        elif isinstance(container, ConditionalContainer):
            if to_filter(container.filter)():
                return traverse(container.content)
        elif isinstance(container, Float):
            if traverse(container.content):
                return True
        elif isinstance(container, Container):
            for c in container.get_children():
                if traverse(c):
                    return True
        return False

    return traverse(app.layout.container)


def bold_character(word: str, index: int) -> HTML:
    """
    Returns an HTML object with the character at the specified index in bold.

    :param word: The word to modify
    :param index: The index of the character to make bold
    :return: HTML object with formatted text
    """
    if index < 0 or index >= len(word):
        return HTML(word)  # Return the word as is if index is out of range

    before = word[:index]
    char = word[index]
    after = word[index + 1:]

    return HTML(f"{before}<b>{char}</b>{after}")


class InstanceList(ButtonList):
    def __init__(self):
        super().__init__(data=[(None, "Empty!")])
        self.instances = []
        # self.headers = ['index', 'cuda', 'model', 'price']

    def add_instance(self, instance: VastInstance):
        self.data.append(instance)
        self.update()

    @fire_and_forget
    async def fetch_instances(self):
        self.data = [(None, "Loading...")]
        self.data = await vast_manager.fetch_instances()
        self.update()
        get_app().invalidate()

    def update(self):
        if not self.data:
            self.data = [(None, "No instances available")]


class OfferList(ButtonList):
    def __init__(self, handler):
        super().__init__(data=[(None, "Empty!")], handler=handler)
        self.offers = []

    @fire_and_forget
    async def fetch_offers(self):
        self.data = [(None, "Loading...")]
        get_app().invalidate()

        self.offers = await vast_manager.fetch_offers()
        self.data = self.offers
        self._process_data()

        self.sort_column(self._cached_headers.index('price'))  # hehe

        get_app().invalidate()


class StyledButton(Button):
    def __init__(self, text, handler, key=None):
        self.keybinding_char_index = -1
        if key and key in text:
            self.keybinding_char_index = text.lower().index(key.lower())

        super().__init__(text,
                         handler=handler,
                         left_symbol='【',
                         right_symbol='】')

        if key:
            kb.add(key)(self._handler)

        # Autosize button
        self.window.width = len(text) + 4

    def _handler(self, event):
        self.handler()

    def _get_text_fragments(self) -> StyleAndTextTuples:
        def handler(mouse_event: MouseEvent) -> None:
            if self.handler is not None and mouse_event.event_type == MouseEventType.MOUSE_UP:
                self.handler()

        fragments = [
            ("class:button.arrow", self.left_symbol, handler),
            ("[SetCursorPosition]", ""),
        ]

        # Add text fragments with bolding if necessary
        for i, char in enumerate(self.text):
            style = "class:button.text"
            if i == self.keybinding_char_index:
                # Change
                style = "class:button.keymap-highlight-char"

            fragments.append((style, char, handler))

        fragments.append(("class:button.text", " ", handler))
        fragments.append(("class:button.arrow", self.right_symbol, handler))

        return fragments

    def __pt_container__(self):
        return self.window


def create_main_layout():
    return Layout(
        HSplit([
            VSplit([
                Frame(instance_list, title="Instances"),
                Frame(status_display, title="Status")
            ]),
            VSplit([
                new_instance_button,
                refresh_button,
                quit_button,
            ], align=HorizontalAlign.CENTER, padding=3)
        ])
    )


def create_dialog_layout():
    return Layout(
        FloatContainer(
            content=Window(),
            floats=[
                Float(
                    Frame(HSplit([offer_list, back_button], width=Dimension(preferred=99999999999)), title="Select an offer"),
                    top=2,
                    left=2,
                    right=2,
                    bottom=2
                )
            ]
        )
    )


def change_state(new_state):
    global state
    if state == new_state:
        return

    state = new_state

    if state == State.MAIN:
        app.layout = create_main_layout()
        app.layout.focus(instance_list)
    elif state == State.DIALOG:
        offer_list.fetch_offers()
        app.layout = create_dialog_layout()
        app.layout.focus(offer_list)


# region Key bindings
def k_c_c(event):
    app.exit()


def k_instance():
    change_state(State.DIALOG)


def k_refresh():
    instance_list.fetch_instances()


def k_quit():
    get_app().exit()


def k_back():
    change_state(State.MAIN)


def k_tab(event):
    event.app.layout.focus_next()


def k_s_tab(event):
    event.app.layout.focus_previous()


# Again as separate functions
def k_up(event):
    event.app.layout.focus_previous()


def k_down(event):
    event.app.layout.focus_next()


def k_left(event):
    current = event.app.layout.current_window
    parent = event.app.layout.get_parent(current)
    if isinstance(parent, VSplit):
        idx = parent.get_child_index(current)
        if idx > 0:
            event.app.layout.focus(to_container(parent.children[idx - 1]))


def k_right(event):
    current = event.app.layout.current_window
    parent = event.app.layout.get_parent(current)
    if isinstance(parent, VSplit):
        idx = parent.get_child_index(current)
        if idx < len(parent.children) - 1:
            event.app.layout.focus(to_container(parent.children[idx + 1]))


# endregion

# Update status display when an instance is selected
def on_instance_select(instance):
    if instance:
        status_display.text = f"Selected Instance:\nID: {instance['id']}\nStatus: {instance['status']}\nIP: {instance['ip']}"
    else:
        status_display.text = "No instance selected"


# Create new instance when an offer is selected
def on_offer_confirm(offer):
    if offer:
        print("A")
        vast_manager.create_instance(offer.id)
        print("B")
        instance_list.fetch_instances()
        print("C")
        change_state(State.MAIN)
        print("D")


state = State.MAIN

kb.add("c-c")(k_c_c)
kb.add("tab")(k_tab)
kb.add("s-tab")(k_s_tab)
kb.add("up")(k_up)
kb.add("down")(k_down)
kb.add("left")(k_left)
kb.add("right")(k_right)

status_display = TextArea(text="No instance selected", read_only=True)
instance_list = InstanceList()
offer_list = OfferList(on_offer_confirm)
new_instance_button = StyledButton("New Instance", handler=k_instance, key='n')
refresh_button = StyledButton("Refresh", handler=k_refresh, key='r')
quit_button = StyledButton("Quit", handler=k_quit, key='q')
back_button = StyledButton("Back", handler=lambda: k_back(), key='escape')

instance_list.on_select = on_instance_select
offer_list.on_select = on_offer_confirm

style = Style.from_dict({
    'frame.border': '#888888',
    'frame.label': 'bg:#ffffff #000000',
    'button': 'bg:#cccccc #000000',
    'button.keymap-highlight-char': 'bg:#999999 bold underline',
    # 'button.focused': 'bg:#007acc #ffffff',
    'radiolist': 'bg:#f0f0f0 #000000',
    'radiolist-selected': 'bg:#007acc #ffffff',
    'textarea': 'bg:#f0f0f0 #000000',
    'textarea.cursor': '#ff0000',
    'scrollbar.background': 'bg:#000000',
    'scrollbar.button': 'bg:#cccccc',
    # 'button-list': 'bg:#f0f0f0',
    # 'button-list-item': 'bg:#000000 #000000',
    'button-list-item-selected': 'bg:#cccccc #000000',
    'row-highlight': 'bg:#cccccc #000000',
    'column-even': 'bg:#050505',
    'column-odd': 'bg:#111111',
    'column-sorted': 'bg:#333333 #ffffff',
})

app = Application(
    key_bindings=kb,
    mouse_support=True,
    full_screen=True,
    style=style,
    layout=create_main_layout(),
)


async def main():
    global app
    change_state(State.DIALOG)
    with patch_stdout(app):
        await app.run_async()
