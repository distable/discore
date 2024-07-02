from typing import Generic, TypeVar, Sequence, Optional, Callable, List, Union, Dict, Any, Tuple
from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.layout import FormattedTextControl, Window, ScrollbarMargin, ConditionalMargin, Dimension
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.filters import Condition
from prompt_toolkit.application import get_app
import logging

from prompt_toolkit.styles import Style

STYLE_SEPARATOR = 'class:separator'
STYLE_HEADER = 'class:header'
STYLE_ROW_HIGHLIGHT = 'class:row-highlight'
STYLE_FOOTER = 'class:footer'
STYLE_COL_ODD = 'class:column-odd'
STYLE_COL_EVEN = 'class:column-even'
STYLE_COL_SORTED = 'class:column-sorted'

_T = TypeVar("_T")
Item = Union[tuple[_T, str], Dict[str, Any], Any]


class ButtonListError(Exception):
    """Custom exception for ButtonList errors."""
    pass


class ButtonList(Generic[_T]):
    open_character = "["
    close_character = "]"
    container_style = "class:button-list"
    default_style = "class:button-list-item"
    selected_style = "class:button-list-item-selected"
    show_scrollbar = True

    def __init__(self,
                 data: Sequence[Item],
                 handler: Optional[Callable] = None,
                 headers: Optional[List[str]] = None,
                 first_item_as_headers: bool = False,
                 separator_width: int = 1
                 ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        if not data:
            raise ButtonListError("ButtonList must be initialized with non-empty values.")

        self.data = data
        self.handler = handler
        self.separator_width = separator_width
        self._headers = headers
        self._first_item_as_headers = first_item_as_headers
        self._selected_index = 0
        self._sort_column = None
        self._sort_ascending = True

        self.style = Style.from_dict({
            'button-list': 'bg:#000080',
            'header': 'bold #ffffff bg:#000080',
            'row': '#ffffff',
            'selected-row': 'bg:#4169E1 #ffffff',
            'footer': 'italic #ffffff',
            'column-even': 'bg:#1a1a1a',
            'column-odd': 'bg:#262626',
        })

        # Cached state
        self._cached_headers = None
        self._cached_keys = None
        self._cached_column_widths = None

        self.control = FormattedTextControl(
            self._get_text_fragments,
            key_bindings=self._create_key_bindings(),
            focusable=True,
        )

        self.window = Window(
            content=self.control,
            style=self.container_style,
            right_margins=[ConditionalMargin(
                margin=ScrollbarMargin(display_arrows=True),
                filter=Condition(lambda: self.show_scrollbar),
            )],
            dont_extend_height=False,
            dont_extend_width=False,
            width=Dimension(preferred=100)
        )

    def _process_data(self):
        if not self.data:
            raise ButtonListError("No values to process.")

        if isinstance(self.data[0], tuple):
            if len(self.data[0]) < 2:
                raise ButtonListError("Tuple items must have at least two elements.")
            keys = ['value', 'display']
        elif isinstance(self.data[0], dict):
            keys = list(self.data[0].keys())
        elif hasattr(self.data[0], '__dict__'):
            keys = list(self.data[0].__dict__.keys())
        else:
            raise ButtonListError(f"Unsupported item type: {type(self.data[0])}")

        headers = self._headers or keys
        if self._first_item_as_headers:
            if self._headers:
                raise ButtonListError("Cannot use both 'headers' and 'first_item_as_headers'.")
            headers = keys
            self.data = self.data[1:]

        self._cached_keys = keys
        self._cached_headers = headers
        return keys, headers

    def _calculate_column_widths(self, keys, headers):
        # We add +2 to the header length to account for the sort indicator
        column_widths = [len(str(header)) + self.separator_width * 2 + 2 for header in headers]

        for item in self.data:
            values = []
            if isinstance(item, tuple):
                values = [str(item[1])] + [''] * (len(headers) - 1)
            elif isinstance(item, dict):
                values = [str(item.get(key, '')) for key in keys]
            elif hasattr(item, '__dict__'):
                values = [str(getattr(item, key, '')) for key in keys]

            # Ensure values has the same length as headers
            values += [''] * (len(headers) - len(values))

            column_widths = [max(current, len(value))
                             for current, value in zip(column_widths, values)]

        self._cached_column_widths = column_widths
        return column_widths

    def _get_headers(self):
        if self._cached_headers is None:
            self._process_data()
        return self._cached_headers

    def _get_keys(self):
        if self._cached_keys is None:
            self._process_data()
        return self._cached_keys

    def _get_column_widths(self):
        if self._cached_column_widths is None:
            keys, headers = self._process_data()
            self._calculate_column_widths(keys, headers)
        return self._cached_column_widths

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        kb.add("up")(lambda event: self._move_cursor(-1))
        kb.add("down")(lambda event: self._move_cursor(1))
        kb.add("left")(lambda event: self.sort_column_dir(-1))
        kb.add("right")(lambda event: self.sort_column_dir(1))
        kb.add("pageup")(lambda event: self._move_cursor(-self._get_page_size()))
        kb.add("pagedown")(lambda event: self._move_cursor(self._get_page_size()))
        kb.add("enter")(lambda event: self._handle_enter())
        kb.add(" ")(lambda event: self._handle_enter())
        kb.add(Keys.Any)(self._find)

        # Sorting for columns 1-9
        for i in range(9):
            @kb.add(str(i + 1))
            def _(event, i=i):
                self.sort_column(i)

        # Sorting for columns 10-20
        shift_number_map = {
            '!': 10,
            '@': 11,
            '#': 12,
            '$': 13,
            '%': 14,
            '^': 15,
            '&': 16,
            '*': 17,
            '(': 18,
            ')': 19,
            '_': 20,  # This is Shift+- for the 20th column
        }

        for key, column in shift_number_map.items():
            @kb.add(key)
            def _(event, column=column):
                if column < len(self._get_headers()):
                    self.sort_column(column - 1)

        return kb

    def _get_page_size(self) -> int:
        app = get_app()
        if app.layout and app.layout.current_window and app.layout.current_window.render_info:
            return len(app.layout.current_window.render_info.displayed_lines)
        return 10  # Default value if unable to determine

    def _handle_enter(self) -> None:
        if not self.data:
            return
        item = self.data[self._selected_index]
        if self.handler:
            try:
                self.handler(item)
            except Exception as e:
                self.logger.error(f"Error in handler: {str(e)}")
                raise ButtonListError(f"Error in handler: {str(e)}")
        elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], Callable):
            try:
                item[0]()
            except Exception as e:
                self.logger.error(f"Error calling item function: {str(e)}")
                raise ButtonListError(f"Error calling item function: {str(e)}")

    def _move_cursor(self, offset: int) -> None:
        if not self.data:
            return
        self._selected_index = max(0, min(len(self.data) - 1, self._selected_index + offset))

    def _find(self, event) -> None:
        if not self.data:
            return
        char = event.data.lower()
        start_index = self._selected_index
        headers = self._get_headers()
        for i in range(len(self.data)):
            index = (start_index + i + 1) % len(self.data)
            item = self.data[index]
            if isinstance(item, tuple):
                text = str(item[1])
            elif isinstance(item, dict):
                text = str(item.get(headers[0], ''))
            elif hasattr(item, '__dict__'):
                text = str(getattr(item, headers[0], ''))
            else:
                continue

            if text.lower().startswith(char):
                self._selected_index = index
                return

    def sort_column(self, column_index: int) -> None:
        headers = self._get_headers()
        if column_index < 0 or column_index >= len(headers):
            # self.logger.error(f"Invalid column index: {column_index}")
            return

        if self._sort_column == column_index:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = column_index
            self._sort_ascending = True

        key = headers[column_index]
        try:
            self.data = sorted(
                self.data,
                key=lambda x: (x.get(key, '') if isinstance(x, dict) else
                               getattr(x, key, '') if hasattr(x, '__dict__') else
                               x[1] if isinstance(x, tuple) else ''),
                reverse=not self._sort_ascending
            )
        except Exception as e:
            self.logger.error(f"Error sorting values: {str(e)}")
            raise ButtonListError(f"Error sorting values: {str(e)}")

    def sort_column_dir(self, direction: int) -> None:
        if self._sort_column is None:
            return

        next_column = self._sort_column + direction
        next_column = next_column % len(self._get_headers())
        self.sort_column(next_column)

    def _get_text_fragments(self) -> StyleAndTextTuples:
        keys, headers = self._process_data()
        column_widths = self._calculate_column_widths(keys, headers)

        fragments: List[Tuple[str, str]] = []

        def add_horizontal_line():
            line = '─' * (sum(column_widths) + len(column_widths) - 1)
            fragments.append(('class:separator', f'─{line}─\n'))

        # TODO ask robot to refactor the function to use this
        def draw(style, text):
            fragments.append((style, text))

        # Calculate total width
        # ----------------------------------------
        total_width = sum(column_widths) + len(column_widths) - 1

        # Draw top border
        # ----------------------------------------
        add_horizontal_line()

        # Draw headers
        # ----------------------------------------
        if headers:
            for i, header in enumerate(headers):
                is_sorted_col = i == self._sort_column
                UP_ARROW_FA = "\uf062"
                DOWN_ARROW_FA = "\uf063"
                UP_CHEVRON_FA = "\uf077"
                DOWN_CHEVRON_FA = "\uf078"
                UP_ARROW_MD = "\uf05d"
                DOWN_ARROW_MD = "\uf045"

                sort_indicator = '  '
                if is_sorted_col and self._sort_ascending: sort_indicator = f'{UP_ARROW_FA} '
                if is_sorted_col and not self._sort_ascending: sort_indicator = f'{DOWN_ARROW_FA} '


                header_text = f'{sort_indicator}{header}'
                fragments.append((STYLE_SEPARATOR, ' ' * self.separator_width))
                fragments.append((STYLE_HEADER, f'{header_text:<{column_widths[i]}}'))
                fragments.append((STYLE_SEPARATOR, ' ' * self.separator_width))

            fragments.append(('', '\n'))

            # Draw header-content separator
            add_horizontal_line()

        # Draw content
        # ----------------------------------------
        for row_index, item in enumerate(self.data):
            is_selected_row = row_index == self._selected_index

            if is_selected_row:
                fragments.append(("[SetCursorPosition]", ""))

            # Collect the fragments for this row
            rowfrags = []
            try:
                if isinstance(item, tuple):
                    column_text_entries = [str(item[1])] + [''] * (len(headers) - 1)
                elif isinstance(item, dict) or hasattr(item, '__dict__'):
                    column_text_entries = [str(item.get(key, '') if isinstance(item, dict) else getattr(item, key, '')) for key in keys]
                else:
                    raise ButtonListError(f"Unsupported item type: {type(item)}")

                # Draw the columns
                for col_index, (column_entry, column_width) in enumerate(zip(column_text_entries, column_widths)):
                    column_style = self.get_column_style(col_index, )
                    style = STYLE_ROW_HIGHLIGHT if is_selected_row else column_style

                    rowfrags.append((style, ' ' * self.separator_width))
                    rowfrags.append((style, f'{column_entry:<{column_width}}'))
                    rowfrags.append((style, ' ' * self.separator_width))

            except Exception as e:
                self.logger.error(f"Error formatting item {row_index}: {str(e)}")
                rowfrags.append(('class:error', f"Error: {str(e)}"))

            fragments.extend(rowfrags)
            fragments.append(('', '\n'))

        # Draw bottom border
        # ----------------------------------------
        add_horizontal_line()

        # Draw footer with item count
        # ----------------------------------------
        footer_text = f' Total items: {len(self.data)} '
        footer_padding = max(0, total_width - len(footer_text))
        fragments.append((STYLE_FOOTER, footer_text + ' ' * footer_padding))

        return fragments

    def get_column_style(self, col_index):
        is_sort_column = col_index == self._sort_column
        if col_index % 2 == 0:
            column_style = STYLE_COL_EVEN
        else:
            column_style = STYLE_COL_ODD

        if is_sort_column:
            column_style += ' ' + STYLE_COL_SORTED

        return column_style

    def __pt_container__(self) -> Window:
        return self.window
