import re

from PyQt5.QtCore import Qt, QEvent, QSize, QTimer
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QTextDocument, QPalette, QAbstractTextDocumentLayout, QGuiApplication
from PyQt5.QtWidgets import QApplication, QListView, QLineEdit, QVBoxLayout, QDialog, QStyledItemDelegate, QLabel, QSizePolicy
from PyQt5.QtWidgets import QStyle, QStyleOptionViewItem
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtWidgets import QDialog, QLineEdit, QVBoxLayout, QLabel

from src.party import maths

CONFIRM_KEYS = [Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab]


class HTMLDelegate(QStyledItemDelegate):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.doc = QTextDocument(self)
		self.doc.setDefaultFont(QApplication.font())
		self.doc.setTextWidth(400)

	def paint(self, painter, option, index):
		painter.save()

		options = QStyleOptionViewItem(option)
		self.initStyleOption(options, index)
		self.doc.setHtml(options.text)
		options.text = ""

		style = QApplication.style() if options.widget is None else options.widget.style()
		style.drawControl(QStyle.ControlElement.CE_ItemViewItem, options, painter)

		ctx = QAbstractTextDocumentLayout.PaintContext()
		# ctx = QTextDocument.PaintContext()

		if option.state & QStyle.StateFlag.State_Selected:
			ctx.palette.setColor(QPalette.ColorRole.Text, option.palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText))
		else:
			ctx.palette.setColor(QPalette.ColorRole.Text, option.palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text))

		textRect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, options)

		if index.column() != 0:
			textRect.adjust(5, 0, 0, 0)

		thefuckyourshitup_constant = 4
		margin = (option.rect.height() - options.fontMetrics.height()) // 2
		margin = margin - thefuckyourshitup_constant
		textRect.setTop(textRect.top() + margin)

		painter.translate(textRect.topLeft())
		painter.setClipRect(textRect.translated(-textRect.topLeft()))
		self.doc.documentLayout().draw(painter, ctx)

		painter.restore()

	def sizeHint(self, option, index):
		return QSize(int(self.doc.idealWidth()), int(self.doc.size().height()))


class PopupSelector(QDialog):
	def __init__(self, items, parent=None, desc=None):
		super().__init__(parent)
		self.selected_row_index = None
		self.items = list(items)
		self.filtered_items = list(items)
		self.filtered_indices = []

		self.setWindowTitle("Popup Selector")
		# self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
		# self.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint)
		# self.setWindowFlag(Qt.WindowType.Popup)

		self.search_line = QLineEdit(self)
		self.search_line.setPlaceholderText("Search...")
		self.search_line.textChanged.connect(self.filter_items)
		self.search_line.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.search_line.installEventFilter(self)

		self.description_label = QLabel(desc, self)
		self.description_label.setWordWrap(True)
		self.description_label.setOpenExternalLinks(True)
		self.description_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		self.description_label.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
		self.description_label.setText(desc)

		self.list_view = QListView(self)
		self.model = QStandardItemModel(self.list_view)
		self.list_view.setSelectionMode(QListView.SelectionMode.SingleSelection)
		self.list_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.list_view.setItemDelegate(HTMLDelegate())
		self.list_view.setModel(self.model)
		self.list_view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
		self.list_view.selectionModel().currentChanged.connect(self.item_selected)
		self.list_view.clicked.connect(self.item_clicked)
		layout = QVBoxLayout()
		layout.setContentsMargins(0, 0, 0, 0)
		if desc is not None:
			layout.addWidget(self.description_label)
		layout.addWidget(self.search_line)
		layout.addWidget(self.list_view)
		self.setLayout(layout)

		self.refresh_items()
		self.search_line.setFocus()
		self.search_line.activateWindow()

		# Focus the dialog / raise to top
		self.setFocus()
		self.activateWindow()

		QTimer.singleShot(100, self.search_line.setFocus)
		# self.move(QApplication.desktop().screen().rect().center() - self.rect().center())

	def refresh_items(self):
		from fuzzywuzzy import fuzz

		filter_text = self.search_line.text()
		search_string = filter_text.lower()
		search_threshold = 70
		self.model.clear()
		self.filtered_items.clear()
		self.filtered_indices.clear()

		for i, item in enumerate(self.items):
			# Remove content between italic
			matching_item = re.sub(r'<i>(.*?)</i>', '', item)

			similarity_score = fuzz.partial_ratio(search_string, matching_item)
			if similarity_score >= search_threshold or search_string == "":
				item_text = item.strip()
				item_item = QStandardItem(item_text)
				item_item.setSelectable(True)
				self.model.appendRow(item_item)
				self.filtered_indices.append(i)
				self.filtered_items.append(item)

		# self.list_view.resizeColumnToContents(0)
		# self.list_view.resizeRowsToContents()
		self.list_view.adjustSize()

		# self.list_view.setFocus()
		self.list_view.setCurrentIndex(self.model.index(0, 0))

		n_selected = len(self.filtered_items)
		if n_selected > 0:
			base_width = 60
			base_height = 60
			min_width = 500
			min_height = self.list_view.sizeHintForRow(0) * n_selected + base_height
			# use screen bounds and min
			max_width = int(QGuiApplication.primaryScreen().geometry().width() * 0.8)
			max_height = int(QGuiApplication.primaryScreen().geometry().height() * 0.8)
			width = self.list_view.sizeHintForColumn(0) + base_width
			height = self.list_view.sizeHintForRow(0) * n_selected + base_height

			width = max(width, min_width)
			height = max(height, min_height)
			width = min(width, max_width)
			height = min(height, max_height)

			self.setMinimumHeight(height)
			self.setMinimumWidth(width)
			self.list_view.setMinimumHeight(height)
			self.list_view.setMinimumWidth(width)
			self.resize(width, height)

			# self.selected_index = maths.clamp(self.selected_index, 0, n_selected - 1)
		else:
			self.selected_row_index = -1

	def keyPressEvent(self, event):
		if event.key() == Qt.Key.Key_Escape:
			# Handle the Escape key press here
			if self.search_line.hasFocus():
				if self.search_line.text() == "":
					self.reject()
				else:
					self.search_line.setFocus()
					self.search_line.clear()
			elif self.list_view.hasFocus():
				self.search_line.setFocus()
				self.search_line.clear()

			pass
		elif event.key() in CONFIRM_KEYS:
			self.select_item()
		else:
			# Handle other key events as needed
			super().keyPressEvent(event)

	def eventFilter(self, obj, event):
		if obj == self.search_line:
			if event.type() == QEvent.Type.FocusIn:
				# Handle FocusIn event
				self.search_line.setFocus()  # Ensure the widget receives focus
			elif event.type() == QEvent.Type.FocusOut:
				# Handle FocusOut event if needed
				pass

			if event.type() == QEvent.Type.KeyPress:
				key = event.key()
				if key == Qt.Key.Key_Down:
					self.list_view.setFocus()
					self.list_view.setCurrentIndex(self.model.index(1, 0))

				return False

		if obj == self.list_view:
			if event.type() == QEvent.Type.KeyPress:
				key = event.key()
				if key == Qt.Key.Key_Up:
					self.list_view.setFocus()
					self.list_view.setCurrentIndex(self.model.index(self.model.rowCount() - 1, 0))
					return False
				elif key in CONFIRM_KEYS:
					self.select_item()

		return super().eventFilter(obj, event)

	def filter_items(self):
		self.refresh_items()
		if len(self.filtered_indices) > 0:
			self.selected_row_index = 0
		else:
			self.selected_row_index = None

	def item_clicked(self, index):
		self.selected_row_index = index.row()
		self.accept()

	def item_selected(self, index):
		self.selected_row_index = index.row()

	def select_item(self):
		self.accept()

	def get_selected_index(self):
		if self.selected_row_index is None or self.selected_row_index == -1: return None
		return self.filtered_indices[self.selected_row_index]

	def get_selected_value(self):
		if self.selected_row_index is None: return None
		return self.filtered_items[self.selected_row_index]


class PopupString(QDialog):
	def __init__(self, parent=None, desc=None, default='', placeholder=None):
		super().__init__(parent)
		self.entered_string = None

		self.setWindowTitle("Popup String")
		self.setModal(True)

		self.input_line = QLineEdit(self)
		self.input_line.setPlaceholderText(placeholder or "Enter string...")
		self.input_line.textChanged.connect(self.validate_input)
		self.input_line.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
		self.input_line.setText(default)

		self.description_label = QLabel(desc, self)
		self.description_label.setWordWrap(False)
		self.description_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
		self.description_label.setOpenExternalLinks(True)
		self.description_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		self.description_label.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
		self.description_label.setText(desc)

		layout = QVBoxLayout()
		layout.setContentsMargins(0, 0, 0, 0)

		if desc is not None:
			layout.addWidget(self.description_label)

		layout.addWidget(self.input_line)
		self.setLayout(layout)

		self.input_line.setFocus()
		self.input_line.activateWindow()
		QTimer.singleShot(100, self.input_line.setFocus)
		self.adjustSize()

	def validate_input(self):
		# You can add input validation logic here if needed
		pass

	def keyPressEvent(self, event):
		if event.key() == Qt.Key.Key_Escape:
			# Handle the Escape key press here
			if self.input_line.hasFocus():
				if self.input_line.text() == "":
					self.reject()
				else:
					self.input_line.setFocus()
					self.input_line.clear()
		elif event.key() in CONFIRM_KEYS:
			self.entered_string = self.input_line.text()
			self.accept()
		else:
			# Handle other key events as needed
			super().keyPressEvent(event)

	def get_entered_string(self):
		return self.entered_string


def show_popup_selector(items, parent=None, title=None, desc=None):
	is_dict = False
	if isinstance(items, dict):
		dic = items
		items = list(items.keys())
		is_dict = True

	popup = PopupSelector(items, parent=parent)
	popup.setFocus()
	popup.setWindowTitle(title)
	result = popup.exec()
	if result != QDialog.DialogCode.Accepted:
		return None

	if is_dict:
		return dic[popup.get_selected_value()]

	if popup.get_selected_index() is not None:
		return items[popup.get_selected_index()]

	return None


def show_string_input(parent=None, desc='', title=None, default=None):
	popup = PopupString(parent=parent, desc=desc, default=default)
	popup.setFocus()
	popup.setWindowTitle(title)
	result = popup.exec()
	if result != QDialog.DialogCode.Accepted:
		return None

	return popup.get_entered_string()
