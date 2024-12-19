from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QSplitter, QHeaderView, QMessageBox, QApplication, QMenu,
    QLabel, QSizePolicy, QLineEdit, QSlider, QHBoxLayout,
    QComboBox, QCheckBox, QToolBar, QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtSvg import QSvgWidget
from chess import Board, InvalidMoveError
import chess.svg

class OpeningBookVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()

        self.opening_name_filter = QLineEdit()
        self.opening_name_filter.setPlaceholderText("Filter by Opening Name")
        self.opening_name_filter.setToolTip("Enter partial or full opening name to filter moves.")
        self.opening_name_filter.textChanged.connect(self.apply_filters)

        self.win_percentage_slider = QSlider(Qt.Horizontal)
        self.win_percentage_slider.setMinimum(0)
        self.win_percentage_slider.setMaximum(100)
        self.win_percentage_slider.setValue(0)
        self.win_percentage_slider.setTickInterval(10)
        self.win_percentage_slider.setTickPosition(QSlider.TicksBelow)
        self.win_percentage_slider.setToolTip("Minimum win percentage for filtered moves.")
        self.win_percentage_slider.valueChanged.connect(self.on_win_percentage_slider_changed)

        self.win_percentage_label = QLabel("Min Win %: 0%")

        self.sorting_combobox = QComboBox()
        self.sorting_combobox.addItem("Sort by Move Order")
        self.sorting_combobox.addItem("Sort by Win %")
        self.sorting_combobox.addItem("Sort by Total Games")
        self.sorting_combobox.addItem("Sort by Move Popularity")
        self.sorting_combobox.setToolTip("Change sorting criteria for displayed moves.")
        self.sorting_combobox.currentIndexChanged.connect(self.apply_filters)

        self.win_checkbox = QCheckBox("Include Wins")
        self.win_checkbox.setChecked(True)
        self.win_checkbox.setToolTip("Include moves with winning outcomes.")
        self.win_checkbox.stateChanged.connect(self.apply_filters)

        self.draw_checkbox = QCheckBox("Include Draws")
        self.draw_checkbox.setChecked(True)
        self.draw_checkbox.setToolTip("Include moves with draw outcomes.")
        self.draw_checkbox.stateChanged.connect(self.apply_filters)

        self.loss_checkbox = QCheckBox("Include Losses")
        self.loss_checkbox.setChecked(True)
        self.loss_checkbox.setToolTip("Include moves with losing outcomes.")
        self.loss_checkbox.stateChanged.connect(self.apply_filters)

        control_layout.addWidget(QLabel("Opening Name:"))
        control_layout.addWidget(self.opening_name_filter)
        control_layout.addWidget(self.win_percentage_label)
        control_layout.addWidget(self.win_percentage_slider)
        control_layout.addWidget(QLabel("Sort by:"))
        control_layout.addWidget(self.sorting_combobox)
        control_layout.addWidget(self.win_checkbox)
        control_layout.addWidget(self.draw_checkbox)
        control_layout.addWidget(self.loss_checkbox)

        splitter = QSplitter(Qt.Horizontal)

        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels([
            "Move", "Win %", "Draw %", "Loss %", "Total Games", "Opening Name"
        ])
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.clicked.connect(self.on_item_clicked)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setUniformRowHeights(True)
        self.tree_view.setExpandsOnDoubleClick(False)
        self.tree_view.header().setStretchLastSection(False)
        self.tree_view.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.on_tree_view_context_menu)
        self.tree_view.expanded.connect(self.on_item_expanded)

        splitter.addWidget(self.tree_view)

        board_layout = QVBoxLayout()
        self.board_widget = QSvgWidget()
        self.board_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        nav_toolbar = QToolBar()
        self.prev_move_action = QAction(QIcon.fromTheme("go-previous"), "Previous Move", self)
        self.prev_move_action.setToolTip("Go to the previous move in the stack.")
        self.prev_move_action.triggered.connect(self.on_prev_move)
        self.next_move_action = QAction(QIcon.fromTheme("go-next"), "Next Move", self)
        self.next_move_action.setToolTip("Go to the next available move.")
        self.next_move_action.triggered.connect(self.on_next_move)
        self.reset_board_action = QAction(QIcon.fromTheme("view-refresh"), "Reset Board", self)
        self.reset_board_action.setToolTip("Reset the board to the initial position.")
        self.reset_board_action.triggered.connect(self.on_reset_board)
        nav_toolbar.addAction(self.prev_move_action)
        nav_toolbar.addAction(self.next_move_action)
        nav_toolbar.addAction(self.reset_board_action)

        board_layout.addWidget(nav_toolbar)
        board_layout.addWidget(self.board_widget)

        board_container = QWidget()
        board_container.setLayout(board_layout)

        splitter.addWidget(board_container)
        splitter.setSizes([400, 600])

        main_layout.addLayout(control_layout)
        main_layout.addWidget(splitter)

        self.setLayout(main_layout)

        self.root_board = Board()
        self.positions = {}
        self.current_board = Board()
        self.move_stack = []

    def reset_visualization(self):
        self.board_widget.load(chess.svg.board(board=Board()).encode('UTF-8'))
        self.tree_model.clear()
        self.tree_model.setHorizontalHeaderLabels([
            "Move", "Win %", "Draw %", "Loss %", "Total Games", "Opening Name"
        ])
        self.positions = {}
        self.current_board.reset()
        self.move_stack = []

    def update_opening_book(self, data):
        self.positions = data.get('positions', {})
        self.apply_filters()

    def apply_filters(self):
        QTimer.singleShot(0, self._apply_filters)

    def _apply_filters(self):
        self.tree_model.removeRows(0, self.tree_model.rowCount())
        root_board = Board()
        self.build_tree(root_board, self.tree_model.invisibleRootItem())
        self.tree_view.expandToDepth(0)
        self.update_board()

    def on_win_percentage_slider_changed(self, value):
        self.win_percentage_label.setText(f"Min Win %: {value}%")
        self.apply_filters()

    def build_tree(self, board, parent_item):
        fen = board.fen()
        if fen not in self.positions:
            return
        moves = self.positions[fen]
        filtered_moves = self.filter_moves(moves.items())
        for san, move_data in filtered_moves:
            total_games = move_data['win'] + move_data['draw'] + move_data['loss']
            if total_games == 0:
                continue
            win_pct = (move_data['win'] / total_games) * 100
            draw_pct = (move_data['draw'] / total_games) * 100
            loss_pct = (move_data['loss'] / total_games) * 100
            move_item = QStandardItem(san)
            move_item.setData(board.fen(), Qt.UserRole + 1)
            move_item.setEditable(False)
            win_item = QStandardItem(f"{win_pct:.1f}%")
            win_item.setEditable(False)
            draw_item = QStandardItem(f"{draw_pct:.1f}%")
            draw_item.setEditable(False)
            loss_item = QStandardItem(f"{loss_pct:.1f}%")
            loss_item.setEditable(False)
            total_games_item = QStandardItem(str(total_games))
            total_games_item.setEditable(False)
            opening_name_item = QStandardItem(move_data['name'])
            opening_name_item.setEditable(False)
            win_ratio = win_pct / 100
            r = int((1 - win_ratio) * 255)
            g = int(win_ratio * 255)
            b = 0
            color = QColor(r, g, b, 50)
            for item in [move_item, win_item, draw_item, loss_item, total_games_item, opening_name_item]:
                item.setBackground(color)
            tooltip_text = f"Move: {san}\nWin: {move_data['win']}\nDraw: {move_data['draw']}\nLoss: {move_data['loss']}"
            move_item.setToolTip(tooltip_text)
            parent_item.appendRow([
                move_item, win_item, draw_item, loss_item, total_games_item, opening_name_item
            ])
            move_item.appendRow([QStandardItem() for _ in range(self.tree_model.columnCount())])

    def on_item_expanded(self, index):
        item = self.tree_model.itemFromIndex(index)
        if item.hasChildren():
            is_dummy = True
            for i in range(self.tree_model.columnCount()):
                child_item = item.child(0, i)
                if child_item and child_item.text():
                    is_dummy = False
                    break
            if is_dummy:
                item.removeRow(0)
                self.load_child_nodes(item)

    def load_child_nodes(self, parent_item):
        fen = parent_item.data(Qt.UserRole + 1)
        board = Board(fen)
        move_san = parent_item.text()
        try:
            move = board.parse_san(move_san)
            board.push(move)
        except (ValueError, InvalidMoveError):
            return
        self.build_tree(board, parent_item)

    def filter_moves(self, moves):
        filtered_moves = []
        min_win_percentage = self.win_percentage_slider.value()
        opening_name_filter = self.opening_name_filter.text().lower()
        include_win = self.win_checkbox.isChecked()
        include_draw = self.draw_checkbox.isChecked()
        include_loss = self.loss_checkbox.isChecked()
        for san, move_data in moves:
            total = move_data['win'] + move_data['draw'] + move_data['loss']
            if total == 0:
                continue
            win_percentage = (move_data['win'] / total) * 100
            if win_percentage < min_win_percentage:
                continue
            if not ((include_win and move_data['win'] > 0) or
                    (include_draw and move_data['draw'] > 0) or
                    (include_loss and move_data['loss'] > 0)):
                continue
            if opening_name_filter and opening_name_filter not in move_data['name'].lower():
                continue
            filtered_moves.append((san, move_data))
        sort_index = self.sorting_combobox.currentIndex()
        if sort_index == 1:
            filtered_moves.sort(key=lambda x: (x[1]['win'] / (x[1]['win'] + x[1]['draw'] + x[1]['loss'])), reverse=True)
        elif sort_index == 2 or sort_index == 3:
            filtered_moves.sort(key=lambda x: x[1]['win'] + x[1]['draw'] + x[1]['loss'], reverse=True)
        return filtered_moves

    def on_item_clicked(self, index):
        index = index.sibling(index.row(), 0)
        item = self.tree_model.itemFromIndex(index)
        self.move_stack = []
        self.build_move_stack(item)
        self.update_board()

    def build_move_stack(self, item):
        if item.parent():
            self.build_move_stack(item.parent())
        if item.text() != '':
            self.move_stack.append(item.text())

    def update_board(self):
        board = Board()
        for move_san in self.move_stack:
            try:
                move = board.parse_san(move_san)
                board.push(move)
            except (ValueError, InvalidMoveError):
                QMessageBox.warning(self, "Invalid Move", f"Invalid move '{move_san}' encountered.")
                self.move_stack.remove(move_san)
        self.current_board = board
        last_move = board.peek() if board.move_stack else None
        svg_data = chess.svg.board(board=board, lastmove=last_move).encode('UTF-8')
        self.board_widget.load(svg_data)
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.prev_move_action.setEnabled(len(self.move_stack) > 0)
        fen = self.current_board.fen()
        has_next_move = fen in self.positions and bool(self.filter_moves(self.positions[fen].items()))
        self.next_move_action.setEnabled(has_next_move)

    def on_prev_move(self):
        if self.move_stack:
            self.move_stack.pop()
            self.update_board()

    def on_next_move(self):
        fen = self.current_board.fen()
        if fen in self.positions:
            moves = self.positions[fen]
            filtered_moves = self.filter_moves(moves.items())
            if filtered_moves:
                san, move_data = filtered_moves[0]
                self.move_stack.append(san)
                self.update_board()
            else:
                QMessageBox.information(self, "No Moves Available", "No further moves are available from this position.")
                self.update_navigation_buttons()
        else:
            QMessageBox.information(self, "No Moves Available", "No further moves are available from this position.")
            self.update_navigation_buttons()

    def on_reset_board(self):
        self.move_stack = []
        self.update_board()

    def on_tree_view_context_menu(self, position):
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
        index = index.sibling(index.row(), 0)
        item = self.tree_model.itemFromIndex(index)
        menu = QMenu()
        copy_action = QAction("Copy Move", self)
        copy_action.triggered.connect(lambda: self.copy_move(item.text()))
        menu.addAction(copy_action)
        menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def copy_move(self, move_san):
        clipboard = QApplication.clipboard()
        clipboard.setText(move_san)