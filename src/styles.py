def get_stylesheet():
    return """
    QWidget {
        background: #1e1e1e;
        color: #eeeeee;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
        selection-background-color: #464646;
        selection-color: #ffffff;
        padding: 0;
        margin: 0;
    }

    QToolTip {
        background-color: #2a2a2a;
        color: #eeeeee;
        border: none;
        padding: 4px;
        border-radius: 3px;
        font-size: 9pt;
    }

    QGroupBox {
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 10px;
        background: #1e1e1e;
        font-weight: 600;
        padding: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
        color: #cccccc;
        background: transparent;
        font-size: 10pt;
    }

    QLabel {
        background: transparent;
    }

    QLineEdit, QTextEdit, QPlainTextEdit {
        background: #2a2a2a;
        border: none;
        border-radius: 2px;
        padding: 4px 6px;
    }
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 1px solid #555555;
    }
    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
        background: #252525;
        color: #7a7a7a;
        border: none;
    }

    QPushButton {
        background: #2a2a2a;
        border: none;
        border-radius: 2px;
        padding: 6px 10px;
        color: #eeeeee;
    }
    QPushButton:hover {
        background: #333333;
    }
    QPushButton:pressed {
        background: #2f2f2f;
    }
    QPushButton:disabled {
        background: #252525;
        color: #7a7a7a;
        border: none;
    }

    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #555555;
        background: #2a2a2a;
        border-radius: 2px;
    }
    QCheckBox::indicator:hover {
        border: 1px solid #666666;
    }
    QCheckBox::indicator:checked {
        background: #555555;
    }
    QCheckBox::indicator:disabled {
        border: 1px solid #333333;
        background: #252525;
    }

    QComboBox {
        background: #2a2a2a;
        border: none;
        border-radius: 2px;
        padding: 4px;
    }
    QComboBox:focus {
        border: 1px solid #555555;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: none;
        background: #2a2a2a;
        border-top-right-radius: 2px;
        border-bottom-right-radius: 2px;
    }
    QComboBox::down-arrow {
        image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAICAYAAAAVcYlCAAAAAElFTkSuQmCC);
    }
    QComboBox QAbstractItemView {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        selection-background-color: #464646;
        selection-color: #ffffff;
    }

    QProgressBar {
        border: 1px solid #2a2a2a;
        border-radius: 2px;
        background: #252525;
        text-align: center;
        color: #eeeeee;
    }
    QProgressBar::chunk {
        background-color: #444444;
        border-radius: 2px;
    }

    QTabWidget {
        border: none;
    }
    QTabBar::tab {
        background: #2a2a2a;
        border: none;
        padding: 10px 15px;
        margin-right: 2px;
        border-top-left-radius: 2px;
        border-top-right-radius: 2px;
        font-weight: 500;
        color: #cccccc;
        font-size: 10pt;
        min-width: 220px;
    }
    QTabBar::tab:selected {
        background: #333333;
        color: #ffffff;
    }
    QTabBar::tab:hover {
        background: #2f2f2f;
    }
    QTabBar::tab:!selected {
        color: #aaaaaa;
    }
    QTabWidget::pane {
        border: 1px solid #2a2a2a;
        top: -1px;
    }

    QFrame[frameShape="4"] {
        border-top: 1px solid #444444;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    QFrame[frameShape="5"] {
        border-left: 1px solid #444444;
        margin-left: 10px;
        margin-right: 10px;
    }

    QTreeView, QTableView {
        background: #2a2a2a;
        border: 1px solid #2a2a2a;
        gridline-color: #3a3a3a;
        selection-background-color: #464646;
        selection-color: #ffffff;
        alternate-background-color: #262626;
    }
    QHeaderView::section {
        background: #2a2a2a;
        border: none;
        padding: 4px;
        font-weight: 600;
        color: #eeeeee;
    }

    QToolBar {
        background: #2a2a2a;
        border-bottom: 1px solid #2a2a2a;
        spacing: 4px;
    }
    QToolBar QToolButton {
        background: transparent;
        border: none;
        margin: 4px;
        padding: 4px;
    }
    QToolBar QToolButton:hover {
        background: #333333;
        border-radius: 2px;
    }

    QMenuBar {
        background: #2a2a2a;
        border-bottom: 1px solid #2a2a2a;
    }
    QMenuBar::item {
        background: transparent;
        padding: 4px 8px;
        color: #cccccc;
    }
    QMenuBar::item:selected {
        background: #353535;
    }

    QMenu {
        background: #2a2a2a;
        border: 1px solid #2a2a2a;
    }
    QMenu::item {
        padding: 4px 20px;
        border-radius: 2px;
        color: #eeeeee;
    }
    QMenu::item:selected {
        background: #353535;
    }

    QScrollBar:horizontal {
        height: 12px;
        background: #2a2a2a;
        margin: 0px;
    }
    QScrollBar::handle:horizontal {
        background: #3a3a3a;
        min-width: 20px;
        border-radius: 2px;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
        background: none;
    }

    QScrollBar:vertical {
        width: 12px;
        background: #2a2a2a;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #3a3a3a;
        min-height: 20px;
        border-radius: 2px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
        background: none;
    }

    QSplitter::handle {
        background: #2a2a2a;
    }

    *:disabled {
        color: #7a7a7a;
        background: #252525;
        border-color: #252525;
    }
    """