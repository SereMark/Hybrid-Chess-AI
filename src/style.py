stylesheet = """
QMainWindow {
    background-color: #f5f7f8;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10pt;
}

QTabWidget::pane {
    border: 1px solid #d3d3d3;
    background: #ffffff;
    border-radius: 6px;
    margin: 4px;
    padding: 4px;
}
QTabBar::tab {
    background: #ececec;
    border: 1px solid #d3d3d3;
    border-bottom-color: #d3d3d3;
    padding: 8px 16px;
    margin-right: -1px;
    min-width: 120px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #2f2f2f;
}
QTabBar::tab:selected,
QTabBar::tab:hover {
    background: #ffffff;
    border-bottom: none;
}
QTabBar::tab:selected {
    color: #0078d4;
    font-weight: 600;
}

QGroupBox {
    border: 1px solid #d3d3d3;
    border-radius: 6px;
    margin-top: 20px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 10px;
    color: #333333;
    font-weight: 600;
    background: transparent;
}

QLabel {
    color: #2f2f2f;
    font-weight: 500;
}

QTextEdit {
    background-color: #ffffff;
    border: 1px solid #d3d3d3;
    border-radius: 6px;
}

QProgressBar {
    border: 1px solid #d3d3d3;
    border-radius: 6px;
    background: #f2f2f2;
    text-align: center;
    height: 16px;
}
QProgressBar::chunk {
    background-color: #2196f3;
    margin: 1px;
    border-radius: 5px;
}

QPushButton {
    background-color: #0078d4;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #005c9f;
}
QPushButton:disabled {
    background-color: #b8b8b8;
    color: #f0f0f0;
}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {
    border: 1px solid #d3d3d3;
    border-radius: 6px;
    background-color: #ffffff;
    padding: 5px;
    color: #2f2f2f;
}
QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {
    background-color: #ebebeb;
    color: #9f9f9f;
}

QCheckBox {
    spacing: 6px;
    color: #333333;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border-left: 1px solid #d3d3d3;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
QComboBox QAbstractItemView {
    border: 1px solid #d3d3d3;
    selection-background-color: #e3f2fd;
    background-color: #ffffff;
}

QScrollBar:horizontal {
    height: 14px;
    background: #f2f2f2;
    border: 1px solid #d3d3d3;
    border-radius: 7px;
}
QScrollBar::handle:horizontal {
    background: #cfcfcf;
    min-width: 25px;
    border-radius: 7px;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    background: none;
}

QScrollBar:vertical {
    width: 14px;
    background: #f2f2f2;
    border: 1px solid #d3d3d3;
    border-radius: 7px;
}
QScrollBar::handle:vertical {
    background: #cfcfcf;
    min-height: 25px;
    border-radius: 7px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    background: none;
}
"""