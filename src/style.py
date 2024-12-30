stylesheet = """
QMainWindow {
    background-color: #fafafa;
}

QTabWidget::pane {
    border: 1px solid #c2c2c2;
    background: #ffffff;
}
QTabBar::tab {
    background: #e9e9e9;
    border: 1px solid #c2c2c2;
    border-bottom-color: #c2c2c2;
    padding: 6px 12px;
    min-width: 100px;
    margin-right: -1px;
}
QTabBar::tab:selected,
QTabBar::tab:hover {
    background: #ffffff;
}
QTabBar::tab:selected {
    border-bottom: none;
}

QGroupBox {
    border: 1px solid #c2c2c2;
    border-radius: 4px;
    margin-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 6px;
    background-color: transparent;
    color: #444444;
    font-weight: bold;
}

QLabel {
    color: #111111;
}

QTextEdit {
    background-color: #ffffff;
    border: 1px solid #c2c2c2;
    border-radius: 4px;
}

QProgressBar {
    border: 1px solid #b9b9b9;
    border-radius: 4px;
    background: #f4f4f4;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #76a9ea;
    width: 10px;
    margin: 0.5px;
}

QPushButton {
    background-color: #0078d4;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #005fa3;
}
QPushButton:disabled {
    background-color: #b8b8b8;
    color: #eaeaea;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #c2c2c2;
    border-radius: 4px;
    background-color: #ffffff;
    padding: 4px;
    color: #111111;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background-color: #f0f0f0;
    color: #8f8f8f;
}

QCheckBox {
    spacing: 6px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #c2c2c2;
}

QComboBox QAbstractItemView {
    border: 1px solid #c2c2c2;
    selection-background-color: #e8f0fe;
}
"""