from ..common.config import ASSETS_DIR
import os

UP_ARROW_IMAGE = os.path.join(ASSETS_DIR, 'up-arrow.png').replace("\\", "/")
DOWN_ARROW_IMAGE = os.path.join(ASSETS_DIR, 'down-arrow.png').replace("\\", "/")

DARK_STYLESHEET = f"""
/* General Application Styles */
QMainWindow {{
    background-color: #121212;
}}

QWidget {{
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12pt;
}}

/* Menu Bar and Menus */
QMenuBar {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    spacing: 10px;
}}

QMenuBar::item {{
    padding: 8px 16px;
}}

QMenuBar::item:selected {{
    background-color: #2C2C2C;
}}

QMenu {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    border: 1px solid #2C2C2C;
    padding: 5px 0;
}}

QMenu::item {{
    padding: 8px 16px;
}}

QMenu::item:selected {{
    background-color: #2962FF;
    color: #FFFFFF;
}}

/* Status Bar */
QStatusBar {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    border-top: 1px solid #2C2C2C;
    font-size: 10pt;
}}

/* Tooltips */
QToolTip {{
    background-color: #333333;
    color: #FFFFFF;
    border: 1px solid #2962FF;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 10pt;
}}

/* Message Boxes */
QMessageBox, QDialog {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    font-size: 12pt;
}}

QMessageBox QLabel, QDialog QLabel {{
    background-color: #1F1F1F;
    color: #E0E0E0;
}}

QMessageBox QPushButton, QDialog QPushButton {{
    background-color: #2C2C2C;
    border: 1px solid #2C2C2C;
    color: #E0E0E0;
    padding: 6px 12px;
    border-radius: 6px;
}}

QMessageBox QPushButton:hover, QDialog QPushButton:hover {{
    background-color: #3C3C3C;
    border-color: #2962FF;
}}

QMessageBox QPushButton:pressed, QDialog QPushButton:pressed {{
    background-color: #2962FF;
    border-color: #1F1F1F;
    color: #FFFFFF;
}}

/* Scroll Bars */
QScrollBar:vertical, QScrollBar:horizontal {{
    background: #1F1F1F;
    border: 1px solid #2C2C2C;
    border-radius: 4px;
    min-width: 12px;
    min-height: 12px;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #424242;
    min-width: 20px;
    min-height: 20px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: #616161;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background: none;
    border: none;
    height: 0;
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid #2C2C2C;
    background: #1F1F1F;
    border-radius: 6px;
}}

QTabBar::tab {{
    background: #2C2C2C;
    border: 1px solid #2C2C2C;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background: #2962FF;
    border-color: #2962FF;
    color: #FFFFFF;
}}

QTabBar::tab:hover {{
    background: #3C3C3C;
    color: #FFFFFF;
}}

/* Checkboxes and Radio Buttons */
QCheckBox, QRadioButton {{
    spacing: 8px;
    font-size: 12pt;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px;
    height: 18px;
}}

QCheckBox::indicator {{
    background: #2C2C2C;
    border: 2px solid #2962FF;
    border-radius: 4px;
}}

QCheckBox::indicator:checked {{
    background: #2962FF;
}}

QRadioButton::indicator {{
    background: #2C2C2C;
    border: 2px solid #2962FF;
    border-radius: 9px;
}}

QRadioButton::indicator:checked {{
    background: #2962FF;
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 8px;
    background: #2C2C2C;
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    background: #2962FF;
    border: 1px solid #2962FF;
    width: 16px;
    margin: -4px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: #3D85FF;
}}

QSlider::sub-page:horizontal {{
    background: #2962FF;
    border-radius: 4px;
}}

QSlider::add-page:horizontal {{
    background: #2C2C2C;
    border-radius: 4px;
}}

/* Progress Bar */
QProgressBar {{
    border: 2px solid #2C2C2C;
    border-radius: 8px;
    text-align: center;
    background-color: #1F1F1F;
    height: 20px;
}}

QProgressBar::chunk {{
    background-color: #2962FF;
    border-radius: 8px;
}}

/* Spin Box */
QSpinBox, QDoubleSpinBox {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    border: 2px solid #2C2C2C;
    border-radius: 8px;
    padding: 6px 12px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    background-color: #2C2C2C;
    border: none;
    border-radius: 4px;
    width: 18px;
    height: 18px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background-color: #2962FF;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: #2C2C2C;
    border: none;
    border-radius: 4px;
    width: 18px;
    height: 18px;
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #2962FF;
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 10px;
    height: 10px;
    image: url({UP_ARROW_IMAGE});
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 10px;
    height: 10px;
    image: url({DOWN_ARROW_IMAGE});
}}

/* Tool Buttons */
QToolButton {{
    background-color: #1F1F1F;
    border: 2px solid #2C2C2C;
    padding: 6px;
    border-radius: 6px;
}}

QToolButton:hover {{
    background-color: #2C2C2C;
    border-color: #2962FF;
}}

QToolButton:pressed {{
    background-color: #2962FF;
    border-color: #1F1F1F;
}}

/* Dock Widgets */
QDockWidget {{
    background-color: #1F1F1F;
    border: 2px solid #2C2C2C;
}}

QDockWidget::title {{
    background-color: #121212;
    color: #E0E0E0;
    padding: 8px;
    font-weight: bold;
    border-bottom: 2px solid #2C2C2C;
}}

/* Table Views */
QTableView {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    gridline-color: #2C2C2C;
    border: 2px solid #2C2C2C;
    selection-background-color: #2962FF;
    selection-color: #FFFFFF;
}}

QHeaderView::section {{
    background-color: #2C2C2C;
    color: #E0E0E0;
    padding: 8px;
    border: 1px solid #2962FF;
    font-weight: bold;
}}

QTableView::item:selected {{
    background-color: #2962FF;
    color: #FFFFFF;
}}

/* Tree Views */
QTreeView {{
    background-color: #1F1F1F;
    color: #E0E0E0;
    border: 2px solid #2C2C2C;
}}

QTreeView::item:selected {{
    background-color: #2962FF;
    color: #FFFFFF;
}}

/* Enhanced Scroll Bars */
QScrollBar:horizontal, QScrollBar:vertical {{
    background: #1F1F1F;
    border: 1px solid #2C2C2C;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal, QScrollBar::handle:vertical {{
    background: #424242;
    min-width: 20px;
    min-height: 20px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {{
    background: #616161;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: none;
    border: none;
    height: 0;
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
"""

MOVE_HISTORY_STYLESHEET = """
QListWidget {
    background-color: #2B2B2B;
    color: #E0E0E0;
    border: 1px solid #444444;
    border-radius: 6px;
    padding: 8px;
    margin: 10px;
}
QListWidget::item {
    padding: 10px;
    margin: 4px 0;
    border-radius: 4px;
    background-color: #1E1E1E;
    border: 1px solid #333333;
}
QListWidget::item:hover {
    background-color: #3A3A3A;
}
QListWidget::item:selected {
    background-color: #555555;
    color: #FFFFFF;
    border: 1px solid #777777;
}
"""

BUTTON_STYLE = """
QPushButton {
    background-color: #1E1E1E;
    color: #E0E0E0;
    border: none;
    padding: 10px 15px;
    margin: 5px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #333333;
}
QPushButton:pressed {
    background-color: #2962FF;
    border-color: #1F1F1F;
    color: #FFFFFF;
}
QPushButton:focus {
    outline: none;
    border: 2px solid #2962FF;
}
"""