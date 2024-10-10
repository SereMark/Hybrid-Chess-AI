from PyQt5.QtWidgets import QLineEdit, QLabel, QPushButton, QComboBox, QSpinBox, QHBoxLayout

def create_labeled_input(label_text, default_text="", button_text="Browse", file_filter=""):
    label = QLabel(label_text)
    input_field = QLineEdit(default_text)
    browse_button = QPushButton(button_text)
    layout = QHBoxLayout()
    layout.addWidget(input_field)
    layout.addWidget(browse_button)
    return label, input_field, browse_button, layout

def create_labeled_combo(label_text, items=None):
    label = QLabel(label_text)
    combo_box = QComboBox()
    if items:
        combo_box.addItems(items)
    layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(combo_box)
    return label, combo_box, layout

def create_labeled_spinbox(label_text, min_value=1, max_value=180, default_value=10):
    label = QLabel(label_text)
    spin_box = QSpinBox()
    spin_box.setRange(min_value, max_value)
    spin_box.setValue(default_value)
    layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(spin_box)
    return label, spin_box, layout