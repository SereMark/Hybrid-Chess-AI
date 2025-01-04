from PyQt5.QtWidgets import QWidget, QTextEdit, QProgressBar, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QFileDialog, QSizePolicy, QStyle, QFrame
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QColor

class BaseTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self.log_text_edit = None
        self.progress_bar = None
        self.remaining_time_label = None
        self.save_checkpoints_checkbox = None
        self.setContentsMargins(10, 10, 10, 10)
        self.init_ui_state = True
        self.showing_logs = True
        self.intro_label = None
        self.control_group = None
        self.progress_group = None
        self.log_group = None
        self.visualization_group = None
        self.start_button = None
        self.stop_button = None
        self.pause_button = None
        self.resume_button = None
        self.show_logs_button = None
        self.show_graphs_button = None
        self.start_new_button = None
        self.toggle_buttons_layout = None
        self.visualization = None

    def create_subtab_layout(self, layout, intro_text, progress_title, log_title, visualization_title, visualization_widget=None):
        self.intro_label = QLabel(intro_text)
        self.intro_label.setWordWrap(True)
        self.intro_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.intro_label)
        self.control_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(self.control_group)
        cg_layout.setSpacing(10)
        layout.addWidget(self.control_group)
        self.toggle_buttons_layout = None
        self.start_new_button = None
        separator = self.create_separator()
        layout.addWidget(separator)
        self.progress_group = QGroupBox(progress_title)
        pg_layout = QVBoxLayout(self.progress_group)
        pg_layout.setSpacing(10)
        pg_layout.addLayout(self.create_progress_layout())
        self.log_group = QGroupBox(log_title)
        lg_layout = QVBoxLayout(self.log_group)
        lg_layout.setSpacing(10)
        self.log_text_edit = self.create_log_text_edit()
        lg_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(lg_layout)
        if visualization_widget is not None:
            self.visualization_group = self.create_visualization_group(visualization_widget, visualization_title)
        layout.addWidget(self.progress_group)
        layout.addWidget(self.log_group)
        if self.visualization_group is not None:
            layout.addWidget(self.visualization_group)
        self.progress_group.setVisible(False)
        self.log_group.setVisible(False)
        if self.visualization_group is not None:
            self.visualization_group.setVisible(False)

    def setup_subtab(self, main_layout, intro_text, progress_title, log_title, visualization_title, visualization_widget, control_buttons_config, start_new_text, spacing=15, show_logs_default=True, show_graphs_default=False):
        main_layout.setSpacing(spacing)
        self.create_subtab_layout(main_layout, intro_text, progress_title, log_title, visualization_title, visualization_widget)
        cg_layout = self.control_group.layout()
        control_buttons_layout = self.create_control_buttons(
            control_buttons_config.get("start_text", ""),
            control_buttons_config.get("stop_text", ""),
            control_buttons_config.get("start_callback", None),
            control_buttons_config.get("stop_callback", None),
            pause_text=control_buttons_config.get("pause_text", None),
            resume_text=control_buttons_config.get("resume_text", None),
            pause_callback=control_buttons_config.get("pause_callback", None),
            resume_callback=control_buttons_config.get("resume_callback", None)
        )
        cg_layout.addLayout(control_buttons_layout)
        self.toggle_buttons_layout = self.create_log_graph_buttons(self.show_logs_view, self.show_graphs_view, "Show Logs", "Show Graphs", show_logs_default)
        cg_layout.addLayout(self.toggle_buttons_layout)
        self.start_new_button = self.create_start_new_button(start_new_text, control_buttons_config.get("start_new_callback", None))
        cg_layout.addWidget(self.start_new_button)
        self.setLayout(main_layout)
        self.show_logs_button.setVisible(False)
        self.show_graphs_button.setVisible(False)
        self.start_new_button.setVisible(False)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)

    def start_worker(self, worker_class, *args, **kwargs):
        if self.thread is not None and self.thread.isRunning():
            return False
        self.thread = QThread()
        self.worker = worker_class(*args, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_worker_finished)
        self.worker.log_update.connect(self.handle_log_update)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)
        self.worker.paused.connect(self.on_worker_paused)
        self.thread.start()
        return True

    def handle_log_update(self, level, message):
        color = self.get_color_for_level(level)
        self.log_text_edit.setTextColor(QColor(color))
        self.log_text_edit.append(message)

    def get_color_for_level(self, level):
        color_map = {
            'DEBUG': 'gray',
            'INFO': 'black',
            'WARNING': 'orange',
            'ERROR': 'red',
            'CRITICAL': 'darkred'
        }
        return color_map.get(level.upper(), 'black')

    def pause_worker(self):
        if self.worker:
            self.worker.pause()

    def resume_worker(self):
        if self.worker:
            self.worker.resume()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            if self.pause_button:
                self.pause_button.setEnabled(False)
            if self.resume_button:
                self.resume_button.setEnabled(False)

    def on_worker_paused(self, is_paused):
        if self.pause_button and self.resume_button:
            self.pause_button.setEnabled(not is_paused)
            self.resume_button.setEnabled(is_paused)

    def on_worker_finished(self):
        self.worker = None
        self.thread = None
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        if self.start_new_button:
            self.start_new_button.setVisible(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat("Progress: {}%".format(value))

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText("Time Left: {}".format(time_left_str))

    def create_log_text_edit(self):
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlaceholderText("Logs will appear here...")
        return text_edit

    def create_progress_layout(self):
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: N/A")
        self.remaining_time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.remaining_time_label)
        return layout

    def create_browse_layout(self, line_edit, browse_button):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        return layout

    def create_interval_widget(self, prefix, input_field, suffix):
        widget = QWidget()
        widget_layout = QHBoxLayout(widget)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        prefix_label = QLabel(prefix)
        widget_layout.addWidget(prefix_label)
        widget_layout.addWidget(input_field)
        widget_layout.addWidget(QLabel(suffix))
        widget_layout.addStretch()
        return widget

    def create_visualization_group(self, visualization_widget, title):
        group = QGroupBox(title)
        vis_layout = QVBoxLayout(group)
        visualization_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vis_layout.addWidget(visualization_widget)
        return group

    def create_control_buttons(self, start_text, stop_text, start_callback, stop_callback, pause_text=None, resume_text=None, pause_callback=None, resume_callback=None):
        layout = QHBoxLayout()
        self.start_button = QPushButton(start_text)
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stop_button = QPushButton(stop_text)
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        if start_callback:
            self.start_button.clicked.connect(start_callback)
        if stop_callback:
            self.stop_button.clicked.connect(stop_callback)
        if pause_text and resume_text and pause_callback and resume_callback:
            self.pause_button = QPushButton(pause_text)
            self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.resume_button = QPushButton(resume_text)
            self.resume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            layout.addWidget(self.pause_button)
            layout.addWidget(self.resume_button)
            self.pause_button.clicked.connect(pause_callback)
            self.resume_button.clicked.connect(resume_callback)
        layout.addStretch()
        return layout

    def browse_file(self, input_field, title, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, title, input_field.text(), file_filter)
        if file_path:
            input_field.setText(file_path)

    def browse_dir(self, input_field, title):
        dir_path = QFileDialog.getExistingDirectory(self, title, input_field.text())
        if dir_path:
            input_field.setText(dir_path)

    def toggle_widget_state(self, widgets, state=None, attribute="enabled"):
        if not isinstance(widgets, list):
            widgets = [widgets]
        for widget in widgets:
            if attribute == "enabled":
                widget.setEnabled(not widget.isEnabled() if state is None else state)
            elif attribute == "visible":
                widget.setVisible(not widget.isVisible() if state is None else state)

    def setup_batch_size_control(self, automatic_batch_size_checkbox, batch_size_input):
        automatic_batch_size_checkbox.toggled.connect(lambda checked: self.toggle_widget_state([batch_size_input], state=not checked, attribute="enabled"))
        self.toggle_widget_state([batch_size_input], state=not automatic_batch_size_checkbox.isChecked(), attribute="enabled")

    def setup_checkpoint_controls(self, save_checkpoints_checkbox, checkpoint_type_combo, interval_widgets):
        self.save_checkpoints_checkbox = save_checkpoints_checkbox
        save_checkpoints_checkbox.stateChanged.connect(lambda state: self.on_checkpoint_enabled_changed(state, checkpoint_type_combo, interval_widgets))
        checkpoint_type_combo.currentTextChanged.connect(lambda text: self.on_checkpoint_type_changed(text, interval_widgets))
        self.on_checkpoint_enabled_changed(save_checkpoints_checkbox.checkState(), checkpoint_type_combo, interval_widgets)

    def on_checkpoint_enabled_changed(self, state, checkpoint_type_combo, interval_widgets):
        is_enabled = state == Qt.Checked
        self.toggle_widget_state([checkpoint_type_combo], state=is_enabled, attribute="enabled")
        self.on_checkpoint_type_changed(checkpoint_type_combo.currentText(), interval_widgets)

    def on_checkpoint_type_changed(self, text, interval_widgets):
        is_enabled = self.save_checkpoints_checkbox.isChecked()
        t = text.lower()
        for key, widget in interval_widgets.items():
            visible = is_enabled and (key == t)
            self.toggle_widget_state([widget], state=visible, attribute="visible")

    def create_log_graph_buttons(self, logs_callback, graphs_callback, logs_text="Show Logs", graphs_text="Show Graphs", logs_default=True):
        layout = QHBoxLayout()
        self.show_logs_button = QPushButton(logs_text)
        self.show_logs_button.setCheckable(True)
        self.show_logs_button.setChecked(logs_default)
        self.show_logs_button.clicked.connect(logs_callback)
        self.show_graphs_button = QPushButton(graphs_text)
        self.show_graphs_button.setCheckable(True)
        self.show_graphs_button.setChecked(not logs_default)
        self.show_graphs_button.clicked.connect(graphs_callback)
        self.show_logs_button.clicked.connect(lambda: self.show_graphs_button.setChecked(not self.show_logs_button.isChecked()))
        self.show_graphs_button.clicked.connect(lambda: self.show_logs_button.setChecked(not self.show_graphs_button.isChecked()))
        layout.addWidget(self.show_logs_button)
        layout.addWidget(self.show_graphs_button)
        return layout

    def create_start_new_button(self, text, callback):
        button = QPushButton(text)
        if callback:
            button.clicked.connect(callback)
        return button

    def create_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def show_logs_view(self):
        if self.show_logs_button and self.show_logs_button.isChecked():
            if self.show_graphs_button:
                self.show_graphs_button.setChecked(False)
            if self.log_group:
                self.log_group.setVisible(True)
            if self.visualization_group:
                self.visualization_group.setVisible(False)
            self.showing_logs = True

    def show_graphs_view(self):
        if self.show_graphs_button and self.show_graphs_button.isChecked():
            if self.show_logs_button:
                self.show_logs_button.setChecked(False)
            if self.log_group:
                self.log_group.setVisible(False)
            if self.visualization_group:
                self.visualization_group.setVisible(True)
            self.showing_logs = False