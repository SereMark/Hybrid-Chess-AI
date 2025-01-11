from PyQt5.QtWidgets import QWidget, QTextEdit, QProgressBar, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QFileDialog, QSizePolicy, QStyle, QFrame
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QColor

class BaseTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Threading and Worker Setup
        self.thread = None
        self.worker = None

        # UI Elements
        self.log_text_edit = None
        self.progress_bar = None
        self.remaining_time_label = None
        self.save_checkpoints_checkbox = None
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

        # State Management
        self.init_ui_state = True

    def setup_subtab(self, main_layout, intro_text, progress_title, log_title, visualization_title, visualization_widget, control_buttons_config):
        # Main Layout Configuration
        main_layout.setSpacing(15)

        # Introduction Label
        self.intro_label = QLabel(intro_text)
        self.intro_label.setWordWrap(True)
        self.intro_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.intro_label)

        # Control Group (Actions)
        self.control_group = QGroupBox("Actions")
        cg_layout = QVBoxLayout(self.control_group)
        cg_layout.setSpacing(10)
        main_layout.addWidget(self.control_group)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Progress Group
        self.progress_group = QGroupBox(progress_title)
        pg_layout = QVBoxLayout(self.progress_group)
        pg_layout.setSpacing(10)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.remaining_time_label = QLabel("Time Left: N/A")
        self.remaining_time_label.setAlignment(Qt.AlignCenter)
        pg_layout.addWidget(self.progress_bar)
        pg_layout.addWidget(self.remaining_time_label)
        self.progress_group.setLayout(pg_layout)
        main_layout.addWidget(self.progress_group)
        self.progress_group.setVisible(False)

        # Log Group
        self.log_group = QGroupBox(log_title)
        lg_layout = QVBoxLayout(self.log_group)
        lg_layout.setSpacing(10)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Logs will appear here...")
        lg_layout.addWidget(self.log_text_edit)
        self.log_group.setLayout(lg_layout)
        main_layout.addWidget(self.log_group)
        self.log_group.setVisible(False)

        # Visualization Group
        self.visualization_group = None
        if visualization_widget is not None:
            self.visualization_group = QGroupBox(visualization_title)
            vis_layout = QVBoxLayout(self.visualization_group)
            visualization_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            vis_layout.addWidget(visualization_widget)
            main_layout.addWidget(self.visualization_group)
            self.visualization_group.setVisible(False)

        # Control Buttons
        layout = QHBoxLayout()

        # Start and Stop Buttons
        self.start_button = QPushButton(control_buttons_config.get("start_text", ""))
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stop_button = QPushButton(control_buttons_config.get("stop_text", ""))
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # Button Callbacks
        if control_buttons_config.get("start_callback"):
            self.start_button.clicked.connect(control_buttons_config["start_callback"])
        if control_buttons_config.get("stop_callback"):
            self.stop_button.clicked.connect(control_buttons_config["stop_callback"])

        # Pause and Resume Buttons
        if all(key in control_buttons_config for key in ["pause_text", "resume_text", "pause_callback", "resume_callback"]):
            self.pause_button = QPushButton(control_buttons_config["pause_text"])
            self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.resume_button = QPushButton(control_buttons_config["resume_text"])
            self.resume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            layout.addWidget(self.pause_button)
            layout.addWidget(self.resume_button)
            self.pause_button.clicked.connect(control_buttons_config["pause_callback"])
            self.resume_button.clicked.connect(control_buttons_config["resume_callback"])

        layout.addStretch()
        cg_layout.addLayout(layout)

        # Toggle Buttons
        self.show_logs_button = QPushButton("Show Logs")
        self.show_logs_button.setCheckable(True)
        self.show_logs_button.setChecked(True)
        self.show_logs_button.clicked.connect(self.show_logs_view)

        self.show_graphs_button = QPushButton("Show Graphs")
        self.show_graphs_button.setCheckable(True)
        self.show_graphs_button.setChecked(False)
        self.show_graphs_button.clicked.connect(self.show_graphs_view)

        # Synchronize Toggle Buttons
        self.show_logs_button.clicked.connect(lambda: self.show_graphs_button.setChecked(not self.show_logs_button.isChecked()))
        self.show_graphs_button.clicked.connect(lambda: self.show_logs_button.setChecked(not self.show_graphs_button.isChecked()))

        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(self.show_logs_button)
        toggle_layout.addWidget(self.show_graphs_button)
        cg_layout.addLayout(toggle_layout)

        # Start New Button
        self.start_new_button = QPushButton("Reconfigure Parameters")
        self.start_new_button.clicked.connect(control_buttons_config.get("start_new_callback", None))
        cg_layout.addWidget(self.start_new_button)

        # Default Visibility and States
        self.show_logs_button.setVisible(False)
        self.show_graphs_button.setVisible(False)
        self.start_new_button.setVisible(False)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(False)
        if hasattr(self, 'resume_button'):
            self.resume_button.setEnabled(False)

        # Final Layout Setup
        self.setLayout(main_layout)

    def show_logs_view(self):
        if self.show_logs_button and self.show_logs_button.isChecked():
            if self.show_graphs_button:
                self.show_graphs_button.setChecked(False)
            if self.log_group:
                self.log_group.setVisible(True)
            if self.visualization_group:
                self.visualization_group.setVisible(False)

    def show_graphs_view(self):
        if self.show_graphs_button and self.show_graphs_button.isChecked():
            if self.show_logs_button:
                self.show_logs_button.setChecked(False)
            if self.log_group:
                self.log_group.setVisible(False)
            if self.visualization_group:
                self.visualization_group.setVisible(True)

    def start_worker(self, worker_class, *args, **kwargs):
        if self.thread is not None and self.thread.isRunning():
            return False

        self.thread = QThread()
        self.worker = worker_class(*args, **kwargs)
        self.worker.moveToThread(self.thread)

        # Thread and Worker Connections
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_worker_finished)
        self.worker.paused.connect(self.on_worker_paused)
        self.worker.log_update.connect(self.handle_log_update)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.time_left_update.connect(self.update_time_left)

        self.thread.start()
        return True

    def on_worker_finished(self):
        self.worker = None
        self.thread = None

        # Update UI Elements
        if self.pause_button:
            self.pause_button.setEnabled(False)
        if self.resume_button:
            self.resume_button.setEnabled(False)
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.start_new_button:
            self.start_new_button.setVisible(True)

    def on_worker_paused(self, is_paused):
        if self.pause_button and self.resume_button:
            self.pause_button.setEnabled(not is_paused)
            self.resume_button.setEnabled(is_paused)

    def handle_log_update(self, level, message):
        color_map = {'DEBUG': 'gray', 'INFO': 'black', 'WARNING': 'orange', 'ERROR': 'red', 'CRITICAL': 'darkred'}
        self.log_text_edit.setTextColor(QColor(color_map.get(level.upper(), 'black')))
        self.log_text_edit.append(message)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")

    def update_time_left(self, time_left_str):
        self.remaining_time_label.setText(f"Time Left: {time_left_str}")

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

    def toggle_widget_state(self, widgets, state=None, attribute="enabled"):
        if not isinstance(widgets, list):
            widgets = [widgets]
        for widget in widgets:
            if attribute == "enabled":
                widget.setEnabled(not widget.isEnabled() if state is None else state)
            elif attribute == "visible":
                widget.setVisible(not widget.isVisible() if state is None else state)

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

    def browse_file(self, input_field, title, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, title, input_field.text(), file_filter)
        if file_path:
            input_field.setText(file_path)

    def browse_dir(self, input_field, title):
        dir_path = QFileDialog.getExistingDirectory(self, title, input_field.text())
        if dir_path:
            input_field.setText(dir_path)

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