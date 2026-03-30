import os

from PySide6.QtCore import QAbstractTableModel, QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pacman_globals import (
    ACTION_NAMES,
    ACTIONS,
    ALPHA,
    CELL_SIZE,
    EPS_DECAY,
    EPS_MIN,
    EPS_START,
    GAMMA,
    MA_WINDOW,
    QTABLE_VIEW_MAX_ROWS,
    QTABLE_VIEW_REFRESH_MS,
    RUN_TIMER_INTERVAL_MS,
    SPACE_STEP_INTERVAL_MS,
    STEPS_PER_TICK,
    QTABLE_PATH,
)
from pacman_qlearning import PacmanWorld

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    _HAS_MPL = True
except Exception:
    FigureCanvas = None
    Figure = None
    FuncFormatter = None
    MaxNLocator = None
    _HAS_MPL = False


class WorldView(QWidget):
    GHOST_COLORS = [QColor("#ff4f64"), QColor("#61e7ff"), QColor("#ffb347")]

    def __init__(self, world: PacmanWorld):
        super().__init__()
        self.world = world
        self.setFixedSize(world.width * CELL_SIZE, world.height * CELL_SIZE)
        self.setFocusPolicy(Qt.StrongFocus)

    def _draw_walls(self, p: QPainter):
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.layout[y][x] != "#":
                    continue
                x0 = x * CELL_SIZE
                y0 = y * CELL_SIZE
                p.fillRect(x0, y0, CELL_SIZE, CELL_SIZE, QColor("#1f53d5"))
                p.setPen(QPen(QColor("#2f7bff"), 1))
                p.drawRect(x0, y0, CELL_SIZE, CELL_SIZE)

    def _draw_pellets(self, p: QPainter):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#f7f4a5"))
        r = CELL_SIZE * 0.10
        for px, py in self.world.pellets:
            cx = px * CELL_SIZE + CELL_SIZE / 2
            cy = py * CELL_SIZE + CELL_SIZE / 2
            p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

    def _draw_pacman(self, p: QPainter):
        px, py = self.world.pacman
        x0 = px * CELL_SIZE + 5
        y0 = py * CELL_SIZE + 5
        x1 = x0 + CELL_SIZE - 10
        y1 = y0 + CELL_SIZE - 10

        mouth_angles = {0: 120, 1: 30, 2: 300, 3: 210}
        start = mouth_angles.get(self.world.last_action, 30)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#ffd84d"))
        p.drawPie(x0, y0, x1 - x0, y1 - y0, int(start * 16), int(300 * 16))

    def _draw_ghost(self, p: QPainter, gx: int, gy: int, color: QColor):
        x0 = gx * CELL_SIZE + 5
        y0 = gy * CELL_SIZE + 5
        x1 = x0 + CELL_SIZE - 10
        y1 = y0 + CELL_SIZE - 10

        p.setPen(Qt.NoPen)
        p.setBrush(color)

        p.drawPie(x0, y0, x1 - x0, y1 - y0, 0, int(180 * 16))

        body_top = int((y0 + y1) / 2)
        p.drawRect(x0, body_top, x1 - x0, (y1 - 6) - body_top)

        step = (x1 - x0) / 4.0
        points = [
            (x0, y1 - 6),
            (x0 + step, y1),
            (x0 + 2 * step, y1 - 6),
            (x0 + 3 * step, y1),
            (x1, y1 - 6),
        ]
        p.drawPolygon(QPolygonF([QPointF(x, y) for (x, y) in points]))

        p.setBrush(QColor("white"))
        eye_r = 3
        lx = x0 + (x1 - x0) * 0.35
        rx = x0 + (x1 - x0) * 0.65
        ey = y0 + (y1 - y0) * 0.38
        p.drawEllipse(int(lx - eye_r), int(ey - eye_r), 2 * eye_r, 2 * eye_r)
        p.drawEllipse(int(rx - eye_r), int(ey - eye_r), 2 * eye_r, 2 * eye_r)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.fillRect(0, 0, self.width(), self.height(), QColor("#070714"))
        self._draw_walls(p)
        self._draw_pellets(p)
        self._draw_pacman(p)

        for index, (gx, gy) in enumerate(self.world.ghosts):
            color = self.GHOST_COLORS[index % len(self.GHOST_COLORS)]
            self._draw_ghost(p, gx, gy, color)

        p.end()


class RewardPlot(QWidget):
    def __init__(self, world: PacmanWorld):
        super().__init__()
        self.world = world
        self.setFixedSize(380, 120)
        self._last_signature: tuple[int, float] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_MPL:
            self._fallback = QLabel("Matplotlib not installed.\nRun: pip install matplotlib")
            self._fallback.setAlignment(Qt.AlignCenter)
            self._fallback.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            layout.addWidget(self._fallback)
            self.canvas = None
            self.fig = None
            self.ax = None
            self._line = None
            return

        dpi = 100
        self.fig = Figure(figsize=(self.width() / dpi, self.height() / dpi), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self._line, = self.ax.plot([], [], linewidth=1.8)
        self.ax.grid(True, alpha=0.25)

        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)

        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        def _fmt_steps(x, _pos):
            n = int(x)
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.0f}k"
            return str(n)

        self.ax.xaxis.set_major_formatter(FuncFormatter(_fmt_steps))
        self.ax.tick_params(axis="both", labelsize=8)

        self.fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.28)
        self.refresh(force=True)

    def refresh(self, force: bool = False):
        if not _HAS_MPL or self.canvas is None:
            return

        history = self.world.reward_ma_history
        if not history:
            self._line.set_data([], [])
            self.ax.set_xlim(0, MA_WINDOW)
            self.canvas.draw_idle()
            return

        signature = (len(history), float(history[-1]))
        if (not force) and self._last_signature == signature:
            return
        self._last_signature = signature

        n = len(history)
        xs = [i * MA_WINDOW for i in range(n)]
        ys = history
        self._line.set_data(xs, ys)

        self.ax.set_xlim(0, max(MA_WINDOW, int((n - 1) * MA_WINDOW)))
        ymin = min(ys)
        ymax = max(ys)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0
        pad = 0.08 * (ymax - ymin)
        self.ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas.draw_idle()


class QTableModel(QAbstractTableModel):
    _STATE_COLUMNS = ["dx_g", "dy_g", "d_g_bin", "d_p_bin", "pellet_dir", "walls"]

    def __init__(self):
        super().__init__()
        self._rows: list[tuple[tuple[int, int, int, int, int, int], list[float]]] = []

    def set_rows(self, rows: list[tuple[tuple[int, int, int, int, int, int], list[float]]]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, _parent=None) -> int:
        return len(self._rows)

    def columnCount(self, _parent=None) -> int:
        return 11

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            cols = self._STATE_COLUMNS + ["Q(U)", "Q(R)", "Q(D)", "Q(L)", "best"]
            return cols[section] if 0 <= section < len(cols) else None
        return str(section)

    def data(self, index, role: int = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        state, q = self._rows[index.row()]
        col = index.column()

        if 0 <= col <= 5:
            return str(state[col])

        if 6 <= col <= 9:
            return f"{q[col - 6]:.3f}"

        if col == 10:
            best = max(range(4), key=lambda a: q[a])
            return ACTION_NAMES[best]

        return None


class QTableDialog(QDialog):
    def __init__(self, world: PacmanWorld, parent=None):
        super().__init__(parent)
        self.world = world
        self.setWindowTitle("Q-table")
        self.setMinimumSize(720, 480)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.model = QTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.table, 1)
        self.setLayout(layout)

        self._timer = QTimer(self)
        self._timer.setInterval(QTABLE_VIEW_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self.refresh()

    def closeEvent(self, event):
        self._timer.stop()
        return super().closeEvent(event)

    def refresh(self):
        items = list(self.world.agent.q_table.items())
        items.sort(key=lambda kv: max(kv[1]), reverse=True)

        total = len(items)
        shown = min(total, QTABLE_VIEW_MAX_ROWS)
        rows = items[:shown]

        self.lbl_info.setText(
            f"states learned: {total}    showing: {shown}    refresh: {QTABLE_VIEW_REFRESH_MS}ms"
        )
        self.model.set_rows(rows)


class MainWindow(QMainWindow):
    def __init__(self, qtable_path: str = QTABLE_PATH):
        super().__init__()
        self.setWindowTitle("Pacman Q-Learning (PySide6)")

        self.qtable_path = qtable_path
        self.world = PacmanWorld()
        self.view = WorldView(self.world)
        self.reward_plot = RewardPlot(self.world)
        self.q_table_dialog: QTableDialog | None = None

        if os.path.exists(self.qtable_path):
            try:
                self.world.agent.load(self.qtable_path)
            except Exception:
                pass

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.setInterval(RUN_TIMER_INTERVAL_MS)

        self.space_timer = QTimer(self)
        self.space_timer.timeout.connect(self.single_step)
        self.space_timer.setInterval(SPACE_STEP_INTERVAL_MS)

        self.btn_run = QPushButton("Run")
        self.btn_step = QPushButton("Step")
        self.btn_undo = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset")
        self.btn_save_q = QPushButton("Save Q-table")
        self.btn_load_q = QPushButton("Load Q-table")
        self.btn_view_q = QPushButton("View Q-table")
        self.btn_reset_q = QPushButton("Reset Q-table")

        self.combo_auto = QComboBox()
        self.combo_auto.addItems(["Q-Learning", "Q-Table", "Random"])

        self.chk_auto_restart = QCheckBox("Auto-Restart")
        self.chk_auto_restart.setChecked(True)

        self.chk_eps_greedy = QCheckBox("epsilon-greedy")
        self.chk_eps_greedy.setChecked(True)
        self.chk_eps_greedy.toggled.connect(self.on_toggle_eps_greedy)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(4)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setValue(ALPHA)
        self.spin_alpha.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setDecimals(4)
        self.spin_gamma.setSingleStep(0.01)
        self.spin_gamma.setRange(0.0, 0.9999)
        self.spin_gamma.setValue(GAMMA)
        self.spin_gamma.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_eps_start = QDoubleSpinBox()
        self.spin_eps_start.setDecimals(4)
        self.spin_eps_start.setSingleStep(0.05)
        self.spin_eps_start.setRange(0.0, 1.0)
        self.spin_eps_start.setValue(EPS_START)
        self.spin_eps_start.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_eps_min = QDoubleSpinBox()
        self.spin_eps_min.setDecimals(4)
        self.spin_eps_min.setSingleStep(0.01)
        self.spin_eps_min.setRange(0.0, 1.0)
        self.spin_eps_min.setValue(EPS_MIN)
        self.spin_eps_min.valueChanged.connect(self.on_hyperparams_changed)

        self.spin_eps_decay = QDoubleSpinBox()
        self.spin_eps_decay.setDecimals(6)
        self.spin_eps_decay.setSingleStep(0.0001)
        self.spin_eps_decay.setRange(0.9, 0.999999)
        self.spin_eps_decay.setValue(EPS_DECAY)
        self.spin_eps_decay.valueChanged.connect(self.on_hyperparams_changed)

        self.lbl_status = QLabel()
        self.lbl_status.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_status.setFont(QFont("Arial", 10))
        self.lbl_status.setMinimumHeight(170)

        self.lbl_plot_title = QLabel("Avg reward per 500 steps")
        self.lbl_plot_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(self.single_step)
        self.btn_undo.clicked.connect(self.undo_last_step)
        self.btn_reset.clicked.connect(self.reset_world)
        self.btn_save_q.clicked.connect(self.save_q_table)
        self.btn_load_q.clicked.connect(self.load_q_table)
        self.btn_view_q.clicked.connect(self.open_q_table_viewer)
        self.btn_reset_q.clicked.connect(self.reset_q_table)

        for b in (
            self.btn_run,
            self.btn_step,
            self.btn_undo,
            self.btn_reset,
            self.btn_save_q,
            self.btn_load_q,
            self.btn_view_q,
            self.btn_reset_q,
        ):
            b.setMinimumHeight(26)

        controls = QVBoxLayout()

        top_buttons = QHBoxLayout()
        top_buttons.setSpacing(6)
        top_buttons.addWidget(self.btn_run)
        top_buttons.addWidget(self.btn_step)
        top_buttons.addWidget(self.btn_undo)
        top_buttons.addWidget(self.btn_reset)
        controls.addLayout(top_buttons)

        controls.addSpacing(6)
        controls.addWidget(QLabel("Policy"))
        controls.addWidget(self.combo_auto)
        controls.addWidget(self.chk_auto_restart)

        controls.addSpacing(6)
        controls.addWidget(QLabel("Hyperparameters"))
        hyper_form = QFormLayout()
        hyper_form.setContentsMargins(0, 0, 0, 0)
        hyper_form.setHorizontalSpacing(8)
        hyper_form.setVerticalSpacing(4)
        hyper_form.addRow("ALPHA:", self.spin_alpha)
        hyper_form.addRow("GAMMA:", self.spin_gamma)
        hyper_form.addRow("EPS_START:", self.spin_eps_start)
        hyper_form.addRow("EPS_MIN:", self.spin_eps_min)
        hyper_form.addRow("EPS_DECAY:", self.spin_eps_decay)
        controls.addLayout(hyper_form)
        controls.addWidget(self.chk_eps_greedy)

        controls.addSpacing(6)
        qtable_buttons = QHBoxLayout()
        qtable_buttons.setSpacing(6)
        qtable_buttons.addWidget(self.btn_save_q)
        qtable_buttons.addWidget(self.btn_load_q)
        qtable_buttons.addWidget(self.btn_view_q)
        qtable_buttons.addWidget(self.btn_reset_q)
        controls.addLayout(qtable_buttons)

        controls.addSpacing(10)
        controls.addWidget(self.lbl_status)
        controls.addSpacing(6)
        controls.addWidget(self.lbl_plot_title)
        controls.addWidget(self.reward_plot)
        controls.addSpacing(10)
        controls.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self.view)
        root.addLayout(controls)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.on_hyperparams_changed()
        self.update_status_panel()

    def update_status_panel(self):
        mode = self.combo_auto.currentText()
        done_text = "yes" if self.world.done else "no"
        info = self.world.last_info

        exploration_rate = self.world.agent.epsilon if self.world.agent.use_epsilon_greedy else 0.0

        self.lbl_status.setText(
            f"steps: {self.world.steps}\n"
            f"alpha: {self.world.agent.alpha:.3f}   gamma: {self.world.agent.gamma:.3f}\n"
            f"epsilon: {self.world.agent.epsilon:.3f} (min {self.world.agent.epsilon_min:.2f}, decay {self.world.agent.epsilon_decay:.6f})\n"
            f"exploration rate: {exploration_rate * 100.0:.1f}%\n"
            f"policy: {mode}\n"
            f"auto-restart: {self.chk_auto_restart.isChecked()}\n"
            f"states learned: {len(self.world.agent.q_table)}\n"
            f"score: {self.world.score:.2f}\n"
            f"ma500(r): {self.world.reward_ma:.3f}\n\n"
            f"last action: {ACTION_NAMES.get(self.world.last_action, '-')}\n"
            f"last reward: {self.world.last_reward:.2f}\n"
            f"pellet eaten: {info.pellet} | invalid: {info.invalid}\n"
            f"caught: {info.caught} | done: {done_text}\n"
            f"pellets collected: {self.world.collected_pellets}"
        )

    def on_toggle_eps_greedy(self, checked: bool):
        self.world.agent.use_epsilon_greedy = bool(checked)
        self.update_status_panel()

    def on_hyperparams_changed(self):
        eps_min = float(self.spin_eps_min.value())
        if self.spin_eps_start.value() < eps_min:
            self.spin_eps_start.blockSignals(True)
            self.spin_eps_start.setValue(eps_min)
            self.spin_eps_start.blockSignals(False)

        self.world.agent.set_hyperparams(
            alpha=float(self.spin_alpha.value()),
            gamma=float(self.spin_gamma.value()),
            eps_start=float(self.spin_eps_start.value()),
            eps_min=float(self.spin_eps_min.value()),
            eps_decay=float(self.spin_eps_decay.value()),
            set_current_eps_to_start=False,
        )
        self.update_status_panel()

    def _one_step(self):
        mode = self.combo_auto.currentText()

        if mode == "Q-Learning":
            self.world.step_learning()
        else:
            action = self.world.choose_auto_action(mode)
            self.world.step_manual(action)

        self.update_status_panel()
        self.view.update()
        if self.world.reward_ma_dirty:
            self.reward_plot.refresh()
            self.world.reward_ma_dirty = False

        if self.world.done and self.timer.isActive():
            if self.chk_auto_restart.isChecked():
                randomize_positions = self.combo_auto.currentText() == "Q-Learning"
                self.world.reset(clear_learning_curve=False, randomize_positions=randomize_positions)
                self.view.update()
                self.update_status_panel()
            else:
                self.timer.stop()
                self.btn_run.setText("Run")

    def on_tick(self):
        mode = self.combo_auto.currentText()
        loops = STEPS_PER_TICK if mode == "Q-Learning" else 1
        for _ in range(loops):
            self._one_step()
            if self.world.done:
                break

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("Run")
            return

        if self.world.done:
            randomize_positions = self.combo_auto.currentText() == "Q-Learning"
            self.world.reset(clear_learning_curve=False, randomize_positions=randomize_positions)
            self.view.update()
            self.update_status_panel()

        self.space_timer.stop()
        self.timer.start()
        self.btn_run.setText("Pause")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and (not self.timer.isActive()):
                self.single_step()
                if not self.space_timer.isActive():
                    self.space_timer.start()
            return True

        if event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                self.space_timer.stop()
            return True

        return super().eventFilter(obj, event)

    def single_step(self):
        if self.timer.isActive():
            return
        self._one_step()

    def undo_last_step(self):
        if self.timer.isActive():
            return

        if not self.world.undo_last_step():
            return

        self.update_status_panel()
        self.view.update()
        self.reward_plot.refresh(force=True)
        if self.q_table_dialog is not None:
            self.q_table_dialog.refresh()

    def reset_world(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        self.world = PacmanWorld()
        self.world.agent.use_epsilon_greedy = bool(self.chk_eps_greedy.isChecked())
        self.world.agent.set_hyperparams(
            alpha=float(self.spin_alpha.value()),
            gamma=float(self.spin_gamma.value()),
            eps_start=float(self.spin_eps_start.value()),
            eps_min=float(self.spin_eps_min.value()),
            eps_decay=float(self.spin_eps_decay.value()),
            set_current_eps_to_start=True,
        )

        self.view.world = self.world
        self.reward_plot.world = self.world
        if self.q_table_dialog is not None:
            self.q_table_dialog.world = self.world
            self.q_table_dialog.refresh()

        self.view.update()
        self.update_status_panel()
        self.reward_plot.refresh(force=True)
        self.world.reward_ma_dirty = False

        self.btn_run.setText("Run")
        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def save_q_table(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Q-table",
            self.qtable_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            if was_running:
                self.timer.start()
                self.btn_run.setText("Pause")
            return

        try:
            self.world.agent.save(path)
            self.qtable_path = path
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def load_q_table(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Q-table",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            if was_running:
                self.timer.start()
                self.btn_run.setText("Pause")
            return

        try:
            self.world.agent.load(path)
            self.world.clear_undo()
            self.qtable_path = path

            self.chk_eps_greedy.setChecked(bool(self.world.agent.use_epsilon_greedy))
            self.spin_alpha.setValue(float(self.world.agent.alpha))
            self.spin_gamma.setValue(float(self.world.agent.gamma))
            self.spin_eps_start.setValue(float(max(self.world.agent.epsilon, self.world.agent.epsilon_min)))
            self.spin_eps_min.setValue(float(self.world.agent.epsilon_min))
            self.spin_eps_decay.setValue(float(self.world.agent.epsilon_decay))

            self.update_status_panel()
            if self.q_table_dialog is not None:
                self.q_table_dialog.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def reset_q_table(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        answer = QMessageBox.question(
            self,
            "Reset Q-table",
            "Delete all learned Q-values from memory?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer == QMessageBox.Yes:
            self.world.agent.reset_q_table()
            self.world.clear_undo()
            self.update_status_panel()
            if self.q_table_dialog is not None:
                self.q_table_dialog.refresh()

        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")

    def open_q_table_viewer(self):
        if self.q_table_dialog is None:
            self.q_table_dialog = QTableDialog(self.world, self)
        else:
            self.q_table_dialog.world = self.world
            self.q_table_dialog.refresh()

        self.q_table_dialog.show()
        self.q_table_dialog.raise_()
        self.q_table_dialog.activateWindow()
