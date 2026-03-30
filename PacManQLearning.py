import argparse
import sys

from PySide6.QtWidgets import QApplication

from pacman_globals import QTABLE_PATH
from pacman_gui import MainWindow
from pacman_qlearning import train_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Pacman Q-learning with GUI and headless training")
    parser.add_argument("--train", type=int, default=0, help="number of training episodes (headless)")
    parser.add_argument("--view", action="store_true", help="start GUI viewer")
    parser.add_argument("--qtable", type=str, default=QTABLE_PATH, help="q-table json path")
    parser.add_argument("--workers", type=int, default=1, help="number of worker processes for training")
    parser.add_argument("--chunk-episodes", type=int, default=2000, help="episodes per worker chunk")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train > 0:
        train_agent(
            episodes=args.train,
            qtable_path=args.qtable,
            workers=args.workers,
            chunk_episodes=args.chunk_episodes,
        )

    if args.view or args.train == 0:
        app = QApplication(sys.argv)
        win = MainWindow(qtable_path=args.qtable)
        win.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
