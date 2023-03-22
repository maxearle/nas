from view import MainWindow, IOPanel, PlotPanel, SettingsControlPanel
from controller import Controller
from model import TdmsDir, Model
from PyQt6.QtWidgets import QApplication
import sys
import logging

#RUN THIS ONE

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    model = Model()
    io = IOPanel()
    plots = PlotPanel()
    settings = {
        "Samples per Batch":(100000, 'line'),
        "Sample Rate":(250000, 'line'),
        "Event Threshold": (-0.1, 'line'),
        "Reject high range traces":(True, 'check'),
        "Correct trace slope":(True, 'check'),
        "Gap tolerance":(100, 'line'),
        "Event Berth": (100, 'line'),
        "Loop delay": (250, 'line')
    }
    buttons = ["Start", "Accept event", "Reject event", "Keep accepting", "Keep rejecting", "Pause","Next batch", "Finish"]
    cfg = SettingsControlPanel(settings, buttons)
    w = MainWindow(title = "Translocation Finder: Earle Edition",top = io,mid = plots,bott = cfg)
    ctrlr = Controller(w,model)
    app.exec()