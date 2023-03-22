from PyQt6.QtWidgets import QMainWindow, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QLabel, QFrame, QCheckBox, QMessageBox
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
import sys

def next_n_squared(val: int) -> int:
    """Returns the first n for which n**2 > val."""
    n = 1
    while True:
        if n**2 > val:
            return n
        else:
            n += 1

class MplCanvas(FigureCanvasQTAgg):
    """MPL/QT Canvas with some default characteristics and some axes."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class IOPanel(QWidget):
    """Input/Output panel class for input directory and output filepath selection."""
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self._create_input_panel()
        self.layout.addWidget(VSeparator())
        self._create_output_panel()

    def _create_input_panel(self):
        """Initialises input side of IOPanel with a browse button and a line editor."""
        panel_title = QLabel("Input Directory")
        self.input_dir = QLineEdit()
        self.in_browse_button = QPushButton("Browse")

        self.input_layout = QVBoxLayout()
        self.input_layout.addWidget(panel_title)
        self.input_layout.addWidget(self.input_dir)
        self.input_layout.addWidget(self.in_browse_button)

        self.layout.addLayout(self.input_layout)

    def _create_output_panel(self):
        """Initialises output side of IOPanel with a browse button and a line editor."""
        panel_title = QLabel("Output Filepath")
        panel_title.setText("Output Filepath")
        self.output_path = QLineEdit()
        self.out_browse_button = QPushButton("Browse")

        self.output_layout = QVBoxLayout()
        self.output_layout.addWidget(panel_title)
        self.output_layout.addWidget(self.output_path)
        self.output_layout.addWidget(self.out_browse_button)

        self.layout.addLayout(self.output_layout)

    def get_input_dir(self) -> str:
        """Convenience method returns whatever is entered in input dir line edit."""
        return self.input_dir.text()

    def set_input_dir(self, text: str):
        """Convenience method sets input dir line edit text."""
        self.input_dir.setText(text)

    def get_output_path(self) -> str:
        """Convenience method gets whatever is in output path line edit."""
        return self.output_path.text()

    def set_output_path(self, text: str):
        """Convenience method sets output path line edit text."""
        self.output_path.setText(text)

    def lock(self):
        self.input_dir.setReadOnly(True)
        self.output_path.setReadOnly(True)
        self.in_browse_button.setEnabled(False)
        self.out_browse_button.setEnabled(False)

    def unlock(self):
        self.input_dir.setReadOnly(True)
        self.output_path.setReadOnly(True)
        self.in_browse_button.setEnabled(False)
        self.out_browse_button.setEnabled(False)

class PlotPanel(QWidget):
    """Widget with a left and right hand plot."""
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self._create_left_plot()
        self.layout.addWidget(VSeparator())
        self._create_right_plot()


    def _create_left_plot(self):
        """Initialises left hand plot"""
        self.l_layout = QVBoxLayout()
        self.l_canvas = MplCanvas(self)
        self.l_label = QLabel("Trace Plot: ?/?")
        self.l_canvas.axes.autoscale(True)
        self.l_layout.addWidget(self.l_label)
        self.l_layout.addWidget(self.l_canvas)
        self.layout.addLayout(self.l_layout)

    def _create_right_plot(self):
        """Initialises right hand plot"""
        self.r_layout = QVBoxLayout()
        self.r_canvas = MplCanvas(self)
        self.r_label = QLabel("Event Plot: ?/?")
        self.r_canvas.axes.autoscale(True)
        self.r_layout.addWidget(self.r_label)
        self.r_layout.addWidget(self.r_canvas)
        self.layout.addLayout(self.r_layout)

    def l_clear_and_plot(self, ydata, xdata = None, **kwargs):
        """Clear left hand plot and plot new data on it"""
        self.l_canvas.axes.cla()
        if xdata is None:
            self.l_canvas.axes.plot(ydata, **kwargs)
        else:
            self.l_canvas.axes.plot(xdata, ydata, **kwargs)
        self.l_canvas.draw()
        

    def r_clear_and_plot(self, ydata, xdata = None, **kwargs):
        """Clear right hand plot and plot new data on it"""
        self.r_canvas.axes.cla()
        if xdata is None:
            self.r_canvas.axes.plot(ydata, **kwargs)
        else:
            self.r_canvas.axes.plot(xdata, ydata, **kwargs)
        self.r_canvas.draw()

    def r_vline(self, x, **kwargs):
        """Add vline to right hand plot at specified x"""
        self.r_canvas.axes.axvline(x, **kwargs)
        self.r_canvas.draw()

    def l_vline(self, x, **kwargs):
        """Add vline to left hand plot at specified x"""
        self.l_canvas.axes.axvline(x, **kwargs)
        self.l_canvas.draw()

    def set_r_label(self, str: str):
        self.r_label.setText(str)

    def set_l_label(self, str: str):
        self.l_label.setText(str)

class SettingsControlPanel(QWidget):
    """Widget with programmatically generated settings fields and control buttons."""
    def __init__(self, settings: dict, butt_names: list):
        super().__init__()
        self.panel_layout = QHBoxLayout()
        self.setLayout(self.panel_layout)
        self._create_settings(settings)
        self.panel_layout.addWidget(VSeparator())
        self._create_controls(butt_names)

    def _create_settings(self, settings):
        """Initialises settings panel from initial settings dictionary, placing fields and labels in a square grid."""
        self.settings_layout = QGridLayout()
        field_n = len(settings)
        #Works out dimensions of square grid needed to accommodate all given settings/buttons
        dim = next_n_squared(field_n)
        keys = list(settings.keys())
        #Dictionary for access to settings fields
        self.fields = {keys[i]:SettingField(keys[i], settings[keys[i]][1]) for i in np.arange(field_n)}

        for i, item in enumerate(self.fields.values()):
            position = (i//dim, i%dim)
            self.settings_layout.addWidget(item,position[0], position[1])
            #Initialises field values to those given in initial settings dictionary
            item.set_val(settings[item.field_name()][0])
        
        self.panel_layout.addLayout(self.settings_layout)

    def _create_controls(self, butt_names):
        """Initialises buttons from button names list, placing buttons in a square grid."""
        self.buttons_layout = QGridLayout()
        butt_n = len(butt_names)
        #Works out dimensions of square grid needed to accommodate all given settings/buttons
        dim = next_n_squared(butt_n)
        #Dictionary for access to buttons
        self.buttons = {butt_names[i]:QPushButton(butt_names[i]) for i in np.arange(butt_n)}

        for i, button in enumerate(self.buttons.values()):
            position = (i//dim, i%dim)
            self.buttons_layout.addWidget(button, position[0], position[1])

        self.panel_layout.addLayout(self.buttons_layout)

    def __getitem__(self, key):
        """Subscripting method to access settings fields values"""
        return self.fields[key].get_val()

    def settings_keys(self):
        """Get view of settings field keys"""
        return self.fields.keys()

    def button_keys(self):
        """Get view of button keys"""
        return self.buttons.keys()


class MainWindow(QMainWindow):
    closed = pyqtSignal()
    def __init__(self, title: str, top: IOPanel, mid: PlotPanel, bott: SettingsControlPanel):
        """Initialise main window layout and sublayouts."""
        super().__init__()
        self.setWindowTitle(title)
        self.io = top
        self.plots = mid
        self.cfg = bott
        widget = QWidget()
        self.mainLayout = QVBoxLayout()
        self._initialise_layout()
        widget.setLayout(self.mainLayout)
        self.setCentralWidget(widget)

        self.show()

    def _initialise_layout(self):
        """Place child widgets separated by lines"""
        self.mainLayout.addWidget(self.io)
        self.mainLayout.addWidget(HSeparator())
        self.mainLayout.addWidget(self.plots)
        self.mainLayout.addWidget(HSeparator())
        self.mainLayout.addWidget(self.cfg)

    def closeEvent(self, event):
        """Causes window to emit 'closed' signal when closed"""
        self.closed.emit()

class HSeparator(QFrame):
    """Horizontal line separator"""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)

class VSeparator(QFrame):
    """Vertical line separator"""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)

class SettingField(QWidget):
    """A widget comprised of a label/field pair."""
    def __init__(self, label: str, field_type: str = 'line'):
        super().__init__()
        self.widget_layout = QHBoxLayout()
        self.setLayout(self.widget_layout)
        self.label = QLabel(label)
        self.type = field_type
        if field_type == "line":
            self.field = QLineEdit()
            self.field.setFixedWidth(150)
        elif field_type == "check":
            self.field = QCheckBox()
        else:
            raise ValueError(f"Invalid argument, field type should be 'line' or 'check', not {field_type}")

        self.widget_layout.addWidget(self.label)
        self.widget_layout.addWidget(self.field)
    
    def get_val(self):
        """Get value of field"""
        if self.type == 'line':
            return self.field.text()
        elif self.type == 'check':
            return self.field.isChecked()

    def set_val(self, val):
        """Set value of field"""
        if self.type == 'line':
            self.field.setText(str(val))
        elif self.type == 'check':
            self.field.setChecked(val)

    def field_name(self) -> str:
        """Get name of field"""
        return self.label.text()

    def __repr__(self) ->str:
        return f"Setting Field of type {self.type} with label {self.field_name()} and current value {self.get_val()}"

class ErrorDialog(QMessageBox):
    def __init__(self, msg: str):
        super().__init__()
        self.setWindowTitle("Error")
        self.setText(msg)
        self.exec()

class AllDone(QMessageBox):
    def __init__(self, msg: str):
        super().__init__()
        self.setWindowTitle("Finished")
        self.setText(msg)
        self.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    settings = {"Length":("",'line'), "Time":("","line"), "Value":(True, 'check')}
    button_names = ["Next", "Last", "Accept", "Reject"]
    w = MainWindow("Hey",IOPanel(), PlotPanel(), SettingsControlPanel(settings, button_names))
    w.closed.connect(AllDone("It works!"))
    app.exec()