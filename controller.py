from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QTimer
from view import ErrorDialog, AllDone
from model import EventError
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

def is_nan_ignore_None(val) -> bool:
    """Convenience function to get around the fact that np.isnan gets confused if you throw it a None"""
    if val is None:
        return False
    elif np.isnan(val):
        return True
    else:
        return False

class Controller():
    """This class is the home of all the spaghetti in this program. It ties together the view part of the program(GUI) and the underlying
    model (the data processing). It's not perfect in places, would be more readable if some changes to the UI and model were refactored as methods of the
    MainWindow and Model classes rather than implementing them here."""
    def __init__(self, view, model):
        self._v = view
        self._m = model
        self.grp_name = "current_data" #Name of base group in output hdf5 file
        self.accepted_count = 0 #Tracks events accepted so far
        self.current_trace_n = 0 #Tracks batches of samples seen so far
        self.n_trace = 0 #Total number of traces to be looked at
        self.accepted_events = 0
        self.rejected_events = 0
        self.accept_timer = QTimer(self._v, interval = int(self._v.cfg["Loop delay"]), timeout = self.accept_event)
        self.reject_timer = QTimer(self._v, interval = int(self._v.cfg["Loop delay"]), timeout = self.reject_event)
        self._connect_IO_buttons()
        self._connect_start_button()

    #INITIALISATION FUNCTIONS THAT RUN SUCCESSFULLY ONLY ONCE

    def _connect_IO_buttons(self):
        self._v.io.in_browse_button.clicked.connect(self.open_dir_dialog)
        self._v.io.out_browse_button.clicked.connect(self.open_file_dialog)

    def _initialise_data_and_display(self):
        self._m.open_tdms_dir(self._v.io.get_input_dir())
        logging.info(f"{len(self._m.tdms)} samples found.")
        self._m.make_output_file(self._v.io.get_output_path())
        self._m.add_group(self.grp_name, attrs = {"sample_rate":int(self._v.cfg["Sample Rate"])})

        self.n_trace = ceil(len(self._m.tdms)/int(self._v.cfg["Samples per Batch"]))
        self._v.plots.set_l_label(f"Trace Plot: {self.current_trace_n}/{self.n_trace}")
        self.process_next()
        self.update_trace_plot()
        self.next_event()

    def _start_process(self):
        """Checks given paths are valid and connects the rest of the buttons"""
        if self._m.check_path_existence(self._v.io.get_input_dir()) and self._m.check_path_existence(os.path.dirname(self._v.io.get_output_path())):
            logging.info("Valid input directory provided")
            self._v.io.lock()
            self._v.cfg.buttons["Start"].setEnabled(False)
            self._initialise_data_and_display()
            self._connect_cfg_buttons()
        else:
            ErrorDialog("Invalid directory or output path entered; please select a valid directory and try again.")

    
    def _connect_start_button(self):
        """Start button is connected first, before the rest which are only connected if valid filepaths are given"""
        self._v.cfg.buttons["Start"].clicked.connect(self._start_process)

        
    def _connect_cfg_buttons(self):
        self._v.cfg.buttons["Next batch"].clicked.connect(self.next_valid_batch)
        self._v.cfg.buttons["Accept event"].clicked.connect(self.accept_event)
        self._v.cfg.buttons["Reject event"].clicked.connect(self.reject_event)
        self._v.cfg.buttons["Keep accepting"].clicked.connect(self.start_accepting)
        self._v.cfg.buttons["Keep rejecting"].clicked.connect(self.start_rejecting)
        self._v.cfg.buttons["Pause"].clicked.connect(self.pause)
        self._v.cfg.buttons["Finish"].clicked.connect(self.finish)

    #DATA PROCESSING FUNCTIONS THAT UPDATE MODEL STATE

    def next_valid_batch(self):
        self.process_next()
        self.update_trace_plot()
        self.update_l_label()
        self.next_event()

    def next_event(self):
        """Loads next event from batch into model memory, if there are none/ no more moves onto next valid batch which calls this function again."""
        try:
            self._m.next_event(int(self._v.cfg["Event Berth"]))
            self.update_event_plot()
            self.update_r_label()

            for pos in self._m.event_boundaries[int(self._m.current_event_index)]:
                self._v.plots.l_vline(pos/int(self._v.cfg["Sample Rate"]), c='r')

        except EventError:
            logging.info("Event index is NaN, moving to next batch.")
            self.next_valid_batch()

    def process_next(self):
        """Processes the next batch of data and handles any rejections on account of missing events or high range etc.
        the program will stay in this loop until a valid batch is found."""
        while True:
            self._m.update_next_n(int(self._v.cfg["Samples per Batch"]))
            if is_nan_ignore_None(self._m.last_read):
                self.finish()
            self.current_trace_n += 1
            if self._v.cfg["Reject high range traces"]:
                trace_attrs = self._m.gen_trace_props()
                if trace_attrs["range"] > 10:
                    logging.info("Rejecting high range trace...")
                    continue
            if self._v.cfg["Correct trace slope"]:
                logging.info("Correcting trace slope")
                self._m.correct_slope()
            self._m.update_event_boundaries(float(self._v.cfg["Event Threshold"]), int(self._v.cfg["Gap tolerance"]))
            if len(self._m.event_boundaries) == 0:
                logging.info("No events in batch, moving on...")
                continue
            break
        logging.debug(f"{len(self._m.current_data)} samples loaded.")

    #EVENT HANDLING FUNCTIONS

    def accept_event(self):
        """Creates new dataset for event on plot and moves onto next"""
        logging.info("Creating new dataset for accepted event.")
        self.accepted_count += 1
        berth = int(self._v.cfg["Event Berth"])
        sample_rate = int(self._v.cfg["Sample Rate"])
        event_attrs = self._m.gen_event_attrs(berth, sample_rate)
        self._m.create_dataset(self.grp_name,f"Event_No_{self.accepted_count}",self._m.event_data,event_attrs)
        self.accepted_events += 1
        self.next_event()

    def reject_event(self):
        logging.info("Rejecting event.")
        self.rejected_events += 1
        self.next_event()

    #EVENT HANDLING LOOP FUNCTIONS

    def start_accepting(self):
        """Starts timer to accept every event after user-defined pause"""
        self.accept_timer = QTimer(self._v, interval = int(self._v.cfg["Loop delay"]), timeout = self.accept_event)
        self.accept_timer.start()
        self.update_r_label()
        for name, button in self._v.cfg.buttons.items():
            if name == "Pause":
                continue
            button.setEnabled(False)
            
    
    def start_rejecting(self):
        """Starts timer to reject every event after user-defined pause"""
        self.reject_timer = QTimer(self._v, interval = int(self._v.cfg["Loop delay"]), timeout = self.reject_event)
        self.reject_timer.start()
        self.update_r_label()
        for name, button in self._v.cfg.buttons.items():
            if name == "Pause":
                continue
            button.setEnabled(False)

    def pause(self):
        logging.info("Pause signal sent")
        logging.debug(f"Accept loop: {self.accept_timer.isActive()}, Reject loop: {self.reject_timer.isActive()}")
        if self.accept_timer.isActive():
            self.accept_timer.stop()
        if self.reject_timer.isActive():
            self.reject_timer.stop()
        logging.debug(f"After pause; Accept loop: {self.accept_timer.isActive()}, Reject loop: {self.reject_timer.isActive()}")
        self.update_r_label()
        for name, button in self._v.cfg.buttons.items():
            button.setEnabled(True)

    #GUI STUFF

    def open_dir_dialog(self):
        dir_name = QFileDialog.getExistingDirectory(self._v, 'Select Directory Containing Input Files')
        self._v.io.set_input_dir(dir_name)

    def open_file_dialog(self):
        dir_name = QFileDialog.getExistingDirectory(self._v, 'Select Directory for Output File')
        self._v.io.set_output_path(os.path.join(dir_name, "EVENTS.hdf5"))
    
    def update_trace_plot(self):
        """Updates left plot with trace and generates real-time axis"""
        t = self._m.gen_timescale(self._m.current_data, int(self._v.cfg["Sample Rate"]))
        self._v.plots.l_clear_and_plot(self._m.current_data, t)

    def update_event_plot(self):
        """Similar to above method for the trace plot"""
        t = self._m.gen_timescale(self._m.event_data, int(self._v.cfg["Sample Rate"]))
        self._v.plots.r_clear_and_plot(self._m.event_data, t)

    def update_l_label(self):
        self._v.plots.set_l_label(f"Trace Plot: {self.current_trace_n}/{self.n_trace}")

    def update_r_label(self):
        """Updates right plot label with current event number / events in batch as well as the current status of the accept/reject loops.
        Later may implement counter of number of events seen, reject, accepted."""
        if self.accept_timer.isActive():
            self._v.plots.set_r_label(f"Event Plot: {self._m.current_event_index + 1}/{len(self._m.event_boundaries)}; A: {self.accepted_events}, R: {self.rejected_events}; Accepting...")
        if self.reject_timer.isActive():
            self._v.plots.set_r_label(f"Event Plot: {self._m.current_event_index + 1}/{len(self._m.event_boundaries)}; A: {self.accepted_events}, R: {self.rejected_events}; Rejecting...")
        self._v.plots.set_r_label(f"Event Plot: {self._m.current_event_index + 1}/{len(self._m.event_boundaries)}; A: {self.accepted_events}, R: {self.rejected_events}")

    #CLEANUP

    def finish(self):
        """Closes output file and the window"""
        self._v.close()
        self._m.output.close()
        AllDone(f"All done! {len(self._m.tdms.file_list)} tdms files read, {self.accepted_count} events saved.")