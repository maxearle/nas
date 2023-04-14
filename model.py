from glob import glob
import os
import numpy as np
import nptdms as nt
import pandas as pd
import h5py as h
from scipy.optimize import curve_fit
import logging

class TdmsDir():
    """ Class handling the reading of TDMS files in a directory, so that they can all be accessed with one object."""
    def __init__(self, root_directory: str):
        self.file_list = sorted(glob(os.path.join(root_directory, '*.tdms')))
        self._imap = self._gen_imap()

    def _gen_imap(self) -> 'np.array':
        n_files = len(self.file_list)
        return np.cumsum([self.i_n_samples(i) for i in np.arange(n_files)])

    def i_read_file(self, index: int, start: int | None = None, end: int | None = None) -> 'np.array[float]':
        file = nt.TdmsFile.open(self.file_list[index])
        grp = file.groups()[0]
        chan = grp.channels()[0]
        if not ((start is None) | (end is None)):
            data = chan[start:end]
        elif not (start is None):
            data = chan[start:]
        elif not (end is None):
            data = chan[:end]
        else:
            data = chan[:]
        file.close()
        return data

    def i_n_samples(self, index: int) -> int:
        file = nt.TdmsFile.open(self.file_list[index])
        grp = file.groups()[0]
        chan = grp.channels()[0]
        n_samples = len(chan)
        file.close()
        return n_samples

    def i_read_files(self, indices: list[int]):
        return [nt.TdmsFile.read(self.file_list(i)) for i in indices]

    def _i_find_file(self, i: int) -> int:
        return np.searchsorted(self._imap, i, side = 'right')

    def __getitem__(self, subscript: int | slice) -> float | np.ndarray:
        if isinstance(subscript, int):
            if subscript < self._imap[0]:
                data = self.i_read_file(0)
                return data[subscript]
            else:
                file_n = self._i_find_file(subscript)
                new_sub = subscript % self._imap[file_n - 1]
                data = self.i_read_file(file_n)
                return data[new_sub]
        elif isinstance(subscript, slice):
            logging.debug(f"Loading data from slice {subscript.start}:{subscript.stop}")
            if subscript.start is None:
                first = 0
            else:
                first = self._i_find_file(subscript.start)
            logging.debug(f"First file: {first}")
            if subscript.stop is None:
                last = len(self.file_list) - 1
            else:
                last = self._i_find_file(subscript.stop)
            logging.debug(f"Last file: {last}")
            d = np.array([])
            count = 0
            for i in np.arange(first, last + 1):
                if i == 0 and subscript.start is None:
                    strt = 0
                elif i == (len(self.file_list) - 1) and subscript.stop is None:
                    strt = subscript.start - self._imap[first - 1]
                    end = None
                elif i == 0 and first == last:
                    strt = subscript.start
                    end = subscript.stop
                else:
                    #BEGIN Spaghetti
                    strt = subscript.start - self._imap[first - 1]
                    end = subscript.stop - self._imap[last - 1]
                if first == last:
                    logging.debug(f"Interpreting as: Read file {first} from {strt} to {end}")
                    chunk_data = self.i_read_file(i, start = strt, end = end)
                elif i == first:
                    chunk_data = self.i_read_file(i, start = strt)
                elif i == last:
                    chunk_data = self.i_read_file(i, end = end)
                else:
                    chunk_data = self.i_read_file(i)
                logging.debug(f"Loaded chunk of data with length {len(chunk_data)}")
                d = np.concatenate([d, chunk_data])
                #END Spaghetti
            data = d
            return data

    def sample_location(self, subscript: int) -> tuple[int,str,int]:
        file_n = self._i_find_file(subscript)
        file = self.file_list(file_n)
        sample = subscript % self._imap[file_n - 1]
        return (file_n, file, sample)

    def __len__(self) -> int:
        return self._imap[-1]

    def __repr__(self) -> str:
        msg = f"TdmsDir Object managing {len(self.file_list)} files, total of {len(self)} samples."
        return msg

    def __enter__(self):
        return self

class Model():
    def __init__(self):
        self.tdms = None
        self.current_data = None
        self.event_boundaries = None
        self.current_event_index = None
        self.event_data = None
        #For last_read, None means no data read yet and NaN means all data has been read
        self.last_read = None
        self.output = None

    def open_tdms_dir(self, fpath):
        self.tdms = TdmsDir(fpath)

    def check_path_existence(self,path: str):
        return os.path.exists(path)

    def gen_timescale(self, data: np.ndarray, sample_rate: int):
        return np.arange(len(data))/sample_rate

    def make_output_file(self,path):
        if self.check_path_existence(path):
            os.remove(path)
            logging.info("File already exists, deleting to replace with new one.")
        self.output = h.File(path, 'a')

    def add_group(self, grp: str, attrs: dict | None = None):
        self.output.create_group(grp)
        if attrs is None:
            return
        for key, value in attrs.items():
            self.output[grp].attrs[key] = value

    def create_dataset(self, grp: str, name: str,  data: np.ndarray, attrs: dict | None = None):
        self.output[grp].create_dataset(name, data=data)
        if attrs is None:
            pass
        for key, value in attrs.items():
            self.output[grp][name].attrs[key] = value

    def update_next_n(self, n: int) -> np.ndarray:
        if self.last_read is None:
            self.current_data = self.tdms[0:n]
            self.last_read = n-1
        elif np.isnan(self.last_read):
            return None
        elif n > (len(self.tdms) - self.last_read):
            self.current_data = self.tdms[self.last_read:]
            self.last_read = np.nan
        else:
            self.current_data = self.tdms[self.last_read:self.last_read + n]
            self.last_read += n

    def slope_fix(self, data):
        def line(x, a, b):
            return a*x + b
        xdata = np.arange(len(data))
        popt, pcov = curve_fit(line, xdata, data)
        return (data - line(xdata,*popt), pcov)

    def gen_trace_props(self) -> dict:
        props = {
            "trace_length":len(self.current_data),
            "std":np.std(self.current_data),
            "mean":np.mean(self.current_data),
            "range":(np.max(self.current_data) - np.min(self.current_data))
        }

        try:
            props["fit_goodness"]=np.sqrt(np.sum(np.diag(self.slope_fix(self.current_data)[1])))
        except:
            logging.debug("Couldn't find linearity for trace, assigning NaN")
            props["fit_goodness"]=np.nan
        return props

    def gen_event_attrs(self, berth: int, sample_rate: float) -> dict:
        cropped_event = self.event_data[berth:-(berth-1)]
        logging.debug(f"Generating event attrs for cropped event of length {len(cropped_event)}")
        c_e_l = len(cropped_event)
        c_e_a = np.sum(cropped_event)
        attrs = {
            "samples":c_e_l,
            "duration_s":c_e_l/sample_rate,
            "ecd":c_e_a/sample_rate,
            "mean":np.mean(cropped_event),
            "ffap":np.sum(cropped_event[:(c_e_l)//5])/c_e_a,
            "lfap":np.sum(cropped_event[(-(c_e_l)//5):])/c_e_a,
        }
        try:
            attrs["linearity"] = np.sqrt(np.sum(np.diag(self.slope_fix(cropped_event)[1])))
        except:
            logging.debug("Couldn't find linearity for this event, assigning NaN")
            attrs["linearity"] = np.nan

        return attrs

    def correct_slope(self):
        self.current_data = self.slope_fix(self.current_data)[0]


    def update_event_boundaries(self, thresh: float, tol: int):
        def get_lims(hits):
            runs = np.diff(hits)
            lims = np.where(runs > 1)[0]
            limits = []
            for i in np.arange(len(lims) + 1):
                if i == 0:
                    left = hits[0]
                try:
                    right = hits[lims[i]]
                    limits.append([left, right])
                    left = hits[lims[i]+1]
                except IndexError:
                    right = hits[-1]
                    limits.append([left, right])
            return limits
        def merge(list, dist = 100):
            pairs = list.copy()
            space = np.array([pairs[i+1][0] - pairs[i][1] for i in np.arange(len(pairs) - 1)])
            merge_locs = np.where(space > dist)[0].astype(int)
            if len(merge_locs) == 0:
                return pairs
            new_list = []
            for i, loc in enumerate(merge_locs):
                if i == 0:
                    left = pairs[0][0]
                try:
                    right = pairs[loc][1]
                    new_list.append([left, right])
                    left = pairs[loc + 1][0]
                except IndexError:
                    right = pairs[-1][1]
                    new_list.append([left, right])
            return new_list
        hits = np.where(self.current_data < thresh)[0]
        if len(hits) == 0:
            self.event_boundaries = []
            self.current_event_index = None
            return
        lims = get_lims(hits)
        logging.debug(f"Lims are: {lims}")
        merged = merge(lims, tol)
        logging.debug(f"Merged lims are: {merged}")
        self.event_boundaries = merged
        self.current_event_index = None

    def next_event(self, berth: int):
        if self.current_event_index is None:
            self.current_event_index = 0
        else:
            self.current_event_index += 1
        logging.info(f"Selected event {self.current_event_index + 1} of {len(self.event_boundaries)}")
        try:
            curr_event_boundaries = self.event_boundaries[self.current_event_index]
            logging.debug("Index valid, updating boundaries")
            self.event_data = self.current_data[(curr_event_boundaries[0] - berth):(curr_event_boundaries[1] + berth)]
        except IndexError:
            logging.debug("Invalid event boundary selected")
            self.event_data = np.nan
            raise EventError("Run out of events for this batch.")
        logging.debug(f"Event boundaries are {curr_event_boundaries}")

class EventError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)       