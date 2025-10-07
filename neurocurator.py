import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile

from scipy import interpolate, signal
import torch

from joblib import Parallel, delayed

import io

import os
import re
from collections import Counter


class Neurocurator:
    def __init__(self):
        """
        Initializes a Hippie object with the given waveforms and spike times.
        waveforms: a pandas dataframe with columns representing waveform points and rows representing units
        spike_times: spikeData object from braingeneerspy
        sampling_rate: the sampling rate of the data
        metadata_obs: a pandas dataframe with metadata of the neurons
        isi_distribution: a pandas dataframe with the interspike interval distribution of the neurons
        waveform_embeddings: a pandas dataframe with the waveform embeddings of the neurons
        isi_distribution_embedding: a pandas dataframe with the interspike interval distribution embeddings of the neurons
        waveform_umap: a pandas dataframe with the waveform umap of the neurons
        isi_distribution_umap: a pandas dataframe with the interspike interval distribution umap of the neurons
        """
        self.waveforms = None
        self.spike_times_train = None
        self.templates = None
        self.sampling_rate = None
        self.metadata_obs = None
        self.isi_distribution = None
        self.acgs = None
        self.embeddings = None
        self.umap = None
        # self.connectivity_map = None

    def load_curation_file(self, qm_path):
        """Load data from a curation file."""
        if not os.path.exists(qm_path):
            raise FileNotFoundError(f"Curation file not found: {qm_path}")
            
        try:
            with zipfile.ZipFile(qm_path, "r") as f_zip:
                if "qm.npz" not in f_zip.namelist():
                    raise ValueError(f"Invalid curation file: missing qm.npz in {qm_path}")
                    
                qm = f_zip.open("qm.npz")
                data = np.load(qm, allow_pickle=True)
                
                # Validate required keys
                required_keys = ["train", "fs"]
                for key in required_keys:
                    if key not in data:
                        raise KeyError(f"Missing required data '{key}' in curation file")
                        
                # Extract data
                spike_times = data["train"].item()
                fs = data["fs"]
                train = [times / fs for _, times in spike_times.items()]
                
                config = data["config"].item() if "config" in data else None
                
                if "neuron_data" not in data:
                    raise KeyError("Missing required 'neuron_data' in curation file")
                neuron_data = data["neuron_data"].item()
                
        except zipfile.BadZipFile:
            raise ValueError(f"File is not a valid zip archive: {qm_path}")
        
        return train, neuron_data, config, fs

    def load_acqm(self, acqm_path):
        """
        Reads the acqm file and returns the waveforms and spike times of the neurons in the Hippie object.
        """
        train, neuron_data, config, fs = self.load_curation_file(acqm_path)
        # Sampling rate just in case
        self.sampling_rate = fs
        # Extract spike trains
        self.spike_times_train = train
        # Convert spike times to ms
        self.spike_times_train = [np.array(spike_times) * 1000 for spike_times in self.spike_times_train]
        # Extract waveforms
        self.waveforms = self.extract_waveforms(neuron_data)
        # Extract ISI distribution
        self.isi_distribution = self.compute_isi_distribution()
        # Extract autocorrelogram
        self.acgs = self.compute_autocorrelogram(train)
        # Extract position
        self.metadata_obs = self.extract_neuron_positions(neuron_data)

        #Validate data integrity
        self.validate_data_integrity()
        #Don't return anything, just set the attributes

        
    def extract_waveforms(self, neuron_data, n_datapoints=50):
        """
        Extracts the waveforms of the neurons in the Hippie object.
        """
        datapoints_before = int(n_datapoints / 5 * 2)
        datapoints_after = n_datapoints - datapoints_before

        neuron_dataframe = pd.DataFrame()
        neuron_array = []
        for neuron_id, neuron in neuron_data.items():
            # print(neuron_id,neuron)
            try:
                neuron_waveforms = neuron["waveforms"]
                neuron_flag = "waveforms"
                # Calculate mean waveform
                mean_waveform = np.mean(neuron_waveforms, axis=0)
                # Add mean waveform to dataframe
                neuron_array.append(mean_waveform)
            except:
                neuron_waveforms = neuron["template"]
                neuron_flag = "template"
                neuron_array.append(neuron_waveforms)


        if neuron_flag == "template":
            print("No waveforms, using template instead")

        # We would need to cut the neuron into X points 20 before the minimum and 30 after the minimum
        # We will use the minimum as the reference point
        neuron_cut = []
        for neuron in neuron_array:
            min_idx = np.argmin(neuron)
            try:
                neuron_cut.append(
                    neuron[min_idx - datapoints_before : min_idx + datapoints_after]
                )
            except:
                # If it goes out of bounds we will interpolate the waveform but now just fill with zeros
                neuron_cut.append(np.zeros(n_datapoints))

        neuron_dataframe = pd.DataFrame(neuron_cut)
        return neuron_dataframe

    def extract_neuron_positions(self, neuron_data):
        """
        Extracts the position of the neurons in the Hippie object.
        """
        position_array = []
        metadata = pd.DataFrame()
        for neuron_id, neuron in neuron_data.items():
            x, y = neuron["position"][0], neuron["position"][1]
            position_array.append([x, y])
        metadata["x"] = [x[0] for x in position_array]
        metadata["y"] = [x[1] for x in position_array]
        return metadata


    def compute_single_isi(self, spike_times):
        """
        Calculate interspike intervals (ISI) from an array of spike times.

        Parameters:
        spike_times (np.ndarray): A 1D numpy array of spike times (in seconds or ms).

        Returns:
        np.ndarray: A 1D array of interspike intervals.
        """
        # Ensure spike times is sorted
        spike_times.sort()
        # Ensure there are more than 2 spike times
        if len(spike_times) < 2:
            return np.array([])

        # Calculate the difference between successive spike times
        isi = np.diff(spike_times)
        # print(isi)
        return isi

    def compute_all_isis(self, spike_times_train):
        """
        Calculates the interspike intervals of the neurons in the Hippie object.
        Returns:
        isi: a list of numpy arrays with the interspike intervals of the neurons
        """
        isi = []
        for idx, neuron in enumerate(spike_times_train):
            if isinstance(neuron, np.ndarray):
                neuron = neuron.flatten()
            # If neuron is list, convert it to numpy array
            elif isinstance(neuron, list):
                neuron = np.array(neuron)
            # If neuron is not a numpy array, raise an error
            else:
                raise ValueError("neuron is not a numpy array")
            # Calculate ISI
            isi_unit = self.compute_single_isi(neuron)
            isi.append(isi_unit)
        return isi

    def compute_isi_distribution(self, time_window=100):
        """
        Extracts the interspike interval distribution of the neurons in the Hippie object.
        time_window: the time window in which to calculate the interspike interval distribution in milliseconds
        It will create a histogram with 1 ms bins
        """
        interspike_intervals_object = self.compute_all_isis(
            self.spike_times_train
        )

        def compute_hist(isi):
            if len(isi) == 0:
                return np.zeros(time_window)
            return np.histogram(isi[isi < time_window], bins=time_window)[0]
        
        isi_distribution = Parallel(n_jobs=-1)(
            delayed(compute_hist)(isi) for isi in interspike_intervals_object
        )
        return pd.DataFrame(isi_distribution)

    def compute_autocorrelogram(
        self,
        spike_trains,
        bin_size_ms=1,
        window_size_ms=100,
        normalize=False,
        remove_central_bin=True,
    ):
        """
        Extract autocorrelograms from a list of spike trains and return a DataFrame
        where each row is a neuron and each column is a time bin count.

        Parameters:
        -----------
        spike_trains : list of lists
            A list where each element is a list of spike times in milliseconds.
        bin_size_ms : float, default=1
            Size of each bin in milliseconds.
        window_size_ms : float, default=100
            Size of the window for computing autocorrelogram in milliseconds.
            The resulting histogram will span [-window_size_ms, window_size_ms].
        normalize : bool, default=False
            If True, normalize the autocorrelogram by dividing by the number of spikes.
        remove_central_bin : bool, default=True
            If True, remove the central bin (lag=0) to exclude self-comparisons.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame where:
            - Each row corresponds to a neuron (spike train)
            - Each column corresponds to a time bin
            - Column names are the bin centers in milliseconds
            - Values are the counts (or normalized counts) in each bin
        """
        # Set up the window parameters
        ccg_win = [-window_size_ms, window_size_ms]

        # Create sparse binary trains for each spike train
        max_time = 0
        for train in spike_trains:
            if len(train) > 0:
                max_time = max(max_time, max(train))

        # Add a buffer to ensure we capture all relationships
        max_time += window_size_ms * 2

        # Bin size in time units
        bin_size = bin_size_ms

        # Calculate number of bins
        n_bins = int(np.ceil(max_time / bin_size)) + 1

        # Create binary spike trains (sparse arrays)
        sparse_trains = []
        for train in spike_trains:
            sparse_train = np.zeros(n_bins, dtype=int)
            if len(train) > 0:
                # Convert spike times to bin indices
                bin_indices = (np.array(train) / bin_size).astype(int)
                # Ensure indices are within bounds
                valid_indices = bin_indices[bin_indices < n_bins]
                sparse_train[valid_indices] = 1
            sparse_trains.append(sparse_train)

        # Calculate lags array
        lags = np.arange(ccg_win[0], ccg_win[1] + bin_size, bin_size)
        n_lags = len(lags)

        # Initialize results storage
        all_acgs = np.zeros((len(spike_trains), n_lags))

        # For each spike train, calculate its autocorrelogram
        for i, sparse_train in enumerate(sparse_trains):
            # Use the ccg function to calculate the ACG
            counts, _ = self.compute_cross_correlation(
                sparse_train, sparse_train, ccg_win=ccg_win, bin_size=bin_size
            )

            # Find and remove the central bin if requested
            if remove_central_bin:
                central_idx = np.where(lags == 0)[0]
                if len(central_idx) > 0:
                    counts[central_idx[0]] = 0

            # Normalize if requested
            if normalize and np.sum(sparse_train) > 0:
                counts = counts / np.sum(sparse_train)

            all_acgs[i] = counts

        # Create the DataFrame
        column_names = [f"{x:.2f}" for x in lags]
        result_df = pd.DataFrame(all_acgs, columns=column_names)

        return result_df

    def compute_cross_correlation(self, bt1, bt2, ccg_win=[-10, 10], t_lags_shift=0, bin_size=1):
        """
        Calculate cross-correlation between two binary spike trains.

        Parameters:
        -----------
        bt1 : numpy.ndarray
            First binary spike train.
        bt2 : numpy.ndarray
            Second binary spike train.
        ccg_win : list, default=[-10, 10]
            Window for cross-correlation in milliseconds [min, max].
        t_lags_shift : int, default=0
            Shift to apply to time lags.
        bin_size : float, default=1
            Size of each bin in milliseconds.

        Returns:
        --------
        tuple
            (counts, lags) - cross-correlation counts and corresponding time lags.
        """
        if np.all((np.array(ccg_win) / bin_size) % 1) != 0:
            raise ValueError("The window and shift must be multiples of the bin size")

        left_edge, right_edge = np.subtract(np.array(ccg_win) / bin_size, t_lags_shift)
        left_edge = int(left_edge)
        right_edge = int(right_edge)
        lags = np.arange(ccg_win[0], ccg_win[1] + bin_size, bin_size)

        pad_width = min(max(-left_edge, 0), max(right_edge, 0))
        bt2_pad = np.pad(bt2, pad_width=pad_width, mode="constant")

        cross_corr = signal.fftconvolve(bt2_pad, bt1[::-1], mode="valid")
        return np.round(cross_corr), lags


    def compute_waveform_features(self, waveform, mid=20, right_only=True, fs=20000.0):
        """
        From Sury's code
        measure the waveform features for both positive
        and negative spikes.
        waveform: a list or array of waveform
        mid: the index of the trough
        fs: sampling frequency
        return:
            a dictionary of features including
            left peak (index, value),
            right peak (index, value),
            trough to peak time (time in ms),
            fwhm (full width half maximum value in ms),
            fwhm_x (the left and right x values of the fwhm)
        """
        # 0. flip the waveform if it is positive
        if waveform[mid] > 0:
            waveform = -waveform
        # 1. measure the left and the right peak
        left = waveform[: mid + 1][::-1]
        right = waveform[mid:]
        ldiff = left[1:] - left[:-1]
        rdiff = right[1:] - right[:-1]
        lpeak_ind_list = np.where(ldiff < 0)[0]
        lpeak_ind, rpeak_ind = 0, len(waveform) - 1
        for ind in lpeak_ind_list:
            if waveform[mid - ind] > 0:
                lpeak_ind = mid - ind
                break
        lpeak_value = waveform[lpeak_ind]
        rpeak_ind_list = np.where(rdiff < 0)[0]
        for ind in rpeak_ind_list:
            if waveform[mid + ind] > 0:
                rpeak_ind = mid + ind
                break
        rpeak_value = waveform[rpeak_ind]
        lpeak = (lpeak_ind, lpeak_value)
        rpeak = (rpeak_ind, rpeak_value)
        # print(f"left peak {lpeak}, right peak {rpeak}")
        # 2. measure the trough to peak time (peak taken as the max peak either left or right)
        if not right_only:
            post_hyper = lpeak if lpeak_value > rpeak_value else rpeak
        else:
            post_hyper = rpeak
        trough_to_peak = abs(post_hyper[0] - mid) / (fs / 1000)  # make it in ms
        # print(f"{trough_to_peak} ms")
        # 3. measure the full width half maximum (FWHM) from the depolarization trough to the baseline
        half_amp = waveform[mid] / 2.0
        # print(f"half amplitude {half_amp}")
        # print(waveform[lpeak_ind: mid])
        # print(waveform[mid: rpeak_ind+1])
        xx_left = np.arange(lpeak_ind, mid + 1)
        xx_right = np.arange(mid, rpeak_ind + 1)
        # interpolate the left peak to trough waveform to find the half amplitude point
        fl = interpolate.interp1d(waveform[lpeak_ind : mid + 1], xx_left)
        fr = interpolate.interp1d(waveform[mid : rpeak_ind + 1], xx_right)

        # Check if half_amp is within the range of the original data
        if (
            np.min(waveform[lpeak_ind : mid + 1])
            <= half_amp
            <= np.max(waveform[lpeak_ind : mid + 1])
        ):
            inter_1 = fl(half_amp)
        else:
            inter_1 = None  # or some other appropriate value

        if (
            np.min(waveform[mid : rpeak_ind + 1])
            <= half_amp
            <= np.max(waveform[mid : rpeak_ind + 1])
        ):
            inter_2 = fr(half_amp)
        else:
            inter_2 = None  # or some other appropriate value

        fwhm_x = np.array([inter_1, inter_2])
        # Breaking :(
        try:
            fwhm = (abs(inter_2) - abs(inter_1)) / (fs / 1000)
        except:
            fwhm = 0
        # print(f"fwhm {fwhm} ms")
        features = {
            "lpeak": lpeak,
            "rpeak": rpeak,
            "trough_to_peak": trough_to_peak,
            "fwhm": fwhm,
            "fwhm_x": fwhm_x,
        }
        return features

    def compute_all_waveform_features(self):
        """
        Calculates the waveform features of the neurons in the Hippie object.

        """
        waveforms_lpeak = []
        waveforms_rpeak = []
        waveforms_trough_to_peak = []
        waveforms_fwhm = []
        waveforms_fwhm_x = []
        for idx, waveform in self.waveforms.iterrows():
            waveform = waveform.to_numpy()
            features = self.compute_waveform_features(waveform)
            waveforms_lpeak.append(features["lpeak"])
            waveforms_rpeak.append(features["rpeak"])
            waveforms_trough_to_peak.append(features["trough_to_peak"])
            waveforms_fwhm.append(features["fwhm"])
            waveforms_fwhm_x.append(features["fwhm_x"])

        self.metadata_obs["lpeak"] = waveforms_lpeak
        self.metadata_obs["rpeak"] = waveforms_rpeak
        self.metadata_obs["trough_to_peak"] = waveforms_trough_to_peak
        self.metadata_obs["fwhm"] = waveforms_fwhm
        self.metadata_obs["fwhm_x"] = waveforms_fwhm_x
        return self.metadata_obs

    def compute_firing_rate(self):
        """
        Calculates the most basic type of firing rate of the neurons in the Hippie object.
        """
        spike_train = self.spike_times_train
        firing_rate = []
        for idx, neuron in enumerate(spike_train):
            firing_rate.append(len(neuron) / (max(neuron) - min(neuron)))
        self.metadata_obs["firing_rate"] = firing_rate
        return self.metadata_obs
    
    def compute_minimum_isi(self):
        """
        Calculates the minimum interspike interval of the neurons in the Hippie object.
        """
        spike_train = self.spike_times_train
        minimum_isi = []
        for idx, neuron in enumerate(spike_train):
            isi = np.diff(neuron)
            minimum_isi.append(min(isi))
        self.metadata_obs["minimum_isi"] = minimum_isi
        return self.metadata_obs


    def set_experiment_condition(self, column_name, condition):
        """
        Sets the biological batch of the neurons in the Hippie object.
        """
        self.metadata_obs[column_name] = condition

    def validate_data_integrity(self):
        """Validate that all required data structures are properly loaded."""
        validation_results = {
            "waveforms": self.waveforms is not None and not self.waveforms.empty,
            "spike_times": self.spike_times_train is not None and len(self.spike_times_train) > 0,
            "sampling_rate": self.sampling_rate is not None and self.sampling_rate > 0,
            "metadata": self.metadata_obs is not None and not self.metadata_obs.empty,
            "isi_distribution": self.isi_distribution is not None and not self.isi_distribution.empty,
            "autocorrelograms": self.acgs is not None and not self.acgs.empty
        }
        
        missing = [k for k, v in validation_results.items() if not v]
        if missing:
            print(f"Warning: The following data structures are missing or empty: {', '.join(missing)}")
            
        return all(validation_results.values())
    
    def get_waveform_embeddings(self):
        return self.waveform_embeddings

    def get_isi_distribution_embedding(self):
        return self.isi_distribution_embedding

    def get_waveform_umap(self):
        return self.waveform_umap

    def get_isi_distribution_umap(self):
        return self.isi_distribution_umap

    def get_connectivity_map(self):
        """
        Returns the connectivity map of the neurons in the Hippie object. The connectivity map is a pandas dataframe with the following columns:
        - neuron1: the id of the first neuron
        - neuron2: the id of the second neuron
        - weight: the weight of the connection between the two neurons
        """
        return self.connectivity_map
    
    def calculate_average_sttc_SD(self, delta=10): #TODO: Change it to work with spike package from alex spaeth
        """
        Calculate average STTC of the neurons in the SpikeData object."""
        sttc = self.spike_times_train.spike_time_tilings(delt=delta)
        mean_sttc = np.mean(sttc, axis=0)
        self.metadata_obs["avg_sttc"] = mean_sttc
        return self.metadata_obs



    def calculate_firing_rate_SD(self, unit="Hz"): #TODO: Change it to work with spike package from alex spaeth
        """
        Calculates the firing rate of the neurons in the Hippie object.
        """
        # Calculate firing rate
        firing_rate = self.spike_times_train.rates(unit)
        self.metadata_obs["firing_rate"] = firing_rate
        return self.metadata_obs

    def calculate_metadata_SD(self):#TODO: Change it to work with spike package from alex spaeth
        """
        Calculates the metadata of the neurons in the Hippie object.
        """
        # Add column with average sttc
        self.calculate_average_sttc()
        # Add column with average firing rate
        self.calculate_firing_rate()
        # Add column with waveform features
        self.compute_all_waveform_features()
    
    def load_nwb_spike_times(self, nwb_path):
 
        from pynwb import NWBHDF5IO #lazy import 

        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()

            #PyNWB exposes per-unit spike times via get_unit_spike_times(i) in seconds
            spikes_ms = []
            n_units = len(nwb.units.id.data)
            for i in range(n_units):
                st_sec = nwb.units.get_unit_spike_times(i)  #seconds (ragged vector)
                st_ms = np.asarray(st_sec, dtype=float) * 1000.0
                spikes_ms.append(st_ms)

        #fill object; downstream ISI/ACG code can run as-is
        self.spike_times_train = spikes_ms

    def load_nwb_waveforms(self, nwb_path, n_datapoints=50, candidates=("waveform_mean", "spike_waveforms")):
        
        from pynwb import NWBHDF5IO #lazy import
        
        #Candidates is the name of the columns to look for first

        #this functionensures all waveforms have the same number of points n, which is 50
        #since the other functions expect 50
        #if waveform is shorter, it pads with zeros at the end
        #if waveform is longer, it cuts around the minimum point
        #
        def cut_or_pad_to_n(wf, n=50):
            wf = np.asarray(wf, float).flatten()
            if wf.size == 0:
                return np.zeros(n, float)
            mid = int(np.argmin(wf))
            left = int(n * 2/5)
            right = n - left
            lo = max(0, mid - left)
            hi = min(wf.size, lo + n)
            seg = wf[lo:hi]
            if seg.size < n:
                seg = np.pad(seg, (0, n - seg.size))
            return seg

        #the function starts here by opening the nwb file
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()
            #check if there is indeed a units table from where we can extract waveforms
            if nwb.units is None:
                self.waveforms = pd.DataFrame()
                return self.waveforms

            #check which of the candidate columns exists to use that one
            cols = set(nwb.units.colnames)
            key = None
            for k in candidates:
                if k in cols:
                    key = k
                    break

            #here it loops through the units id there is no waveform column it fills everything with zeros and terminates
            #if we have spike waveforms it takes all points and averages them (uses cut and pad function)
            #if it has waveform_mean it just uses that as the vector
            wf_rows = []
            n_units = len(nwb.units.id.data)
            for i in range(n_units):
                if key is None:
                    #no waveform column inside of units units = fills all with zeros
                    wf_rows.append(np.zeros(n_datapoints))
                    continue

                if key == "spike_waveforms":
                    #here it averages all the points and adds it
                    wfs = nwb.units.get_unit_spike_waveforms(i)
                    if wfs is None or len(wfs) == 0:
                        wf_rows.append(np.zeros(n_datapoints))
                    else:
                        mean_wf = np.mean(np.asarray(wfs, float), axis=0)
                        wf_rows.append(cut_or_pad_to_n(mean_wf, n=n_datapoints))
                else:
                    # if it already has the averge it just adds it
                    row = np.asarray(nwb.units[key][i], float)
                    wf_rows.append(cut_or_pad_to_n(row, n=n_datapoints))
        #define the universal waveform dataframe with the computed rows
        self.waveforms = pd.DataFrame(wf_rows)
        return self.waveforms
    

    def load_phy_curated(self, zip_path, n_datapoints=50, window_ms=100, bin_ms=1, good_only=True):

        #load .npy straight from the zip
        def _np_from_zip(zf, name):
            with zf.open(name) as f:
                return np.load(io.BytesIO(f.read()), allow_pickle=True)

        #read and decode a text file (e.g., params.py)
        def _text_from_zip(zf, name):
            with zf.open(name) as f:
                return f.read().decode("utf-8", errors="ignore")

        #map basename -> full path (handles nested dirs)
        def _index_by_basename(zf):
            m = {}
            for p in zf.namelist():
                base = p.split("/")[-1]
                if base not in m:
                    m[base] = p
            return m

        #center on trough, cut/pad to fixed length
        def _cut_or_pad_to_n(wf, n=50):
            wf = np.asarray(wf, float).flatten()
            if wf.size == 0:
                return np.zeros(n, float)
            mid = int(np.argmin(wf))
            left = int(n * 2/5)
            lo = max(0, mid - left)
            hi = min(wf.size, lo + n)
            seg = wf[lo:hi]
            if seg.size < n:
                seg = np.pad(seg, (0, n - seg.size))
            return seg

        with zipfile.ZipFile(zip_path, "r") as zf:
            by_base = _index_by_basename(zf)
            #fail if one of the data is missing
            REQUIRED = {"spike_times.npy", "spike_clusters.npy", "templates.npy", "params.py"}
            missing = [r for r in REQUIRED if r not in by_base]
            if missing:
                raise ValueError(f"Missing required files {missing} inside {zip_path}")

            #load each one of the datasets into a variable
            
            spike_times = _np_from_zip(zf, by_base["spike_times.npy"])
            spike_clusters = _np_from_zip(zf, by_base["spike_clusters.npy"])
            templates = _np_from_zip(zf, by_base["templates.npy"])  # [nTemplates, nTime, nChannels]
            params_txt = _text_from_zip(zf, by_base["params.py"])

            m = re.search(r"sample_rate\s*=\s*([0-9.]+)", params_txt)
            if not m:
                raise ValueError("Could not parse sample_rate from params.py")
            sample_rate = float(m.group(1))  # Hz

            channel_map = _np_from_zip(zf, by_base["channel_map.npy"]) if "channel_map.npy" in by_base else None
            channel_positions = _np_from_zip(zf, by_base["channel_positions.npy"]) if "channel_positions.npy" in by_base else None
            spike_templates = _np_from_zip(zf, by_base["spike_templates.npy"]) if "spike_templates.npy" in by_base else None

            #'good' filter from cluster_info.tsv
            allow_cids = None
            if "cluster_info.tsv" in by_base and good_only:
                import csv
                good_ids = set()
                with zf.open(by_base["cluster_info.tsv"]) as f:
                    text = f.read().decode("utf-8", errors="ignore").splitlines()
                    rdr = csv.DictReader(text, delimiter="\t")
                    for row in rdr:
                        grp = (row.get("group") or row.get("KSLabel") or "").strip().lower()
                        if grp == "good":
                            try:
                                good_ids.add(int(row["cluster_id"]))
                            except Exception:
                                pass
                if good_ids:
                    allow_cids = good_ids

            #creating per-unit arrays
            unique_c = np.unique(spike_clusters)
            unit_times_ms = []
            wf_rows = []
            xs, ys = [], []

            #main code, iterate iver the units
            for cid in unique_c:
                if allow_cids is not None and int(cid) not in allow_cids:
                    continue

                idx = np.where(spike_clusters == cid)[0]
                if idx.size == 0:
                    continue

                #change to milliseconds
                times_ms = (spike_times[idx].astype(np.float64) / sample_rate) * 1000.0
                unit_times_ms.append(times_ms)

                #use dominant template for this unit
                #(robust to (n,1) shapes / object dtype)
                #pickong the waaveform
                if spike_templates is not None:
                    vals = np.asarray(spike_templates)[idx]
                    vals = np.array(vals, dtype=np.int64).reshape(-1)   #flatten + ensure ints
                    if vals.size > 0:
                        tmpl = int(np.bincount(vals).argmax())          #mode of template IDs
                    else:
                        tmpl = 0
                else:
                    tmpl = int(cid) if 0 <= int(cid) < templates.shape[0] else 0

                #clamp in case tmpl is out of range
                if not (0 <= int(tmpl) < templates.shape[0]):
                    tmpl = 0

                #best channel = deepest trough
                tmpl_wf = templates[tmpl]  # [time, channels]
                best_ch = int(np.argmin(np.min(tmpl_wf, axis=0)))
                ch_wf = tmpl_wf[:, best_ch]
                seg = _cut_or_pad_to_n(ch_wf, n=n_datapoints)

                #map to probe channel and (x,y) if available
                probe_ch = best_ch
                if channel_map is not None and best_ch < channel_map.size:
                    probe_ch = int(channel_map[best_ch])
                if channel_positions is not None and 0 <= probe_ch < channel_positions.shape[0]:
                    xs.append(float(channel_positions[probe_ch, 0]))
                    ys.append(float(channel_positions[probe_ch, 1]))
                else:
                    xs.append(np.nan); ys.append(np.nan)

                wf_rows.append(seg)

        #set attributes
        self.waveforms = pd.DataFrame(wf_rows)
        self.spike_times_train = [np.asarray(st, float) for st in unit_times_ms]
        self.metadata_obs = pd.DataFrame({"x": xs, "y": ys})
        self.sampling_rate = sample_rate

        #compute isi and acg with the already existing methods
        self.isi_distribution = self.compute_isi_distribution(time_window=window_ms)
        self.acgs = self.compute_autocorrelogram(
            self.spike_times_train,
            bin_size_ms=bin_ms,
            window_size_ms=window_ms,
            normalize=False,
            remove_central_bin=True,
        )

        self.validate_data_integrity()
        return

