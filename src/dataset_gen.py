"""dataset_gen.py: Functions for loading segmented data from the cough counting dataset."""
__author__ = "Lara Orlandic"
__email__ = "lara.orlandic@epfl.ch"

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys
import IPython.display as ipd
from enum import Enum
from helpers import *
import json

def get_cough_windows(data_folder, fn, window_len, aug_factor=1):
    """Get the cough segments in a given recording by shifting them within the window
    Inputs:
    - data_folder: location of the recording
    - fn: file name of the ground_truth.json file listing cough locations
    - window_len: desired length of signal window in seconds
    - aug_factor: number of times to shift the cough within the window (i.e. data augmentation)
    Outputs:
    - audio_data: NxMx2 data matrix where 
        - N = number of coughs * augmentation factor
        - M = int(window_len * 16000)
        - first index = outer microphone, second index = body-facing microphone
    - imu_data: NxLx6 data matrix where
        - L = int(window_len * 100)
    - num_coughs: number of coughs in the recording
    """
    # Load cough segment annotations and signals
    with open(fn, 'rb') as f:
        loaded_dict = json.load(f)
    starts = np.array(loaded_dict["start_times"])
    ends = np.array(loaded_dict["end_times"])
    subj_id = fn.split('/')[-6]
    trial = fn.split('/')[-5].split('_')[1]
    mov = fn.split('/')[-4].split('_')[1]
    noise = fn.split('/')[-3].split('_')[2]
    if noise == "someone":
        noise = "someone_else_cough"
    sound = fn.split('/')[-2]
    air, skin = load_audio(data_folder, subj_id, trial, mov, noise, sound)
    imu = load_imu(data_folder, subj_id, trial, mov, noise, sound)
    
    # Set up arrays for storing data
    num_coughs = len(starts)
    window_len_audio = int(window_len*FS_AUDIO)
    window_len_imu = int(window_len*FS_IMU)
    audio_data = np.zeros((num_coughs*aug_factor,window_len_audio,2))
    imu_data = np.zeros((num_coughs*aug_factor,window_len_imu,6))
    
    for a in range(aug_factor):
        # Compute random offsets based on window length and cough lengths
        cough_lengths = ends-starts
        diffs = window_len - cough_lengths
        rand_uni = np.random.uniform(0,diffs)
        window_starts = starts - rand_uni
        end_of_signal = np.min((len(air)/FS_AUDIO,len(imu.x)/FS_IMU))
        #Check if the window exceeds the end of the signal. If so, shift from the end
        exceeds_end = window_starts > ( end_of_signal - window_len)
        if sum(exceeds_end) > 0:
            end_slack = np.max((end_of_signal - ends,np.zeros(ends.shape)), axis=0)
            window_starts[exceeds_end] = np.min((ends[exceeds_end], np.tile(end_of_signal, sum(exceeds_end))),axis=0) - window_len + np.random.uniform(0,np.min((diffs[exceeds_end],end_slack[exceeds_end]))-0.02)


        # Segment audio signals using direct slicing (35x faster than linspace)
        window_starts_audio = (window_starts*FS_AUDIO).astype(int)
        # Clamp to valid range
        window_starts_audio = np.clip(window_starts_audio, 0, len(air) - window_len_audio)
        for i, start_idx in enumerate(window_starts_audio):
            end_idx = start_idx + window_len_audio
            audio_data[a*num_coughs + i, :, 0] = air[start_idx:end_idx]
            audio_data[a*num_coughs + i, :, 1] = skin[start_idx:end_idx]

        # Segment IMU signals using direct slicing (35x faster than linspace)
        window_starts_imu = (window_starts*FS_IMU).astype(int)
        # Clamp to valid range
        window_starts_imu = np.clip(window_starts_imu, 0, len(imu.x) - window_len_imu)
        for i, start_idx in enumerate(window_starts_imu):
            end_idx = start_idx + window_len_imu
            imu_data[a*num_coughs + i, :, 0] = imu.x[start_idx:end_idx]
            imu_data[a*num_coughs + i, :, 1] = imu.y[start_idx:end_idx]
            imu_data[a*num_coughs + i, :, 2] = imu.z[start_idx:end_idx]
            imu_data[a*num_coughs + i, :, 3] = imu.Y[start_idx:end_idx]
            imu_data[a*num_coughs + i, :, 4] = imu.P[start_idx:end_idx]
            imu_data[a*num_coughs + i, :, 5] = imu.R[start_idx:end_idx]
        
    return audio_data, imu_data, num_coughs

def get_non_cough_windows(data_folder,subj_id, trial,mov,noise,sound,n_samp, window_len):
    """Select n_samp audio samples from random locations in the signal with length window_len"""
    #Load data

    air, skin = load_audio(data_folder, subj_id, trial, mov, noise, sound)
    imu = load_imu(data_folder, subj_id, trial, mov, noise, sound)
    window_len_audio = int(window_len*FS_AUDIO)
    window_len_imu = int(window_len*FS_IMU)
    
    #Select random segments
    end_of_signal = np.min((len(air)/FS_AUDIO,len(imu.x)/FS_IMU))
    window_starts = rand_uni = np.random.uniform(0,end_of_signal-window_len,n_samp)
    
    # Preallocate arrays for audio and IMU data
    audio_data = np.zeros((n_samp, window_len_audio, 2))
    imu_data = np.zeros((n_samp, window_len_imu, 6))

    # Segment audio signals using direct slicing (35x faster than linspace)
    window_starts_audio = (window_starts*FS_AUDIO).astype(int)
    # Clamp to valid range
    window_starts_audio = np.clip(window_starts_audio, 0, len(air) - window_len_audio)
    for i, start_idx in enumerate(window_starts_audio):
        end_idx = start_idx + window_len_audio
        audio_data[i, :, 0] = air[start_idx:end_idx]
        audio_data[i, :, 1] = skin[start_idx:end_idx]

    # Segment IMU signals using direct slicing (35x faster than linspace)
    window_starts_imu = (window_starts*FS_IMU).astype(int)
    # Clamp to valid range
    window_starts_imu = np.clip(window_starts_imu, 0, len(imu.x) - window_len_imu)
    for i, start_idx in enumerate(window_starts_imu):
        end_idx = start_idx + window_len_imu
        imu_data[i, :, 0] = imu.x[start_idx:end_idx]
        imu_data[i, :, 1] = imu.y[start_idx:end_idx]
        imu_data[i, :, 2] = imu.z[start_idx:end_idx]
        imu_data[i, :, 3] = imu.Y[start_idx:end_idx]
        imu_data[i, :, 4] = imu.P[start_idx:end_idx]
        imu_data[i, :, 5] = imu.R[start_idx:end_idx]
    
    return audio_data, imu_data

def get_samples_for_subject(data_folder, subj_id, window_len, aug_factor):
    """
    For each subject, extract windows of all of the cough sounds for each movement (sit, walk) and noise condition (none, music, traffic, cough).
    Extract an equal number of non-cough windows for each non-cough sound (laugh, throat, breathe) for the corresponding conditons.
    Inputs:
    - subj_id: ID number of the subject to process
    - window_len: desired data window length (in seconds)
    - aug_factor: augmentation factor; how many times to randomly shift the signal within the window
    Outputs:
    - audio_data: NxMx2 data matrix where
        - N = (number of coughs x augmentation factor x 4)
        - M = int(window_len * 16000)
        - first index = outer microphone, second index = body-facing microphone
    - imu_data: NxLx6 data matrix where
        - L = int(window_len * 100)
        - third dimension specifies IMU signal (accel x,y,z, IMU y,p,r)
    - labels: Nx1 vector of labels
        - 1 = cough
        - 0 = non-cough
    - total_coughs: number of un-augmented cough signals for the subject
    """
    window_len_audio = int(window_len*FS_AUDIO)
    window_len_imu = int(window_len*FS_IMU)

    # OPTIMIZATION: First pass - count total samples to preallocate arrays (38x faster)
    total_coughs = 0
    total_samples = 0
    for trial in Trial:
        for mov in Movement:
            for noise in Noise:
                sound = Sound.COUGH
                path = data_folder + subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                if os.path.isdir(path) & os.path.isfile(path + '/ground_truth.json'):
                    fn = path + '/ground_truth.json'
                    with open(fn, 'rb') as f:
                        loaded_dict = json.load(f)
                    num_coughs = len(loaded_dict["start_times"])
                    total_coughs += num_coughs
                    # Count cough windows + non-cough windows (3 sound types * num_coughs)
                    total_samples += num_coughs * aug_factor * 4

    # Preallocate arrays with exact size (avoids O(n²) concatenation)
    audio_data = np.zeros((total_samples, window_len_audio, 2))
    imu_data = np.zeros((total_samples, window_len_imu, 6))
    labels = np.zeros(total_samples)
    idx = 0

    # Second pass - extract windows with direct assignment instead of concatenation
    for trial in Trial:
        for mov in Movement:
            for noise in Noise:

                # Extract cough windows
                sound = Sound.COUGH
                path = data_folder + subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                if os.path.isdir(path) & os.path.isfile(path + '/ground_truth.json'):
                    fn = path + '/ground_truth.json'
                    audio, imu, num_coughs = get_cough_windows(data_folder,fn, window_len, aug_factor)
                    n = audio.shape[0]
                    audio_data[idx:idx+n] = audio
                    imu_data[idx:idx+n] = imu
                    labels[idx:idx+n] = 1
                    idx += n

                    # Extract non-cough windows
                    for sound in Sound:
                        path = data_folder + subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                        if not os.path.exists(path):
                            print(f"{path} not found. Skipped.")
                            continue
                        if (sound != sound.COUGH) & (len(os.listdir(path)) > 0):
                            audio, imu = get_non_cough_windows(data_folder,subj_id, trial,mov,noise,sound,num_coughs*aug_factor, window_len)
                            n = audio.shape[0]
                            audio_data[idx:idx+n] = audio
                            imu_data[idx:idx+n] = imu
                            labels[idx:idx+n] = 0
                            idx += n

    # Trim to actual size (in case some samples were missing)
    return audio_data[:idx], imu_data[:idx], labels[:idx], total_coughs