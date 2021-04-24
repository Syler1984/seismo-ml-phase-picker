from obspy.core import read
import config.vars as config
import h5py
import re

import matplotlib.pyplot as plt
import numpy as np


def compose(filename, p_picks, s_picks, noise_picks):
    """
    Composes an hdf5 file from processed data
    :param filename:    string - full name of resulting hdf5 file
    :param p_picks:     list   - list of processes p-wave picks
    :param s_picks:     list   - list of processes s-wave picks
    :param noise_picks: list   - list of processes noise picks
    :return:
    """
    # Creating datasets
    initial_set = []

    if True:
        length = len(p_picks) + len(s_picks) + len(noise_picks)
    else:
        length = len(p_picks) + len(s_picks)

    X = np.zeros((length, config.required_trace_length, 3))
    Y = np.zeros((length))
    Z = []
    YYZ = np.zeros((length, 3, config.required_trace_length))


    index = 0
    local_index = 0

    while local_index < len(p_picks):
        transposed_list = np.zeros((config.required_trace_length, 3))
        inner_index = 0
        while inner_index < config.required_trace_length:
            transposed_list[inner_index][0] = p_picks[local_index][0][0][0][inner_index]
            transposed_list[inner_index][1] = p_picks[local_index][1][0][0][inner_index]
            transposed_list[inner_index][2] = p_picks[local_index][2][0][0][inner_index]

            inner_index += 1

        X[local_index, :, :] = transposed_list
        Y[local_index] = config.p_code
        Z.append(p_picks[local_index][0][2])
        YYZ[local_index, 0, :] = p_picks[local_index][0][0][1]
        YYZ[local_index, 1, :] = p_picks[local_index][0][0][2]
        YYZ[local_index, 2, :] = p_picks[local_index][0][0][3]

        index += 1
        local_index += 1


    index = 0
    local_index = 0
    while local_index < len(s_picks):
        transposed_list = np.zeros((config.required_trace_length, 3))
        inner_index = 0
        while inner_index < config.required_trace_length:
            transposed_list[inner_index][0] = s_picks[local_index][0][0][0][inner_index]
            transposed_list[inner_index][1] = s_picks[local_index][1][0][0][inner_index]
            transposed_list[inner_index][2] = s_picks[local_index][2][0][0][inner_index]

            inner_index += 1

        X[local_index + len(p_picks), :, :] = transposed_list
        Y[local_index + len(p_picks)] = config.s_code
        Z.append(s_picks[local_index][0][2])
        YYZ[local_index + len(p_picks), 0, :] = s_picks[local_index][0][0][1]
        YYZ[local_index + len(p_picks), 1, :] = s_picks[local_index][0][0][2]
        YYZ[local_index + len(p_picks), 2, :] = s_picks[local_index][0][0][3]

        index += 1
        local_index += 1
    '''
    local_index = 0
    while local_index < len(noise_picks) and local_index < len(noise_picks):
        transposed_list = np.zeros((config.required_trace_length, 3))
        inner_index = 0
        while inner_index < config.required_trace_length:
            transposed_list[inner_index][0] = noise_picks[local_index][0][0][0][inner_index]
            transposed_list[inner_index][1] = noise_picks[local_index][1][0][0][inner_index]
            transposed_list[inner_index][2] = noise_picks[local_index][2][0][0][inner_index]
            inner_index += 1


        X[local_index + len(p_picks) + len(s_picks), :, :] = transposed_list
        Y[local_index + len(p_picks) + len(s_picks)] = config.noise_code

        ascii_string = noise_picks[local_index][0][1].encode("ascii", "ignore")
        Z.append(ascii_string)
        # Z.append(noise_picks[local_index][0][2])
        YYZ[local_index + len(p_picks) + len(s_picks), 0, :] = noise_picks[local_index][0][0][1]
        YYZ[local_index + len(p_picks) + len(s_picks), 1, :] = noise_picks[local_index][0][0][2]
        YYZ[local_index + len(p_picks) + len(s_picks), 2, :] = noise_picks[local_index][0][0][3]

        local_index += 1
    '''
    file = h5py.File(filename, "w")

    dset1 = file.create_dataset('X', data=X)
    dset2 = file.create_dataset('Y', data=Y)
    dset3 = file.create_dataset('YYZ', data=YYZ)

    if config.save_ids:
        dset6 = file.create_dataset(config.ids_dataset_name, data=Z)

    file.close()

    return None


def process_pick_list(list, is_noise_from_wave=False):
    """
    Processes a list of waveform picks. Only works with actual phases, does not process noise list
    :param is_noise_from_wave:
    :param list: List of waveform picks
    :return: List of processed data
    """
    # Get stats
    pick_phase_hint = list[0]
    all_actual_picks = list[1]
    pick_stats = all_actual_picks[0]
    all_picks = all_actual_picks[1:]

    result_list = []
    # Parse events
    for event in all_picks:
        event_stats = event[0]
        slice_groups = event[1:]

        # Parse slice group
        for group in slice_groups:
            group_stats = group[0]
            picks = group[1:]

            # Parse actual slices
            if len(picks) != len(config.archive_channels_order):
                continue

            picks_ordered = []
            for channel_name in config.archive_channels_order:
                channel_found = False
                for file in picks:
                    file_name_split = file.split('/')
                    file_name = file_name_split[len(file_name_split) - 1]
                    # Check file channel
                    name_split = file_name.split('.')
                    channel = name_split[2][len(name_split[2]) - 1]

                    if channel == channel_name:
                        picks_ordered.append(file)
                        channel_found = True
                        break
                if not channel_found:
                    break

            if len(picks_ordered) != len(config.archive_channels_order):
                continue

            # Process slices
            processed_slices = []
            for slice_file in picks_ordered:
                file_name_split = slice_file.split('/')
                file_name = file_name_split[len(file_name_split) - 1]
                name_split = file_name.split('.')
                spip = name_split[2]
                is_acc = False
                # Check if its accelerogramm
                if spip in config.acc_codes:
                    is_acc = True

                processed = process(slice_file, group_stats.file_format, noise=False, is_acc=is_acc,
                                    is_noise_from_wave=is_noise_from_wave)

                if processed is None:
                    processed_slices = None
                    break

                # Save pick and filename and event ID
                processed_slices.append([processed, slice_file, int(group_stats.event_id)])

            if processed_slices is not None:

                if config.normalization_enabled:
                    max = 0.0
                    for channel_slice in processed_slices:
                        for x in channel_slice[0]:
                            if abs(x) > max:
                                max = x

                    for channel_slice in processed_slices:
                        for i in range(0, len(channel_slice[0])):
                            channel_slice[0][i] = float(channel_slice[0][i]) / max

                result_list.append(processed_slices)

    return result_list


def process(filename, file_format="MSEED", rand=0, noise=False, is_acc=False, is_noise_from_wave=False):
    """
    Processes a pick file to be suitable for hdf5 packing
    :param filename:    string - filename
    :param file_format: string - format of the file, default: miniSEED "MSEED"
    :return: list of samples
    """
    st = read(filename, file_format)

    # Is acceleration based
    if noise:
        is_acc = False
        regex_filter = re.search(r'\.[a-zA-Z]{3}', filename)
        type_of_file = str(regex_filter.group(0)[1:4])
        if type_of_file in config.acc_codes:
            is_acc = True

        if config.ignore_acc and is_acc:
            return None

    # Resampling
    if st[0].stats.sampling_rate < config.required_df:
        return None
    if st[0].stats.sampling_rate != config.required_df:
        resample(st, config.required_df)

    # Detrend
    if config.detrend:
        st.detrend(type='linear')

    # Acceleration to velocity
    if is_acc:
        for trace in st.traces:
            trace.integrate()

    # Bandpass filtering
    # st.filter(type='bandpass', freqmin=3.0, freqmax=20.0)

    # High-pass filtering
    if config.highpass_filter_df > 1:
        st.filter("highpass", freq=config.highpass_filter_df)

    # Slice offset
    if not is_noise_from_wave:
        if not noise:
            start_time = st[0].stats.starttime
            end_time = st[0].stats.endtime
            offset = (config.slice_duration - config.slice_size)/2
            st = st.slice((start_time + offset), (end_time - offset))
        else:
            start_time = st[0].stats.starttime
            end_time = st[0].stats.endtime
            duration = float(end_time - start_time)
            offset = (duration - config.slice_size) / 2
            st = st.slice((start_time + offset), (end_time - offset))
    else:
        start_time = st[0].stats.starttime
        offset = config.noise_offset
        st = st.slice((start_time + offset), (start_time + offset + config.slice_size))

    # Normalize
    if config.normalization_enabled and (is_noise_from_wave or noise):
        st.normalize(global_max=config.global_max_normalizing)

    # Check that size is accurate
    #resize(st, config.required_trace_length)

    if True: #noise: #True: #noise:
        a_pick = st[0].data

        if a_pick.size < config.required_trace_length:
            if a_pick.size < config.required_trace_length/2:
                return None

            a_new_pick = np.zeros(config.required_trace_length)
            a_new_pick[:a_pick.size] = a_pick
            a_new_pick[a_pick.size:] = np.flip(a_pick[-(config.required_trace_length - a_pick.size):])

        if a_pick.size > config.required_trace_length:
            a_new_pick = np.zeros(config.required_trace_length)
            a_new_pick[:config.required_trace_length] = a_pick[:config.required_trace_length]

        return a_new_pick

    return st[0].data


def resample(stream, df):
    """
    Resamples trace to required sampling rate
    :param trace: trace to resample
    :param df:
    :return:
    """
    stream.resample(df)


def resize(stream, size):
    """
    Cuts list of data size
    :param data:
    :return:
    """
    if stream[0].data.size == size:
        return

    if stream[0].data.size > size:
        stream[0].data = stream[0].data[:size]
    # else: throw exception?


def sample(stream):
    """
    Gets samples list from trace
    :param stream:
    :return:
    """
    # Normalize (by maximum amplitude)
    stream.normalize()
    normalized_data = stream.data

    # Check df and resample if necessary
    resampled_data = []
    if int(stream.trace.stats.sampling_rate) == config.required_df:
        resampled_data = normalized_data
    resampled_data = normalized_data

    # Check length
    result_data = []

    i = 0
    while i < config.required_sample_length:
        if i < len(resampled_data):
            result_data.append(resampled_data[i])
        else:
            # Log message about adding zeroes to result data
            result_data.append(0)

    return result_data


def clear_process_pick_list(list, is_noise_from_wave=False):
    """
    Processes a list of waveform picks. Only works with actual phases, does not process noise list
    :param is_noise_from_wave:
    :param list: List of waveform picks
    :return: List of processed data
    """
    # Get stats
    pick_phase_hint = list[0]
    all_actual_picks = list[1]
    pick_stats = all_actual_picks[0]
    all_picks = all_actual_picks[1:]

    result_list = []
    # Parse events
    for event in all_picks:
        event_stats = event[0]
        slice_groups = event[1:]

        # Parse slice group
        for group in slice_groups:
            group_stats = group[0]
            picks = group[1:]

            # Parse actual slices
            if len(picks) != len(config.archive_channels_order):
                continue

            picks_ordered = []
            for channel_name in config.archive_channels_order:
                channel_found = False
                for file in picks:
                    file_name_split = file.split('/')
                    file_name = file_name_split[len(file_name_split) - 1]
                    # Check file channel
                    name_split = file_name.split('.')
                    channel = name_split[2][len(name_split[2]) - 1]

                    if channel == channel_name:
                        picks_ordered.append(file)
                        channel_found = True
                        break
                if not channel_found:
                    break

            if len(picks_ordered) != len(config.archive_channels_order):
                continue

            # Process slices
            processed_slices = []
            for slice_file in picks_ordered:
                file_name_split = slice_file.split('/')
                file_name = file_name_split[len(file_name_split) - 1]
                name_split = file_name.split('.')
                spip = name_split[2]
                is_acc = False
                # Check if its accelerogramm
                if spip in config.acc_codes:
                    is_acc = True

                processed = clear_process(slice_file, group_stats.file_format, noise=False, is_acc=is_acc,
                                          is_noise_from_wave=is_noise_from_wave, stats=group_stats, event_id=int(group_stats.event_id))

                if processed is None:
                    processed_slices = None
                    break

                # Save pick and filename and event ID
                processed_slices.append([processed, slice_file, int(group_stats.event_id)])

            if processed_slices is not None:
                '''if config.normalization_enabled:
                    max = 0.0
                    for channel_slice in processed_slices:
                        for x in channel_slice[0]:
                            if abs(x) > max:
                                max = x

                    for channel_slice in processed_slices:
                        for i in range(0, len(channel_slice[0])):
                            channel_slice[0][i] = float(channel_slice[0][i]) / max'''

                result_list.append(processed_slices)

    return result_list


def clear_process(filename, file_format="MSEED", rand=0, noise=False, is_acc=False,
                  is_noise_from_wave=False, stats=None, event_id=-1, plot_path=None):
    """
    Processes a pick file to be suitable for hdf5 packing
    :param filename:    string - filename
    :param file_format: string - format of the file, default: miniSEED "MSEED"
    :return: list of samples
    """
    st = read(filename, file_format)

    # Is acceleration based
    if noise:
        is_acc = False
        regex_filter = re.search(r'\.[a-zA-Z]{3}', filename)
        type_of_file = str(regex_filter.group(0)[1:4])
        if type_of_file in config.acc_codes:
            is_acc = True

        if config.ignore_acc and is_acc:
            return None

    # Resampling
    # if st[0].stats.sampling_rate < config.required_df:
        # return None

    if st[0].stats.sampling_rate != config.required_df:
        resample(st, config.required_df)

    # Detrend
    if config.detrend:
        st.detrend(type='linear')

    # Acceleration to velocity
    if is_acc:
        for trace in st.traces:
            trace.integrate()

    # Bandpass filtering
    # st.filter(type='bandpass', freqmin=3.0, freqmax=20.0)

    # High-pass filtering
    if config.highpass_filter_df > 1:
        st.filter("highpass", freq=config.highpass_filter_df)

    # Label gen functions
    def spike(l=60, m=0.99):
        '''
        only even l values are accepted
        '''
        y = np.zeros(l)

        h = int(l / 2)
        y[:h] = np.linspace(0.0, m, h)

        step = y[1] - y[0]
        y[h:] = np.linspace(m - step, 0.0, h)

        return y

    def quad_spike(l=60, s=30, m=0.99):
        y1 = np.linspace(0.0, m, s)
        y3 = np.linspace(m, 0.0, s)

        y = np.zeros(l)
        if l > 2 * s:
            y2 = np.zeros(l - 2 * s)
            y2[:] = m

            y[:s] = y1
            y[s:-s] = y2
            y[-s:] = y3
        else:
            y[:s] = y1
            y[-s:] = y3

        return y

    def gen_labels(*args, l=6000):
        arr = np.zeros(l)

        for x in args:
            # if x + 30 >= arr.shape[0]:
                # continue

            start = x - 30
            end = x + 30

            if start < 0:
                start = 0
            if end >= arr.shape[0]:
                end = arr.shape[0] - 1

            s_start = start + 30 - x
            s_end = end + 30 - x
            arr[start : end] = spike()[s_start : s_end]

        return arr

    def gen_detection(*args, l=6000):
        arr = np.zeros(l)

        start = min(args)
        end = max(args)

        length = end - start
        if length < 30:
            arr[start : start + 30] = spike(30)
        else:
            arr[start: end] = quad_spike(length)

        return arr

    def center_slice(*n_arrs, l=6000):
        if l > n_arrs[0].shape[0]:
            return n_arrs

        size = n_arrs[0].shape[0]
        c_size = size - l

        l_cut = r_cut = int(c_size / 2)
        if size % 2 == 1:
            r_cut += 1

        if r_cut == 0:
            result = [x[l_cut:] for x in n_arrs]
        else:
            result = [x[l_cut:-r_cut] for x in n_arrs]

        return result

    if noise:
        x = st[0].data
        length = x.shape[0]

        # Extend
        if length < config.required_trace_length:
            if length < config.required_trace_length / 2:
                return None

            if not noise:
                return None

            n_x = np.zeros(config.required_trace_length)
            n_x[:length] = x
            n_x[length:] = np.flip(x[-(config.required_trace_length - length):])
            x = n_x
            length = x.shape[0]

        p_label = np.zeros(length)
        s_label = np.zeros(length)
        detection_label = np.zeros(length)

        # Slice
        result_arr = center_slice(x, p_label, s_label, detection_label, l=config.required_trace_length)

        # Normalize pick
        result_arr[0] = result_arr[0] / np.abs(result_arr[0]).max()

        '''
        import matplotlib.pyplot as plt

        if noise:
            plt.plot(x)
            plt.savefig(plot_path)
            plt.clf()
        '''

        return result_arr

    if not is_noise_from_wave:
        # Prepare for slicing and label gen
        x = st[0].data
        alt_phase = stats.alt_phase
        alt_phase_time = stats.alt_phase_time
        phase = stats.phase_hint
        phase_time = stats.phase_time

        start_time = st[0].stats.starttime
        end_time = st[0].stats.endtime

        # print("PHASE TIME: {}, START: {}, END: {}".format(phase_time, start_time, end_time))
        # if alt_phase is not None:
            # print("ALT PHASE: {}".format(alt_phase))

        length = x.shape[0]

        p_mark = None
        s_mark = None

        marks = []

        # Extend
        if length < config.required_trace_length:
            if length < config.required_trace_length / 2:
                return None

            if not noise:
                return None

            n_x = np.zeros(config.required_trace_length)
            n_x[:length] = x
            n_x[length:] = np.flip(x[-(config.required_trace_length - length):])
            x = n_x
            length = x.shape[0]

        # Generate labels
        if phase_time is not None:
            p_time = phase_time - start_time
            p_mark = p_time

            if phase_time > end_time:
                phase_label = np.zeros(length)
            elif p_time <= 0:
                phase_label = np.zeros(length)
            else:
                # phase_label_mark = int(p_time*config.required_df)
                # phase_label = gen_labels(phase_label_mark, l=length)
                # marks.append(phase_label_mark)
                phase_label_mark = int(p_time*config.required_df)
                phase_label = np.zeros(length)
                phase_label[phase_label_mark] = 0.99
                marks.append(phase_label_mark)
        else:
            phase_label = np.zeros(length)

        if alt_phase_time is not None:
            p_time = alt_phase_time - start_time
            s_mark = p_time

            if alt_phase_time > end_time:
                alt_phase_label = np.zeros(length)
            elif p_time <= 0:
                alt_phase_label = np.zeros(length)
            else:
                # phase_label_mark = int(p_time*config.required_df)
                # alt_phase_label = gen_labels(phase_label_mark, l=length)
                # marks.append(phase_label_mark)
                phase_label_mark = int(p_time * config.required_df)
                alt_phase_label = np.zeros(length)
                alt_phase_label[phase_label_mark] = 0.99
                marks.append(phase_label_mark)
        else:
            alt_phase_label = np.zeros(length)

        # Determine phase
        p_label = phase_label
        s_label = alt_phase_label

        if phase == 'S':
            tmp = s_mark
            s_mark = p_mark
            p_mark = tmp

            p_label = alt_phase_label
            s_label = phase_label

        # Generate detection
        detection_label = np.zeros(length)
        if len(marks) == 1:
            detection_label = gen_detection(marks[0], marks[0] + 200, l=length)
        if len(marks) >= 2:
            detection_label = gen_detection(*marks, l=length)

        # Slice
        result_arr = center_slice(x, p_label, s_label, detection_label, l=config.required_trace_length)

        # Normalize pick
        result_arr[0] = result_arr[0] / np.abs(result_arr[0]).max()

        return result_arr

    return [st[0].data]