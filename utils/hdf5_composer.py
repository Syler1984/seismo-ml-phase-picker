from obspy.core import read
import config.vars as config
import h5py
import re


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
    print('NOISE: ' + str(len(noise_picks)))
    print('P: ' + str(len(p_picks)))
    print('S: ' + str(len(s_picks)))
    initial_set = []

    X = []
    Y = []
    Z = []

    index = 0
    local_index = 0
    while local_index < len(p_picks):
        transposed_list = []
        inner_index = 0
        while inner_index < len(p_picks[local_index][0][0]) and \
                inner_index < len(p_picks[local_index][1][0]) and \
                inner_index < len(p_picks[local_index][2][0]):
            transposed_list.append([p_picks[local_index][0][0][inner_index],
                                    p_picks[local_index][1][0][inner_index],
                                    p_picks[local_index][2][0][inner_index]])
            inner_index += 1

        X.append(transposed_list)
        Y.append(config.p_code)
        Z.append(p_picks[local_index][0][2])
        index += 1
        local_index += 1

    index = 0
    local_index = 0
    while local_index < len(s_picks):
        transposed_list = []
        inner_index = 0
        while inner_index < len(s_picks[local_index][0][0]) and \
                inner_index < len(s_picks[local_index][1][0]) and \
                inner_index < len(s_picks[local_index][2][0]):
            transposed_list.append([s_picks[local_index][0][0][inner_index],
                                    s_picks[local_index][1][0][inner_index],
                                    s_picks[local_index][2][0][inner_index]])
            inner_index += 1

        X.append(transposed_list)
        Y.append(config.s_code)
        ascii_string = s_picks[local_index][0][1].encode("ascii", "ignore")
        Z.append(ascii_string)
        index += 1
        local_index += 1

    index = 0
    local_index = 0

    while local_index < len(noise_picks) and local_index < len(s_picks):
        transposed_list = []
        inner_index = 0
        while inner_index < len(noise_picks[local_index][0][0]) and \
                inner_index < len(noise_picks[local_index][1][0]) and \
                inner_index < len(noise_picks[local_index][2][0]):
            transposed_list.append([noise_picks[local_index][0][0][inner_index],
                                    noise_picks[local_index][1][0][inner_index],
                                    noise_picks[local_index][2][0][inner_index]])
            inner_index += 1

        X.append(transposed_list)
        Y.append(config.noise_code)
        ascii_string = noise_picks[local_index][0][1].encode("ascii", "ignore")
        Z.append(ascii_string)
        index += 1
        local_index += 1



    file = h5py.File(filename, "w")

    dset1 = file.create_dataset('X', data=X)
    dset2 = file.create_dataset('Y', data=Y)

    if config.save_ids:
        dset3 = file.create_dataset(config.ids_dataset_name, data=Z)

    file.close()

    return None


def process(filename, file_format="MSEED", rand=0, noise=False):
    """
    Processes a pick file to be suitable for hdf5 packing
    :param filename:    string - filename
    :param file_format: string - format of the file, default: miniSEED "MSEED"
    :return: list of samples
    """
    st = read(filename, file_format)

    # Is acceleration based
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

    # High-pass filtering
    if config.highpass_filter_df > 1:
        st.filter("highpass", freq=config.highpass_filter_df)
    # st.filter("bandpass", freqmin=2, freqmax=20)

    # Slice offset
    if not noise:
        start_time = st[0].stats.starttime
        end_time = st[0].stats.endtime
        offset = (config.slice_duration - config.slice_size)/2
        st = st.slice((start_time + offset), (end_time - offset))

    # Normalize
    if config.normalization_enabled:
        st.normalize(global_max=config.global_max_normalizing)

    # Check that size is accurate
    resize(st, config.required_trace_length)

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
    if len(stream[0].data) == size:
        return

    if len(stream[0]) > size:
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
