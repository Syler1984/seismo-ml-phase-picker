import numpy as np
from scipy.signal import find_peaks
from .h5_tools import write_batch


def cut_spans_to_slices(cut_spans, start_time, end_time):
    """
    Utility function for reversing list of time spans to cut from the stream to list of time spans which should
    be provided to ObsPy stream.slice method.
    :param cut_spans: time spans to cut from the stream
    :param start_time: stream start time
    :param end_time: stream end time
    :return: list of time span tuples: [(start, end), ...]
    """
    slice_spans = [[start_time, end_time]]

    for cut_span in cut_spans:

        i = 0
        while i < len(slice_spans):

            slice_span = slice_spans[i]
            if cut_span[0] < slice_span[0] and slice_span[1] < cut_span[1]:
                slice_spans.pop(i)
                continue
            elif cut_span[0] < slice_span[0] < cut_span[1]:
                slice_span[0] = cut_span[1]
            elif cut_span[0] < slice_span[1] < cut_span[1]:
                slice_span[1] = cut_span[0]
            elif slice_span[0] < cut_span[0] and cut_span[1] < slice_span[1]:
                new_span = [slice_span[0], cut_span[0]]
                slice_span[0] = cut_span[1]
                slice_spans.insert(i, new_span)
                i += 1

            i += 1

    return slice_spans


def preprocess_streams(streams, start_datetime, end_datetime, cut_spans):
    """
    Preprocess streams: filter, detrend, interpolate, trim, cut
    :param streams: Input streams dictionary in format {channel: stream}
    :param start_datetime: Start datetime to trim all the streams
    :param end_datetime: End datetime to trim all the streams
    :param cut_spans: Time spans to cut from the streams
    :return: List of cut streams dictionaries
    """
    # Filter, highpass, interpolate
    for _, stream in streams.items():
        stream.detrend(type = "linear")
        stream.filter(type = "highpass", freq = 2)

        frequency = 100.  # TODO: Init this from argv
        required_dt = 1. / frequency
        dt = stream[0].stats.delta

        if dt != required_dt:
            stream.interpolate(frequency)

    # Trim
    max_start_time = max([stream[0].stats.starttime for _, stream in streams.items()])
    min_end_time = min([stream[-1].stats.endtime for _, stream in streams.items()])

    max_start_time = max([max_start_time, start_datetime])
    min_end_time = min([min_end_time, end_datetime])

    cut_streams = {}
    for channel, stream in streams.items():
        cut_streams[channel] = stream.slice(max_start_time, min_end_time)
    streams = cut_streams
    del cut_streams

    slices = cut_spans_to_slices(cut_spans, max_start_time, min_end_time)

    result_streams = []
    for span in slices:

        stream_group = {}
        for channel, stream in streams.items():
            stream_group[channel] = stream.slice(span[0], span[1])

        result_streams.append(stream_group)

    return result_streams


def trim_traces(traces):
    """
    Trim traces to the same length
    :param traces:
    :return:
    """
    max_start_time = max([x.stats.starttime for x in traces])
    min_end_time = min([x.stats.endtime for x in traces])

    trimmed = [x.slice(max_start_time, min_end_time) for x in traces]

    return trimmed


def count_batches(traces, batch_size):
    """
    Counts batches amount.
    :param traces:
    :param batch_size:
    :return:
    """
    trace_length = traces[0].data.shape[0]

    batch_count = trace_length // batch_size
    last_batch = trace_length % batch_size
    if last_batch:
        batch_count += 1

    return batch_count, last_batch


def sliding_window(data, length, shift):
    """
    Returns sliding windows NumPy array
    :param data:
    :param length:
    :param shift:
    :return:
    """
    count = np.floor((data.shape[0] - length) / shift + 1).astype(int)
    shape = [count, length]

    strides = [data.strides[0] * shift, data.strides[0]]

    # TODO: Fix negative dimensions error!
    windows = np.lib.stride_tricks.as_strided(data, shape, strides)

    return windows.copy()


def normalize_windows(windows):
    """
    Normalizes windows by global absolute maximum
    :param windows:
    :return:
    """
    length = windows.shape[0]

    for i in range(length):

        global_max = np.max(np.abs(windows[i, :, :]))
        windows[i, :, :] = windows[i, :, :] / global_max


def scan_batch(model, batch):
    """
    Scans batch and returns predicted scores.
    :param model:
    :param batch:
    :return:
    """
    # TODO: Get number of features and shift from parameters
    windows = [sliding_window(x.data, 400, 10) for x in batch]  # get windows
    min_length = min([x.shape[0] for x in windows])  # trim windows

    # Convert to NumPy array
    data = np.zeros((min_length, 400, len(windows)))
    for i in range(len(windows)):
        data[:, :, i] = windows[i][:min_length]
    windows = data

    normalize_windows(windows)

    # TODO: Maybe keep model verbal and use it's own tools to help with the progress bar
    scores = model.predict(windows, verbose = False, batch_size = 500)

    return scores


def get_positives(scores, label, other_labels, threshold):
    """

    :param scores:
    :param label:
    :param other_labels:
    :param threshold:
    :return:
    """
    avg_window_half_size = 100
    positives = []

    x = scores[:, label]
    peaks = find_peaks(x, distance = 10_000, height = [threshold, 1.])

    for i in range(len(peaks[0])):

        start_id = peaks[0][i] - avg_window_half_size
        if start_id < 0:
            start_id = 0

        end_id = start_id + avg_window_half_size*2
        if end_id > len(x):
            end_id = len(x) - 1
            start_id = end_id - avg_window_half_size*2

        # Get mean values
        peak_mean = x[start_id : end_id].mean()

        means = []
        for idx in other_labels:

            means.append(scores[:, idx][start_id : end_id].mean())

        is_max = True
        for m in means:

            if m > peak_mean:
                is_max = False

        if is_max:
            positives.append([peaks[0][i], peaks[1]['peak_heights'][i]])
    
    return positives


def get_windows(batch, n_window, n_features, shift):
    """
    Returns window by its number in the batch.
    :return:
    """

    # TODO: Get number of features and shift from parameters
    windows = [sliding_window(x.data, n_features, shift) for x in batch]  # get windows
    min_length = min([x.shape[0] for x in windows])  # trim windows

    # Convert to NumPy array
    data = np.zeros((min_length, 400, len(windows)))
    for i in range(len(windows)):
        data[:, :, i] = windows[i][:min_length]
    windows = data

    normalize_windows(windows)

    return windows[n_window]

    """
    n_channels = len(batch)
    window = np.zeros((1, n_features, n_channels))
    start_pos = shift * n_window
    for i, trace in enumerate(batch):
        window[0, :, i] = trace.data[start_pos : start_pos + n_features]

    return window
    """


def restore_scores(_scores, shape, shift):
    """
    Restores scores to original size using linear interpolation.

    Arguments:
    scores -- original 'compressed' scores
    shape  -- shape of the restored scores
    shift  -- sliding windows shift
    """
    new_scores = np.zeros(shape)
    for i in range(1, _scores.shape[0]):

        for j in range(_scores.shape[1]):

            start_i = (i - 1) * shift
            end_i = i * shift
            if end_i >= shape[0]:
                end_i = shape[0] - 1

            new_scores[start_i : end_i, j] = np.linspace(_scores[i - 1, j], _scores[i, j], shift + 1)[:end_i - start_i]

    return new_scores


def plot_wave_scores(file_token, wave, scores,
                     start_time, predictions, right_shift = 0,
                     channel_names = ['N', 'E', 'Z'],
                     score_names = ['P', 'S', 'N']):
    """
    Plots waveform and prediction scores as an image
    """
    channels_num = wave.shape[1]
    classes_num = scores.shape[1]
    scores_length = scores.shape[0]

    # TODO: Make figure size dynamically chosen, based on the input length
    fig = plt.figure(figsize = (9.8, 7.), dpi = 160)
    axes = fig.subplots(channels_num + classes_num, 1, sharex = True)

    # Plot wave
    for i in range(channels_num):

        axes[i].plot(wave[:, i], color = '#000000', linewidth = 1.)
        axes[i].locator_params(axis = 'both', nbins = 4)
        axes[i].set_ylabel(channel_names[i])

    # Process events and ticks
    freq = 100.  # TODO: implement through Trace.stats
    labels = {'p': 0, 's': 1}  # TODO: configure labels through options
    # TODO: make sure that labels are not too close.
    ticks = [100, scores_length - 100]
    events = {}

    for label, index in labels.items():

        label_events = []
        for pos, _ in predictions[label]:

            pos += right_shift
            label_events.append(pos)
            ticks.append(pos)

        events[index] = label_events

    # Plot scores
    for i in range(classes_num):

        axes[channels_num + i].plot(scores[:, i], color = '#0022cc', linewidth = 1.)

        if i in events:
            for pos in events[i]:
                axes[channels_num + i].plot([pos], scores[:, i][pos], 'r*', markersize = 7)

        axes[channels_num + i].set_ylabel(score_names[i])

    # Set x-ticks
    for ax in axes:
        ax.set_xticks(ticks)

    # Configure ticks labels
    xlabels = []
    for pos in axes[-1].get_xticks():

        c_time = start_time + pos/freq
        micro = c_time.strftime('%f')[:2]
        xlabels.append(c_time.strftime('%H:%M:%S') + f'.{micro}')

    axes[-1].set_xticklabels(xlabels)

    # Add date text
    date = start_time.strftime('%Y-%m-%d')
    fig.text(0.095, 1., date, va = 'center')

    # Finalize and save
    fig.tight_layout()
    fig.savefig(file_token + '.jpg')
    fig.clear()


def print_scores(data, scores, predictions, file_token, window_length = 400):
    """
    Prints scores and waveforms.
    """
    right_shift = window_length // 2

    shapes = [d.data.shape[0] for d in data] + [scores.shape[0]]
    shapes = set(shapes)

    if len(shapes) != 1:
        raise AttributeError('All waveforms and scores must have similar length!')

    length = shapes.pop()

    waveforms = np.zeros((length, len(data)))
    for i, d in enumerate(data):
        waveforms[:, i] = d.data

    # Shift scores
    shifted_scores = np.zeros((length, len(data)))
    shifted_scores[right_shift:] = scores[:-right_shift]

    plot_wave_scores(file_token, waveforms, shifted_scores, data[0].stats.starttime, predictions,
                     right_shift = right_shift)

    # TODO: Save predictions samples in .csv ?

    np.save(f'{file_token}_wave.npy', waveforms)
    np.save(f'{file_token}_scores.npy', shifted_scores)


def predict_streams(model, streams, frequency = 100., params = None, progress_bar = None):
    """
    Predicts streams and returns scores
    :param frequency:
    :param model:
    :param streams:
    :param batch_size:
    :return:
    """
    # TODO: Implement progress bar through class
    # Check if every stream in the group has equal number of traces
    trace_counts = [len(x) for _, x in streams.items()]
    if len(np.unique(np.array(trace_counts))) != 1:
        raise AttributeError('Not equal number of traces in the stream!')

    trace_count = trace_counts[0]

    # Progress bar
    progress_bar.change_max('traces', trace_count)
    progress_bar.set_progress(0, level = 'traces')

    for i in range(trace_count):

        progress_bar.set_progress(i, level = 'traces')

        # Grab current traces
        traces = [x[i] for _, x in streams.items()]
        traces = trim_traces(traces)
        batch_count, last_batch = count_batches(traces, params['batch_size'])

        # Progress bar
        progress_bar.change_max('batches', batch_count)
        progress_bar.set_progress(0, level = 'batches')

        for b in range(batch_count):

            progress_bar.set_progress(b, level = 'batches')

            # Get batch data
            c_batch_size = params['batch_size']
            if b == batch_count - 1 and last_batch:
                c_batch_size = last_batch

            start_sample = params['batch_size'] * b
            end_sample = start_sample + c_batch_size
            start_time = traces[0].stats.starttime
            slice_start = start_time + start_sample / frequency
            slice_end = start_time + end_sample / frequency

            # Progress bar
            progress_bar.set_postfix_arg('start', slice_start)
            progress_bar.set_postfix_arg('end', slice_end)
            progress_bar.print()

            batch = [trace.slice(slice_start, slice_end) for trace in traces]

            # Predict
            scores = scan_batch(model, batch)

            # TODO: Should i restore scores?
            # 1. Try restoring scores and look at results
            # 2. Drop scores restoring, do not forget to change parameters for pick detection (window size, etc.)
            # 3. Compare results!
            scores = restore_scores(scores, (len(batch[0]), len(params['model_labels'])), 10)

            # Find positives
            predicted_labels = {}
            for p_label_name, p_label in params['positive_labels'].items():

                other_labels = []
                for m_label_name, m_label in params['model_labels'].items():
                    if m_label_name != p_label_name:
                        other_labels.append(m_label)

                positives = get_positives(scores,
                                          p_label,
                                          other_labels,
                                          threshold = params['threshold'])

                predicted_labels[p_label_name] = positives

            for label, predictions in predicted_labels.items():
                Y = np.zeros(1, dtype = 'int')
                Y[0] = params['model_labels'][label]

                P = np.zeros(1)

                for position, prob in predictions:

                    # TODO: Get number of features and shift from parameters
                    X = get_windows(batch, position, 400, 10)

                    P[0] = prob
                    write_batch(params['out_hdf5'], 'X', X)
                    write_batch(params['out_hdf5'], 'Y', Y)
                    write_batch(params['out_hdf5'], 'P', P)

                # TODO: Extract additional info about positives, e.g. sample position, timestamp, channel data, P.
            
            # TODO: Put them into the array
            if False:
                print_scores(batch, scores, predicted_labels, f't{i}_b{b}')

    return []
