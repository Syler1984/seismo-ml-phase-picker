import numpy as np


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


def predict_streams(model, streams, batch_size = 500_000, frequency = 100.):
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
    for i in range(trace_count):

        # Grab current traces
        traces = [x[i] for _, x in streams.items()]
        traces = trim_traces(traces)
        batch_count, last_batch = count_batches(traces, batch_size)

        for b in range(batch_count):

            # Get batch data
            c_batch_size = batch_size
            if b == batch_count - 1 and last_batch:
                c_batch_size = last_batch

            start_sample = batch_size * b
            end_sample = start_sample + c_batch_size
            start_time = traces[0].stats.starttime
            slice_start = start_time + start_sample / frequency
            slice_end = start_time + end_sample / frequency

            batch = [trace.slice(slice_start, slice_end) for trace in traces]

            # Predict
            scores = scan_batch(model, batch)

            print('***\t\t' * 9)
            print('SCORES TYPE: ', type(scores))
            print('SCORES: ')
            print(scores)
            print('***\t\t' * 9)

    return []
