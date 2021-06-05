def cut_spans_to_slices(cut_spans, start_time, end_time):

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

    print('-' * 35)
    print('start: ', max_start_time)
    print('end: ', min_end_time)
    print('cut_spans:')
    for span in cut_spans:
        print(span)
    print('\t\t***' * 5)
    print('slice_spans:')
    for span in slices:
        print(span)
    print('-' * 35)
    # TODO: write function which takes streams time span and cut_spans and return list of spans which
    #       should be passed to stream.slices. Basically, list of slice spans

    # TODO: slice all the streams and return them as a list of dictionaries
