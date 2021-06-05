def preprocess_streams(streams, start_datetime, end_datetime, cut_spans):

    cut_streams = {}
    for channel, stream in streams.items():
        cut_streams[channel] = stream.slice(start_datetime, end_datetime)

    stream.detrend(type = "linear")
    stream.filter(type = "highpass", freq = 2)

    frequency = 100.
    required_dt = 1. / frequency
    dt = stream[0].stats.delta

    if dt != required_dt:
        stream.interpolate(frequency)

    # TODO: grab streams maximum start_time and minimum end time

    # TODO: trim all the streams to it

    # TODO: write function which takes streams time span and cut_spans and return list of spans which
    #       should be passed to stream.slices. Basically, list of slice spans

    # TODO: slice all the streams and return them as a list of dictionaries
