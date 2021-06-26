import argparse
import sys
from obspy.core.utcdatetime import UTCDateTime
from obspy import read

from utils.ini_tools import parse_ini
import utils.seisan_tools as st
import utils.predict_tools as pt
from utils.converter import date_str

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Default params
default_model_weights = {
    'seismo': 'weights/seismo_sakh_2014_2019.h5',
    'favor': None,
    'cnn': None
}

day_length = 60. * 60 * 24

# TODO: Make sure every parameter is used!
params = {
    'config': 'config.ini',
    'channel_order': ['N', 'E', 'Z'],
    'threshold': 0.95,
    'mulplt': None,
    'batch_size': 500_000,
    'start': None,
    'end': None,
    'slice_range': 2.,  # seconds before and after event
    'archive_path': None,
    's_path': None,
    'seisan': None,
    'frequency': 100.,
    'out': 'noise',
    'debug': False,
    'seismo': False,
    'favor': False,
    'cnn': False,
    'weights': None,
    'out_hdf5': 'data.hdf5',
    'p_event_before': 10.,
    'p_event_after': 60. * 60 * 1,
    's_event_before': 60. * 10,
    's_event_after': 60. * 10,
    'default_event_before': 60. * 60 * 1,
    'default_event_after': 60. * 60 * 1,
    'model_labels': {'p': 0, 's': 1, 'n': 2},
    'positive_labels': {'p': 0, 's': 1}
}

# Only this params can be set via script arguments
param_aliases = {
    'config': ['--config', '-c'],
    'threshold': ['--threshold'],
    'mulplt': ['--mulplt'],
    'batch_size': ['--batch-size'],
    'start': ['--start', '-s'],
    'end': ['--end', '-e'],
    'slice_range': ['--slice_range', '--range'],
    'archive_path': ['--archive_path', '-a'],
    's_path': ['--s_path'],
    'seisan': ['--seisan'],
    'out': ['--out', '-o'],
    'debug': ['--debug', '-d'],
    'seismo': ['--seismo'],
    'favor': ['--favor'],
    'cnn': ['--cnn'],
    'weights': ['--weights', '-w']
}

# Help messages
param_help = {
    'config': 'Path to .ini config file',
    'threshold': 'Positive prediction threshold, default: 0.95',
    'mulplt': 'Path to MULPLT.DEF file',
    'batch_size': 'Batch size for sliding windows memory, default: 500 000 samples',
    'start': 'start date in ISO 8601 format:\n'
             '{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}\n'
             'or\n'
             '{year}-{month}-{day}T{hour}:{minute}:{second}\n'
             'or\n'
             '{year}-{month}-{day}\n'
             'default: beginning of this month',
    'end': 'end date in ISO 8601 format:\n'
           '{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}\n'
           'or\n'
           '{year}-{month}-{day}T{hour}:{minute}:{second}\n'
           'or\n'
           '{year}-{month}-{day}\n'
           'default: now',
    'slice_range': 'Slicing range in seconds before and after wave arrival',
    'archive_path': 'Path to Seisan archive directory',
    's_path': 'Path to s-files database directory (e.g. "/seismo/seisan/REA/IMGG_/")',
    'seisan': 'Path to SEISAN.DEF',
    'out': 'Output path, default: "wave_picks"',
    'debug': 'Enable debug info output',
    'seismo': 'Load default Seismo-Transformer',
    'favor': 'Load fast-attention Seismo-Transformer',
    'cnn': 'Load fast-attention Seismo-Transformer with CNN',
    'weights': 'Path to model weights file'
}

# Param actions
param_actions = {
    'debug': 'store_true',
    'seismo': 'store_true',
    'favor': 'store_true',
    'cnn': 'store_true'
}


if __name__ == '__main__':

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Seisan database waveform picker, creates ' +
                                                 'a h5 dataset with P, S and Noise waveforms stored. ' +
                                                 'Or append if dataset file already created.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    for k in param_aliases:

        if k in param_actions:
            parser.add_argument(*param_aliases[k], help=param_help[k], action = param_actions[k])
        else:
            parser.add_argument(*param_aliases[k], help=param_help[k])

    args = parser.parse_args()
    args = vars(args)

    params_set = []

    for k in args:

        if args[k] is not None:
            params[k] = args[k]
            params_set.append(k)

    params = parse_ini(params['config'], params_set, params=params)

    # Convert types
    params['threshold'] = float(params['threshold'])

    # Parse dates
    def parse_date_param(d_params, name):

        if name not in d_params:
            return None
        if d_params[name] is None:
            return None

        try:
            return UTCDateTime(d_params[name])
        except Exception:
            print(f'Failed to parse {name}, value: {d_params[name]}.')


    start_date = parse_date_param(params, 'start')
    end_date = parse_date_param(params, 'end')

    if end_date is None:
        end_date = UTCDateTime()
    if start_date is None:
        start_date = UTCDateTime() - 30 * 24 * 60 * 60

    # Parse MULPLT.DEF
    mulplt_parsed = st.parse_mulplt(params['mulplt'])

    # Parse stations
    stations = st.process_seisan_def_mulplt(params['seisan'], mulplt_parsed)
    stations = st.order_stations(stations, params['channel_order'])

    # Fix archive_path
    if params['archive_path'][-1] != '/':
        params['archive_path'] += '/'

    # Fix s-files dir path
    if params['s_path'][-1] != '/':
        params['s_path'] += '/'

    # Batch size fix
    params['batch_size'] = int(params['batch_size'])

    # Initialize output directory
    if not os.path.isdir(params['out']):

        if os.path.isfile(params['out']):
            print(f'--out: {params["out"]} is not a directory!')
            sys.exit(0)

        os.makedirs(params['out'])

    if params['out'][-1] != '/':
        params['out'] += '/'

    # Loading model
    from models import seismo_load

    if params['seismo']:
        print('Loading default Seismo-Transformer')
        # Check model weights
        if not params['weights']:
            if not default_model_weights['seismo']:
                raise AttributeError('No model weights provided!')
            else:
                params['weights'] = default_model_weights['seismo']

        model = seismo_load.load_transformer(params['weights'])
    elif params['favor']:
        print('Loading fast-attention Seismo-Transformer')
        # Check model weights
        if not params['weights']:
            if not default_model_weights['favor']:
                raise AttributeError('No model weights provided!')
            else:
                params['weights'] = default_model_weights['favor']

        model = seismo_load.load_favor(params['weights'])
    elif params['cnn']:
        print('Loading fast-attention Seismo-Transformer with CNN')
        # Check model weights
        if not params['weights']:
            if not default_model_weights['cnn']:
                raise AttributeError('No model weights provided!')
            else:
                params['weights'] = default_model_weights['cnn']

        model = seismo_load.load_cnn(params['weights'])
    else:
        raise AttributeError('No model specified!')

    # Prepare for archive files gathering
    current_dt = start_date

    current_end_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day, 23, 59, 59))
    if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
        current_end_dt = end_date

    if params['debug']:
        print(f'DEBUG: start = {start_date}',
              f'DEBUG: end = {end_date}', sep='\n')

    current_month = -1
    s_events = []
    true_positives = []

    while current_dt < end_date:

        if params['debug']:
            print(f'DEBUG: current_date = {current_dt} current_end_date: {current_end_dt}')

        # Parse all s_files once per month
        if current_dt.month != current_month:
            s_base_path = params['s_path']
            s_base_path += f'{current_dt.year:0>4d}/{current_dt.month:0>2d}/'
            current_month = current_dt.month
            s_events = st.parse_s_dir(s_base_path, stations, params)

            # Add time_span for every event
            true_positives = []
            for event_file in s_events:
                for event_group in event_file:
                    for event in event_group:

                        event_time = event['utc_datetime']

                        if not event_time:
                            continue

                        if event['phase'] == 'P':
                            event['time_span'] = (event_time - params['p_event_before'],
                                                  event_time + params['p_event_after'])
                        elif event['phase'] == 'S':
                            event['time_span'] = (event_time - params['s_event_before'],
                                                  event_time + params['s_event_after'])
                        else:
                            event['time_span'] = (event_time - params['default_event_before'],
                                                  event_time + params['default_event_after'])

                        true_positives.append(event['time_span'])

        if not s_events:
            s_events = []
        if not true_positives:
            true_positives = []

        # Filter out today's true_positives
        current_true_positives = []
        for span_start, span_end in true_positives:

            is_in = False
            if current_dt <= span_start and span_end <= current_end_dt:
                is_in = True
            elif span_start < current_dt < span_end:
                is_in = True
            elif span_start < current_end_dt < span_end:
                is_in = True

            if is_in:
                current_true_positives.append((span_start, span_end))

        # Predict
        # TODO: Read archives for every group, var name: stations (code for archive path get from model integration)
        # TODO: Then read archives, trim archives
        # TODO: Write function which takes time span and arrays of current_true_positives time spans and returns
        #           array of timespans to which i should cut the base timespan.

        for archive_list in stations:

            # Archives path and meta data
            archive_data = st.archive_to_path(archive_list, current_dt,
                                           params['archive_path'], params['channel_order'])

            # Check if streams path are valid
            from os.path import isfile
            valid = True
            for channel in params['channel_order']:
                if not isfile(archive_data[channel]):
                    valid = False
                    break

            if not valid:
                continue

            # Read data
            streams = {}
            try:
                for channel in params['channel_order']:
                    streams[channel] = read(archive_data[channel])
            except Exception:  # TODO: Replace with less general built-in exceptions
                continue

            streams = pt.preprocess_streams(streams, current_dt, current_end_dt, current_true_positives)  # preprocess data

            for stream_group in streams:

                group_scores = pt.predict_streams(model, stream_group, params = params)

                # FOR EVERY STREAM GROUP:
                # Check if stream traces number is equal
                # Per every trace:
                #   Count batches
                #   Loop for every batch, maybe, do this in separate function:
                #     Predict. Find peaks. Find positives. Save them into h5py.

        # Shift date
        current_dt += 24 * 60 * 60
        current_end_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day, 23, 59, 59))
        current_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day))

        if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
            current_end_dt = end_date

