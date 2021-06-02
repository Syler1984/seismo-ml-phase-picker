import argparse
import sys
from obspy.core.utcdatetime import UTCDateTime

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.ini_tools import parse_ini
from utils.seisan_tools import process_seisan_def, process_stations_file

# Default params
default_model_weights = {
    'seismo': 'weights/seismo_sakh_2014_2019.h5',
    'favor': None,
    'cnn': None
}

day_length = 60. * 60 * 24

params = {
    'config': 'config.ini',
    'start': None,
    'end': None,
    'slice_range': 2.,  # seconds before and after event
    'archive_path': None,
    's_path': None,
    'seisan_def': None,
    'stations': None,
    'allowed_channels': [
        ['SHN', 'SHE', 'SHZ'],
        ['BHN', 'BHE', 'BHZ'],
    ],
    'frequency': 100.,
    'out': 'wave_picks',
    'debug': False,
    'seismo': False,
    'favor': False,
    'cnn': False,
    'weights': None,
    'out_hdf5': 'data.hdf5'
}

# Only this params can be set via script arguments
param_aliases = {
    'config': ['--config', '-c'],
    'start': ['--start', '-s'],
    'end': ['--end', '-e'],
    'slice_range': ['--slice_range', '--range'],
    'archive_path': ['--archive_path', '-a'],
    's_path': ['--s_path'],
    'seisan_def': ['--seisan_def'],
    'stations': ['--stations'],
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
    'seisan_def': 'Path to SEISAN.DEF',
    'stations': 'Path to stations file',
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

    # Parse stations
    stations = None
    if params['stations']:
        stations = process_stations_file(params['stations'])

    if not stations:
        stations = process_seisan_def(params['seisan_def'], params['allowed_channels'])

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
        if not params['weights'] and not default_model_weights['seismo']:
            raise AttributeError('No model weights provided!')

        model = seismo_load.load_transformer(params['weights'])
    elif params['favor']:
        print('Loading fast-attention Seismo-Transformer')
        # Check model weights
        if not params['weights'] and not default_model_weights['seismo']:
            raise AttributeError('No model weights provided!')

        model = seismo_load.load_favor(params['weights'])
    elif params['cnn']:
        print('Loading fast-attention Seismo-Transformer with CNN')
        # Check model weights
        if not params['weights'] and not default_model_weights['seismo']:
            raise AttributeError('No model weights provided!')

        model = seismo_load.load_cnn(params['weights'])
    else:
        raise AttributeError('No model specified!')

    model.summary()
