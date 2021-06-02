import argparse
import os
import sys
from obspy.core.utcdatetime import UTCDateTime
import numpy as np

from utils.ini_tools import parse_ini
from utils.seisan_tools import process_seisan_def, process_stations_file, parse_s_dir, slice_archives
from utils.converter import date_str
from utils.h5_tools import write_batch


# TODO: This script should:
#           get all data from either config/vars.py
#           or through another data source.
#           should also initialize by default alot of it's parameters.
#       Then is should get all stations from SEISAN.DEF with suitable channels.
#       Go through all s-files in specified time span (just look at integration as inspiration source).
#       Return all archives names and slice center timestamps.
#           Decide what to do with them later:
#               either do a directory archive or actually save them in some other format.
#               What i want to save:
#                   Waveforms, labels, event-ids, s-file path, starttime/endtime, exact event timestamp,
#                   frequency, magnitude, depth, distance (if it has any other labels in it (like P + S)).
#               Also i want to save a config file to have a reference to how to reproduce the archive or just
#                   refresh my memory on the old archive.
#       Maybe have some csv file with some data or can we just save it into the hdf5?
#       Also how about zip or tar.gz this whole archive after we done?
#       Also save some data for a currently processed date..
#       VERY IMPORTANT: when processing an s-file: group processing by stations (and channel/device type).
#           Then, when dealing with archive, check if some picks will be in the same file, like P and S together.
#           Then save them like a single file.
#       Then in following .csv file or whatever just use same path to two different events_instances
#           .csv should be again done for events and grouped by files!

# TODO: sort all code into different modules?
# TODO: Maybe write my own little rep for .ini parsing


if __name__ == '__main__':

    # Default params
    day_length = 60. * 60 * 24
    params = {'config': 'config.ini',
              'start': None,
              'end': None,
              'slice_range': 2.,  # seconds before and after event
              'min_magnitude': None,
              'max_magnitude': None,
              'min_depth': None,
              'max_depth': None,
              'min_distance': None,
              'max_distance': None,
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
              'out_hdf5': 'data.hdf5',
              'out_csv': 'data.csv',
              'phase_labels': {'P': 0, 'S': 1, 'N': 2},
              'noise_picking': True,
              'noise_picker_phase': 'P',
              'noise_phase_before': 4.}

    # Only this params can be set via script arguments
    param_aliases = {'config': ['--config', '-c'],
                     'start': ['--start', '-s'],
                     'end': ['--end', '-e'],
                     'slice_range': ['--slice_range', '--range'],
                     'min_magnitude': ['--min_magnitude'],
                     'max_magnitude': ['--max_magnitude'],
                     'min_depth': ['--min_depth'],
                     'max_depth': ['--max_depth'],
                     'min_distance': ['--min_distance'],
                     'max_distance': ['--max_distance'],
                     'archive_path': ['--archive_path', '-a'],
                     's_path': ['--s_path'],
                     'seisan_def': ['--seisan_def'],
                     'stations': ['--stations'],
                     'out': ['--out', '-o'],
                     'debug': ['--debug', '-d']}

    # Help messages
    param_help = {'config': 'Path to .ini config file',
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
                  'min_magnitude': 'Minimal event magnitude allowed',
                  'max_magnitude': 'Maximal event magnitude allowed',
                  'min_depth': 'Minimal event depth allowed',
                  'max_depth': 'Maximal event depth allowed',
                  'min_distance': 'Minimal reading distance to the epicenter allowed',
                  'max_distance': 'Maximal reading distance to the epicenter allowed',
                  'archive_path': 'Path to Seisan archive directory',
                  's_path': 'Path to s-files database directory (e.g. "/seismo/seisan/REA/IMGG_/")',
                  'seisan_def': 'Path to SEISAN.DEF',
                  'stations': 'Path to stations file',
                  'out': 'Output path, default: "wave_picks"',
                  'debug': 'Enable debug info output? 1 - enable, default: 0'}

    # Param actions
    param_actions = {'debug': 'store_true'}

    # TODO: also implement params defaults and types.
    # TODO: add parameter for stations file

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

    # Initialize .csv and .hdf5 output files

    # TODO: go through every day and get all picks/dates
    current_dt = start_date

    current_end_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day, 23, 59, 59))
    if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
        current_end_dt = end_date

    if params['debug']:
        print(f'DEBUG: start = {start_date}',
              f'DEBUG: end = {end_date}', sep='\n')

    current_month = -1
    s_events = []

    # Fix archive_path
    if params['archive_path'][-1] != '/':
        params['archive_path'] += '/'

    while current_dt < end_date:

        if params['debug']:
            print(f'DEBUG: current_date = {current_dt} current_end_date: {current_end_dt}')

        # Get s-files dir path
        s_base_path = params['s_path']
        if s_base_path[-1] != '/':
            s_base_path += '/'

        s_base_path += f'{current_dt.year:0>4d}/{current_dt.month:0>2d}/'

        # Parse all s_files once per month
        if current_dt.month != current_month:
            current_month = current_dt.month
            s_events = parse_s_dir(s_base_path, stations, params)

        if not s_events:
            s_events = []

        # Go through every event and parse everything which happend today
        for event_file in s_events:
            for event_group in event_file:
                for event in event_group:

                    # Compare dates

                    # Get path to archive:
                    # archives_path/code/station/station.code.location.channel.year.julday
                    # where channel = instrument + algorithm + actuall channel (last letter in allowed_channels)
                    base_path = f'{params["archive_path"]}{event["code"]}/{event["station"]}/'

                    # Get relevant channels
                    ch_tag = f'{event["instrument"]}{event["algorithm"]}'
                    channels = None
                    for ch_group in params['allowed_channels']:

                        if ch_tag == ch_group[0][:2]:
                            channels = ch_group
                            break

                    if not channels:
                        if params['debug']:
                            print(f'DEBUG: Skipping event: {event["s_path"]} - channels not allowed!')
                        continue

                    # Get archive files paths
                    path_template = f'{event["station"]}.{event["code"]}.{event["location"]}.' + \
                                    r'{channel}.' + \
                                    f'{event["year"]}.{event["utc_datetime"].julday:0>3}'

                    archive_files = [base_path + path_template.format(channel = ch) for ch in channels]

                    # Get slice range
                    slice_start = event['utc_datetime'] - params['slice_range']
                    slice_end = event['utc_datetime'] + params['slice_range']

                    # Check if dates are valid
                    if slice_start < current_dt or slice_end < current_dt:
                        continue
                    if slice_start > current_end_dt or slice_end > current_end_dt:
                        continue

                    # TODO: make support for multi-day slices. For now, just skip them.
                    if slice_start.julday != slice_end.julday:
                        continue

                    # TODO: go though every trace in a stream
                    #  and check that slice_start and slice_end are in one discontinued trace
                    traces = slice_archives(archive_files, slice_start, slice_end, params['frequency'])

                    if not traces:
                        if params['debug']:
                            print('DEBUG: Skipping, no traces sliced!')
                        continue

                    # Slice noise?
                    noise_phase = None
                    if params['noise_picking']:

                        phase = event['phase'].strip()

                        if phase == params['noise_picker_phase']:

                            # Get slice range
                            slice_start = event['utc_datetime'] - params['noise_phase_before'] - params['slice_range']
                            slice_end = event['utc_datetime'] - params['noise_phase_before'] + params['slice_range']

                            # Check if dates are valid
                            if slice_start < current_dt or slice_end < current_dt:
                                continue
                            if slice_start > current_end_dt or slice_end > current_end_dt:
                                continue

                            # TODO: make support for multi-day slices. For now, just skip them.
                            if slice_start.julday != slice_end.julday:
                                continue

                            # TODO: go though every trace in a stream
                            #  and check that slice_start and slice_end are in one discontinued trace
                            noise_phase = slice_archives(archive_files, slice_start, slice_end, params['frequency'])

                    # Convert to NumPy arrays
                    trace_length = int(params['frequency'] * params['slice_range'] * 2)
                    ch_num = len(traces)
                    X = np.zeros((trace_length, ch_num))
                    Y = np.zeros(1)

                    skip = False
                    for i, tr in enumerate(traces):

                        if tr.data.shape[0] < trace_length:
                            skip = True
                            break
                        X[:, i] = tr.data[:trace_length]

                    if skip:
                        continue

                    if event['phase'].strip() in params['phase_labels']:
                        phase_code = params['phase_labels'][event['phase'].strip()]
                    else:
                        continue

                    Y[0] = phase_code

                    np_id = [f'{event["id"]}_l{event["line_number"]}_s{event["station"]}']
                    np_id = np.array(np_id, dtype = object)

                    X_shape = [1, 0, 0]
                    X_shape[1:] = list(X.shape)
                    X_shape = tuple(X_shape)

                    # Normalize
                    X_max = np.max(np.abs(X))
                    X = X / X_max

                    # Reshape to (1, -1, -1)
                    X = X.reshape(X_shape)

                    # Prepare noise
                    if noise_phase:

                        X_noise = np.zeros((trace_length, ch_num))
                        Y_noise = np.zeros(1)

                        for i, tr in enumerate(noise_phase):
                            X_noise[:, i] = tr.data[:trace_length]

                        Y_noise[0] = params['phase_labels']['N']

                        np_noise_id = np.array(['NOISE'], dtype = object)

                        X_max = np.max(np.abs(X_noise))
                        X_noise = X_noise / X_max

                        X_noise = X_noise.reshape(X_shape)

                    # Write to hdf5
                    write_batch(params['out_hdf5'], 'X', X)
                    write_batch(params['out_hdf5'], 'Y', Y)
                    write_batch(params['out_hdf5'], 'ID', np_id, string = True)

                    if noise_phase:

                        write_batch(params['out_hdf5'], 'X', X_noise)
                        write_batch(params['out_hdf5'], 'Y', Y_noise)
                        write_batch(params['out_hdf5'], 'ID', np_noise_id, string = True)

                    if params['debug']:
                        print(f'DEBUG: HDF5 data saved (LINE {event["line_number"]}) EVENT DT: {event["utc_datetime"]}')
                        # print('DEBUG: archive files: ', archive_files)

                    # Cut traces and save them into hdf5 (buffered), also save metadata, generate unique id.
                    # i suggest: event_id + station + random 4-digit number

                    # Save trace data
                    # X set - data
                    # Y set - label 
                    # ID set - ID string
                    # LINE set - number of the line in corresponding S file, use event['line_number']
                    # TODO: add phase_labels param
                    #  and check, if phase is not in there, then do not pick
                    # TODO: add gather_noise and noise_picker_phase (default P), and picks noise before this phase
                    #  also create function which gets archive_files list, start time, end time and returns slices
                    # TODO: save absolute line number for every event

                    # Save trace_meta_data

                    # print('EVENT: ', event)
                    # print('FILES: ', archive_files)

        # Shift date
        current_dt += 24 * 60 * 60
        current_end_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day, 23, 59, 59))
        current_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day))

        if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
            current_end_dt = end_date
