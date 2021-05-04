import argparse
import os
import sys
import re
from obspy.core.utcdatetime import UTCDateTime
import obspy.core as oc
from obspy import read
import h5py as h5
import pandas as pd
import numpy as np

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

def remove_chars(line, chars=' \t', quotes='\'\"', comments=None):
    """
    Removes all specified characters but leaves quotes intact. Removes comments if comment character is specified.
    """
    new_line = ''
    quote_stack = ''
    remove_comments = (type(comments) is list) or (type(comments) is str)

    for c in line:

        if remove_comments and len(quote_stack) == 0 and c in comments:
            break

        if len(quote_stack) == 0 and c in chars:
            continue

        if c in quotes:
            if len(quote_stack) == 0 or c != quote_stack[-1]:
                quote_stack += c
            elif len(quote_stack) != 0:
                quote_stack = quote_stack[:-1]

            continue

        new_line += c

    return new_line


def parse_value(key, value):
    """
    Parses .ini value and returns tuple of value and type.
    """
    # Parse dictionary
    if value[0] == '{':

        value = value[1:-1]
        v_split = value.split(',')

        d = {}
        for x in v_split:
            x_split = x.split(':')
            d[x_split[0]] = x_split[1]

        return key, d, 'dictionary'

    # Parse single variable or list
    typ = 'var'
    if key[-2:] == '[]':
        typ = 'list'
        key = key[:-2]

    split = value.split(',')

    if len(split) == 1:
        return key, value, typ
    else:
        return key, split, 'list'


def parse_line(line):
    """
    Parses single .ini file line. Returns tuple of key, value and type of param.
    """
    # Trim line
    line = line.strip(' \t\n')

    # Check if empty
    if len(line) == 0:
        return None, None, None

    # Check if comment
    if line[0] in '#;':
        return None, None, 'comment'

    # Check if section: and ignore it
    if line[0] == '[':
        return None, None, 'section'

    # Remove all whitespaces unless in quotes and remove inline comments
    line = remove_chars(line, comments=';#')

    # Get key
    split = line.split('=')
    if len(split) < 2:
        return None, None, None

    key = split[0]
    val = line[len(key) + 1:]

    # Check value type
    key, val, typ = parse_value(key, val)

    return key, val, typ


def parse_ini(filename, params_set=None, params=None):
    """
    Parses .ini file.
    """
    var_dictionary = {}
    if params:
        var_dictionary = params

    with open(filename, 'r') as f:

        for line in f:

            key, value, typ = parse_line(line)

            if not typ or typ == 'comment' or typ == 'section':
                continue

            if key in params_set:
                continue

            if typ is not None:

                if typ == 'var':
                    var_dictionary[key] = value

                if typ == 'list':

                    if type(value) is not list:
                        value = [value]

                    if key in var_dictionary:
                        var_dictionary[key] += value
                    else:
                        var_dictionary[key] = value

                if typ == 'dictionary':
                    var_dictionary[key] = value

    return var_dictionary


def process_stations_file(path):
    return None


def process_seisan_def(path, allowed_channels):
    """
    Read SEISAN.DEF file and get list of all the stations with allowed channels.
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    records = []
    tag = 'ARC_CHAN'
    for line in lines:

        if line[:len(tag)] != tag:
            continue

        entry = line[len(tag):].split()
        station = line[40:45].strip()
        channel = line[45:48]
        code = line[48:50]
        location = line[50:52]
        start_date = entry[2] if len(entry) >= 3 else None
        end_date = entry[3] if len(entry) >= 4 else None

        records.append([station, channel, code, location, start_date, end_date])

    grouped_records = group_archives(records)

    grouped_records = filter_by_channel(grouped_records, allowed_channels)

    # TODO: oreder channels?

    return grouped_records


def filter_by_channel(archives, allowed_channels):
    """
    Filters out archive groups which are not in allowed_channels.
    :param archives:
    :param allowed_channels:
    :return:
    """

    # Collections compare function
    import collections
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    lens = []  # Get all unique allowed channels length
    for l in [len(x) for x in allowed_channels]:
        if l not in lens:
            lens.append(l)

    result_archives = []
    for group in archives:

        if len(group) not in lens:
            continue

        gr_channels = [x[1] for x in group]
        is_present = False
        for ch in allowed_channels:

            if compare(gr_channels, ch):
                is_present = True
                break

        if is_present:
            result_archives.append(group)

    return result_archives


def group_archives(archives):
    """
    Takes archive definitions list and groups them together by stations and channel types.
    :param archives:
    :return:
    """
    grouped = []
    grouped_ids = []
    for i in range(len(archives)):

        if i in grouped_ids:
            continue

        current_group = [archives[i]]
        group_tag = archives[i][0] + archives[i][1][:2] + archives[i][2] + archives[i][3]

        for j in range(i + 1, len(archives)):

            if j in grouped_ids:
                continue

            current_tag = archives[j][0] + archives[j][1][:2] + archives[j][2] + archives[j][3]

            if current_tag == group_tag:
                grouped_ids.append(j)
                current_group.append(archives[j])

        grouped_ids.append(i)
        grouped.append(current_group)

    return grouped


def date_str(year, month, day, hour=0, minute=0, second=0., microsecond=None):
    """
    Creates an ISO 8601 string.
    """
    # Get microsecond if not provided
    if microsecond is None:
        if type(second) is float:
            microsecond = int((second - int(second)) * 1000000)
        else:
            microsecond = 0

    # Convert types
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    microsecond = int(microsecond)

    # ISO 8601 template
    tmp = '{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}'

    return tmp.format(year=year, month=month, day=day,
                      hour=hour, minute=minute, second=second, microsecond=microsecond)


def parse_s_file(path):
    """
    Parses s-file and returns all its events readings.
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    head = lines[0]

    # Find events table
    table_head = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO AIN AR TRES W  DIS CAZ7'

    for i, l in enumerate(lines):

        if l[:len(table_head)] == table_head:
            events_table = lines[i + 1:]
            events_table_line_num = i + 1 + 1  # + 1 - because number should start with 1

    # Parse head
    magnitude = head[55:59].strip()
    if len(magnitude):
        magnitude = float(magnitude)
    else:
        magnitude = None

    magnitude_type = head[59]

    if magnitude_type != 'L':
        print(f'In file "{path}": unsupported magnitude type "{magnitude_type}"! Skipping..')
        return

    depth = head[38:43].strip()  # in Km
    if len(depth):
        depth = float(depth)
    else:
        depth = None
    
    # Parse ID
    event_id = None
    q_id = re.compile(r'\bID:')
    for l in lines:

        f_iter = q_id.finditer(l)

        found = False
        for match in f_iter:
            span = match.span()

            if span == (57, 60):
                found = True
                break

        if not found:
            continue
        
        event_id = l[span[1] : span[1] + 14]
 
        break

    year = None
    month = None
    day = None
    if event_id:
        year = int(event_id[:4])
        month = int(event_id[4:6])
        day = int(event_id[6:8])

    # Parse events
    events = []
    for i, l in enumerate(events_table):

        if not len(l.strip()):
            continue

        station = l[1:6]
        instrument = l[6]
        channel = l[7]
        phase = l[10:14]
        hour = int(l[18:20])
        minute = int(l[20:22])
        second = float(l[22:28])
        distance = float(l[70:75])

        if second >= 60.:
            
            minute_add = second // 60
            second = (second % 60)

            minute += minute_add
            minute = int(minute)

        if minute >= 60:

            if hour != 23:
                minute = 0
                hour += 1
            else:
                minute = 59
       
            minute = int(minute)
            hour = int(hour)

        utc_datetime = UTCDateTime(date_str(year, month, day, hour, minute, second))
        
        events.append({'station': station,
                       'instrument': instrument,
                       'channel': channel,
                       'phase': phase,
                       'year': year,
                       'month': month,
                       'day': day,
                       'hour': hour,
                       'minute': minute,
                       'second': second,
                       'distance': distance,
                       'magnitude': magnitude,
                       'depth': depth,
                       's_path': path,
                       'utc_datetime': utc_datetime,
                       'id': event_id,
                       'line_number': events_table_line_num + i})

    return events


def group_events(events):
    """
    Groups events by ID, station, instrument and channel code.
    """
    
    grouped = []
    grouped_ids = []

    for i, e in enumerate(events):

        if i in grouped_ids:
            continue

        group_tag = e['id'] + e['station'] + e['instrument']
        current_group = [e]
        grouped_ids.append(i)
        
        for j in range(i + 1, len(events)):

            if j in grouped_ids:
                continue
            
            e2 = events[j]
            e2_tag = e2['id'] + e2['station'] + e2['instrument']
            if e2_tag == group_tag:
                grouped_ids.append(j)
                current_group.append(e2)

        grouped.append(current_group)

    return grouped


def filter_events(events, stations):
    """ 
    Filters out phase lines with stations not defined in stations list. Also adds code and location to events.
    """
    stations_tags = []
    for s in stations:

        # Station tag = [station_name, instrument, code, location, algorythm]
        algorythm = '',
        if len(s[0][1]) == 3:
            algorythm = s[0][1][1]
        
        stations_tags.append([s[0][0], s[0][1][0], s[0][2], s[0][3], algorythm])

    filtered = []
    for group in events:
        
        event = group[0]
        for tag in stations_tags:
        
            if event['station'] == tag[0] and event['instrument'] == tag[1]:
                filtered.append(group)

                # Add code and location
                for e in group:
                    e['code'] = tag[2]
                    e['location'] = tag[3]
                    e['algorythm'] = tag[4]

                break

    return filtered


def parse_s_dir(path, stations):
    """
    Scans path directory, parses all s-files and returns filtered events_list grouped by stations.
    :param path - path to the directory
    :param stations - grouped stations list to filter out all stations which are not in this list.
    """
    if path[-1] != '/':
        path += '/'

    files = os.listdir(path)
    files = [f'{path}{f}' for f in files]
    
    all_events = []
    for f in files:

        events = parse_s_file(f)

        if not events:
            continue

        events = group_events(events)
        events = filter_events(events, stations)
        
        if len(events):
            all_events.append(events)

    return all_events


def slice_archives(archives, start, end, frequency):
    """
    This function reads archives from provided list of filenames and slices them.
    :param archives: List of full archives file names.
    :param start: obspy UTCDateTime slice start time.
    :param end: obspy UTCDateTime slice end time.
    :return: List of obspy Trace objects.
    """
    traces = []
    for a_f in archives:
        # Check if archive files exist
        if not os.path.isfile(a_f):
            print(f'Archive file does not exist {a_f}; Skipping phase.')
            return None

        # Load archive
        st = read(a_f)

        # TODO: Continuity check
        
        # Pre-process
        # TODO: Add some params check (e.g. highpass? Hz? Normalize? detrend?)
        st.detrend(type = "linear")
        st.filter(type = "highpass", freq = 2)

        # Convert frequency
        required_dt = 1. / frequency
        dt = st[0].stats.delta
        if dt != required_dt:
            st.interpolate(frequency)

        # Slice
        st = st.slice(start, end)

        # If there are not exactly 1 trace, something is wrong
        if len(st) != 1:
            return None

        traces.append(st[0])

    if len(traces) != len(archives):
        return None

    return traces


def write_batch(path, dataset, batch, string = False):
    """
    Writes batch to h5 file
    :param path: Path to h5 file
    :param dataset: Name of the dataset
    :param batch: Data
    :param string: True - if data should be VL string
    """
    with h5.File(path, 'a') as file:

        first = True
        if dataset in file.keys():
            first = False

        if not first:
            file[dataset].resize((file[dataset].shape[0] + batch.shape[0]), axis = 0)
            file[dataset][-batch.shape[0]:] = batch
        else:
            if not string:
                maxshape = list(batch.shape)
                maxshape[0] = None
                maxshape = tuple(maxshape)
                file.create_dataset(dataset, data = batch, maxshape = maxshape, chunks = True)
            else:
                dt = h5.string_dtype(encoding='utf-8')
                file.create_dataset(dataset, data = batch, maxshape = (None,), chunks = True, dtype = dt)


if __name__ == '__main__':

    # Default params
    day_length = 60. * 60 * 24
    params = {'config': 'config.ini',
              'start': None,
              'end': None,
              'slice_range': 2., # seconds before and after event
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
              'allowed_channels': [['SHZ', 'SHN', 'SHE']],
              'frequency': 100.,
              'out': 'wave_picks',
              'debug': 0,
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

    # TODO: also implement params defaults and types.
    # TODO: add parameter for stations file

    # Setup argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    for k in param_aliases:
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
    # stations: GROUPS of ['VAL', 'SHZ', 'IM', '00', '20141114', '20170530']

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
            s_events = parse_s_dir(s_base_path, stations)

        # Go through every event and parse everything which happend today
        for event_file in s_events:
            for event_group in event_file:
                for event in event_group:
                   
                    # Compare dates

                    # Get path to archive:
                    # archives_path/code/station/station.code.location.channel.year.julday
                    # where channel = instrument + algorythm + actuall channel (last letter in allowed_channels)
                    base_path = f'{params["archive_path"]}{event["code"]}/{event["station"]}/'
                    
                    # Get relevant channels
                    ch_tag = f'{event["instrument"]}{event["algorythm"]}'
                    channels = None
                    for ch_group in params['allowed_channels']:
                        
                        if ch_tag == ch_group[0][:2]:
                            channels = ch_group
                            break

                    if not channels:
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

                    for i, tr in enumerate(traces):

                        X[:, i] = tr.data[:trace_length]

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
                        print('DEBUG: archive files: ', archive_files)
                    

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
