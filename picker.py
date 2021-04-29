import argparse
import os
import re
from obspy.core.utcdatetime import UTCDateTime


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
    for l in events_table:

        if not len(l.strip()):
            continue

        station = l[1:6]
        instrument = l[6]
        channel = l[7]
        phase = l[10:14]
        hour = int(l[18:20])
        minute = int(l[20:22])
        second = float(l[22:25])
        distance = float(l[70:75])

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
                       'id': event_id})

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
    Filters out phase lines with stations not defined in stations list.
    """
    stations_tags = []
    for s in stations:

        # Station tag = [station_name, instrument]
        stations_tags.append([s[0][0], s[0][1][0]])

    filtered = []
    for group in events:
        
        event = group[0]
        for tag in stations_tags:
        
            if event['station'] == tag[0] and event['instrument'] == tag[1]:
                filtered.append(group)
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


if __name__ == '__main__':

    # Default params
    day_length = 60. * 60 * 24
    params = {'config': 'config.ini',
              'start': None,
              'end': None,
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
              'allowed_channels': [['SHE', 'SHN', 'SHZ']],
              'out': 'wave_picks',
              'debug': 0}

    # Only this params can be set via script arguments
    param_aliases = {'config': ['--config', '-c'],
                     'start': ['--start', '-s'],
                     'end': ['--end', '-e'],
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

    # TODO: initialize output dir and all subdirs.

    # TODO: go through every day and get all picks/dates
    current_dt = start_date

    current_end_dt = None
    if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
        current_end_dt = end_date

    if params['debug']:
        print(f'DEBUG: start = {start_date}',
              f'DEBUG: end = {end_date}', sep='\n')

    current_month = -1
    s_events = []

    while current_dt < end_date:

        if params['debug']:
            print(f'DEBUG: current_date = {current_dt}')

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
                    
                    print(event)
                    

        # Shift date
        current_dt += 24 * 60 * 60
        current_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day))

        current_end_dt = None
        if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
            current_end_dt = end_date
