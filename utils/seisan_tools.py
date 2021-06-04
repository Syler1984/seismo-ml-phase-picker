from obspy.core.utcdatetime import UTCDateTime
import re
import os
from .converter import date_str
from obspy import read


def process_stations_file(path):
    return None


def group_archives(archives):
    """
    Takes archive definitions list and groups them together by stations and channel types.
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


def filter_by_channel(archives, allowed_channels):
    """
    Filters out archive groups which are not in allowed_channels.
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


def process_seisan_def(path, allowed_channels):
    """
    Read SEISAN.DEF file and get list of all the stations with allowed channels.
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

    return grouped_records


def group_by(lst, column, comp_margin = None):
    """
    Groups list entities by column values.
    """
    sorted_values = []
    result_list = []
    current_value = None

    for i in range(0, len(lst)):
        x = lst[i]
        if x[column][0:comp_margin] in sorted_values or x[column][0:comp_margin] == current_value:
            continue

        current_value = x[column][0:comp_margin]
        current_list = []
        for j in range(i, len(lst)):
            y = lst[j]
            if y[column][0:comp_margin] != current_value:
                continue

            current_list.append(y)

        sorted_values.append(current_value)
        result_list.append(current_list)

    return result_list


def process_archives_list(lst):
    """
    Processes output of parse_seisan_def: combines into lists of three channeled entries.
    """
    lst = group_by(lst, 0)
    result = []
    for x in lst:
        channel_group = group_by(x, 1, 2)
        for y in channel_group:
            location_group = group_by(y, 3)

            for z in location_group:
                if len(z) == 3:
                    result.append(z)
    return result


def process_seisan_def_mulplt(path, mulplt_data = None, allowed_channels = None):
    """
    Parses seisan.def file and returns grouped lists like:
    [station, channel, network_code, location_code, archive start date, archive end date (or None)].
    """
    data = []

    if mulplt_data is not None:
        stations_channels = [x[0] + x[1] + x[2] for x in mulplt_data]

    with open(path, "r") as f:
        lines = f.readlines()
        tag = "ARC_CHAN"

        for line in lines:
            if line[:len(tag)] == tag:
                entry = line[len(tag):].split()
                station = line[40:45].strip()
                channel = line[45:48]
                code = line[48:50]
                location = line[50:52]
                start_date = entry[2] if len(entry) >= 3 else None
                end_date = entry[3] if len(entry) >= 4 else None

                if mulplt_data is not None:
                    if station + channel not in stations_channels:
                        continue

                parsed_line = [station, channel, code, location, start_date, end_date]

                if allowed_channels:

                    is_channel_allowed = False
                    for ch in allowed_channels:
                        if ch == channel[:len(ch)]:
                            is_channel_allowed = True

                    if is_channel_allowed:
                        data.append(parsed_line)

                else:
                    data.append(parsed_line)

    return process_archives_list(data)


def parse_s_file(path, params):
    """
    Parses s-file and returns all its events readings.
    :param path:
    :return:
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return
    except FileNotFoundError:
        return

    d_path = '02-0422-22D.S201601'
    h_path = path.split('/')[-1]

    if d_path == h_path:
        print(f'FOUND {path}')

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
    loc = head[21]

    if loc != 'L':
        print(f'In file "{path}": unsupported locale type "{loc}"! Skipping..')
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

        try:
            station = l[1:6].strip()
            instrument = l[6]
            channel = l[7]
            phase = l[10:14].strip()
            hour = int(l[18:20].strip())
            minute = int(l[20:22].strip())
            second = float(l[22:28].strip())
            distance = float(l[70:75].strip())
        except ValueError as e:
            continue

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

        if hour >= 24:
            continue

        utc_datetime = UTCDateTime(date_str(year, month, day, hour, minute, second))

        # Events filtering
        # TODO: Replace all min/max code for something better, that does full min and max support and checks for None values
        if params['min_magnitude'] and (not magnitude or magnitude < float(params['min_magnitude'])):
            if params['debug']:
                print(f'DEBUG: Skipping event in {path}. Reason: low magnitude ({magnitude}).')
            return


        if params['max_depth'] and (not depth or depth > float(params['max_depth'])):
            if params['debug']:
                print(f'DEBUG: Skipping event in {path}. Reason: high depth ({depth}).')
            return

        if params['max_distance'] and (not distance or distance > float(params['max_distance'])):
            if params['debug']:
                print(f'DEBUG: Skipping event in {path}. Reason: high distance ({distance}).')
            return

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

    if d_path == h_path:
        print('RETURN:')
        for i, s in enumerate(events):
            print(f'{i}: {s}')
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


def filter_events(events, stations, db = False):
    """
    Filters out phase lines with stations not defined in stations list. Also adds code and location to events.
    """
    stations_tags = []
    for s in stations:

        # Station tag = [station_name, instrument, code, location, algorithm]
        algorithm = '',
        if len(s[0][1]) == 3:
            algorithm = s[0][1][1]

        stations_tags.append([s[0][0], s[0][1][0], s[0][2], s[0][3], algorithm])

    if db:
        print('TAGS:')
        for x in stations_tags:
            print(x)
        print('')

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
                    e['algorithm'] = tag[4]

                break

    return filtered


def parse_s_dir(path, stations, params):
    """
    Scans path directory, parses all s-files and returns filtered events_list grouped by stations.
    :param path - path to the directory
    :param stations - grouped stations list to filter out all stations which are not in this list.
    """
    if path[-1] != '/':
        path += '/'

    try:
        files = os.listdir(path)
        files = [f'{path}{f}' for f in files]
    except FileNotFoundError:
        return

    all_events = []
    for f in files:

        events = parse_s_file(f, params)

        # TODO: remove this debug output
        d_path = '02-0422-22D.S201601'
        h_path = f.split('/')[-1]
        if d_path == h_path and events:
            print(f'\n\nFOUND LENGTH = {len(events)}')

        if not events:
            continue

        events = group_events(events)

        if d_path == h_path:
            print(f'GROUP LENGTH {len(events)}')

        if d_path == h_path:
            print('\nSTATIONS:')
            for x in stations:
                print(x)
            print('')

        if d_path == h_path:
            events = filter_events(events, stations, True)
        else:
            events = filter_events(events, stations)

        if d_path == h_path:
            print(f'FILTERED LENGTH {len(events)}')

        if len(events):
            all_events.append(events)

        if d_path == h_path:
            print(f'FINAL LENGTH: {len(events)}\n\n')

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
        try:
            st = read(a_f)
        except TypeError:
            print(f'TypeError while obspy.read file {a_f}; Skipping phase.')
            return None

        # TODO: Continuity check

        # Pre-process
        # TODO: Add some params check (e.g. highpass? Hz? Normalize? detrend?)
        st.detrend(type = "linear")
        try:
            st.filter(type = "highpass", freq = 2)
        except ValueError:
            print(f'ValueError while filtering archive {a_f}; Skipping phase.')
            return None

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

        print('DEBUG: TRACE: ', a_f)
        traces.append(st[0])

    if len(traces) != len(archives):
        return None

    return traces


def parse_mulplt(path):
    """
    Parses MULPLT.DEF file and returns list of lists like: [station, channel type (e.g. SH), channel (E, N or Z)].
    """
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        tag = "#DEFAULT CHANNEL"

        for line in lines:
            if line[:len(tag)] == tag:
                # entry[0] - station, ..[1] - type, ..[2] - channel
                entry = line[len(tag):].split()
                data.append(entry)
    return data


def order_stations(stations, order):

    ordered_list = []
    for station_group in stations:

        ordered_group = []
        for channel in order:
            for station in station_group:

                if station[1][-1] == channel:
                    ordered_group.append(station)

        if len(ordered_group) == len(order):
            ordered_list.append(ordered_group)

    return ordered_list
