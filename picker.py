import argparse


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

        if not check_channel(channel, allowed_channels):
            continue

        records.append([station, channel, code, location, start_date, end_date])

    grouped_records = group_archives(records)
    grouped_records = filter_by_channel(grouped_records, allowed_channels)


def filter_by_channel(archives, allowed_channels):
    """
    Filters out archive groups which are not in allowed_channels.
    :param archives:
    :param allowed_channels:
    :return:
    """

    lens = [] # Get all unique allowed channels length
    for l in [len(x) for x in allowed_channels]:
        if l not in lens:
            lens.append(l)

    result_archives = []
    for group in archives:

        if len(group) not in lens:
            continue


    # TODO: compare length to group length
    # TODO: compare channels


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


def check_channel(channel, allowed_channels):
    """
    Returns True if channel is in one of the allowed_channels list.
    :param channel:
    :param allowed_channels:
    :return:
    """

    for ch_group in allowed_channels:

        if channel in ch_group:
            return True

    return False


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
              'out': 'wave_picks'}

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
                     'out': ['--out', '-o']}

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
                  'out': 'Output path, default: "wave_picks"'}

    # TODO: also implement params defaults and types.
    # TODO: add parameter for stations file

    # Setup argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    for k in param_aliases:
        parser.add_argument(*param_aliases[k], help=param_help[k])

    args = parser.parse_args()
    args = vars(args)

    if 'stations' in args:
        stations = process_stations_file(args['stations'])

    if not stations:
        stations = process_seisan_def(args['seisan_def'])

    # TODO: initialize output dir and all subdirs.

    # TODO: read stations if specified, create and save new one, if don't

    # TODO: go through every day and get all picks/dates
