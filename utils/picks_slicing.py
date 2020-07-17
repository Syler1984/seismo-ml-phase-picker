import os
import obspy.io.nordic.core as nordic_reader
from obspy.core import read
import logging
import utils.seisan_reader as seisan
import config.vars as config
from obspy.io.mseed import InternalMSEEDError
import matplotlib.pyplot as plt

def get_stations(nordic_file_names, output_level=0):
    """
    Get all stations from provided S-files
    :param nordic_file_names:   list    List of nordic file full names
    :param output_level:        int     0 - min output, 5 - max output, default - 0
    :return:
    """
    stations = []
    for file in nordic_file_names:
        new_stations = get_event_stations(file, output_level)

        if new_stations == -1:
            continue

        for x in new_stations:
            if x not in stations:
                stations.append(x)

    return sorted(stations)


def get_event_stations(reading_path, output_level=0):
    """
    Reads S-file and gets all stations from it
    :param reading_path:    string  path to REA database
    :param output_level:    int     0 - min output, 5 - max output, default - 0
    :return: 
    """
    if output_level >= 5:
        logging.info('Reading file: ' + reading_path)

    try:
        events = nordic_reader.read_nordic(reading_path, True)  # Events tuple: (event.Catalog, [waveforms file names])
    except nordic_reader.NordicParsingError as error:
        if output_level >= 2:
            logging.warning('In ' + reading_path + ': ' + str(error))
        return -1
    except ValueError as error:
        if output_level >= 2:
            logging.warning('In ' + reading_path + ': ' + str(error))
        return -1
    except AttributeError as error:
        if output_level >= 2:
            logging.warning('In ' + reading_path + ': ' + str(error))
        return -1

    stations = []
    for event in events[0].events:
        try:
            if len(event.picks) > 0:  # Only files with picks have stations data
                for pick in event.picks:
                    stations.append(pick.waveform_id.station_code)
        except ValueError as error:
            if output_level >= 2:
                logging.warning('In ' + reading_path + ': ' + str(error))
            continue

    return stations


def slice_from_reading(reading_path, waveforms_path, slice_duration=5, archive_definitions=[], output_level=0):
    """
    Reads S-file on reading_path and slice relevant waveforms in waveforms_path
    :param reading_path:        string    path to S-file
    :param waveforms_path:      string    path to folder with waveform files
    :param slice_duration:      int       duration of the slice in seconds
    :param archive_definitions: list      list of archive definition tuples (see utils/seisan_reader.py)
    :param output_level:        int       0 - min output, 5 - max output, default - 0
    :return: -1                                  -    corrupted file
             [(obspy.core.trace.Trace, string)]  -    list of slice tuples: (slice, name of waveform file)
    """
    if output_level >= 5:
        logging.info('Reading file: ' + reading_path)

    try:
        events = nordic_reader.read_nordic(reading_path, True)  # Events tuple: (event.Catalog, [waveforms file names])
    except nordic_reader.NordicParsingError as error:
        if output_level >= 2:
            logging.warning('In ' + reading_path + ': ' + str(error))
        return -1
    except ValueError as error:
        if output_level >= 2:
            logging.warning('In ' + reading_path + ': ' + str(error))
        return -1
    except AttributeError as error:
        if output_level >= 2:
            logging.warning('In ' + reading_path + ': ' + str(error))
        return -1

    index = -1
    slices = []
    picks_line = "STAT SP IPHASW"
    for event in events[0].events:
        index += 1

        f = open(reading_path)
        l = [line.strip() for line in f]

        id = None
        picks_started = False
        picks_amount = len(event.picks)
        picks_read = 0
        picks_distance = []
        for line in l:
            if picks_started and picks_read < picks_amount and len(line) >= 74:
                try:
                    dist = float(line[70:74])
                except ValueError as e:
                    dist = None
                picks_distance.append(dist)

            if len(line) > 73:
                title = line[0:6]
                if title == "ACTION":
                    id_title = line[56:59]
                    if id_title == "ID:":
                        id_str = line[59:73]
                        id = int(id_str)

            if len(line) > 25:
                if line[0:len(picks_line)] == picks_line:
                    picks_started = True

        # Min magnitude check
        if len(event.magnitudes) > 0:
            if event.magnitudes[0].mag < config.min_magnitude:
                continue

        # Max depth check
        if len(event.origins) > 0:
            if event.origins[0].depth is None:
                continue
            if event.origins[0].depth > config.max_depth:
                continue

        try:
            if len(event.picks) > 0:  # Only for files with picks
                if output_level >= 3:
                    logging.info('File: ' + reading_path + ' Event #' + str(index) + ' Picks: ' + str(len(event.picks)))

                picks_index = -1
                for pick in event.picks:
                    if output_level >= 3:
                        logging.info('\t' + str(pick))

                    picks_index += 1

                    if picks_index < len(picks_distance) and picks_distance[picks_index] is not None:
                        if picks_distance[picks_index] > config.max_dist:
                            continue

                    # Check phase
                    if pick.phase_hint != 'S' and pick.phase_hint != 'P':
                        logging.info('\t' + 'Neither P nor S phase. Skipping.')
                        continue

                    if output_level >= 3:
                        logging.info('\t' + 'Slices:')

                    # Checking archives
                    found_archive = False
                    if len(archive_definitions) > 0:
                        station = pick.waveform_id.station_code
                        station_archives = seisan.station_archives(archive_definitions, station)

                        channel_slices = []
                        for x in station_archives:
                            if x[4] <= pick.time:
                                if x[5] is not None and pick.time > x[5]:
                                    continue
                                else:
                                    archive_file_path = seisan.archive_path(x, pick.time.year, pick.time.julday,
                                                                            config.archives_path, output_level)

                                    if os.path.isfile(archive_file_path):
                                        arch_st = read(archive_file_path)

                                        #arch_st.normalize(global_max=config.global_max_normalizing)  # remove that
                                        #arch_st.filter("highpass", freq=config.highpass_filter_df)  # remove that
                                        # line later
                                        for trace in arch_st:
                                            if trace.stats.starttime > pick.time or pick.time + slice_duration >= trace.stats.endtime:
                                                logging.info('\t\tArchive ' + archive_file_path +
                                                             ' does not cover required slice interval')
                                                continue

                                            shifted_time = pick.time - config.static_slice_offset
                                            end_time = shifted_time + slice_duration

                                            found_archive = True

                                            trace_slice = trace.slice(shifted_time, end_time)
                                            if output_level >= 3:
                                                logging.info('\t\t' + str(trace_slice))

                                            trace_file = x[0] + str(x[4].year) + str(x[4].julday) + x[1] + x[2] + x[3]
                                            event_id = x[0] + str(x[4].year) + str(x[4].julday) + x[2] + x[3]
                                            slice_name_station_channel = (trace_slice, trace_file, x[0], x[1], event_id,
                                                                          pick.phase_hint, id_str)

                                            #print("ID " + str(id_str))
                                            #if id_str == '20140413140958':
                                                #print(x[0])
                                                #if True:#x[0] == 'NKL':
                                                    #trace.integrate()
                                                    #trace_slice.integrate()
                                                    #trace.normalize()
                                                    #trace_slice.normalize()
                                                    #print('FOUND ID! NORMALIZED')
                                                    #print('ARCHIVE: ' + archive_file_path)
                                                    #print('FILE: ' + trace_file)
                                                    #print('SLICE: ' + str(trace_slice))
                                                    #print('TIME: ' + str(shifted_time) + ' till ' + str(end_time))
                                                    #print('TRACE: ' + str(trace))
                                                    #print('DATA: ' + str(trace_slice.data))

                                                    #trace_slice.filter("highpass", freq=config.highpass_filter_df)
                                                    #patho = "/seismo/seisan/WOR/chernykh/plots/part/"
                                                    #patho2 = "/seismo/seisan/WOR/chernykh/plots/whole/"

                                                    #plt.plot(trace_slice.data)
                                                    #plt.ylabel('Amplitude')
                                                    #plt.savefig(patho + trace_file)
                                                    #plt.figure()

                                                    #plt.plot(trace.data)
                                                    #plt.ylabel('Amplitude')
                                                    #plt.savefig(patho2 + trace_file)
                                                    #plt.figure()

                                            if len(trace_slice.data) >= 400:
                                                channel_slices.append(slice_name_station_channel)

                    # Read and slice waveform
                    if found_archive:
                        if len(channel_slices) > 0:
                            slices.append(channel_slices)
                        continue

        except ValueError as error:
            if output_level >= 2:
                logging.warning('In ' + reading_path + ': ' + str(error))
            continue

    return sort_slices(slices)


def save_traces(traces, save_dir, file_format="MSEED"):
    """
    Saves trace/name tuples list to a file
    :param traces:      [(obspy.core.trace.Trace, string)]    list of slice tuples: (slice, name of waveform file)
    :param save_dir:    string                                save path
    :param file_format: string                                format of same wave file, default - miniSEED "MSEED"
    """
    for event in traces:
        if config.dir_per_event and len(event) > 0:
            base_dir_name = event[0][4]
            if len(event[0]) == 7 and event[0][6] is not None:
                base_dir_name = event[0][6]
            dir_name = base_dir_name
            index = 0
            while os.path.isdir(save_dir + '/' + dir_name):
                dir_name = base_dir_name + str(index)
                index += 1
            os.mkdir(save_dir + '/' + dir_name)
        for x in event:
            try:
                if config.dir_per_event:
                    file_name = x[1] + '.' + x[3] + '.' + x[5]
                    index = 0
                    while os.path.isfile(save_dir + '/' + dir_name + '/' + file_name):
                        file_name = x[1] + '.' + x[5] + '.' + str(index)
                        index += 1
                    x[0].write(save_dir + '/' + dir_name + '/' + file_name, format=file_format)
                else:
                    file_name = x[1] + '.' + x[5]
                    index = 0
                    while os.path.isfile(save_dir + '/' + file_name):
                        file_name = x[1] + '.' + x[5] + '.' + str(index)
                        index += 1

                    x[0].write(save_dir + '/' + file_name, format=file_format)
            except InternalMSEEDError:
                logging.warning(str(InternalMSEEDError))
            except OSError:
                logging.warning(str(OSError))


def get_picks_stations_data(path_array):
    data = []
    for x in path_array:
        stat_picks = get_single_picks_stations_data(x)
        if type(stat_picks) == list:
            data.extend(stat_picks)

    return data


def get_single_picks_stations_data(nordic_path):
    """
    Returns all picks for stations with corresponding pick time in format: [(UTC start time, Station name)]
    :param nordic_path: string  path to REA database
    :return:
    """
    try:
        events = nordic_reader.read_nordic(nordic_path, True)  # Events tuple: (event.Catalog, [waveforms file names])
    except nordic_reader.NordicParsingError as error:
        if config.output_level >= 2:
            logging.warning('In ' + nordic_path + ': ' + str(error))
        return -1
    except ValueError as error:
        if config.output_level >= 2:
            logging.warning('In ' + nordic_path + ': ' + str(error))
        return -1
    except AttributeError as error:
        if config.output_level >= 2:
            logging.warning('In ' + nordic_path + ': ' + str(error))
        return -1

    index = -1
    slices = []
    for event in events[0].events:
        index += 1

        try:
            if len(event.picks) > 0:  # Only for files with picks
                for pick in event.picks:
                    slice_station = (pick.time, pick.waveform_id.station_code)
                    slices.append(slice_station)

        except ValueError as error:
            if config.output_level >= 2:
                logging.warning('In ' + nordic_path + ': ' + str(error))
            continue

    return slices


def sort_slices(slices):
    """
    Sorts slices by station and then by channel (but it removes all non-unique station, channel pairs)
    :param slices: slices in format: [[trace, filename, station, channel], ...]
    :return: Sorted slices in the same format: [[trace, filename, station, channel], ...]
    """
    result = []
    for x in slices:
        sorted = []
        semi_sorted = []
        # Sort by stations
        x.sort(key=lambda y: y[2])

        # Sort by channels
        found_channels = []
        current_station = x[0][2]
        for y in x:
            if current_station != y[2]:
                current_station = y[2]
                found_channels = []
            if y[3][-1] in found_channels:
                continue
            if y[3][-1] in config.archive_channels_order:
                found_channels.append(y[3][-1])
                semi_sorted.append(y)

        current_station = ""
        index = 0
        for y in semi_sorted:
            if y[2] != current_station:
                current_station = y[2]
                for channel in config.archive_channels_order:
                    sorting_index = index
                    while sorting_index < len(semi_sorted) and semi_sorted[sorting_index][2] == current_station:
                        if semi_sorted[sorting_index][3][-1] == channel:
                            sorted.append(semi_sorted[sorting_index])
                            break
                        sorting_index += 1
            index += 1

        result.append(sorted)

    return result
