import os
import sys
import obspy.io.nordic.core as nordic_reader
from obspy.core import read
from obspy.core import utcdatetime
import logging


def utcdatetime_from_string(string):
    """
    Converts string of formats: YYYYMMDD, YYYYMMDDHHmm, YYYYMMDDHHmmss to obspy.core.utcdatetime.UTCDateTime
    :param string: string   datetime string of either formats: YYYYMMDD, YYYYMMDDHHmm, YYYYMMDDHHmmss
    :return:       obspy.core.utcdatetime.UTCDateTime
                   None     - conversion failed
    """
    if len(string) in [8, 14]:
        return utcdatetime.UTCDateTime(string)
    elif len(string) == 12:
        new_string = string + "00"
        return utcdatetime.UTCDateTime(new_string)
    return None


def utcdatetime_from_tuple(date):
    """
    Creates obspy.core.utcdatetime.UTCDateTime from tuple of integers like (year, month, day)
    :param date:
    :return:
    """
    line = str(date[0])
    if date[1] < 10:
        line += '0'
    line += str(date[1])
    if date[2] < 10:
        line += '0'
    line += str(date[2])
    return utcdatetime_from_string(line)


def default_path(path):
    """
    Converts path to default form (with no slash at the end)
    :param path: string - path to convert
    :return:     string - result path
    """
    while path[len(path) - 1] == '/' or path[len(path) - 1] == '\\':
        path = path[0:-1]

    return path


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
    tmp = '{year}-{month:0>2d}-{day:0>2d}T{hour:0>2d}:{minute:0>2d}:{second}.{microsecond}'

    return tmp.format(year = year, month = month, day = day,
                      hour = hour, minute = minute, second = second, microsecond = microsecond)