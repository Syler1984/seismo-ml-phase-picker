# GENERAL PATH VARIABLES
home_directory = '/opt/seisan/WOR/chernykh/phases/'  # Home directory (serves as prefix to every data saving path)
path = '/opt/seisan/'  # Seisan root
db_name = 'IMGG_'  # Database name
wav_path = 'WAV/'  # WAV files subdir name
rea_path = 'REA/'  # S-files subdir name
seisan_definitions_path = '/opt/seisan/DAT/SEISAN.DEF' # '/seismo/seisan/DAT/SEISAN.DEF' # Path to def file (used for finding stations definitions)
archives_path = '/opt/archive' # '/seismo/archive/' # Path to archives
stations_save_path = home_directory + 'stations'  # Where to save stations list (station-picker)
stations_load_path = home_directory + 'stations'  # Leave empty if want to generate stations list in process

# CALCULATED GENERAL PATH VARIABLES
readings_path = path + rea_path  # Partial path to S-files
waveforms_path = path + wav_path  # Partial path to WAV files
full_readings_path = '/opt/seisan/REA/IMGG_' # readings_path + db_name # Full path to S-files
full_waveforms_path = '/opt/seisan/WAV/IMGG_' # waveforms_path + db_name # Full path to WAV files

# OUTPUT PARAMETERS
output_level = 5  # 0 - minimal output, 5 - maximum output

# SLICING PARAMETERS
slice_duration = 100  # Slice duration in seconds
static_slice_offset = slice_duration/2  # Center the pick

archive_channels_order = ['N', 'E', 'Z']  # Specifies channels to pick

dir_per_event = True  # If True - creates subdir for every event

min_magnitude = 1.0  # Minimal magnitude of events allowed
max_depth = 50000  # Maximum depth allowed
max_dist = 300.0  # Maximum distance to station allowed

seconds_high_precision = True  # If true - seconds in picks will take 6 symbols, else - 5

save_dir = home_directory + 'SACKHALIN_100_SEC'  # Where to save picks

picking_stats_file = 'stats'
event_stats_file = 'stats'
picks_stats_file = 'stats'

# If True - will rewrite all duplicate events unless last picking ended with error
rewrite_duplicates = True
# If True - will rewrite all duplicate events instead of ignoring them even if last picking ended with error
explicit_rewrite_duplicates = False

# NOISE PICKING
noise_slice_duration = 100  # Slice duration in seconds
noise_static_slice_offset = noise_slice_duration/2  # Center the pick

max_noise_picks = 10000  # Max amount of noise examples to pick per script run

start_date = [2019, 5, 1]  # Starting date for noise picker
end_date = [2020, 6, 1]    # End date for noise picker

tolerate_events_in_same_day = False  # If False - noise picker will ignore days when actual recorded events happend
event_tolerance = 15  # Number of seconds around noise trace which should not contain any events

noise_save_dir = home_directory + 'SACKHALIN_100_SEC_NOISE'  # Where to save noise picks
if len(noise_save_dir) == 0:  # Grab path from picks_slicing is not set
    noise_save_dir = save_dir

event_time_threshold = 900  # How long noise cannot be picked around event in seconds

# TO-DO: move month length to utils/ as function
month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# HDF5 PARAMETERS
hdf5_file_name = home_directory + 'sachalin_100.hdf5'  # Name to save composed hdf5 file under

required_df = 100  # Required frequency of hdf5 traces for GPD
required_trace_length = 10000  # Required amount of samples (basically length*frequency)

# hdf5_array_length = 100  # Amount of S/P-picks and noise picks

# Codes to be used in Y dataset to characterize p/s/noise picks
p_code = 0  # Code for p-wave picks
s_code = 1  # Code for s-wave picks
noise_code = 2  # Code for noise picks

ignore_acc = True  # If true will skip all accelerogramms
acc_codes = ['ENE', 'ENN', 'ENZ', 'HNE', 'HNN', 'HNZ']  # Codes for accelerogramms

save_ids = True  # If true, will add new dataset with events IDs
ids_dataset_name = 'Z'  # Name for dataset with events IDs

order_of_channels = ['Z', 'N', 'E']  # Order of channels for hdf5 set

detrend = True  # If True, perform detrend

highpass_filter_df = 2  # High-pass filter frequency, if 1, filter disabled

normalization_enabled = True  # If true - normalize picks

global_max_normalizing = True  # If true, will normalize waveform by all traces in stream

p_picks_dir_per_event = True  # Are p-wave picks organized with sub-dirs for each event
p_picks_path = home_directory + 'WAV_P'  # Path to p-wave picks root
p_file_extension = 'P'        # File extension of p-wave picks
p_file_postfix_indexing = True  # If True - p-wave pick files can have indexing after extension (e.g. "filename.P.102")

s_picks_dir_per_event = True  # Are s-wave picks organized with sub-dirs for each event
s_picks_path = home_directory + 'WAV_S'  # Path to s-wave picks root
s_file_extension = 'S'        # File extension of s-wave picks
s_file_postfix_indexing = True  # If True - s-wave pick files can have indexing after extension (e.g. "filename.S.102")

noise_picks_hdf5_path = ''  # Path to noise picks
if len(noise_picks_hdf5_path) == 0:  # Grab path from noise-picker parameters if not set
    noise_picks_hdf5_path = noise_save_dir + '/'

slice_offset_end = 0    # Max value of random slice offset in seconds (negatively shifts start of waveform
#                      ..slicing from 1 to slice_offset seconds)
slice_offset_start = 0

slice_size = 59  # In seconds

wave_noise_offset_picking = True  # Slice noise from wave-picks
noise_offset = 2  # Offset for noise picks from the 0 point of the wave-pick

# MESSAGES
picks_help_message = """Usage: python seismo-phase-picker.py [options]
Options: 
-h, --help \t\t : print this help message and exit
-s, --save \t arg : directory to save picks traces
-r, --rea \t arg : s-files database main directory
-w. --wav \t arg : waveforms database main directory
This script slices a list of events picks waveforms, according to picks in s-files. 
It tries to get picks from continious archives if avaliable, rather than from WAV database."""

noise_help_message = """Usage: python noise-picker.py [options]
Options: 
-h, --help \t\t : print this help message and exit
-s, --save \t arg : directory to save noise picks
-r, --rea \t arg : s-files database main directory
-l, --load \t arg : path to stations from which to pick noise (leave empty if want to generate stations list during this script execution)
-d, --def \t arg : full path to SEISAN.DEF (including filename)
-s, --start \t arg : start date in format DD.MM.YYYY
-e, --end \t arg : end date in format DD.MM.YYYY
-m, --max_picks \t arg : maximum number of noise picks
-a, --archives \t arg : path to continious archives files directory
--output_level \t arg : logging level from 1 to 5
--offset \t arg : maximum picks offset 
--duration \t arg : pick duration
This script slices a list of noise picks wich passed STA/LTA trigger but not described in any of s-files."""

stations_help_message = """Usage: python stations-picker.py [options]
Options: 
-h, --help \t\t : print this help message and exit
-s, --save \t arg : full path for generated stations list file
-r, --rea \t arg : s-files database main directory
This script generates list of all stations, which registered atleast one event in current database."""

hdf5_creator_help_message = """Usage: python hdf5-creator.py [options]
Options: 
-h, --help \t\t : print this help message and exit"""
