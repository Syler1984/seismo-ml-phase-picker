# seismo-ml-phase-picker
**seismo-ml-phase-picker** is a series of scripts for analysing seisan database and forming 
a set of waveform picks of events and noise picks 

## Gathering waveforms

```
python picker.py -c data/config.ini -s 2020-02-01 -e 2020-03-01
```

**picker.py** designed to provide basic functionality for creating a 
*.h5* dataset from a *Seisan* database. An outcome of running **picker.py**
is a *.h5* file with three datasets: **X** - containing waveforms,
**Y** - containing labels (0 - for P-waves, 1 - for S-waves, 2 - for noise) and
**ID** - containing strings of information, which can be used to identify
waveforms origins.