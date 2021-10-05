# replay-behavior
a little script for reconstructing human demonstraction videos from iGibson BEHAVIOR dataset

## Pre-requisits

BEHAVIOR recommends using `gsutil` to download the dataset. Install `gsutil` by following [these instructions](https://cloud.google.com/storage/docs/gsutil_install)

```
pip install h5py numpy matplotlib opencv-python tqdm
```

## Run

Run the script:

```
python video_generator.py
```
