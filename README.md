# replay-behavior
a little script for reconstructing human demonstraction videos from iGibson BEHAVIOR dataset

## Prerequisite

BEHAVIOR recommends using `gsutil` to download the dataset. Install `gsutil` by following [these instructions](https://cloud.google.com/storage/docs/gsutil_install).

The script requires the following packages:

```
pip install h5py numpy matplotlib opencv-python tqdm
```

## Run the script

The script downloads the `.hdf5` human behavior data files for the activities specified in list `chosen_activities`, reconstruct images (up to `len_frames` for each video), and generate MP4 videos from those images.

```
python video_generator.py
```
