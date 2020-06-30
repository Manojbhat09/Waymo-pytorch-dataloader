
# Waymo pytorch Dataloader

Pytorch dataloader for object detection tasks
- Quick attach to your kitti training files
- Has kitti format of calibration and label object
- Uses only top lidar and all 5 images
- Only serial dataloader
Please feel free to send pull requests if you have any changes

## Installation

```bash
git clone Waymo-pytorch-dataloader
git clone https://github.com/gdlg/simple-waymo-open-dataset-reader
```
or recursively download the subrepository like 
```bash
git clone Waymo-pytorch-dataloader --recursive
```

Directly use the dataloader in your script like:
```python
DATA_PATH = '/home/jupyter/waymo-od/waymo_dataset'
LOCATIONS = ['location_sf']

dataset = WaymoDataset(DATA_PATH, LOCATIONS, 'train', True, "new_waymo")

frame, idx = dataset.data, dataset.count
calib = dataset.get_calib(frame, idx)
pts =  dataset.get_lidar(frame, idx)
target = dataset.get_label(frame, idx)
```


## License

This code is released under the Apache License, version 2.0. This projects incorporate some parts of the [Waymo Open Dataset code](https://github.com/waymo-research/waymo-open-dataset/blob/master/README.md) (the files `simple_waymo_open_dataset_reader/*.proto`) and is licensed to you under their original license terms. See `LICENSE` file for details.
