# lidar-align

Small python tool for re-aligning misaligned lidars in a concatenated "supercloud"
dataset.

## Known limitations

This tool was made for one specific use-case, so it's not very flexible. Following
limitations apply:

1. The data is expected to contain a `label` field to color the points,
which is not strictly necessary to use the tool. If you don't have this data,
just fill your PCD files with field `label` of all zeroes.
2. The color map used to color the points according to `label` field is hardcoded,
and supports classes 0-17. If there are more classes than 18, it assigns color
`[128, 128, 128]` to those classes.

## Usage

### 1. Calibrate with open3D GUI

The calibration tool expects a single PCD format file with input, that has
fields `x,y,z,sensor_id,label`.

Here `sensor_id` is a `uint8` value that represents the index of the sensors
from which the data came from.

`label` is a `uint8` value that represents the class index of the point. 

Secondly, it expects transformation matrices for all of the sensors, that were
used for concatenating the separate sensor frames into a single composite
pointcloud (our input PCD). In other words, each matrix maps the sensor's
own coordinate space into global space (the composite cloud space).

The matrices are provided as a single string format input, where each sensor's
matrix is formatted as a row-major 3x4 affine transformation, separated by
semicolons. The matrix elements are: `[r_1, r_2, r_3, t_x, r_4, r_5, r_6, t_y, r_7, r_8, r_9, t_z]`,
where the components are Rotation `R` (3x3 matrix) and translation `T` (3x1 matrix).

Translation represents the XYZ offset of this sensor relative to the supercloud origin.

Example invocation:

```sh
export CALIBRATION_STRING="1,0,0,0,0,1,0,0,0,0,1,0;0.997292,0.0680656,-0.027833,0.0239639,-0.0669792,0.997021,0.0382632,-1.10121,0.0303544,-0.0362953,0.99888,0.0138038;-0.886873,-0.461957,0.007202,-0.208495,0.461488,-0.885015,0.0614494,0.606671,-0.0220131,0.0578215,0.998084,-0.0851192;0.922887,-0.384039,-0.0281881,-2.26607,0.385063,0.920876,0.0609422,0.634961,0.00255362,-0.067097,0.997743,0.110059;-0.855559,0.517699,0.00258428,-0.200085,-0.517377,-0.855184,0.0313555,-1.71323,0.0184428,0.0254893,0.999505,-0.0186664;0.913619,0.406555,0.00338869,-2.2398,-0.406505,0.913295,0.0254494,-1.69465,0.00725159,-0.0246285,0.99967,0.201087;-0.996481,0.0105359,-0.0831498,-3.12797,-0.0111834,-0.99991,0.00732328,-0.415618,-0.0830654,0.00822729,0.996509,0.449154"
python fix_sensor_calibration.py \
   path/to/file.pcd \
   $CALIBRATION_STRING
```

The tool saves `calibration_fixed.json` to your CWD when ENTER is pressed within
tool.

### 2. Apply calibration matrix to the whole dataset

This takes as input your dataset you want to recalibrate (directory of `.pcd`
files). Again fields `x,y,z,sensor_id,label` are expected. It also takes same
calibration string used in step 1, a path to an output directory and the
recalibration matrix saved in step 1.

```sh
python apply_calibration_correction.py path/to/pcd/dataset/ \
    $CALIBRATION_STRING \
    path/to/output/dir/ \
    --calibration_json calibration_fixed.json
```
