# Code for Pile Penetration Depth Estimation using 3d CV techniques
The techniques include classical CV methods such as optical flow, ML related techniques including segmentation and feature point detection using DOPE mode. And lastly kalman filter (WIP).

## Data source

1. Download you can download video.zip from https://drive.google.com/drive/folders/1buvw2SAH0mHDW_G-bmm7Y2XeSQVAug-c?usp=sharing for pile scene
2. Generate the pile images from blender using pile.blend
3. In case you want to test with animated scene generated rendered video from animation.blend

## How to run
### Running optical flow without segmentation
`python opflow_with_target.py`

### Running optical flow with segmentation
`python opflow_with_target_with_segmentation.py`

### Training model
1. Open pile.blend file and run the script to generated images for training model
2. Place the generated image folder inside the project folder and run `python preprocess.py` to process the image metadata (transforms.json) to output feature points cooridnates for each frame in results.json file.
3. Make sure to update the path on 'model/main.py' to point to correct results.json and generated image folder path. Run `python ./model/main.py` to train the model.

### Running pile tracking using model and segmentation
`python ./model/track_pile_with_model.py`

## Note
Segmentation still requires us to finetune and pinpoint a few coordinates (mostly in the middle of the scenes) to extract the object in these coordinates. Passing `showFirstFrame=True` in `remove_background` method will popup how the segmentation look likes in the first scene of the video, which should be helpful in finetuning this.
