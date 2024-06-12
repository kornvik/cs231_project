# Code for Pile Penetration Depth Estimation using 3d CV techniques
The techniques include classical CV methods such as optical flow, ML related techniques including segmentation and feature point detection using DOPE mode. And lastly kalman filter (WIP).

## Example Results 
Tracking pile using optical flow and segmentation

![AGV_vUf92t0uZKI2mPAFzYgzRdow9Qu0647VZG2SfxWejsQmsxV574Od4XZwX8o5mJVRM_iP8iadcWA0XoCNXUqHYH9jdXRijMhoXNtu5XJzKeFIxR3h017v4YW8](https://github.com/kornvik/cs231_project/assets/40062331/30bc183f-376c-48a3-aaca-576c65541750)

Tracking pile using DOPE Model (using animated scene from animation.blend)

![AGV_vUcvLTbTQF90UNkQv6EgkiuHD3eV-Zo7ZdP27YNSQacObi857KqVxfEKOCw5ifgcrA5VvvG-YA_Q7vWyGmYHSqD7YVPga0VXveqaKkDowx2jILd00DOpYg9m](https://github.com/kornvik/cs231_project/assets/40062331/6827adcd-2321-4128-bb71-6cde73e48620)



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
