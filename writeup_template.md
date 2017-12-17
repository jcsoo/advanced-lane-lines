**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal_2_before]: ./camera_cal/calibration2.jpg "Original"
[cal_2_after]: ./camera_out/calibration2.jpg "Undistorted"

[pipeline_0]: ./examples/pipeline_0.jpg "Original Image"

[pipeline_1a]: ./examples/pipeline_1a.jpg "Distortion Corrected Image"
[pipeline_1b]: ./examples/pipeline_1b.jpg "Cropped Image"

[pipeline_2]: ./examples/pipeline_2.jpg "Colorspace Conversion"

[pipeline_3a]: ./examples/pipeline_3a.jpg "H: Hue Similarity Threshold"
[pipeline_3b]: ./examples/pipeline_3b.jpg "S: Saturation Vertical Line Convolution"
[pipeline_3c]: ./examples/pipeline_3c.jpg "V_S: Value Vertical Line Convolution"
[pipeline_3c]: ./examples/pipeline_3d.jpg "V_T: Value Threshold"

[pipeline_4a]: ./examples/pipeline_4a.jpg "A: (Dilate H) AND S"
[pipeline_4b]: ./examples/pipeline_4b.jpg "B: S AND V_S"
[pipeline_4b]: ./examples/pipeline_4c.jpg "C: V_S AND V_T"

[pipeline_5]: ./examples/pipeline_5.jpg "Combined: A + B + C"

[pipeline_6]: ./examples/pipeline_6.jpg "Apply Perspective Transform"

[pipeline_7]: ./examples/pipeline_7.jpg "Line Fitting"

[pipeline_8]: ./examples/pipeline_8.jpg "Curve Radius and Lane Position"

[pipeline_9]: ./examples/pipeline_9.jpg "Draw Lane Image"

[pipeline_10]: ./examples/pipeline_10.jpg "Draw Information Overlays"

# Project Description

## Approach

## Camera Calibration

## Pipeline

### 0: Original Image

### 1: Distortion Correction and Cropping

### 2: Colorspace Conversion to HSV

### 3: Basic Feature Extraction

### 4: Feature Combination

### 5: Feature Unification

### 6: Perspective Transform

### 7: Line Fitting

#### Without Priors - Simple Windowing

#### With Priors - Curve-based Windowing

#### Filtering Logic

### 8: Curve Radius and Lane Position

### 9: Draw Lane Image

### 10: Draw Information Overlays

### Video 1 - Project Video ###

[Project Video](./examples/pv.mpg)

### Video 2 - Challenge Video ###

[Challenge Video](./examples/cv.mpg)

### Video 3 - Harder Challenge Video ###

[Harder Challenge Video](./examples/hcv.mpg)

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

Provided.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration is handled in the `calibrate.py` module. Initial calibration uses `calibrate()`, which
reads all images from the calibration file directory and uses `cv2.findChessboardCorners()` and 
`cv2.cornerSubPix()` to detect all of the corners and construct the object point and image point arrays.
Then, `cv2.calibrateCamera()` is used to generate a correction matrix, which is stored in the file
`calibrate.p`.

The module also includes an image loader class that reads the correction data and implements methods to 
correct the incoming images.

Original Calibration Image
----

![alt text][cal_2_before]

Corrected Calibration Image
----

![alt text][cal_2_after]

### Pipeline (single images)

The core image pipeline has several steps.


#### 1. Provide an example of a distortion-corrected image.

See Pipeline Phase 1 

#### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

See Pipeline Phase 2

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

See Pipeline Phase 6

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

See Pipeline Phase 7

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

See Pipeline Phase 8

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

See Pipeline Phase 9

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

### Video 1 - Project Video ###

[Project Video](./examples/pv.mpg)

### Video 2 - Challenge Video ###

[Challenge Video](./examples/cv.mpg)

### Video 3 - Harder Challenge Video ###

[Harder Challenge Video](./examples/hcv.mpg)

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### Phases 3, 4 and 5 (Pixel Classification)

#### Phase 7 (Line Fitting)

#### Phase 8 (Curve Radius and Lane Position)



