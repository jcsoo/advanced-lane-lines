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

[calibration_0]: ./examples/calibration0.jpg "Original"
[calibration_1]: ./examples/calibration1.jpg "Undistorted"

[pipeline_0]: ./examples/pipeline_0.jpg "Original Image"

[pipeline_1a]: ./examples/pipeline_1a.jpg "Distortion Corrected Image"
[pipeline_1b]: ./examples/pipeline_1b.jpg "Cropped Image"

[pipeline_2a]: ./examples/pipeline_2a.jpg "Histogram Equalization"
[pipeline_2b]: ./examples/pipeline_2b.jpg "Colorspace Conversion"

[pipeline_3a]: ./examples/pipeline_3a.jpg "H: Hue Similarity Threshold"
[pipeline_3b]: ./examples/pipeline_3b.jpg "S: Saturation Vertical Line Convolution"
[pipeline_3c]: ./examples/pipeline_3c.jpg "V_S: Value Vertical Line Convolution"
[pipeline_3d]: ./examples/pipeline_3d.jpg "V_T: Value Threshold"

[pipeline_4a]: ./examples/pipeline_4a.jpg "A: (Dilate H) AND S"
[pipeline_4b]: ./examples/pipeline_4b.jpg "B: S AND V_S"
[pipeline_4c]: ./examples/pipeline_4c.jpg "C: V_S AND V_T"

[pipeline_5]: ./examples/pipeline_5.jpg "Combined: A + B + C"

[pipeline_6]: ./examples/pipeline_6.jpg "Apply Perspective Transform"

[pipeline_7a]: ./examples/pipeline_7a.jpg "Line Fitting - No Priors"
[pipeline_7b]: ./examples/pipeline_7b.jpg "Line Fitting - With Priors"

[pipeline_8]: ./examples/pipeline_8.jpg "Curve Radius and Lane Position"

[pipeline_9]: ./examples/pipeline_9.jpg "Draw Lane Image"

[pipeline_10]: ./examples/pipeline_10.jpg "Draw Information Overlays"

# Project Description

## Approach and Philosophy

This implementation followed the basic approach from the class notes, with an initial
pixel classification phase followed by windowed polynomial curve fitting.

For this particular project, I started initially with a Sobel filter based approach for
pixel classification but decided to focus on a technique using a combination of threshold
filtering and varying-width vertical line filters. I chose this approach because it is
amenable to implementation in real time on smaller embedded camera systems, perhaps even 
incrementally at the line or block level.

I kept the basic lane fitting approach with a few modifications to improve intial fitting and
to provide more feedback on the quality of the fit. There are a number of improvements that
I can foresee (using splines instead of polynomials, using iterative pixel neighbor
techniques, and using RANSAC or other progressive fitting techniques) but it seemed to me
that implementing them in NumPy / OpenCV would be challenging and beyond the scope of this project.

I have implemented some simple techniques for simple temporal averaging and quality filtering,
but I have chosen to keep the amount of averaging minimal. Turning off filtering exposes
the quality of the underlying feature extraction and the quality of the underlying data. For
this project it's more interesting to see how well the code performs, and in a real system
it would be better to provide the raw data to upper layers which can then decide on a
filtering policy with more information than this subsystem would have available.

## Camera Calibration

Camera calibration is handled in the `calibrate.py` module. Initial calibration uses `calibrate()`, which
reads all images from the calibration file directory and uses `cv2.findChessboardCorners()` and 
`cv2.cornerSubPix()` to detect all of the corners and construct the object point and image point arrays.
Then, `cv2.calibrateCamera()` is used to generate a correction matrix, which is stored in the file
`calibrate.p`.

The module also includes an image loader class that reads the correction data and implements methods to 
correct the incoming images.

Original Calibration Image
----

![alt text][calibration_0]

Corrected Calibration Image
----

![alt text][calibration_1]

## Pipeline

### 0: Original Image

![original][pipeline_0]

### 1: Distortion Correction and Cropping

![][pipeline_1a]

![][pipeline_1b]

### 2: Histogram Equalization and Colorspace Conversion to HSV

![][pipeline_2a]

![][pipeline_2b]

### 3: Basic Feature Extraction

![][pipeline_3a]

![][pipeline_3b]

![][pipeline_3c]

![][pipeline_3d]

### 4: Feature Combination

![][pipeline_4a]
![][pipeline_4b]
![][pipeline_4c]

### 5: Feature Unification

![][pipeline_5]

### 6: Perspective Transform

![][pipeline_6]

### 7: Line Fitting

#### Without Priors - Simple Windowing

![][pipeline_7a]

#### With Priors - Curve-based Windowing

![][pipeline_7b]

#### Filtering Logic


### 8: Curve Radius and Lane Position

### 9: Draw Lane Image
![][pipeline_9]

### 10: Draw Information Overlays
![][pipeline_10]

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



