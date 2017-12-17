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

[pipeline_2a]: ./examples/pipeline_2a.jpg "Global Histogram Equalization"
[pipeline_2b]: ./examples/pipeline_2b.jpg "Slice Histogram Equalization"

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

For this project, the only area of interest is the section of the image containing the road surface. The top
420 and bottom 80 pixels are cropped, leaving a 1280 x 220 image with only 30% of the original pixels.

![][pipeline_1b]

### 2: Histogram Equalization and Colorspace Conversion to HSV

See `process_image()` in `conv.py`.

In order to adjust for lighting conditions, two equalization methods were investigated in the `equalize_hist`
function on conv.py:

```
def equalize_hist(img):
    if False:
        # Global Histogram Equalization
        img[:,:,1] = cv2.equalizeHist(img[:,:,1])
        img[:,:,2] = cv2.equalizeHist(img[:,:,2])

    if False:
        # Striped Histogram Equalization
        steps = 4
        step = int(img.shape[0] / steps)
        for i in range(steps):
            img_row = img[(i * step):((i + 1) * step),:,:]
            img_row[:,:,1] = cv2.equalizeHist(img_row[:,:,1])
            img_row[:,:,2] = cv2.equalizeHist(img_row[:,:,2])
    return img
```

The first variant uses `cv2.equalizeHist()` on the S and V channels across the entire image, and the second
does the same except across four horizontal slices of the image.

![][pipeline_2a]

![][pipeline_2b]

In practice, performance did not seem to be improved when using this type of equalization, possibly
because the histogram was affected by parts of the image besides the road surface. A future version
could use an application-specific equalization method designed to equalize the road surface of the
entire image to the approximate gray level of the area immediately in front of the car.

In the final version, no equalization or contrast adjustment was performed, and the image was 
immediately converted to a floating point HSV colorspace.

### 3: Basic Feature Extraction

See `process_image()` in `conv.py`.

For the first phase of the pixel classification section of the pipeline, a number of per-channel
binary threshold filters and 2D convolutions were applied.

All of these filters can be used to produce a continuous signal, but are currently binarized for
use in later stages. The continuous-valued output is displayed below if available to give an indication
of the performance of each filter.

#### 3a: Hue Threshold

In order to match the yellow used in some lane lines, a simple threshold filter was used to
match all pixels within specified range:

```
    h = filter_thresh(img[:,:,0], (0.075, 0.100))
```

This results in a fairly crude image that tends to match other areas not associated with the yellow lane
line.

![][pipeline_3a]

The relatively low quality of this filter is acceptable because this will mainly be used to filter lines produced from the other channels. This would be useful for identifying whether a lane marker is yellow.

#### 3b: S: Saturation Vertical Line Convolution

This first-level channel is produced by using a simple vertical line filter on the Saturation channel. The code for this is as follows (in `conv.py`):

```
def filter_stripes(img, bias, mul, wmul=1, deriv=True):
    img = np.concatenate([
        filter_vline(img[0:80,:], (2, 6 * wmul), deriv),
        filter_vline(img[80:150,:], (2, 6 * wmul), deriv),
        filter_vline(img[150:220,:], (2, 6 * wmul), deriv),
    ])
    img += bias
    img[img < 0] = 0.0

    if mul is None:
        img /= img.max()
    else:
        img *= mul

    return img
    

def filter_vline(img, size, deriv=True):
    kernel = (np.concatenate([
        np.zeros(size, np.float32),
        np.ones(size, np.float32),
        np.ones(size, np.float32),
        np.zeros(size, np.float32),
    ], axis=1) - 0.5) / (size[0] * size[1] * 4)
    if deriv:
        kernel = cv2.GaussianBlur(kernel, (5, 5), 0, 0)    
        k2 = np.array([[-0.5, -0.5, 0.5, 0.5]])
        kernel = cv2.filter2D(kernel, -1, k2)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
```

which produces:

![][pipeline_3b]

In this particular case, the vertical line filter for all slices is set to 6, but it is possible
to vary the width for each slice. With a progressive line width, sensitivity to thinner lane markers
at distance is improved, at a cost of higher false positives. The current line matching system
seems to perform better at extrapolating from fewer markers than at eliminating false positives.

There is also an option to use the derivative of the simple rectangular filter, which does a better
job at only matching true positive-step followed by negative step lines of the appropriate width. This
mode was used for the Value channel filtering discussed in the next section.

The current system is fairly simple, but is designed to allow more sophisticated line filtering in the
future. For instance, one could use the current lane curvatures to generate a series of filters tuned
to match the exact width and orientation of the expected lines. Another approach would be to use
backpropagation and SGD to generate a set of general purpose line filters.

This implementation accepts a `bias` and `mult` parameter and performs a `RELU` activation, similar to
a typical 2D convolution filter in a neural network.

#### 3c: V_S: Value Vertical Line Convolution

This channel is produced by using the vertical-line filter on the Value channel. In this case the derivative
of the rectangle kernel is used in order to be more selective. As can be seen below, this does a good
job of showing the lines as long as highlights from non-road surfaces are ignored:

![][pipeline_3c]

#### 3d: V_T: Value Threshole Filter

This channel is produced by using a simple threshold filter on the Value channel, selecting for values from
60% to 100%:

![][pipeline_3d]

This channel is similar to the H threshold channel in that it has many false positives, highlighting
the lines as well are regions around the lines. Its main use is in conjunction with other filters.

### 4: Feature Combination

This set of output channels is produced through logical combinations of the various first-layer channels.

#### 4a: A: (Dilate H) AND S

This channel is generated by performing a dilation on the H threshold channel (in order to merge nearby
points and reduce noise) and then intersecting with the S vertical line convolution channel:

![][pipeline_4a]

This channel seems to do a good job at identifying yellow lines with relatively few false positives, which
could easily be eliminated with further filtering. 

#### 4b: B: S AND V_S

This channel is generated by intersecting the S vertical line convolution channel and the V vertical
line convolution channel:

![][pipeline_4b]

This also seems to do a good job at identifying yellow lines.

#### 4b: C: V\_S AND V_T

This channel is generated by intersecting the V\_S and V\_T channels. The V\_T channel serves
to eliminate line matches on the V\_S channel produced by features that are darker than
the main road surface such as pavement changes or tar lines.

![][pipeline_4c]

This channel might be good enough on its own in many cases.

### 5: Feature Unification

For the next step, a simple union of the A, B and C channels is performed:

![][pipeline_5]

Note that this operation plus the binarization from the earlier stages potentially
throws out a lot of useful information. The current line matching system can only use
a single binary input channel, but it's easy to see how a different design could use
all three (or more) floating point channels to assign confidence levels to different
pixels. Pixels that match two or more channels can be given much more weight than
ones that only show up in a single channel, and individual channels can be weighted
differently.


### 6: Perspective Transform

In the next stage of the pipeline, a perspective transform is performed to convert to a
rectangular top-down view.

First, a perspective transform matrix and its inverse are generated based on manually determined
geometric parameters (see `process()` and `perspective_transform()` in pipeline.py:

```
    self.M, self.Minv = self.perspective_transform(
        vp = (640, 0), # Vanishing Point
        tl = (520, 40), # Top Left of source parallelogram
        bl = (0, 220), # Bottom Left of source parallelogram
        out = self.warp_shape
    )

    def perspective_transform(self, vp, tl, bl, out):
        # Symmetric perspective transform based on vanishing point, top left, bottom left

        src = np.array([
            (tl[0], tl[1]), 
            (vp[0] + (vp[0] - tl[0]), tl[1]), 
            (bl[0], bl[1]), 
            (vp[0] + (vp[0] - bl[0]), bl[1])],
        np.float32)
        dst = np.array([(0, 0), (out[0], 0), (0, out[1]), (out[0], out[1])], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        return M, Minv    
```

The actual warp and unwarp transformation are performed by a pair of methods using the matrices:

```
    def warp(self, img, size=None):
        if size is None:
            size = self.warp_shape
        return cv2.warpPerspective(img, self.M, size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img, size=None):
        if size is None:
            size = self.img_shape
        return cv2.warpPerspective(img, self.Minv, size, flags=cv2.INTER_LINEAR)
```

The output of this stage is a representation of the lane lines:

![][pipeline_6]

In this case, the lines are curving to the right.

It is interesting to note that the lines are not completely parallel, and that there appears to be
some distortion on the bottom right. The perspective transform was manually selected using
`straight\_lines1.jpg` and `straight\_lines2.jpg`, but it could be that there is further distortion
that was not accounted for or that the vanshing point was set incorrectly.

This distortion may be one reason why the calculated curve radius appears to be a bit off.

### 7: Line Fitting

Line fitting was performed using the basic approaches from the class lectures. The code was modified
to allow for fitting a single line at a time rather than both lines at once, and to perform the fit
in world coordinates as well as in screen coordinates.

Both variants were also updated to generate and return a histogram of candidate points in the Y axis.
This is used for basic validity checking: fits generated with a small number of bins tend to be
extremely sensitive to small variations and are more likely to produce an incorrect polynomial, so
a fits with a bin count lower than a minimum threshold (currently set to 3 of 8) are discarded.

Earlier variants used a simple count of pixels used for the line fitting, but it proved difficult to
find a good threshold given that this was affected by the width of the line more than the length of the
line - in particular when lines of different colors have different thicknesses due to the different channel
filters described above.

#### Without Priors - Simple Windowing

*Note: I have omitted a sample image because the rendering code for this filter was
removed at an early stage. The implementation is almost identical to the variant described in the class notes. Please let me know if this description is insufficient.*

See `find_line` in `pipeline.py`.

This line fitting process is used when there are no priors or the priors are believed to be unreliable. First,
a histogram of the bottom half of the image is used in order to detect the highest peak on each side of the centerline. Then, a window is generated for each horizontal segment from the bottom of the image to the top. If
pixels are found and meet the configurable threshold, they are added to the list and the window is adjusted.

The only variation from the classroom algorithm is that the shift is recorded, and reused if not enough pixels are found. This improves performance in cases where there is a gap in a curved or angled line.

After all windows are scanned, `np.polyfit(y, x, 2)` is used to fit a polynomial, in both pixel space and
in world space. These polynomials are returned, as well as the number of bins containing matching pixels.

#### With Priors - Curve-based Windowing

*Note: I have omitted a sample image because the rendering code for this filter was
removed at an early stage. The implementation is almost identical to the variant described in the class notes. Please let me know if this description is insufficient.*

See `find_line_with_priors` in `pipeline.py`.

For this variant, pixels are selected from the image if they are within a specified margin of the
given polynomial. If nonzero pixels are found, `np.polyfit()` is used to fit a polynomial in both pixel
space and in world space. These polynomials are returned, as well as the number of bins containing matching pixels.

#### Filtering Logic

*Note: Please see Approach and Philosophy, above.*

After several variations were tried, a fairly simple approach was used for filtering found lines
and potentially averaging them.

When no priors are available (such as at the beginning of a run), `find_line()` is used. If 
the number of matching bins meets the threshold, they are used as priors for future frames.

Additionally, priors are cleared based on a configurable frame counter.

If priors are available, `find_line_with_priors()` is used. If the number of matching bins meets the
threshold, the priors are updated, otherwise the priors from the previous frame are kept and used as
the output. Each line is handled independently.

In addition, a configurable list of previous fits are kept for each lane, and can be averaged together
to produce the output. The current variant has the maximum list length set to `1` to disable temporal
averaging.

There were a number of other approaches considered but not implemented due to time constraints:

 - Instead of simply averaging the polynomials, we could store all of the invidual pixels used
   for matching. 
 - The number of failed `find_line_with_priors()` attempts could be tracked and used to trigger a
   new `find_line` attempt after reaching a threshold.
 - Different actions could be taken based on the bin count returned from `find_line_with_priors()`; 
   a zero could trigger `find_line`, while a partial failed match could simply be retried next frame.
 - Both approaches could use an `inner_margin` / `outer_margin` pair, which could be used to detect
   when pixels leave the predicted path.
 - Instead of the current static path, `find_line_with_priors` could be implemented in such a way to
   iterate through windows, recalculating the best fit after each window if it improves the number of
   pixels matched. A RANSAC based line detector could also be used.
 - The line detectors could be redesigned to use floating point rather than binary inputs, allowing
   them to consider the strength of detection provided by previous layers and weighting multi-layer
   pixels more highly.

### 8: Curve Radius and Lane Position

Curve radius was implemented by calculating the derivative of each line at the bottom of the screen:

```
    def curve_radius(self, fit, y_eval):
        curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        # Example values - px: 1926.74 1908.48
        return curverad
```

This was performed once for each curve in pixel space and after conversion to world space based on 
measurements of the warped lines.

Lane width was calculated by finding the difference of the positions of each line in pixel space and in
world space. Offset was calculated from the screen center to the average of the two lines.

Lane width estimates matched up well with the expected numbers - roughly 3.8 meters across. Line curvature
did not match as well, with a prediction of roughly 400m for the curve in the project video vs an expected 
value of about 1000m. The curve estimates were particularly noisy.

### 9: Draw Lane Image

The lane image was drawn in canvas space and then re-projected and merged onto the original cropped image.

![][pipeline_9]

### 10: Draw Information Overlays

A line of text containing useful information such as the number of bins used in the last match and the
curve details was drawn at the top of the image.

![][pipeline_10]

### Video 1 - Project Video ###

This variant performed well with the project video. Some noise is visible in lane projections, but 
nothing major that would not be filtered out with some time averaging. Note that there are no major
excursions over the first bridge and only minor ones over the second.

[Project Video](./examples/pv.mp4)

### Video 2 - Challenge Video ###

This video was the main focus of effort in the late stages of this project. The various combinations
of filters proved useful in suppressing matches on the non-lane markings such as the pavement transitions
and nearby vehicles, and also allowed increasing sensitivity in order to pick up faint dashed markers
further down the road.

The biggest improvement came from implementing the bin filtering system, which prevented line detection from extrapolating large excursions from partial lines nearing the area under the bridge.

Note that there are only minor glitches under the bridge, that the transition over the pavement edge is 
handled cleanly, and that the vehicle in the next lane at the end of the video does not affect the line
detection.

[Challenge Video](./examples/cv.mp4)

### Video 3 - Harder Challenge Video ###

An earlier variant performed well on this video, but the current version does not do as well - probably because
the thresholding on some filters was turned down to improve sensitivity in the Challenge video. Dynamic
threshold changes based on pavement brightness would help significantly.

The other issue is that the line detectors are not tuned agressively enough to handle this case, particuarly
because they can not dynamically calculate the best fit windows to sample. Also, it might be better to use
splines for fitting curves that can change direction.

There are also limitations to approach based on the fact that we are primarily using NumPy and OpenCV. There are
a few iterative pixel-neighbor techniques that are promising but far to slow to implement in Python, and not
worth the effort to implement natively for this particular problem.

[Harder Challenge Video](./examples/hcv.mp4)

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

With more time, I would have liked to implement completely different line fitting algorithms, probably
based on RANSAC or a similar approach.

In general, I felt held back by implementing in Python and Numpy - there were a number of things I wanted to
try that would have been very inefficient using an iterative approach, but weren't practical using the
amount of NumPy that I know.

As mentioned in Approach and Philosophy, I think there are a lot of ways I could have smoothed the output
in a smart way, but they seemed less interesting than figuring out the lower layers.

My ultimate goal was to build a pixel classification layer that I could actually use backpropagation and SGD
to train. You can see this in the architecture, but unfortunately I didn't have the time to really get beyond the start of that idea. I may try this on my own in the future.

This particular pipeline's biggest weakness is that there are several key parameters (mainly threshold values) that are dependent on the specific lighting conditions - as seen on the Harder Challenge Video.

 It's likely
that this could be addressed by making these thesholds more dynamic - based on the overall lighting conditions,
the lighting of the road surface in front of the car, and even the road surface right near the area being classified.