import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
from moviepy.editor import VideoFileClip

from calibrate import ImageLoader

PIPELINE_CFG = {
    'calibration_path': 'calibration.p',
    'mask' : (0.08, 0.40),
    'font' : cv2.FONT_HERSHEY_SIMPLEX,
}

WHITE = (255, 255, 255)

class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loader = ImageLoader(cfg['calibration_path'])
        self.img_shape = None
        self.M = None
        self.Minv = None
        self.fit_left = None
        self.fit_right = None
        self.num_left = 0
        self.num_right = 0

    def load_image(self, path):
        return self.loader.load_bgr(path)

    def mask_image(self, img, mask):        
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        if mask is None:
            return img

        imshape = img.shape

        mask_w, mask_h = mask

        vertices = np.array([[
            (0,imshape[0]),
            (imshape[1] * (0.5 - mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1] * (0.5 + mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1], imshape[0])
        ]], dtype=np.int32)


        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image        

    def mask_image_binary(self, img, mask):        
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        if mask is None:
            return img

        imshape = img.shape

        mask_w, mask_h = mask

        vertices = np.array([[
            (0,imshape[0]),
            (imshape[1] * (0.5 - mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1] * (0.5 + mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1], imshape[0])
        ]], dtype=np.int32)


        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        ignore_mask_color = 1
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image            

    def sobel(self, img):
        k_size = 3
        sx_thresh = None #(20, 100), 
        sy_thresh = None #(0, 10), 
        mag_thresh = (75, 150)
        dir_thresh = (0.7, 1.3)    

        # Convert to HLS color space and separate the L channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        
        # l_channel = l_channel / l_channel.mean()

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=k_size) # Take the derivative in x
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=k_size) # Take the derivative in 1

        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal

        mag_sobel = np.sqrt(sobelx**2 + sobely**2) # Magnitude of gradient
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx) # Direction of gradient
        
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
        scaled_mag = np.uint8(255*mag_sobel/np.max(mag_sobel))

        gradx= np.zeros_like(scaled_sobelx)
        if sx_thresh:
            gradx[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        grady = np.zeros_like(scaled_sobely)
        if sy_thresh:
            grady[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

        mag_binary = np.zeros_like(mag_sobel)
        if mag_thresh:
            mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1

        dir_binary = np.zeros_like(dir_sobel)
        if dir_thresh:
            dir_binary[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1
        
        combined = np.zeros_like(img[:,:,0])
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1    
        # combined[(mag_binary == 1)] = 1
        return combined
       
        
    def color_threshold(self, img):
        s_thresh=(150, 255)

        img = np.copy(img)
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        s_channel = hls[:,:,2]
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        if s_thresh:
            s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        return s_binary

    def color_yellow(self, img):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # channel = img[:,:,0]
        return self.image_threshold(img, [
            (0, 128),
            (0, 255),
            (150, 255),
        ])        

    def color_white(self, img):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # channel = img[:,:,0]
        return self.image_threshold(img, [
            (0, 255),
            (200, 255),
            (0, 255),
        ])        




    def image_threshold(self, img, thresh):
        b0 = self.channel_threshold(img[:,:,0], thresh[0])
        b1 = self.channel_threshold(img[:,:,1], thresh[1])
        b2 = self.channel_threshold(img[:,:,2], thresh[2])
        return b0 & b1 & b2

    def channel_threshold(self, channel, thresh):
        binary = np.zeros_like(channel)
        if thresh:
            binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
        return binary



    def combined(self, img):
        sobel = self.sobel(img)
        s_binary = self.color_threshold(img)

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sobel)
        combined_binary[(s_binary == 1) | (sobel == 1)] = 1        
        return combined_binary

    def find_base_lines(self, img):
        # Histogram of Warped Image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find midpoint
        midpoint = np.int(histogram.shape[0]/2)
        # Find left peak
        leftx_base = np.argmax(histogram[:midpoint])
        # Find right peak
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def find_line(self, img, x_base, nwindows=9, margin=100, minpix=20):
        binary_warped = img

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        x_current = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 

        num_x = np.count_nonzero(x)

        if num_x > 0 :
            fit = np.polyfit(y, x, 2)
        else:
            fit = None
        
        return fit, num_x

    def find_lines(self, img):
        left_x, right_x = self.find_base_lines(img)

        left_fit, left_num = self.find_line(img, left_x)
        right_fit, right_num = self.find_line(img, right_x)

        return left_fit, right_fit, left_num, right_num

    def find_lines_orig(self, img, plot=False):
        ### From Class Notes

        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 20

        binary_warped = img
        # Find Lane Lines
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Histogram of Warped Image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find midpoint
        midpoint = np.int(histogram.shape[0]/2)
        # Find left peak
        leftx_base = np.argmax(histogram[:midpoint])
        # Find right peak
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        if leftx.any() & lefty.any():
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = self.fit_left

        if rightx.any() & righty.any():
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = self.fit_right

        if plot:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)    
            plt.show()
        return (left_fit, right_fit)    

    def find_line_with_priors(self, img, fit, margin=50):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = (
            (nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - margin)) &
            (nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + margin))
        )         

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]        

        num = np.count_nonzero(x)

        if num > 0:
            fit = np.polyfit(y, x, 2)
        else:
            fit = None

        return fit, num

    def find_lines_with_priors(self, img, left_fit, right_fit, margin=50, min_points=500, plot=False):
        binary_warped = img
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        lfit, lnum = self.find_line_with_priors(img, left_fit, margin)
        rfit, rnum = self.find_line_with_priors(img, right_fit, margin)

        if lnum < min_points:
            lfit = left_fit
        
        if rnum < min_points:
            rfit = right_fit

        return lfit, rfit, lnum, rnum

        
    def plot_fits(self, binary_warped, left_fit, right_fit):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Generate x and y values for plotting
        # Create an image to draw on and an image to show the selection window

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        window_img = np.zeros_like(img)
        # Color in left and right line pixels
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])                                    
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0) 
        plt.show()

               


    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self, image, window_width, window_height, margin):        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    def window_conv(self, img):
        # window settings
        window_width = 50 
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching

        warped = img
        window_centroids = self.find_window_centroids(warped, window_width, window_height, margin)

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
            # output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
            output = template + warpage
        
        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show()        
        return (None, None, output)

    def curve_radius_px(self, left_fit, right_fit, y_eval):
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        # Example values: 1926.74 1908.48
        return left_curverad, right_curverad

    def curve_radius_m(self, left_fit, right_fit, y_eval):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension        

        # # Fit new polynomials to x,y in world space
        # left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        # right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

        # # Calculate the new radii of curvature
        # left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        # right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # # Example values: 632.1 m    626.2 m
        # return left_curverad, right_curverad

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, self.img_shape, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, self.img_shape, flags=cv2.INTER_LINEAR)

    def draw_unwarped(self, undist, warped, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            print("No lines")
            return undist
        if self.num_left == 0 or self.num_right == 0:
            return undist

        image = undist
        binary_warped = warped

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # print(left_fitx.shape, ploty.shape)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = self.unwarp(color_warp)

        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def draw_text(self, img, text, org, scale, color):      
        org = [org[0], org[1]]
        shape = img.shape
        if org[0] < 0:
            org[0] = shape[1] + org[0]
        if org[1] < 0:
            org[1] = shape[0] + org[1]
        org = (org[0], org[1])
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)  

    def draw_poly(self, img, points, closed, color, thickness=1):
        cv2.polylines(img, [np.array(points, np.int32)], closed, color, thickness)

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

    def process_video(self, img):
        return cv2.cvtColor(self.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)

    def process(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Image is BGR Colorspace
        img_h, img_w = img.shape[:2]        

        self.base_lines = None          

        if self.img_shape is None:
            self.img_shape = (img_w, img_h)

        if self.M is None:
            self.M, self.Minv = self.perspective_transform(
                vp = (640, 420),
                tl = (540, 450),
                bl = (30, 615),
                out = self.img_shape,
            )

        # self.view(self.warp(img))

        if False:
            img_sobel = self.sobel(img)
            img_color = self.color_threshold(img)
            zero_channel = np.zeros_like(img_color) # create a zero color channel
            # img = np.array(cv2.merge((zero_channel, img, zero_channel)),np.uint8) # make window pixels green
            warpage = np.dstack((img_sobel, img_color, zero_channel)) * 255 # making the original road pixels 3 color channels
            out = cv2.addWeighted(img_orig, 0.5, warpage.astype(np.uint8), 1, 0)
            return out


        # img_sobel = self.sobel(img)
        # print(img_sobel.dtype)
        #self.view(img_sobel)
        img_yellow = self.color_yellow(img)
        img_white = self.color_white(img)
        
        #img = self.combined(img)
        if True:
            # img_sobel = self.mask_image_binary(img_sobel, self.cfg.get('mask', None))
            img_yellow = self.mask_image_binary(img_yellow, self.cfg.get('mask', None))
            img_white = self.mask_image_binary(img_white, self.cfg.get('mask', None))

        if False:
            zero_channel = np.zeros_like(img_yellow) # create a zero color channel
            img_out = np.dstack((img_yellow, img_white, zero_channel)) * 255 # making the original road pixels 3 color channels
            # img_out = np.dstack((zero_channel, zero_channel, img_sobel)) * 255 # making the original road pixels 3 color channels
            img_out = cv2.addWeighted(img, 0.5, img_out, 0.8, 0)
            self.view(img_out)        

        img_combined = img_yellow | img_white

        if False:
            img_out = np.dstack((img_combined, img_combined, img_combined)) * 255 # making the original road pixels 3 color channels
            img_out = cv2.addWeighted(img, 0.5, img_out, 0.5, 0)
            self.view(img_out)        

        img_warped = self.warp(img_combined)

        # self.view(img_warped)
        # (fit_left, fit_right, out_img) = self.window_conv(img)

        min_points = 200

        fit_left, fit_right = self.fit_left, self.fit_right       
        num_left, num_right = 0, 0 
        # fit_left, num_left = None, 0
        # fit_right, num_right = None, 0


        if fit_left is not None:                
            fit_left, num_left = self.find_line_with_priors(img_warped, fit_left, margin=20)
            # print('left', self.fit_left, fit_left, num_left)
            # if num_left < min_points:
            #     print("Lost left lane 2")
            #     self.num_left = 0
            # else:
            #     self.fit_left = fit_left
            #     self.num_left = num_left

        if fit_right is not None:                
            fit_right, num_right = self.find_line_with_priors(img_warped, fit_right, margin=20)
            # print("right", self.fit_right, fit_right, num_right)
            # if num_right < min_points:
            #     print("Lost right lane 2")
            #     self.num_right = 0
            # else:
            #     self.fit_right = fit_right
            #     self.num_right = num_right

        # if fit_left[0] == fit_right[0]:
        #     print("Both lanes the same")
        #     fit_left, num_left = None, 0
        #     fit_right, num_right = None, 0


        if fit_left is None or fit_right is None:
            left_x, right_x = self.find_base_lines(img_warped)

            if fit_left is None:
                fit_left, num_left = self.find_line(img_warped, left_x)

            if fit_right is None:
                fit_right, num_right = self.find_line(img_warped, right_x)

        if num_left < min_points:
            print("Lost left")
            print(fit_left, num_left)
        else:
            self.fit_left = fit_left
            self.num_left = num_left

        if num_right < min_points:
            print("Lost Right")
            print(fit_right, num_right)
        else:
            self.fit_right = fit_right
            self.num_right = num_right


        # if num_left < min_points or num_right < min_points:
        #     print("find_lines")
        #     fit_left, fit_right, num_left, num_right = self.find_lines(img_warped)

        #     if num_left > min_points:
        #         if prev_left is None:
        #             prev_left = fit_left
        #         diff_left = prev_left - fit_left                
                    
        #         # print('diff_left', diff_left)                

        #         self.fit_left = fit_left
        #         self.num_left = num_left
        #     else:
        #         print("Lost left lane 1")
        #         self.num_left = 0

        #     if num_right > min_points:
        #         if prev_right is None:
        #             prev_right = fit_right
        #         diff_right = prev_right - fit_right
        #         # print('diff_right', diff_right)

        #         self.fit_right = fit_right
        #         self.num_right = num_right
        #     else:
        #         print("Lost right lane 1")                
        #         self.num_right = 0




            # self.plot_fits(img_warped, self.fit_left, self.fit_right)

        # self.fit_left, self.fit_right, self.num_left, self.num_right = self.find_lines_with_priors(img_warped, self.fit_left, self.fit_right, min_points=min_points)

        # (left_curve, right_curve) = self.curve_radius_px(self.fit_left, self.fit_right, 720)
        # print(left_curve, right_curve)
        
        # print(self.fit_left, self.fit_right)

        img_out = self.draw_unwarped(img, img_warped, self.fit_left, self.fit_right)

        if True:
            # Overlay img_combined
            z = np.zeros_like(img_combined)
            tmp = np.dstack((z, z, img_combined)) * 255 # making the original road pixels 3 color channels
            img_out = cv2.addWeighted(img_out, 0.5, tmp, 0.5, 0)
            # self.view(img_out)        

        # pipeline.view(out_img)
        # self.display_lanes(img, fit)

        # plt.plot(histogram)
        # plt.show()

        #self.draw_text(img, 'Hello, World', (-500, 100), 2.0, WHITE)
        #self.draw_poly(img, [(0,0), (0, 100), (100, 100), (100, 0)], False, WHITE)
        # self.view(out_img)
        return img_out

    def run(self, path):
        return self.process(self.load_image(path))

    def view(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()           

    def process_movie(self, path, out_path):
        #out_path = os.path.join('out_videos', path);
        print(path, out_path)
        # clip = VideoFileClip(path).subclip(0, 5)
        clip = VideoFileClip(path)
        out_clip = clip.fl_image(self.process_video)
        out_clip.write_videofile(out_path, audio=False)

def main(args):
    pipeline = Pipeline(PIPELINE_CFG)

    arg = args[0]
    if os.path.splitext(arg)[1] == '.jpg':
        pipeline.view(pipeline.run(arg))
    elif os.path.splitext(arg)[1] == '.mp4':
        pipeline.process_movie(arg, args[1])

if __name__ == '__main__':
    main(sys.argv[1:])