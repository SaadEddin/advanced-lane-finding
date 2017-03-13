### Project 4 - Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In project 1, a simple pipeline with multiple limitations has been applied. In this project, the goal is to provide a more robust lane detection using more advanced techniques such as color and gradient thresholding, polynomial fitting of 2nd degree to capture curvature in lanes, and apply camera calibration, perspective transform and image distortion/undistortion techniques.

The next sections will cover the steps above one by one: How the camera matrix - needed for calibration - has been obtained, and how it was used in the overall pipeline to find lanes. We will show the result of applying each step on test images, and finally, on a stream of frames from the project video.

#### Camera Calibration

We will use a set of test images of a chess board, to find the camera matix and distortion coeffs. The function `findChessboardCorners()` is used and it needs the number of horizontal and vertical corners in the chess board. For this section, I ran this method on the set of 20 chess board images with different value of nx and ny in the code below.

```python
# Number of inside corners along the x and y axis
nx = 9
ny = 6 

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

# Prepare object points in this format (0,0,0), (1,0,0), ..., (7,5,0)
objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Step through the list and search for chessboard corners
for index, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        imgs_.append(img)
```

For these values (nx = 9 and ny = 6) the `findChessboardCorners()` method returned 17 sets of corners. This is the maximum number of returned set of corners. The figure below shows the detected corners on one of the chessboard images.

![{chess_corners}](figs/chess1.png)

Now the detected corners will be used to get the camera matrix. We use the `CalibrateCamera()` method, which takes a set of points in the original, distorted image and their equivalents in the undistorted images, and applied the pin-hole camera model and projects the points using perspective transformation on an image plane. [More details here](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html).

```python
# Read image and retrieve the size
img = cv2.imread('../camera_cal/calibration1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
size = (gray.shape[1], gray.shape[0])
# Calibrate the camera using the derived object points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                   imgpoints, 
                                                   size, 
                                                   None, 
                                                   None)
# undistort the image using the derived calibration parameters
img_undist = cv2.undistort(img, mtx, dist, None, mtx)
```

The following two figures show how the calibration parameters are applied on two images: A chess board and a traffic image.

![{calibrated_chess}](figs/calibrated_chess.png)
![{calibrated_traffic}](figs/calibrated_traffic.png)
