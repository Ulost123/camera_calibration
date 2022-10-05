import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('C:/Users/PC/OneDrive/desktop/YUJIN/2022-2/Convergence/Calibration/check/*.jpg')

for fname in images:    
    img = cv.imread(fname)
    img = cv.resize(img, (512, 512))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (10, 7), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (10,7), corners2, ret)
        cv.imshow('img', img)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('ret : ', ret, '\nmtx : ', mtx, '\ndist : ', dist, '\nrvecs : ', rvecs, '\ntvecs : ', tvecs)

img_2 = cv.imread('C:/Users/PC/OneDrive/desktop/YUJIN/2022-2/Convergence/Calibration/img/cat.jpg')
img_2 = cv.resize(img_2, (512, 512))
cv.imwrite('C:/Users/PC/OneDrive/desktop/YUJIN/2022-2/Convergence/Calibration/img/cat_512.jpg', img_2)
h, w = img.shape[:2]

new_cam_mat, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)

dst = cv.undistort(img_2, mtx, dist)
dst2 = cv.undistort(img_2, mtx, dist, None, new_cam_mat)

cv.imwrite('C:/Users/PC/OneDrive/desktop/YUJIN/2022-2/Convergence/Calibration/img/test_cat.jpg', dst)
cv.imwrite('C:/Users/PC/OneDrive/desktop/YUJIN/2022-2/Convergence/Calibration/img/test_cat2.jpg', dst2)
