import pdb
import math
import numpy as np
import cv2
import glob
from itertools import permutations

def resize_images_and_combine(scale_percent, img1, img2):
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    return np.concatenate((resized_img1, resized_img2), axis=1)


def calibrate_rectify_undistort(objpoints, imgpointsL, imgpointsR, frameSize, grayL, grayR, imgL, imgR, to_save = False):
    ############## CALIBRATION #######################################################
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
    imgL_undistorted = cv2.undistort(imgL, cameraMatrixL, distL, None, newCameraMatrixL)

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
    imgR_undistorted = cv2.undistort(imgR, cameraMatrixR, distR, None, newCameraMatrixR)

    ########## Stereo Vision Calibration #############################################
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    ########## Stereo Rectification #################################################
    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

    undistortedL= cv2.remap(imgL, stereoMapL[0], stereoMapL[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(imgR, stereoMapR[0], stereoMapR[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    ###PRESSED ESC AND NOW SAVING
    if to_save == True:
        print("Saving parameters!")
        cv_file = cv2.FileStorage('stereoCalib.xml', cv2.FILE_STORAGE_WRITE)
        cv_file.write('newCameraMatrixL',newCameraMatrixL)
        cv_file.write('newCameraMatrixR',newCameraMatrixR)
        cv_file.write('stereoMapL_x',stereoMapL[0])
        cv_file.write('stereoMapL_y',stereoMapL[1])
        cv_file.write('stereoMapR_x',stereoMapR[0])
        cv_file.write('stereoMapR_y',stereoMapR[1])

        np.savez('CameraInfo.npz',
         cameraMatrixL=cameraMatrixL,
         cameraMatrixR=cameraMatrixR,
         distL=distL,
         distR=distR,
         tvecsL=tvecsL,
         tvecsR=tvecsR,
         rvecsL=rvecsL,
         rvecsR=rvecsR)

        cv_file.release()

    return undistortedL, undistortedR
    # return imgL_undistorted, imgR_undistorted


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
chessboardSize = (10,7)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

# capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

num = 0

last_calib_frame = None
while capL.isOpened():
    successL, imgL = capL.read()
    successR, imgR = capR.read()

    copL = imgL
    copR = imgR

    if successL and successR:
        # print(imgL.shape)
        # print(imgR.shape)

        l_and_r = resize_images_and_combine(100, copL, copR)
        if last_calib_frame is None:
            last_calib_frame = np.zeros(l_and_r.shape, dtype=np.uint8)

        cv2.imshow('l_and_r', np.concatenate((l_and_r, last_calib_frame), axis=0))

        k = cv2.waitKey(5)

        if k == 27:
            calibrate_rectify_undistort(objpoints, imgpointsL, imgpointsR, frameSize, grayL, grayR, imgL, imgR, to_save=True)
            break

        elif k == ord('s'):
            print('Testing frame')
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            frameSize = (grayL.shape[1], grayL.shape[0])
            # print(frameSize)
            # Find the chess board corners
            retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
            retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if retL and retR == True:
                print('VALID CALIB')

                objpoints.append(objp)

                cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                imgpointsL.append(cornersL)

                cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                imgpointsR.append(cornersR)



                # Draw and display the corners
                # cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
                # cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
                print(len(objpoints))
                imgL, imgR = calibrate_rectify_undistort(objpoints, imgpointsL, imgpointsR, frameSize, grayL, grayR, imgL, imgR)
                last_calib_frame = resize_images_and_combine(100, imgL, imgR)
                #cv2.imshow('img right', l_and_r)
                cv2.waitKey(1000)

cv2.destroyAllWindows()
