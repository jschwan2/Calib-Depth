import pdb
import numpy as np
import cv2
import glob
from tqdm import tqdm
from scipy.linalg import norm
from scipy import sum, average
import pickle
import matplotlib.path as mplPath
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def shortest_distance(x1, y1, z1, a, b, c, d):
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    # print("Perpendicular distance is", d/e)
    return d/e

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
n_rows = 10
n_cols = 7
n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
n_rows_and_cols = (n_rows, n_cols)
sqr_size = 0.025 # 25mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objpL = np.zeros((n_rows*n_cols,3), np.float32)
objpL[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
objpR = np.zeros((n_rows*n_cols,3), np.float32)
objpR[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)


baseline = 21.8 #Distance between the cameras [cm]

CameraInfo = np.load('CameraInfo.npz')

cameraMatrixL = CameraInfo['cameraMatrixL']
distL = CameraInfo['distL']
rvecsL = CameraInfo['rvecsL']
tvecsL = CameraInfo['tvecsL']

cameraMatrixR = CameraInfo['cameraMatrixR']
distR = CameraInfo['distR']
rvecsR = CameraInfo['rvecsR']
tvecsR = CameraInfo['tvecsR']

cv_file = cv2.FileStorage('stereoCalib.xml', cv2.FILE_STORAGE_READ)

newCameraMatrixL = cv_file.getNode("newCameraMatrixL").mat()
newCameraMatrixR = cv_file.getNode("newCameraMatrixR").mat()
stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

cv_file.release()

capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

# capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

#3D Rotation Matrices - notes: theta is expected in radians
def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                    [ 0, math.cos(theta),-math.sin(theta)],
                    [ 0, math.sin(theta), math.cos(theta)]])

def Ry(theta):
    return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-math.sin(theta), 0, math.cos(theta)]])
  
def Rz(theta):
    return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                    [ math.sin(theta), math.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])

#LCS = local coordinate system
#GCS = global coordinate system (relative to that of grid with known lat and lon)
def convert_from_LCS_to_GCS(point_of_interest_in_LCS, location_of_grid_in_LCS, rot_x, rot_y, rot_z):
    #TRANSLATING ORIGIN OF LCS TO GCS
    translated_point_of_interest_to_GCS_origin = (point_of_interest_in_LCS[0]-location_of_grid_in_LCS[0], 
                                              point_of_interest_in_LCS[1]-location_of_grid_in_LCS[1],
                                              point_of_interest_in_LCS[2]-location_of_grid_in_LCS[2])
    R = Rz(rot_z) * Ry(rot_y) * Rx(rot_x)

    #ROTATING TO FIND POINT OF INTEREST IN GCS
    point_in_GCS = R * translated_point_of_interest_to_GCS_origin

    return point_in_GCS

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def nothing(x):
    pass

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))
       global mouseX,mouseY
       mouseX,mouseY = x,y

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def compare_images(img1, img2):
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def resize_images_and_combine(scale_percent, img1, img2):
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    return np.concatenate((resized_img1, resized_img2), axis=1)

while capL.isOpened():
    successL, imgL = capL.read()
    successR, imgR = capR.read()

    if successL and successR:
        #Undistorting and rectifying both left and right frames
        undistortedL= cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR= cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        h,  w = undistortedR.shape[:2]

        ########## Getting pixel focal point
        cam_mat_to_use = newCameraMatrixL
        fx = cam_mat_to_use[0][0] #in px units
        cam_mat_to_use = newCameraMatrixR
        fx = cam_mat_to_use[0][0]
        ##########

        #FOR BOARD POSE
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        #END BP

        # Convert from 'BGR' to 'RGB'
        gray_left = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)

        cv2.imshow('image', resize_images_and_combine(70, undistortedL, undistortedR))
        k = cv2.waitKey(20) & 0xFF

        if k == ord('a'):
            break

        if k == 27:
            exit()

print('SELECTED FRAME')

try:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_rows = 6
    n_cols = 4
    n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
    n_rows_and_cols = (n_rows, n_cols)
    sqr_size = 0.040 # 40mm
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpL = np.zeros((n_rows*n_cols,3), np.float32)
    objpL[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    objpR = np.zeros((n_rows*n_cols,3), np.float32)
    objpR[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    ################
    ########## BOARD POSE ##########
    retL, cornersL = cv2.findChessboardCorners(grayL, n_rows_and_cols,None)
    retR, cornersR = cv2.findChessboardCorners(grayR, n_rows_and_cols,None)

    #### If found, add object points, image points (after refining them)
    if retL == True:
        print('GOT BOTH')

        # cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        # cv2.drawChessboardCorners(imgR, n_

        # Find the rotation and translation vectors.
        retL,rvecsL, tvecsL = cv2.solvePnP(objpL, cornersL, cameraMatrixL, distL)
        retR,rvecsR, tvecsR = cv2.solvePnP(objpR, cornersR, cameraMatrixR, distR)

        tvecs_realL = tvecsL *  sqr_size
        tvecs_realR = tvecsR *  sqr_size

        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

        #Left
        axisL = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsL, jacL = cv2.projectPoints(axisL, rvecsL, tvecsL, cameraMatrixL, distL)
        imgL = draw(imgL,cornersL,imgptsL)
        bxL, byL, bzL = tvecs_realL
        bsbxL = round(bxL[0], 3)
        bsbyL = round(byL[0], 3)
        bsbzL = round(bzL[0], 3)
        mystrL = "xL: " + str(bxL) + " yL: " + str(byL) + " zL: " + str(bzL)
        print(mystrL)
        cv2.putText(imgL,mystrL, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollL, pitchL, yawL = rvecsL
        bsrollL = round(math.degrees(rollL), 2)
        bspitchL = round(math.degrees(pitchL), 2)
        bsyawL = round(math.degrees(yawL), 2)
        oristrdegL = "L roll: " + str(rollL) + " pitch: " + str(pitchL) + " yaw: " + str(yawL) + ""
        cv2.putText(imgL,oristrdegL, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        # #Right

        axisR = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsR, jacR = cv2.projectPoints(axisR, rvecsR, tvecsR, cameraMatrixR, distR)
        imgR = draw(imgR,cornersR,imgptsR)
        bxR, byR, bzR = tvecs_realR
        bsbxR = round(bxR[0], 3)
        bsbyR = round(byR[0], 3)
        bsbzR = round(bzR[0], 3)
        mystrR = "xR: " + str(bxR) + " yR: " + str(byR) + " zR: " + str(bzR)
        print(mystrR)
        cv2.putText(imgR, mystrR, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollR, pitchR, yawR = rvecsR
        bsrollR = round(math.degrees(rollR), 2)
        bspitchR = round(math.degrees(pitchR), 2)
        bsyawR = round(math.degrees(yawR), 2)
        oristrdegR = "R roll: " + str(rollR) + " pitch: " + str(pitchR) + " yaw: " + str(yawR) + ""
        cv2.putText(imgR,oristrdegR, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        
        #VIEW BOTH
        # cv2.imshow('BOARD POSES',resize_images_and_combine(70, imgL, imgR))
    else:
        print('CANT SEE 6X4 CHESSBOARD IN BOTH LEFT AND RIGHT')

except:
    print("Failed")


########## BOARD POSE ##########

corners_for_left = []
corners_for_right = []

try:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_rows = 10
    n_cols = 7
    n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
    n_rows_and_cols = (n_rows, n_cols)
    sqr_size = 0.025 # 40mm
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpL = np.zeros((n_rows*n_cols,3), np.float32)
    objpL[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    objpR = np.zeros((n_rows*n_cols,3), np.float32)
    objpR[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    ################
    ########## BOARD POSE ##########
    retL, cornersL = cv2.findChessboardCorners(grayL, n_rows_and_cols,None)
    retR, cornersR = cv2.findChessboardCorners(grayR, n_rows_and_cols,None)

    #### If found, add object points, image points (after refining them)
    if retL == True:

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        corners_for_left = cornersL
        corners_for_right = cornersR

        # Find the rotation and translation vectors.
        retL,rvecsL, tvecsL = cv2.solvePnP(objpL, cornersL, cameraMatrixL, distL)
        retR,rvecsR, tvecsR = cv2.solvePnP(objpR, cornersR, cameraMatrixR, distR)

        tvecs_realL = tvecsL *  sqr_size
        tvecs_realR = tvecsR *  sqr_size

        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

        #Left
        axisL = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsL, jacL = cv2.projectPoints(axisL, rvecsL, tvecsL, cameraMatrixL, distL)
        imgL = draw(imgL,cornersL,imgptsL)
        bxL, byL, bzL = tvecs_realL
        bxL = round(bxL[0], 3)
        byL = round(byL[0], 3)
        bzL = round(bzL[0], 3)
        mystrL = "xL: " + str(bxL) + " yL: " + str(byL) + " zL: " + str(bzL)
        print(mystrL)
        cv2.putText(imgL,mystrL, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollL, pitchL, yawL = rvecsL
        rollL = round(math.degrees(rollL), 2)
        pitchL = round(math.degrees(pitchL), 2)
        yawL = round(math.degrees(yawL), 2)
        oristrdegL = "L roll: " + str(rollL) + " pitch: " + str(pitchL) + " yaw: " + str(yawL) + ""
        cv2.putText(imgL,oristrdegL, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        # #Right

        axisR = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsR, jacR = cv2.projectPoints(axisR, rvecsR, tvecsR, cameraMatrixR, distR)
        imgR = draw(imgR,cornersR,imgptsR)
        bxR, byR, bzR = tvecs_realR
        bxR = round(bxR[0], 3)
        byR = round(byR[0], 3)
        bzR = round(bzR[0], 3)
        mystrR = "xR: " + str(bxR) + " yR: " + str(byR) + " zR: " + str(bzR)
        print(mystrR)
        cv2.putText(imgR, mystrR, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollR, pitchR, yawR = rvecsR
        rollR = round(math.degrees(rollR), 2)
        pitchR = round(math.degrees(pitchR), 2)
        yawR = round(math.degrees(yawR), 2)
        oristrdegR = "R roll: " + str(rollR) + " pitch: " + str(pitchR) + " yaw: " + str(yawR) + ""
        cv2.putText(imgR,oristrdegR, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        
        #VIEW BOTH
        # cv2.imshow('BOARD POSES',resize_images_and_combine(70, imgL, imgR))
        cv2.waitKey(5000)
    else:
        print('CANT SEE CHESSBOARD IN BOTH LEFT AND RIGHT')
    ######### END BOARD POSE ###########
except:
    pass


########## UNDISTORTED:
try:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_rows = 6
    n_cols = 4
    n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
    n_rows_and_cols = (n_rows, n_cols)
    sqr_size = 0.040 # 40mm
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpL = np.zeros((n_rows*n_cols,3), np.float32)
    objpL[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    objpR = np.zeros((n_rows*n_cols,3), np.float32)
    objpR[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    ################
    ########## BOARD POSE ##########
    retL, cornersL = cv2.findChessboardCorners(grayL, n_rows_and_cols,None)
    retR, cornersR = cv2.findChessboardCorners(grayR, n_rows_and_cols,None)

    #### If found, add object points, image points (after refining them)
    if retL == True:
        print('GOT BOTH')

        # cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        # cv2.drawChessboardCorners(imgR, n_

        # Find the rotation and translation vectors.
        retL,rvecsL, tvecsL = cv2.solvePnP(objpL, cornersL, cameraMatrixL, distL)
        retR,rvecsR, tvecsR = cv2.solvePnP(objpR, cornersR, cameraMatrixR, distR)

        tvecs_realL = tvecsL *  sqr_size
        tvecs_realR = tvecsR *  sqr_size

        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

        #Left
        axisL = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsL, jacL = cv2.projectPoints(axisL, rvecsL, tvecsL, cameraMatrixL, distL)
        imgL = draw(imgL,cornersL,imgptsL)
        bxL, byL, bzL = tvecs_realL
        bsbxL = round(bxL[0], 3)
        bsbyL = round(byL[0], 3)
        bsbzL = round(bzL[0], 3)
        mystrL = "xL: " + str(bxL) + " yL: " + str(byL) + " zL: " + str(bzL)
        print(mystrL)
        cv2.putText(imgL,mystrL, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollL, pitchL, yawL = rvecsL
        bsrollL = round(math.degrees(rollL), 2)
        bspitchL = round(math.degrees(pitchL), 2)
        bsyawL = round(math.degrees(yawL), 2)
        oristrdegL = "L roll: " + str(rollL) + " pitch: " + str(pitchL) + " yaw: " + str(yawL) + ""
        cv2.putText(imgL,oristrdegL, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        # #Right

        axisR = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsR, jacR = cv2.projectPoints(axisR, rvecsR, tvecsR, cameraMatrixR, distR)
        imgR = draw(imgR,cornersR,imgptsR)
        bxR, byR, bzR = tvecs_realR
        bsbxR = round(bxR[0], 3)
        bsbyR = round(byR[0], 3)
        bsbzR = round(bzR[0], 3)
        mystrR = "xR: " + str(bxR) + " yR: " + str(byR) + " zR: " + str(bzR)
        print(mystrR)
        cv2.putText(imgR, mystrR, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollR, pitchR, yawR = rvecsR
        bsrollR = round(math.degrees(rollR), 2)
        bspitchR = round(math.degrees(pitchR), 2)
        bsyawR = round(math.degrees(yawR), 2)
        oristrdegR = "R roll: " + str(rollR) + " pitch: " + str(pitchR) + " yaw: " + str(yawR) + ""
        cv2.putText(imgR,oristrdegR, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        
        #VIEW BOTH
        # cv2.imshow('BOARD POSES',resize_images_and_combine(70, imgL, imgR))
    else:
        print('CANT SEE 6X4 CHESSBOARD IN BOTH LEFT AND RIGHT')

except:
    print("Failed")


########## BOARD POSE ##########
grayL = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)

corners_for_left = []
corners_for_right = []

try:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_rows = 10
    n_cols = 7
    n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
    n_rows_and_cols = (n_rows, n_cols)
    sqr_size = 0.025 # 40mm
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpL = np.zeros((n_rows*n_cols,3), np.float32)
    objpL[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    objpR = np.zeros((n_rows*n_cols,3), np.float32)
    objpR[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
    ################
    ########## BOARD POSE ##########
    retL, cornersL = cv2.findChessboardCorners(grayL, n_rows_and_cols,None)
    retR, cornersR = cv2.findChessboardCorners(grayR, n_rows_and_cols,None)

    #### If found, add object points, image points (after refining them)
    if retL == True:

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        corners_for_left = cornersL
        corners_for_right = cornersR

        # Find the rotation and translation vectors.
        retL,rvecsL, tvecsL = cv2.solvePnP(objpL, cornersL, cameraMatrixL, distL)
        retR,rvecsR, tvecsR = cv2.solvePnP(objpR, cornersR, cameraMatrixR, distR)

        tvecs_realL = tvecsL *  sqr_size
        tvecs_realR = tvecsR *  sqr_size

        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

        #Left
        axisL = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsL, jacL = cv2.projectPoints(axisL, rvecsL, tvecsL, cameraMatrixL, distL)
        imgL = draw(imgL,cornersL,imgptsL)
        bxL, byL, bzL = tvecs_realL
        bxL = round(bxL[0], 3)
        byL = round(byL[0], 3)
        bzL = round(bzL[0], 3)
        mystrL = "xL: " + str(bxL) + " yL: " + str(byL) + " zL: " + str(bzL)
        print(mystrL)
        cv2.putText(imgL,mystrL, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollL, pitchL, yawL = rvecsL
        rollL = round(math.degrees(rollL), 2)
        pitchL = round(math.degrees(pitchL), 2)
        yawL = round(math.degrees(yawL), 2)
        oristrdegL = "L roll: " + str(rollL) + " pitch: " + str(pitchL) + " yaw: " + str(yawL) + ""
        cv2.putText(imgL,oristrdegL, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        # #Right

        axisR = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        imgptsR, jacR = cv2.projectPoints(axisR, rvecsR, tvecsR, cameraMatrixR, distR)
        imgR = draw(imgR,cornersR,imgptsR)
        bxR, byR, bzR = tvecs_realR
        bxR = round(bxR[0], 3)
        byR = round(byR[0], 3)
        bzR = round(bzR[0], 3)
        mystrR = "xR: " + str(bxR) + " yR: " + str(byR) + " zR: " + str(bzR)
        print(mystrR)
        cv2.putText(imgR, mystrR, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        rollR, pitchR, yawR = rvecsR
        rollR = round(math.degrees(rollR), 2)
        pitchR = round(math.degrees(pitchR), 2)
        yawR = round(math.degrees(yawR), 2)
        oristrdegR = "R roll: " + str(rollR) + " pitch: " + str(pitchR) + " yaw: " + str(yawR) + ""
        cv2.putText(imgR,oristrdegR, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        
        #VIEW BOTH
        # cv2.imshow('BOARD POSES',resize_images_and_combine(70, imgL, imgR))
        cv2.waitKey(5000)
    else:
        print('CANT SEE CHESSBOARD IN BOTH LEFT AND RIGHT')
    ######### END BOARD POSE ###########
except:
    pass
###################### UNDIST END





all_coord_info = []
count = 0
for corner_num in range(len(corners_for_left)):
    current_corner_L = corners_for_left[corner_num]
    current_corner_R = corners_for_right[corner_num]

    left_xy = (current_corner_L[0][0], current_corner_L[0][1])
    right_xy = (current_corner_R[0][0], current_corner_R[0][1])

    disp_x = abs(left_xy[0] - right_xy[0]) 
    disp_y = abs(left_xy[1] - right_xy[1])
    disparity = disp_x
    depth = (baseline*fx)/disparity #depth in cm

    u = int(left_xy[0] - (w/2))
    v = int(left_xy[1] - (h/2))
    Z = depth
    X = (u * Z)/fx
    Y = -(v * Z)/fx

    font_disp = 300
    fontSize = 1.7
    try:
        bsbpX = round(bsbxL, 2)
        bsbpY = -round(bsbyL, 2)
        bsbpZ = round((bsbzR+bsbzL)/2, 2)
        bpX = round(bxL, 2)
        bpY = -round(byL, 2)
        bpZ = round((bzR+bzL)/2, 2)
        bpPositon = "(Meters) BX: " + str(bpX) + " BY: " + str(bpY) + " BZ: " + str(bpZ)

        bsbproll = round((bsrollL+bsrollR)/2, 2)
        bsbppitch = round((bspitchL+bspitchR)/2, 2)
        bsbpyaw = round((bsyawL+bsyawR)/2, 2)

        bproll = round((rollL+rollR)/2, 2)
        bppitch = round((pitchL+pitchR)/2, 2)
        bpyaw = round((yawL+yawR)/2, 2)
        bpOrientation = "(Degrees) R: " + str(bproll) + " P: " + str(bppitch) + "Y: " + str(bpyaw)
        
        location_of_grid_in_LCS = np.array([[bpX],[bpY],[bpZ]])
        bslocation_of_grid_in_LCS = np.array([[bsbpX],[bsbpY],[bsbpZ]])
    except:
        print('NO GRID INFO: FAILED')

    X = round(X/100 - ((baseline/4)/100), 2)
    Y = round(Y/100 + ((baseline/4)/100), 2)
    Z = round(Z/100, 2)
    point_of_interest = "(Meters) PX: " + str(X) + " PY: " + str(Y) + " PZ: " + str(Z)
    all_coord_info.append([left_xy[0], left_xy[1], disparity, X, Y, Z])
    point_of_interest_in_LCS = np.array([[X],[Y],[Z]])
    
try:
    print('SAVING')
    np.savez('board_corner_info.npz',
        fx = fx,
        baseline = baseline,
        board_position=location_of_grid_in_LCS,
        board_roll=bproll,
        board_yaw=bpyaw,
        board_pitch=bppitch,
        bsboard_position=bslocation_of_grid_in_LCS,
        bsboard_roll=bsbproll,
        bsboard_yaw=bsbpyaw,
        bsboard_pitch=bsbppitch,
        all_coord_info=all_coord_info)
    print('SAVED')
except:
    print('FAILED TO SAVE')

all_plane_info = np.load('board_corner_info.npz')
fx = all_plane_info['fx']
baseline = all_plane_info['baseline']
board_position = all_plane_info['board_position']
board_roll = all_plane_info['board_roll']
board_yaw = all_plane_info['board_yaw']
board_pitch = all_plane_info['board_pitch']
bsboard_position = all_plane_info['bsboard_position']
bsboard_roll = all_plane_info['bsboard_roll']
bsboard_yaw = all_plane_info['bsboard_yaw']
bsboard_pitch = all_plane_info['bsboard_pitch']
all_coord_info = all_plane_info['all_coord_info']

xs = []
ys = []
zs = []


coords_to_use = []
for coord in all_coord_info[::1]:
    left_x = coord[0]
    left_y = coord[1]
    disparity = coord[2]
    X = coord[3]
    Y = coord[4]
    Z = coord[5]
    # if Z < 2.00 and Z > 1.90:
    coords_to_use.append([X*100, Y*100, Z*100])

smallest_x = 10000
smallest_y = 10000
smallest_z = 10000
coord_to_sub = [0,0,0]
for coord in coords_to_use:
    if coord[0] < smallest_x:
        smallest_x = coord[0]
        coord_to_sub[0] = coord[0]
    if coord[1] < smallest_y:
        smallest_y = coord[1]
        coord_to_sub[1] = coord[1]
    if coord[2] < smallest_z:
        smallest_z = coord[2]
        coord_to_sub[2] = coord[2]


for coord in coords_to_use:
    xs.append(coord[0]-coord_to_sub[0])
    ys.append(coord[1]-coord_to_sub[1])
    zs.append(coord[2]-coord_to_sub[2])
    # xs.append(coord[0])
    # ys.append(coord[1])
    # zs.append(coord[2])
print(coords_to_use)

points = [xs, ys, zs]
# subtract out the centroid and take the SVD
svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

# Extract the left singular vectors
left = svd[0]
normal = left[:, -1]

print(left)
print(normal)

point  = np.array([0,0,0])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
a = normal[0]
b = normal[1]
c = normal[2]
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(23), range(20))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

all_dists = []
for i in range(len(xs)):
    distance_from_pt = shortest_distance(xs[i], ys[i], zs[i], a, b, c, d)
    all_dists.append(distance_from_pt)

print("AVG:", sum(all_dists)/len(all_dists))
print("MAX:", max(all_dists))

########
# Create the figure
fig = plt.figure()
# Add an axes
ax = fig.add_subplot(111,projection='3d')
# plot the surface
ax.plot_surface(xx, yy, z, alpha=0.8)
# and plot the point 
ax.scatter(xs , ys , zs,  color='green')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

X = np.array(xs)
Y = np.array(ys)
Z = np.array(zs)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

plt.grid()
plt.show()



