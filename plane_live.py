import pdb
import numpy as np
import cv2
import glob
from tqdm import tqdm
from scipy.linalg import norm
from scipy import sum, average
import math
import pickle
import matplotlib.path as mplPath

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
        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

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
        cv2.imshow('BOARD POSES',resize_images_and_combine(70, imgL, imgR))
    else:
        print('CANT SEE 6X4 CHESSBOARD IN BOTH LEFT AND RIGHT')

except:
    print("Failed")


########## BOARD POSE ##########
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
        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

        # Find the rotation and translation vectors.
        retL,rvecsL, tvecsL = cv2.solvePnP(objpL, cornersL, cameraMatrixL, distL)
        retR,rvecsR, tvecsR = cv2.solvePnP(objpR, cornersR, cameraMatrixR, distR)

        tvecs_realL = tvecsL *  sqr_size
        tvecs_realR = tvecsR *  sqr_size

        cv2.drawChessboardCorners(imgL, n_rows_and_cols, cornersL, retL)
        cv2.drawChessboardCorners(imgR, n_rows_and_cols, cornersR, retR)

        ####################################################### NEW THING
        # Converting rotation vector to rotation matrix
        # np_rodriguesL = np.asarray(rvecsL[:, :], np.float64)
        # rmatrixL = cv2.Rodrigues(np_rodriguesL)[0]
        # tvecs_realL = -np.matrix(rmatrixL).T * np.matrix(tvecs_realL)

        # np_rodriguesR = np.asarray(rvecsR[:, :], np.float64)
        # rmatrixR = cv2.Rodrigues(np_rodriguesR)[0]
        # tvecs_realR = -np.matrix(rmatrixR).T * np.matrix(tvecs_realR)
        #######################################################


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
        cv2.imshow('BOARD POSES',resize_images_and_combine(70, imgL, imgR))
        cv2.waitKey(5000)
    else:
        print('CANT SEE CHESSBOARD IN BOTH LEFT AND RIGHT')
    ######### END BOARD POSE ###########
except:
    pass
###### DEPTH ESTIMATION #######
first_corner = (0,0)
cv2.namedWindow('1')
cv2.setMouseCallback('1',onMouse)
while(1):
    cv2.imshow('1',gray_left)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('1'):
        first_corner = (mouseX,mouseY)
        break

second_corner = (0,0)
cv2.namedWindow('2')
cv2.setMouseCallback('2',onMouse)
while(1):
    cv2.imshow('2',gray_left)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('2'):
        second_corner = (mouseX,mouseY)
        break

third_corner = (0,0)
cv2.namedWindow('3')
cv2.setMouseCallback('3',onMouse)
while(1):
    cv2.imshow('3',gray_left)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('3'):
        third_corner = (mouseX,mouseY)
        break

fourth_corner = (0,0)
cv2.namedWindow('4')
cv2.setMouseCallback('4',onMouse)
while(1):
    cv2.imshow('4',gray_left)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('4'):
        fourth_corner = (mouseX,mouseY)
        break
cv2.destroyAllWindows()

selected_plane_corners = np.array([[first_corner[0], first_corner[1]],
                                    [second_corner[0], second_corner[1]],
                                    [fourth_corner[0], fourth_corner[1]],
                                    [third_corner[0], third_corner[1]]
                                    ])
smallest_x = 1000000000
smallest_y = 1000000000
biggest_x = 0
biggest_y = 0

for point in selected_plane_corners:
    if point[0] < smallest_x:
        smallest_x = point[0]
    if point[0] > biggest_x:
        biggest_x = point[0]

    if point[1] < smallest_y:
        smallest_y = point[1]
    if point[1] > biggest_y:
        biggest_y = point[1]
    print(point)

print('HERE', smallest_x, smallest_y, biggest_x, biggest_y)

poly_path = mplPath.Path(selected_plane_corners)

undistortedL = cv2.rectangle(undistortedL, (smallest_x, smallest_y), (biggest_x, biggest_y), (255, 255, 255), 2)

pixels_within_plane = []
for current_x in range(smallest_x, biggest_x):
    for current_y in range(smallest_y, biggest_y):
        if poly_path.contains_point((current_x, current_y)):
            undistortedL = cv2.circle(undistortedL, (current_x, current_y), radius=1, color=(0, 0, 255), thickness=-1)
            pixels_within_plane.append((current_x, current_y))

undistortedL = cv2.circle(undistortedL, first_corner, radius=1, color=(0, 0, 255), thickness=-1)
undistortedL = cv2.circle(undistortedL, second_corner, radius=1, color=(0, 0, 255), thickness=-1)
undistortedL = cv2.circle(undistortedL, third_corner, radius=1, color=(0, 0, 255), thickness=-1)
undistortedL = cv2.circle(undistortedL, fourth_corner, radius=1, color=(0, 0, 255), thickness=-1)
cv2.destroyAllWindows()
cv2.imshow('undistortedL',undistortedL)
cv2.waitKey(0)
cv2.destroyAllWindows()

all_coord_info = []
count = 0
every_n_coord = 5
pixels_within_plane = pixels_within_plane[::every_n_coord]
for coord in pixels_within_plane:
    count +=1
    print('Pixel', count, 'of', len(pixels_within_plane))
    left_xy = (coord[0], coord[1])

    side_bias = 1.5
    search_box_size_x = 30
    search_box_size_y = 30
    y_search = 10
    x_search = 220
    gray_left_cop = gray_left
    gray_left = cv2.rectangle(gray_left, (left_xy[0]-x_search, left_xy[1]-y_search), (left_xy[0]-int(x_search/side_bias), left_xy[1]+y_search), (255,0,0), 2)
    gray_left = cv2.rectangle(gray_left, (left_xy[0]-search_box_size_x, left_xy[1]-search_box_size_y), (left_xy[0]+search_box_size_x, left_xy[1]+search_box_size_y), (255,0,0), 2)
    gray_left = cv2.circle(gray_left, left_xy, radius=10, color=(0, 0, 255), thickness=-1)
    # cv2.imshow('image',gray_left)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('SEARCHING')

    # normalize to compensate for exposure difference, this may be unnecessary consider disabling it
    gray_left = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)
    normL = normalize(gray_left)
    normR = normalize(gray_right)

    best_score = 10000000000000000000000
    best_x = 0
    best_y = 0

    to_compare_against = normL[left_xy[1]-search_box_size_x:left_xy[1]+search_box_size_y, left_xy[0]-search_box_size_x:left_xy[0]+search_box_size_y]
    # cv2.imshow('image',to_compare_against)
    # cv2.waitKey(0)
    search_area = int(x_search/(side_bias*4))
    print('search_area', search_area)
    for x_change in range(search_area):
        current_x = left_xy[0] - x_search + x_change
        for y_change in range(int(y_search/2)):
            current_y = left_xy[1] - y_search + y_change

            current_crop = normR[current_y-search_box_size_x:current_y+search_box_size_y, current_x-search_box_size_x:current_x+search_box_size_y]
            if to_compare_against.shape != current_crop.shape:
                continue
            n_m, n_0 = compare_images(to_compare_against, current_crop)
            if n_m < best_score:
                best_score = n_m
                best_x = current_x
                best_y = current_y

    # print('WOOT', best_score, best_x, best_y)
    right_xy = (best_x, best_y)
    disp_x = abs(left_xy[0] - right_xy[0]) 
    disp_y = abs(left_xy[1] - right_xy[1])
    disparity = disp_x
    # print("Dx", disp_x)
    # print("Dy", disp_y)
    # print("DISP", disparity)
    # print("FX", fx) # IN px
    # print('BASE', baseline)
    depth = (baseline*fx)/disparity #depth in cm
    # print(depth, "CM")

    u = int(left_xy[0] - (w/2))
    v = int(left_xy[1] - (h/2))
    Z = depth
    X = (u * Z)/fx
    Y = -(v * Z)/fx
    # print("Z", depth/100, "M")
    # print("X", X/100, "M")
    # print("Y", Y/100, "M")

    font_disp = 300
    fontSize = 1.7
    # gray_right = gray_left #Just to keep the point visible
    # gray_right = cv2.circle(gray_right, (best_x, best_y), radius=10, color=(0, 0, 255), thickness=-1)
    # gray_right = cv2.rectangle(gray_right, (best_x-search_box_size, best_y-search_box_size), (best_x+search_box_size, best_y+search_box_size), (255,0,0), 2)
    # gray_right = cv2.putText(img = gray_right,text = "X: " + str(round(X/100, 2)) + "M",org = (10,100),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)
    # gray_right = cv2.putText(img = gray_right,text = "Y: " + str(round(Y/100, 2)) + "M",org = (10,150),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)
    # gray_right = cv2.putText(img = gray_right,text = "Z: " + str(round(Z/100, 2)) + "M",org = (10,200),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)

    # bpX = round((bxR+bxL)/2, 2)
    # bpY = round((byR+byL)/2, 2)

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


    X = round(X/100 - ((baseline/4)/100), 2)
    Y = round(Y/100 + ((baseline/4)/100), 2)
    Z = round(Z/100, 2)
    point_of_interest = "(Meters) PX: " + str(X) + " PY: " + str(Y) + " PZ: " + str(Z)
    all_coord_info.append([left_xy[0], left_xy[1], disparity, X, Y, Z])
    point_of_interest_in_LCS = np.array([[X],[Y],[Z]])
    location_of_grid_in_LCS = np.array([[bpX],[bpY],[bpZ]])
    bslocation_of_grid_in_LCS = np.array([[bsbpX],[bsbpY],[bsbpZ]])

print('SAVING')
np.savez('NEWer1REGESTall_plane_info.npz',
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

# rot_z = math.radians(bproll)
# rot_y = math.radians(bppitch)
# rot_x = math.radians(bpyaw)
# point_in_GCS = convert_from_LCS_to_GCS(point_of_interest_in_LCS, location_of_grid_in_LCS, rot_x, rot_y, rot_z)
# print('ORIG POINT:\n', point_of_interest_in_LCS)
# print('POINT IN GCS:\n', point_in_GCS, '\n')
# print(point_of_interest)
# print(bpPositon)
# print(bpOrientation)

# point_in_GCS_x = round(float(point_in_GCS[0]), 2)
# point_in_GCS_y = round(float(point_in_GCS[1]), 2)
# point_in_GCS_z = round(float(point_in_GCS[2]), 2)

# error = (abs(point_in_GCS_x) + abs(point_in_GCS_y) + abs(point_in_GCS_z))*100

# gray_left = cv2.putText(img = gray_left,text = point_of_interest, org = (10,100),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)
# gray_left = cv2.putText(img = gray_left,text = bpPositon, org = (10,150),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)
# gray_left = cv2.putText(img = gray_left,text = bpOrientation, org = (10,200),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)
# gray_left = cv2.putText(img = gray_left,text = '(Meters) point relative to grid: X: ' + str(point_in_GCS_x) + ', Y: ' + str(point_in_GCS_y) + ' Z: ' + str(point_in_GCS_z), org = (10,250),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)
# gray_left = cv2.putText(img = gray_left,text = '(CM) Error/Dist from GT: ' + str(error), org = (10,300),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = fontSize,color = (255, 255, 255),thickness = 2)



# boardposes = resize_images_and_combine(70, imgL, imgR)
# calculated_3d_point = cv2.cvtColor(resize_images_and_combine(70, gray_left, gray_right) ,cv2.COLOR_GRAY2BGR) 

# # cv2.imshow('image', np.concatenate((boardposes, calculated_3d_point), axis=0))

# gray_left = cv2.cvtColor(gray_left, cv2.COLOR_GRAY2BGR) 
# objpL = np.zeros((n_rows*n_cols,3), np.float32)
# objpL[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
# gray_left_cop = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
# retL, cornersL = cv2.findChessboardCorners(gray_left_cop, n_rows_and_cols,None)
# if retL == True:
#     print('TRUE')
#     cv2.drawChessboardCorners(gray_left, n_rows_and_cols, cornersL, retL)
#     retL,rvecsL, tvecsL = cv2.solvePnP(objpL, cornersL, cameraMatrixL, distL)
#     cv2.drawChessboardCorners(gray_left, n_rows_and_cols, cornersL, retL)
#     axisL = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
#     imgptsL, jacL = cv2.projectPoints(axisL, rvecsL, tvecsL, cameraMatrixL, distL)
#     gray_left = draw(gray_left,cornersL,imgptsL)

# cv2.imshow('Result', gray_left)
# cv2.waitKey(0)
