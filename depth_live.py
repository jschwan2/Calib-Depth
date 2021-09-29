import pdb
import numpy as np
import cv2
import glob
from tqdm import tqdm
from scipy.linalg import norm
from scipy import sum, average

baseline = 21.2 #Distance between the cameras [cm]

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

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

while capL.isOpened():
    successL, imgL = capL.read()
    successR, imgR = capR.read()

    if successL and successR:
        #Undistorting and rectifying both left and right frames
        undistortedL= cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR= cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # print(undistortedR.shape)
        h,  w = undistortedR.shape[:2]
        # l_and_r = resize_images_and_combine(10, undistortedL, undistortedR)
        # cv2.imshow('l_and_r', l_and_r)


        ########## Getting pixel focal point
        cam_mat_to_use = newCameraMatrixL
        fx = cam_mat_to_use[0][0] #in px units
        cam_mat_to_use = newCameraMatrixR
        fx = cam_mat_to_use[0][0]
        ##########

        # Convert from 'BGR' to 'RGB'
        gray_left = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)

        # scale_percent = 30
        # width = int(gray_left.shape[1] * scale_percent / 100)
        # height = int(gray_left.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # resized_img1 = cv2.resize(gray_left, dim, interpolation = cv2.INTER_AREA)
        # resized_img2 = cv2.resize(gray_right, dim, interpolation = cv2.INTER_AREA)

        # disparity_image = depth_map(resized_img1, resized_img2)  # Get the disparity map

        # cv2.imshow('image',disparity_image)

        cv2.imshow('image', resize_images_and_combine(20, undistortedL, undistortedR))
        k = cv2.waitKey(20) & 0xFF

        if k == ord('a'):
            break

        if k == 27:
            exit()

print('SELECTED FRAME')

left_xy = (0,0)
cv2.namedWindow('image')
cv2.setMouseCallback('image',onMouse)
while(1):
    cv2.imshow('image',gray_left)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('b'):
        left_xy = (mouseX,mouseY)
        break

search_box_size = 60
y_search = 100
x_search = 400
gray_left = cv2.rectangle(gray_left, (left_xy[0]-x_search, left_xy[1]-y_search), (left_xy[0]+x_search, left_xy[1]+y_search), (255,0,0), 2)
gray_left = cv2.rectangle(gray_left, (left_xy[0]-search_box_size, left_xy[1]-search_box_size), (left_xy[0]+search_box_size, left_xy[1]+search_box_size), (255,0,0), 2)
gray_left = cv2.circle(gray_left, left_xy, radius=10, color=(0, 0, 255), thickness=-1)
cv2.imshow('image',gray_left)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('SEARCHING')

# normalize to compensate for exposure difference, this may be unnecessary consider disabling it
normL = normalize(gray_left)
normR = normalize(gray_right)

best_score = 10000000000000000000000
best_x = 0
best_y = 0

to_compare_against = normL[left_xy[1]-search_box_size:left_xy[1]+search_box_size, left_xy[0]-search_box_size:left_xy[0]+search_box_size]
# cv2.imshow('image',to_compare_against)
# cv2.waitKey(0)

for x_change in range(x_search*2):
    current_x = left_xy[0] - x_search + x_change
    for y_change in range(y_search*2):
        current_y = left_xy[1] - y_search + y_change

        current_crop = normR[current_y-search_box_size:current_y+search_box_size, current_x-search_box_size:current_x+search_box_size]
        if to_compare_against.shape != current_crop.shape:
            continue
        n_m, n_0 = compare_images(to_compare_against, current_crop)
        # print("Manhattan norm:", n_m, "/ per pixel:", n_m/normL.size)
        # print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/normL.size)
        if n_m < best_score:
            best_score = n_m
            best_x = current_x
            best_y = current_y

print('WOOT', best_score, best_x, best_y)
right_xy = (best_x, best_y)
disp_x = abs(left_xy[0] - right_xy[0]) 
disp_y = abs(left_xy[1] - right_xy[1])
disparity = disp_x
print("Dx", disp_x)
print("Dy", disp_y)
print("DISP", disparity)
print("FX", fx) # IN px
print('BASE', baseline)
depth = (baseline*fx)/disparity #depth in cm
print(depth, "CM")

u = int(left_xy[0] - (w/2))
v = int(left_xy[1] - (h/2))
Z = depth
X = (u * Z)/fx
Y = -(v * Z)/fx
print("Z", depth/2.54, "IN")
print("X", X/2.54, "IN")
print("Y", Y/2.54, "IN")

font_disp = 300

fontSize = 2.3
# gray_right = gray_left #Just to keep the point visible
gray_left = cv2.circle(gray_left, (best_x, best_y), radius=10, color=(0, 0, 255), thickness=-1)
gray_right = cv2.rectangle(gray_right, (best_x-search_box_size, best_y-search_box_size), (best_x+search_box_size, best_y+search_box_size), (255,0,0), 2)
# X
gray_right = cv2.putText(
  img = gray_right,
  text = "X: " + str(round(X/2.54, 2)) + "IN",
  org = (10,100),#(int(right_xy[0]-(font_disp/2)), int(right_xy[1]-font_disp)),
  fontFace = cv2.FONT_HERSHEY_DUPLEX,
  fontScale = fontSize,
  color = (255, 255, 255),
  thickness = 2
)

# Y
gray_right = cv2.putText(
  img = gray_right,
  text = "Y: " + str(round(Y/2.54, 2)) + "IN",
  org = (10,150),#(int(right_xy[0]-(font_disp/2)), int(right_xy[1]-font_disp)),
  fontFace = cv2.FONT_HERSHEY_DUPLEX,
  fontScale = fontSize,
  color = (255, 255, 255),
  thickness = 2
)


# Z
gray_right = cv2.putText(
  img = gray_right,
  text = "Z: " + str(round(depth/2.54, 2)) + "IN", #+ str(round(Z, 2)) + "CM, " + 
  org = (10,200),#(int(right_xy[0]-(font_disp/2)), int(right_xy[1]-font_disp)),
  fontFace = cv2.FONT_HERSHEY_DUPLEX,
  fontScale = fontSize,
  color = (255, 255, 255),
  thickness = 2
)

cv2.imshow('image', resize_images_and_combine(30, gray_left, gray_right))
cv2.waitKey(0)
