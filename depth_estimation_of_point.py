import cv2
import glob
import numpy as np
from tqdm import tqdm
from scipy.linalg import norm
from scipy import sum, average

global mouseX,mouseY

# Set up video capture
left_video = cv2.VideoCapture('L40.MP4')
right_video = cv2.VideoCapture('R40.MP4')

left_path = 'good_calib_data/vast_data/CameraInfoLeft.npz'
right_path = 'vast_data/CameraInfoRight.npz'
right_path = left_path

l_data = np.load(left_path)
r_data = np.load(right_path)

scale_percent = 7

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


def nothing(x):
    pass

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))
       global mouseX,mouseY
       mouseX,mouseY = x,y
cv2.setMouseCallback('WindowName', onMouse)

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')



# Get information about the videos
n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)),
               int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))
fps = int(left_video.get(cv2.CAP_PROP_FPS))

for _ in tqdm(range(n_frames)):
    # Grab the frames from their respective video streams
    for i in range(30):
        ok, left = left_video.read()
        _, right = right_video.read()

    if ok:
        #############
        h,  w = left.shape[:2]
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(l_data['cam_matrix'], l_data['distortion_coeff'], (w, h), 1, (w, h))
        img1_undistorted = cv2.undistort(left, l_data['cam_matrix'], l_data['distortion_coeff'], None, new_cam_matrix)
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(r_data['cam_matrix'], r_data['distortion_coeff'], (w, h), 1, (w, h))
        img2_undistorted = cv2.undistort(right, r_data['cam_matrix'], r_data['distortion_coeff'], None, new_cam_matrix)

        ###############
        new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(r_data['cam_matrix'], r_data['distortion_coeff'], (w, h), 1, (w, h))
        cam_mat_to_use = new_cam_matrix
        fx = cam_mat_to_use[0][0] #in px units
        fy = cam_mat_to_use[1][1] #in px units
        h,  w = right.shape[:2]
        W = 6.17
        H = 4.55
        Fx = fx * W / w
        Fy = fy * H / h

        # Convert from 'BGR' to 'RGB'
        gray_left = cv2.cvtColor(img1_undistorted, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img2_undistorted, cv2.COLOR_BGR2GRAY)

        width = int(gray_right.shape[1] * scale_percent / 100)
        height = int(gray_right.shape[0] * scale_percent / 100)
        dim = (width, height)

        img1_undistorted_gray = cv2.resize(gray_left, dim, interpolation = cv2.INTER_AREA)
        img2_undistorted_gray = cv2.resize(gray_right, dim, interpolation = cv2.INTER_AREA)
        img2_undistorted = cv2.resize(img2_undistorted, dim, interpolation = cv2.INTER_AREA)

        break


stereo = cv2.StereoSGBM_create(minDisparity= cv2.getTrackbarPos('minDisparity','Output'),
                                numDisparities = numDisparities,
                                blockSize = blockSize,
                                uniquenessRatio = uniquenessRatio,
                                speckleWindowSize = speckleWindowSize,
                                speckleRange = speckleRange,
                                disp12MaxDiff = disp12MaxDiff,
                                P1 = P1,#8*3*win_size**2,
                                P2 =P2) #32*3*win_size**2)

left_xy = (0,0)
cv2.namedWindow('image')
cv2.setMouseCallback('image',onMouse)
while(1):
    cv2.imshow('image',gray_left)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        left_xy = (mouseX,mouseY)
        break

search_box_size = 60

y_search = 200
x_search = 600
gray_left = cv2.rectangle(gray_left, (left_xy[0]-x_search, left_xy[1]-y_search), (left_xy[0]+x_search, left_xy[1]+y_search), (255,0,0), 2)
gray_left = cv2.rectangle(gray_left, (left_xy[0]-search_box_size, left_xy[1]-search_box_size), (left_xy[0]+search_box_size, left_xy[1]+search_box_size), (255,0,0), 2)
gray_left = cv2.circle(gray_left, left_xy, radius=10, color=(0, 0, 255), thickness=-1)
cv2.imshow('image',gray_left)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
        n_m, n_0 = compare_images(to_compare_against, current_crop)
        # print("Manhattan norm:", n_m, "/ per pixel:", n_m/normL.size)
        # print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/normL.size)
        if n_m < best_score:
            best_score = n_m
            best_x = current_x
            best_y = current_y

print('WOOT', best_score, best_x, best_y)

right_xy = (best_x, best_y)

# fx = 2.2321826473871864 in cm not px
baseline = 9.9 #Distance between the cameras [cm]
disp_x = abs(left_xy[0] - right_xy[0]) 
disp_y = abs(left_xy[1] - right_xy[1])
disparity = disp_x
print("Dx", disp_x)
print("Dy", disp_y)
print("DISP", disparity)
print("FX", fx) # IN px
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

# X
fontSize = 2.3
gray_right = gray_left
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

# gray_right = cv2.rectangle(gray_right, (left_xy[0]-x_search, left_xy[1]-y_search), (left_xy[0]+x_search, left_xy[1]+y_search), (255,0,0), 2)
# gray_right = cv2.rectangle(gray_right, (right_xy[0]-search_box_size, right_xy[1]-search_box_size), (right_xy[0]+search_box_size, right_xy[1]+search_box_size), (255,0,0), 2)
# gray_right = cv2.circle(gray_right, right_xy, radius=10, color=(0, 0, 255), thickness=-1)
cv2.imshow('image',gray_right)
cv2.waitKey(0)
