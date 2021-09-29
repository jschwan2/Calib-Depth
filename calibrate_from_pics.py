import cv2
import glob
import numpy as np

# Define image format and chessboard dimensions
image_path_format = '/Users/jonathanschwan/Desktop/StereoVision-master/calib2images/calibration/left/*.png'
save_path = 'vast_data/CameraInfoLeft.npz'
test_img = image_path_format[:-5] + "frame100.png"
print(test_img)
# chessboard_size = (9, 6)
chessboard_size = (10,7)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_pts = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
obj_pts[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points (3D) and image points (2D) from all the images.
obj_points = []
img_points = []

# Get list of image paths
images = glob.glob(image_path_format)

usable_count = 0

for img_name in images:
    print(img_name)
    img = cv2.imread(img_name)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_g, (chessboard_size[0], chessboard_size[1]), None)
    if ret:
        usable_count+=1
        obj_points.append(obj_pts)
        # Refine the corner location
        corners2 = cv2.cornerSubPix(img_g, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        print('Usable calib pics:', usable_count)

        # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(100)
    # cv2.destroyAllWindows()

image_path_format = '/Users/jonathanschwan/Desktop/StereoVision-master/calib3images/calibration/left/*.png'
images = glob.glob(image_path_format)
for img_name in images:
    print("NEW", img_name)
    img = cv2.imread(img_name)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_g, (chessboard_size[0], chessboard_size[1]), None)
    if ret:
        usable_count+=1
        obj_points.append(obj_pts)
        # Refine the corner location
        corners2 = cv2.cornerSubPix(img_g, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        print('Usable calib pics:', usable_count)


# image_path_format = '/Users/jonathanschwan/Desktop/StereoVision-master/calib1images/calibration/left/*.png'
# images = glob.glob(image_path_format)
# for img_name in images:
#     print("NEWest", img_name)
#     img = cv2.imread(img_name)
#     img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(img_g, (chessboard_size[0], chessboard_size[1]), None)
#     if ret:
#         usable_count+=1
#         obj_points.append(obj_pts)
#         # Refine the corner location
#         corners2 = cv2.cornerSubPix(img_g, corners, (11, 11), (-1, -1), criteria)
#         img_points.append(corners2)
#         print('Usable calib pics:', usable_count)



# Calibrate camera using both 2D and 3D points
ret, cam_matrix, distortion_coeff, rotation_vecs, translation_vecs = cv2.calibrateCamera(obj_points,
                                                                                         img_points,
                                                                                         img_g.shape[::-1],
                                                                                         None,
                                                                                         None)

# Save camera calibration information for later
np.savez(save_path,
         cam_matrix=cam_matrix,
         distortion_coeff=distortion_coeff,
         rotation_vecs=rotation_vecs,
         translation_vecs=translation_vecs)

# Calculate the re-projection error
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rotation_vecs[i], translation_vecs[i],
                                       cam_matrix, distortion_coeff)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error

print('Total Error: {:.4f}%'.format((100*mean_error) / len(obj_points)))


img = cv2.imread(test_img)
h,  w = img.shape[:2]

new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, distortion_coeff, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, cam_matrix, distortion_coeff, None, new_cam_matrix)

cv2.imwrite('calib2undistorted_left.png', dst)

#cropped the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calib2undistorted_left_cropped.png', dst)
