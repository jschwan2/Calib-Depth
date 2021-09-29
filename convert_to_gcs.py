import numpy as np
import math

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

    print('ORIG POINT:\n', point_of_interest_in_LCS)
    print('POINT IN GCS:\n', point_in_GCS, '\n')
    return point_in_GCS


point_of_interest_in_LCS = np.array([[4],[2],[0]])

location_of_grid_in_LCS = np.array([[-6],[1],[0]])

rot_z = math.radians(45)
rot_y = math.radians(0)
rot_x = math.radians(0)

convert_from_LCS_to_GCS(point_of_interest_in_LCS, location_of_grid_in_LCS, rot_x, rot_y, rot_z)





point_of_interest_in_LCS = np.array([[4],[2],[0]])

location_of_grid_in_LCS = np.array([[1],[2],[0]])

rot_z = math.radians(45)
rot_y = math.radians(0)
rot_x = math.radians(0)

convert_from_LCS_to_GCS(point_of_interest_in_LCS, location_of_grid_in_LCS, rot_x, rot_y, rot_z)