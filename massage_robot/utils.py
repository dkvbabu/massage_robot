
import numpy as np
import scipy


def get_intrinsics(pMatrix, width, height,fov=106.26):
    # 106.26
    aspect = width/height
    #fx = width / (2 * aspect * np.tan(np.radians(fov / 2)))
    #fy = height / (2 * np.tan(np.radians(fov / 2)))

    fx = width * pMatrix[0,0] /  (2 * 1 ) 
    fy = height * pMatrix[1,1] / 2
    #fx = fy * aspect

    #fx = width / (2 * aspect )
    #fy = height / (2 )

    cx = width / 2
    cy = height / 2
    K = np.eye(3)
    K[0,0]  = fx
    K[1,1]  = fy
    K[0,2]  = cx
    K[1,2]  = cy
    K[2,2]  = 1
    return K

def get_extrinsics(view_matrix):

    Tc = np.array([[1,  0,  0,  0],
                   [0,  -1,  0,  0],
                   [0,  0,  -1,  0],
                   [0,  0,  0,  1]])

    T = np.linalg.inv(view_matrix) @ Tc
    #T[3,1:] *= (-1)

    # for test
    T.T[:3,:3] = scipy.spatial.transform.Rotation.from_euler('xyz',[-np.pi,0,np.pi/2]).as_matrix()
    #print(T)
    return T

def get_extrinsics2(view_matrix):

    Tc = np.linalg.inv(np.array([[1,  0,  0,  0],
                   [0,  -1,  0,  0],
                   [0,  0,  -1,  0],
                   [0,  0,  0,  1]]))

    
    RT = np.linalg.inv(Tc @ (view_matrix.T))
    #T[3,1:] *= (-1)

    # for test
    print(RT)
    return RT