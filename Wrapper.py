import cv2
import numpy as np
import argparse
import glob
import math
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def read_image(images):
  image = [cv2.imread(file)
          for file in images]
  return image


def generate_extrinsic_params(K, H):

    l1 = np.linalg.norm(np.matmul(np.linalg.inv(K), H[:, 0]))
    l2 = np.linalg.norm(np.matmul(np.linalg.inv(K), H[:, 1]))
    mean_l1l2 = (np.linalg.norm(l1)+np.linalg.norm(l2))/2
    world_to_camera = np.matmul(np.linalg.inv(K), H)

    det = np.linalg.det(world_to_camera)
    if det < 0:
        E = -world_to_camera/mean_l1l2
    elif det >= 0:
        E = world_to_camera/mean_l1l2
    r1 = E[:, 0]
    r2 = E[:, 1]

    r3 = np.cross(r1, r2)
    t = E[:, 2]

    orthagonal_basis = np.array([r1, r2, r3]).T
    u, s, v = np.linalg.svd(orthagonal_basis)
    R = np.matmul(u, v)

    ext_m = np.hstack((R, t.reshape(3, 1)))
    return ext_m


def generate_param_vec(A, k1, k2):
    intrinsics_reshaped = np.reshape(np.array([A[0][0], 0, A[0][2], A[1][1], A[1][2]]), (5, 1))

    distortion_params = np.array([k1, k2]).reshape(2, 1)
    parameter_vec = np.concatenate([distortion_params, intrinsics_reshaped])
    return parameter_vec


def vectorv(i, j, H):
    i = i-1
    j = j-1
    vij = [H[0, i]*H[0, j], H[0, i]*H[1, j] + H[1, i]
           * H[0, j], H[1, i]*H[1, j], H[0, i]*H[2, j] + H[2, i]*H[0, j], H[1, i]*H[2, j] + H[2, i]*H[1, j], H[2, i]*H[2, j]]
    v = np.array(vij).reshape(6, 1)
    return v


def compute_error(params, corners, homographies):

    A = np.zeros((3, 3))
    A[0, :] = params[2:5]
    A[1, :] = [0, params[5], params[6]]
    A[2, :] = [0, 0, 1]

    K = np.reshape(params[:2], (2, 1))

    points_3d = []
    for i in range(6):
        for j in range(9):

            points_3d.append([21.5*(j+1), 21.5*(i+1), 0, 1])
    points_3d = np.array(points_3d)

    error = np.empty([54, 1])
    for ind in range(len(homographies)):
        H = homographies[ind]
        points_2d = corners[ind]

        R = generate_extrinsic_params(A, H)
        new_points_3d = np.matmul(R, points_3d.T)
        new_points_3d = new_points_3d/new_points_3d[2]
        P = np.matmul(A, R)
        imgpt = np.matmul(P, points_3d.T)
        imgpt = imgpt/imgpt[2]

        u0, v0 = A[0, 2], A[1, 2]

        u, v = imgpt[0], imgpt[1]

        x, y = new_points_3d[0], new_points_3d[1]

        k1, k2 = K[0], K[1]

        u_hat = u+(u-u0)*(k1*(x**2+y**2)+k2*(x**2+y**2)**2)
        v_hat = v+(v-v0)*(k1*(x**2+y**2)+k2*(x**2+y**2)**2)

        project = points_2d
        project = np.reshape(project, (-1, 2))

        reproject = np.reshape(np.array([u_hat, v_hat]), (2, 54)).T

        err = np.linalg.norm(np.subtract(project, reproject), axis=1)

        error = np.vstack((error, err.reshape((54, 1))))
    error = error[54:]
    error = np.reshape(error, (702,))

    return error


def final_error(A, K, homographies, corners):

    points_3d = []
    for i in range(6):
        for j in range(9):

            points_3d.append([21.5*(j+1), 21.5*(i+1), 0])
    points_3d = np.array(points_3d)
    mean = 0
    error = np.zeros([2, 1])
    for i in range(len(homographies)):

        world_to_camera = generate_extrinsic_params(A, homographies[i])

        points_2d, _ = cv2.projectPoints(
            points_3d, world_to_camera[:, 0:3], world_to_camera[:, 3], A, K)
        points_2d = np.array(points_2d)
        errors = np.linalg.norm(np.subtract(
            points_2d[:, 0, :], corners[i][:, 0, :]), axis=1)
        error = np.concatenate(
            [error, np.reshape(errors, (errors.shape[0], 1))])
    error = np.mean(error)

    return error

def main():
  img = ["data/IMG_20170209_042606.jpg", "data/IMG_20170209_042608.jpg", "data/IMG_20170209_042610.jpg", "data/IMG_20170209_042612.jpg", "data/IMG_20170209_042614.jpg", "data/IMG_20170209_042616.jpg", "data/IMG_20170209_042619.jpg", "data/IMG_20170209_042621.jpg", "data/IMG_20170209_042624.jpg", "data/IMG_20170209_042627.jpg", "data/IMG_20170209_042629.jpg", "data/IMG_20170209_042630.jpg", "data/IMG_20170209_042634.jpg"]

  images = read_image(img)


  assert (len(images) > 0), "Unable to read images"

  V = np.ones((1, 6))

  H_list = []
  corner_list = []

  for img in images:

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

      corner_list.append(corners)

      camera_list = []

      for k in [0, 8, 53, 45]:
          camera_list.append(corners[k][0])

      camera_list = np.float32(camera_list)

      points_3d = np.float32([(21.5, 21.5), (21.5*9, 21.5),
                        (21.5*9, 6*21.5), (21.5, 6*21.5)])

      H, _ = cv2.findHomography(points_3d, camera_list)
      H_list.append(H)

      V1 = np.vstack((vectorv(1, 2, H).T, (vectorv(1, 1, H)-vectorv(2, 2, H)).T))

      V = np.vstack((V, V1))


  U, S, L = np.linalg.svd(V[1:])

  b = L[:][5]
  [B11, B12, B22, B13, B23, B33] = b


  c_y = (B12*B13-B11*B23)/(B11*B22-B12**2)

  l = B33-(B13**2+c_y*(B12*B13-B11*B23))/B11


  f_x = math.sqrt(l/B11)

  f_y = math.sqrt(l*B11/(B11*B22-B12**2))


  gama = -1*B12*(f_x**2)*f_y/l


  c_x = gama*c_y/f_y - B13*(f_x**2)/l


  A = np.array([[f_x, gama, c_x], [0, f_y, c_y], [0, 0, 1]])

  print("Pre-distortion K = ", A)


  initial = generate_param_vec(A, 0, 0)
  res = least_squares(compute_error, x0=np.squeeze(initial),
                      method='lm', args=(corner_list, H_list))
  A = np.zeros((3, 3))
  A[0, :] = res.x[2:5]
  A[1, :] = [0, res.x[5], res.x[6]]
  A[2, :] = [0, 0, 1]
  temp = generate_extrinsic_params(A, H)
  print("temp = ",temp)

  print("Post distortion K = ", A)
  K = res.x[:2]
  print("k1,k2 = ", K)

  distortion = np.array([K[0], K[1], 0, 0, 0], dtype=float)

  points_3d = []
  for i in range(6):
      for j in range(9):
          points_3d.append([21.5*(j+1), 21.5*(i+1), 0, 1])

  points_3d = np.array(points_3d)

  for index in range(len(images)):
      undist = cv2.undistort(images[index], A, distortion)
      H = H_list[index]
      final_mean_error = final_error(A, distortion, H_list, corner_list)
      corners = corner_list[index]
      gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
      R = generate_extrinsic_params(A, H)
      new_points_3d = np.matmul(R, points_3d.T)
      new_points_3d = new_points_3d/new_points_3d[2]
      P = np.matmul(A, R)
      imgpt = np.matmul(P, points_3d.T)
      corners2 = imgpt/imgpt[2]

      for k in range(54):
        cv2.circle(undist, (int(corners2[0, k]), int(
            corners2[1, k])), 5, (0, 0, 255), 7)

      cv2.imwrite("data/IMG_UNDISTORTED_" + str(index)+".jpg", undist)



  print("Final Mean Error = ", final_mean_error)

if __name__=='__main__':
    main()
