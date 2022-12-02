import argparse
import pickle
from typing import Callable
import timeit

import numpy as np

SANITY_CHECK = False


def rot_to_axis_angle(R):
    """
    Computes the axis-angle representation of a rotation matrix
    """
    if np.linalg.norm(R - R.T) < 1e-6:
        # symmetric matrix!
        theta = np.arccos((np.trace(R) - 1) / 2)
        theta_sign = np.sign((np.trace(R) - 1.0) / 2)
        theta = theta * theta_sign
        vals, vecs = np.linalg.eig(R)
        for i, v in enumerate(vals):
            if np.allclose(np.real(v), 1.0):
                v = np.real(vecs[:, i])
                v = v / np.linalg.norm(v)
                return np.real(theta) * v
        raise ValueError("No eigenvector with eigenvalue 1!")
    else:
        # use skew-symmetric matrix trick
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        theta = np.arcsin(np.linalg.norm(v) / 2.0)
        v = v / np.linalg.norm(v)
        return theta * v


def axangle_to_rot(v):
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return np.eye(3)
    phi = v / theta
    return (
        np.cos(theta) * np.eye(3)
        + (1 - np.cos(theta)) * phi[:, None] @ phi[None, :]
        + np.sin(theta) * cross_explode(phi)
    )


def cross_explode(x):
    """
    Computes the cross product matrix of x, such that x^T @ y = x cross y
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def rodrigues(phi_theta, v):
    """
    Computes the rotation of theta along axis phi using Rodrigues' formula for v
    """
    R = axangle_to_rot(phi_theta)
    return R @ v


def fast_blocked_inverse(M, blocksize):
    """
    Computes the inverse of a block diagonal matrix M
    """
    inv = np.zeros_like(M)
    for i in range(0, M.shape[0], blocksize):
        inv[i : i + blocksize, i : i + blocksize] = np.linalg.inv(
            M[i : i + blocksize, i : i + blocksize]
        )
    return inv


def fast_blocked_solve(M, x, blocksize):
    """
    Solve a linear system with a block diagonal matrix M
    """
    result = np.zeros(M.shape[0])
    for i in range(0, M.shape[0], blocksize):
        result[i : i + blocksize] = np.linalg.solve(
            M[i : i + blocksize, i : i + blocksize], x[i : i + blocksize]
        )

    return result


def lm(
    campoints: np.array,
    fun: Callable,
    jac: Callable,
    n_cameras,
    n_points,
    lmbda: float = 1.0,
):
    """
    One step of Levenberg-Marquardt algorithm for non-linear least squares optimization (https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
    Input:
        params: initial parameters
        fun: function that computes the residual for each input point, given the parameters
        jac: function that computes the jacobian matrix of the residual wrt params
             Each row i is the gradient dfun(x_i, params)/dparams
             there should be
        x: independent variables
        y: dependent variables

    Output:
        delta: change in parameters
    """

    start_t = timeit.default_timer()

    f: np.array = fun()
    losses = np.square(campoints - f).sum(axis=1)
    A, B, C, JtE, J_gt = jac(losses)

    # in lm, we add lambda term to diag of JtJ
    A = A + lmbda * np.diag(np.diag(A)) + 1e-3 * np.eye(A.shape[0])
    B = B + lmbda * np.diag(np.diag(B)) + 1e-3 * np.eye(B.shape[0])

    # B_inv = np.linalg.inv(B)
    B_inv = fast_blocked_inverse(B, 3)

    neg_CBinv = -1 * (C @ B_inv)

    bE = JtE[: n_cameras * 7] + neg_CBinv @ JtE[n_cameras * 7 :]
    bP = JtE[n_cameras * 7 :]

    delta_E = np.linalg.solve((A + neg_CBinv @ C.T), bE)
    # delta_P = np.linalg.solve(B, bP - C.T @ delta_E)
    delta_P = fast_blocked_solve(B, bP - C.T @ delta_E, 3)

    delta = np.concatenate((delta_E, delta_P))

    print("Time for LM step: ", timeit.default_timer() - start_t)

    if SANITY_CHECK:
        JtJ = J_gt.T @ J_gt
        delta_gt = np.linalg.solve(
            (JtJ + lmbda * np.diag(np.diag(JtJ)) + 1e-3 * np.eye(J_gt.shape[1])),
            -1 * JtE,
        )
        assert np.allclose(delta, delta_gt)
    # squared norm
    # losses = np.square(losses)
    # print("losses", losses[0])
    # print(campoints[0], f[0])
    # CBinv = C.to_numpy() @ np.linalg.inv(B.diag_inv())

    return delta


def dehomogenize(x):
    x = x.copy()
    return x[:-1] / x[-1]


def project(P, pt, f=1):
    """
    Project a 3D point onto the image
    Input:
        P: camera extrinsics matrix, dimension [3, 4]
        pt: 3D points in world coordinates pt=[X, Y, Z]
        f: focal length
    Output:
        (x,y): image point
    """

    pt1 = np.dot(P, [pt[0], pt[1], pt[2], 1.0])
    x = f * (pt1[0] / pt1[2])
    y = f * (pt1[1] / pt1[2])

    return (x, y)


def pi(cam_pose_axangle, cam_translate, point, focal_length):
    """
    Computes the projection of a point onto the image plane
    """
    return (
        dehomogenize(rodrigues(cam_pose_axangle, point) + cam_translate) * focal_length
    )


def jac_pi_phi(cam_pose, point, focal_length):
    """
    Computes the jacobian of pi wrt phi of cam_pose_axangle = R_op exp(phi^) (evaluated at 0) (should return 2x3 matrix)
    """
    cam_rot = cam_pose[:3, :3]
    cam_translate = cam_pose[:3, 3]
    # jac_f
    jac_f = cross_explode(cam_rot @ point) * -1

    # jac T(RP + t)
    jac_T = jac_dehomogenize(cam_rot @ point + cam_translate)

    return focal_length * jac_T @ jac_f


def jac_pi_translate(cam_pose, point, focal_length):
    """
    Computes the jacobian of pi wrt camera translation (should return 2x3 matrix)
    """
    cam_rot = cam_pose[:3, :3]
    cam_translate = cam_pose[:3, 3]

    # jac T(RP + t)
    jac_T = jac_dehomogenize(cam_rot @ point + cam_translate)

    return focal_length * jac_T


def jac_pi_point(cam_pose, point, focal_length):
    """
    Computes the jacobian of pi wrt point estimate P (should return 2x3 matrix)
    """
    #
    cam_rot = cam_pose[:3, :3]
    cam_translate = cam_pose[:3, 3]

    # jac T(RP + t)
    jac_T = jac_dehomogenize(cam_rot @ point + cam_translate)

    return focal_length * jac_T @ cam_rot


def jac_pi_focal(cam_pose, point, focal_length):
    """
    Computes the jacobian of pi wrt focal length (should return 2x1 matrix)
    """
    cam_rot = cam_pose[:3, :3]
    cam_translate = cam_pose[:3, 3]

    jac_f = dehomogenize(cam_rot @ point + cam_translate)

    return jac_f


def jac_sq_norm(x):
    """
    Computes the jacobian of the squared norm of x wrt x
    """
    return 2 * x


def jac_dehomogenize(x):
    j = np.zeros((2, x.shape[0]), dtype=np.float64)
    j[0, 0] = 1.0 / x[2]
    j[1, 1] = 1.0 / x[2]
    j[0, 2] = (-1 * x[0]) / (x[2] ** 2)
    j[1, 2] = (-1 * x[1]) / (x[2] ** 2)
    return j


def homogenize(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=-1)


def solve_ba_problem(problem):
    """
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    """
    solution = problem

    n_cameras = problem["poses"].shape[0]
    n_points = problem["points"].shape[0]
    print("n_cameras", n_cameras)
    print("n_points", n_points)

    problem["observations"] = list(sorted(problem["observations"]))

    # # sanity-check:
    # for cam_id, point_id, x, y in problem["observations"]:
    #     projected = pi(
    #         rot_to_axis_angle(problem["poses"][cam_id, :3, :3]),
    #         problem["poses"][cam_id, :3, 3],
    #         problem["points"][point_id],
    #         problem["focal_lengths"][cam_id],
    #     )

    #     projected_gt = project(
    #         problem["poses"][cam_id],
    #         problem["points"][point_id],
    #         f=problem["focal_lengths"][cam_id],
    #     )
    #     assert np.allclose(projected, projected_gt)

    # camera_rots = np.stack([rot_to_axis_angle(R) for R in problem["poses"][:, :3, :3]])
    # # assert camera_rots.shape == (n_cameras * 3,)

    # camera_positions = problem["poses"][:, :3, 3].copy()
    # # assert camera_positions.shape == (n_cameras * 3,)

    # camera_focal_lengths = problem["focal_lengths"][:, None].copy()
    # # assert camera_focal_lengths.shape == (n_cameras,)

    # image_points = problem["points"].copy()
    # # assert image_points.shape == (problem["points"].shape[0] * 3,)

    gt_points = np.stack(
        [np.array([x, y], dtype=np.float64) for _, _, x, y in problem["observations"]]
    )

    # params = np.concatenate(
    #     [camera_rots, camera_positions, camera_focal_lengths], axis=-1
    # ).flatten()
    # params = np.concatenate([params, image_points.flatten()])
    params = np.zeros((n_cameras * 7 + n_points * 3,), dtype=np.float64)

    # YOUR CODE STARTS

    def fun():
        cam_poses, points, focals = (
            problem["poses"],
            problem["points"],
            problem["focal_lengths"],
        )

        projected_points = []
        for cam_id, point_id, x, y in problem["observations"]:
            projected = project(cam_poses[cam_id], points[point_id], f=focals[cam_id])
            projected_points.append(projected)

        return np.array(projected_points)

    def jac(errs):
        # compute jacobian of error wrt params
        jacobian_matrix = None

        start_t = timeit.default_timer()

        if SANITY_CHECK:
            jacobian_matrix = np.zeros(
                (len(problem["observations"]), len(params)), dtype=np.float64
            )

        JtE = np.zeros((len(params),), dtype=np.float64)
        cam_poses, points, focals = (
            problem["poses"],
            problem["points"],
            problem["focal_lengths"],
        )
        A = np.zeros((7 * n_cameras, 7 * n_cameras), dtype=np.float64)
        B = np.zeros((3 * n_points, 3 * n_points), dtype=np.float64)
        C = np.zeros((7 * n_cameras, 3 * n_points), dtype=np.float64)

        for erridx, (cam_id, point_id, x, y) in enumerate(problem["observations"]):
            # compute jacobian of term wrt params
            # 1) compute jacobian of term wrt cam poses
            projected = project(cam_poses[cam_id], points[point_id], f=focals[cam_id])
            jac_pose = (
                jac_sq_norm(projected - np.array([x, y]))
                @ jac_pi_phi(
                    cam_poses[cam_id],
                    points[point_id],
                    focals[cam_id],
                )
            ).flatten()

            jac_loc = (
                jac_sq_norm(projected - np.array([x, y]))
                @ jac_pi_translate(
                    cam_poses[cam_id],
                    points[point_id],
                    focals[cam_id],
                )
            ).flatten()

            jac_focal = (
                jac_sq_norm(projected - np.array([x, y]))
                @ jac_pi_focal(
                    cam_poses[cam_id],
                    points[point_id],
                    focals[cam_id],
                )
            ).flatten()
            if problem["is_calibrated"]:
                jac_focal = np.zeros_like(jac_focal)

            jac_point = (
                jac_sq_norm(projected - np.array([x, y]))
                @ jac_pi_point(
                    cam_poses[cam_id],
                    points[point_id],
                    focals[cam_id],
                )
            ).flatten()

            jac_camparams = np.concatenate([jac_pose, jac_loc, jac_focal]).flatten()

            # for debugging: (update groundtruth)
            if SANITY_CHECK:
                row_jac = np.zeros(params.shape[-1], dtype=np.float64)
                row_jac[cam_id * 7 : cam_id * 7 + 7] = jac_camparams
                row_jac[
                    n_cameras * 7 + point_id * 3 : n_cameras * 7 + point_id * 3 + 3
                ] = jac_point
                jacobian_matrix[erridx] = row_jac

            # compute err times jac^T for this term
            # JtE += row_jac * errs[erridx]
            JtE[cam_id * 7 : cam_id * 7 + 7] += jac_camparams * errs[erridx]
            JtE[n_cameras * 7 + point_id * 3 : n_cameras * 7 + point_id * 3 + 3] += (
                jac_point * errs[erridx]
            )

            A[cam_id * 7 : cam_id * 7 + 7, cam_id * 7 : cam_id * 7 + 7] += np.outer(
                jac_camparams, jac_camparams
            )
            B[
                point_id * 3 : point_id * 3 + 3, point_id * 3 : point_id * 3 + 3
            ] += np.outer(jac_point, jac_point)
            C[cam_id * 7 : cam_id * 7 + 7, point_id * 3 : point_id * 3 + 3] += np.outer(
                jac_camparams, jac_point
            )

        # make sure blocks are correct:
        if SANITY_CHECK:
            JtJ_gt = jacobian_matrix.T @ jacobian_matrix
            A_gt = JtJ_gt[: n_cameras * 7, : n_cameras * 7]
            B_gt = JtJ_gt[n_cameras * 7 :, n_cameras * 7 :]
            C_gt = JtJ_gt[: n_cameras * 7, n_cameras * 7 :]

            assert np.allclose(A, A_gt)
            assert np.allclose(B, B_gt)
            assert np.allclose(C, C_gt)

            JtE_gt = jacobian_matrix.T @ errs
            assert np.allclose(JtE_gt, JtE)

        print("Jacobian computation time: ", timeit.default_timer() - start_t)

        return A, B, C, -1 * JtE, jacobian_matrix

    lmbda = 1.0
    lastloss = -1 * np.inf
    currloss = np.linalg.norm(fun() - gt_points, axis=-1).mean()

    while abs(lastloss - currloss) > 1e-4:
        step = lm(gt_points, fun, jac, n_cameras, n_points, lmbda)
        if np.linalg.norm(step) < 1e-4:
            break
        camposes_old, points_old, focals_old = (
            problem["poses"].copy(),
            problem["points"].copy(),
            problem["focal_lengths"].copy(),
        )

        # apply update for poses in rotation vector form:
        for i in range(n_cameras):
            phi = step[i * 7 : i * 7 + 3]
            update = axangle_to_rot(phi)
            problem["poses"][i, :3, :3] = problem["poses"][i, :3, :3] @ update
            problem["poses"][i, :3, 3] += step[i * 7 + 3 : i * 7 + 6]
            problem["focal_lengths"][i] += step[i * 7 + 6]
        problem["points"] += step[n_cameras * 7 :].reshape(n_points, 3)

        newloss = np.linalg.norm(fun() - gt_points, axis=-1).mean()
        print(
            "loss",
            currloss,
            "last",
            lastloss,
            "lmbda",
            lmbda,
            "step",
            np.linalg.norm(step),
            "newloss",
            newloss,
        )

        if newloss <= currloss:
            lmbda /= 2
            lastloss = currloss
            currloss = newloss
        else:
            lmbda *= 2
            problem["poses"] = camposes_old
            problem["points"] = points_old
            problem["focal_lengths"] = focals_old

    solution = problem
    return solution


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, "rb"))
    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
