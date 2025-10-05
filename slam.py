#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import ctypes

REPO_ROOT = Path(__file__).resolve().parent

if sys.platform == "darwin":
  lib_dir = REPO_ROOT / "lib" / "macosx"
  sys.path.append(lib_dir.as_posix())
  lib_str = str(lib_dir)

  def _prepend_path(var: str) -> None:
    existing = os.environ.get(var, "")
    parts = [p for p in existing.split(":") if p]
    if lib_str not in parts:
      parts.insert(0, lib_str)
      os.environ[var] = ":".join(parts)

  _prepend_path("DYLD_FALLBACK_LIBRARY_PATH")
  _prepend_path("DYLD_LIBRARY_PATH")
else:
  lib_dir = REPO_ROOT / "lib" / "linux"
  sys.path.append(lib_dir.as_posix())
import time
import cv2
from display import Display2D, Display3D
from frame import Frame, match_frames
import numpy as np


def _preload_native_deps():
  if sys.platform != "darwin":
    return

  ordered = [
    "libg2o_stuff",
    "libg2o_core",
    "libg2o_opengl_helper",
    "libg2o_ext_freeglut_minimal",
    "libg2o_types_slam2d",
    "libg2o_types_slam2d_addons",
    "libg2o_types_sclam2d",
    "libg2o_types_slam3d",
    "libg2o_types_slam3d_addons",
    "libg2o_types_sba",
    "libg2o_types_sim3",
    "libg2o_types_icp",
    "libg2o_types_data",
    "libg2o_solver_cholmod",
    "libg2o_solver_csparse",
    "libg2o_solver_dense",
    "libg2o_solver_eigen",
    "libg2o_solver_pcg",
    "libg2o_solver_slam2d_linear",
    "libg2o_solver_structure_only",
    "libg2o_contrib",
  ]

  loaded = set()
  for name in ordered:
    path = lib_dir / f"{name}.dylib"
    if not path.exists():
      continue
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    loaded.add(path.stem)

  # Load any remaining extension libs to ensure full coverage.
  for extra in sorted(lib_dir.glob("libg2o_*.dylib")):
    if extra.stem in loaded:
      continue
    ctypes.CDLL(str(extra), mode=ctypes.RTLD_GLOBAL)
    loaded.add(extra.stem)


_preload_native_deps()

from pointmap import Map, Point
from helpers import triangulate, poseRt

np.set_printoptions(suppress=True)

class SLAM(object):
  def __init__(self, W, H, K):
    # main classes
    self.mapp = Map()

    # params
    self.W, self.H = W, H
    self.K = K
    self.min_pnp_points = 6

  def _drop_last_frame(self):
    if not self.mapp.frames:
      return
    self.mapp.frames.pop()
    self.mapp.max_frame = max(0, self.mapp.max_frame - 1)

  def process_frame(self, img, pose=None, verts=None):
    start_time = time.time()
    assert img.shape[0:2] == (self.H, self.W)
    frame = Frame(self.mapp, img, self.K, verts=verts)

    if frame.id == 0:
      return

    f1 = self.mapp.frames[-1]
    f2 = self.mapp.frames[-2]

    idx1, idx2, pts1_px, pts2_px = match_frames(f1, f2)
    if len(idx1) < 8:
      print("Matches:  insufficient, dropping frame")
      self._drop_last_frame()
      return

    pts1 = np.asarray(pts1_px, dtype=np.float64)
    pts2 = np.asarray(pts2_px, dtype=np.float64)
    idx1 = np.asarray(idx1, dtype=np.int32)
    idx2 = np.asarray(idx2, dtype=np.int32)

    known_mask = np.array([f2.pts[idx] is not None for idx in idx2])
    known_indices = np.nonzero(known_mask)[0]
    object_points = np.array([f2.pts[idx2[i]].pt for i in known_indices], dtype=np.float64) if len(known_indices) else np.empty((0, 3), dtype=np.float64)
    image_points = pts1[known_indices] if len(known_indices) else np.empty((0, 2), dtype=np.float64)

    pose_estimated = False
    good_mask = np.ones(len(idx1), dtype=bool)

    if len(object_points) >= self.min_pnp_points:
      success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        self.K,
        None,
        iterationsCount=100,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
      )
      if success and inliers is not None and len(inliers) >= self.min_pnp_points:
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        f1.pose = poseRt(R, t)
        pose_estimated = True

        inlier_mask = np.zeros(len(object_points), dtype=bool)
        inlier_mask[inliers.ravel()] = True
        for local_idx, is_inlier in zip(known_indices, inlier_mask):
          if not is_inlier:
            good_mask[local_idx] = False

    if not pose_estimated:
      E, e_mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
      if E is None:
        print("Essential matrix estimation failed; dropping frame")
        self._drop_last_frame()
        return
      _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K)
      mask = pose_mask.ravel().astype(bool)
      if e_mask is not None:
        mask &= e_mask.ravel().astype(bool)
      if mask.sum() < 8:
        print("Pose recovery failed; dropping frame")
        self._drop_last_frame()
        return
      good_mask &= mask
      pts1 = pts1[good_mask]
      pts2 = pts2[good_mask]
      idx1 = idx1[good_mask]
      idx2 = idx2[good_mask]

      R_prev = f2.pose[:3, :3]
      t_prev = f2.pose[:3, 3]
      t_rel = t_rel.reshape(3)
      R_curr = R_rel @ R_prev
      t_curr = R_rel @ t_prev + t_rel
      f1.pose = poseRt(R_curr, t_curr)
    else:
      pts1 = pts1[good_mask]
      pts2 = pts2[good_mask]
      idx1 = idx1[good_mask]
      idx2 = idx2[good_mask]

    if len(idx1) == 0:
      print("Matches: 0 after filtering; dropping frame")
      self._drop_last_frame()
      return

    # add new observations if the point is already observed in the previous frame
    # TODO: consider tradeoff doing this before/after search by projection
    for i,idx in enumerate(idx2):
      if f2.pts[idx] is not None and f1.pts[idx1[i]] is None:
        f2.pts[idx].add_observation(f1, idx1[i])

    # pose optimization
    if pose is None:
      pose_opt = self.mapp.optimize(local_window=5, fix_points=True)
      print("Pose:     %f" % pose_opt)
    else:
      # have ground truth for pose
      f1.pose = pose

    sbp_pts_count = 0

    # search by projection
    if len(self.mapp.points) > 0:
      # project *all* the map points into the current frame
      map_points = np.array([p.homogeneous() for p in self.mapp.points])
      projs = np.dot(np.dot(self.K, f1.pose[:3]), map_points.T).T
      depths = projs[:, 2]
      valid_depth = depths > 0
      projs_2d = projs[:, :2] / depths[:, None]

      # only the points that fit in the frame
      good_pts = valid_depth & \
                 (projs_2d[:, 0] >= 0) & (projs_2d[:, 0] < self.W) & \
                 (projs_2d[:, 1] >= 0) & (projs_2d[:, 1] < self.H)

      for i, p in enumerate(self.mapp.points):
        if not good_pts[i]:
          # point not visible in frame
          continue
        if f1 in p.frames:
          # we already matched this map point to this frame
          # TODO: understand this better
          continue
        for m_idx in f1.kd.query_ball_point(projs_2d[i], 2):
          # if point unmatched
          if f1.pts[m_idx] is None:
            b_dist = p.orb_distance(f1.des[m_idx])
            # if any descriptors within 64
            if b_dist < 64.0:
              p.add_observation(f1, m_idx)
              sbp_pts_count += 1
              break

    # triangulate the points we don't have matches for
    good_pts4d = np.array([f1.pts[i] is None for i in idx1])

    # do triangulation in global frame
    pts4d = triangulate(self.K, f1.pose, f2.pose, f1.kpus[idx1], f2.kpus[idx2])
    good_pts4d &= np.abs(pts4d[:, 3]) != 0
    pts4d /= pts4d[:, 3:]       # homogeneous 3-D coords

    # adding new points to the map from pairwise matches
    new_pts_count = 0
    for i,p in enumerate(pts4d):
      if not good_pts4d[i]:
        continue

      # check parallax is large enough
      # TODO: learn what parallax means
      """
      r1 = np.dot(f1.pose[:3, :3], add_ones(f1.kps[idx1[i]]))
      r2 = np.dot(f2.pose[:3, :3], add_ones(f2.kps[idx2[i]]))
      parallax = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
      if parallax >= 0.9998:
        continue
      """

      # check points are in front of both cameras
      pl1 = np.dot(f1.pose, p)
      pl2 = np.dot(f2.pose, p)
      if pl1[2] < 0 or pl2[2] < 0:
        continue

      # reproject
      pp1 = np.dot(self.K, pl1[:3])
      pp2 = np.dot(self.K, pl2[:3])

      # check reprojection error
      pp1 = (pp1[0:2] / pp1[2]) - f1.kpus[idx1[i]]
      pp2 = (pp2[0:2] / pp2[2]) - f2.kpus[idx2[i]]
      pp1 = np.sum(pp1**2)
      pp2 = np.sum(pp2**2)
      if pp1 > 2 or pp2 > 2:
        continue

      # add the point
      try:
        color = img[int(round(f1.kpus[idx1[i],1])), int(round(f1.kpus[idx1[i],0]))]
      except IndexError:
        color = (255,0,0)
      pt = Point(self.mapp, p[0:3], color)
      pt.add_observation(f2, idx2[i])
      pt.add_observation(f1, idx1[i])
      new_pts_count += 1

    print("Adding:   %d new points, %d search by projection" % (new_pts_count, sbp_pts_count))

    # optimize the map
    if frame.id >= 4 and frame.id % 5 == 0:
      err = self.mapp.optimize(local_window=10, fix_points=False)
      print("Optimize: %f units of error" % err)

    print("Map:      %d points, %d frames" % (len(self.mapp.points), len(self.mapp.frames)))
    print("Time:     %.2f ms" % ((time.time()-start_time)*1000.0))
    # print(np.linalg.inv(f1.pose))


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("%s <video.mp4>" % sys.argv[0])
    exit(-1)

  cap = cv2.VideoCapture(sys.argv[1])

  # camera parameters
  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  F = float(os.getenv("F", "525"))

  disp2d = None
  disp3d = Display3D(W, H, F)




  if os.getenv("SEEK") is not None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("SEEK")))

  if W > 1024:
    downscale = 1024.0/W
    F *= downscale
    H = int(H * downscale)
    W = 1024
  print("using camera %dx%d with F %f" % (W,H,F))

  # camera intrinsics
  K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

  # create 2-D display
  if os.getenv("HEADLESS") is None:
    disp2d = Display2D(W, H)

  slam = SLAM(W, H, K)

  """
  mapp.deserialize(open('map.json').read())
  while 1:
    disp3d.paint(mapp)
    time.sleep(1)
  """

  gt_pose = None
  if len(sys.argv) >= 3:
    gt_pose = np.load(sys.argv[2])['pose']
    # add scale param?
    gt_pose[:, :3, 3] *= 50

  i = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.resize(frame, (W, H))

    print("\n*** frame %d/%d ***" % (i, CNT))
    slam.process_frame(frame, None if gt_pose is None else np.linalg.inv(gt_pose[i]))

    disp3d.paint(slam.mapp)

    if disp2d is not None:
      img = slam.mapp.frames[-1].annotate(frame)
      if i >= 2:
        # disp3d.paint(slam.mapp)
        disp2d.paint(img)

    i += 1
    """
    if i == CNT:
      smap = slam.mapp.serialize()
      print(smap)
      with open('map.json', 'w') as f:
        f.write(smap)
        exit(0)

    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()
      break

  cap.release()
  cv2.destroyAllWindows()

  if disp2d is not None:
    disp2d.close()

  if disp3d is not None:
    disp3d.close()
