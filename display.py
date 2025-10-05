import ctypes
import os
import sys
from pathlib import Path

import pygame
from pygame.locals import DOUBLEBUF


def _preload_pangolin_deps():
    """Ensure Pangolin's shared libraries are available before importing."""
    if sys.platform != "darwin":
        return

    repo_root = Path(__file__).resolve().parent
    lib_root = repo_root / "lib" / "macosx"
    if not lib_root.exists():
        return

    def _ensure_env_path(var: str, path: Path) -> None:
        existing = os.environ.get(var, "")
        parts = [p for p in existing.split(":") if p]
        path_str = str(path)
        if path_str not in parts:
            parts.insert(0, path_str)
            os.environ[var] = ":".join(parts)

    # Make the bundled folder visible to dyld without requiring shell overrides.
    _ensure_env_path("DYLD_FALLBACK_LIBRARY_PATH", lib_root)

    # Load auxiliary dependencies first.
    for name in ("libtinyobj.dylib", "libtinyobj.0.dylib"):
        dep = lib_root / name
        if dep.exists():
            ctypes.CDLL(str(dep), mode=ctypes.RTLD_GLOBAL)

    def _canonical_name(path: Path) -> str:
        stem = path.name.split('.dylib')[0]
        if '.0.' in stem:
            return stem.split('.0.')[0]
        if stem.endswith('.0'):
            return stem[:-2]
        return stem

    def _priority(path: Path) -> int:
        name = path.name
        if name.endswith('.0.dylib'):
            return 0
        if name.count('.') == 1:  # plain .dylib
            return 1
        return 2  # fully versioned (e.g. .0.9.3)

    groups = {}
    for lib_path in lib_root.glob("libpango_*.dylib"):
        groups.setdefault(_canonical_name(lib_path), []).append(lib_path)

    # Load Pangolin components in a dependency-friendly order.
    preferred = [
        "libpango_core",
        "libpango_vars",
        "libpango_image",
        "libpango_geometry",
        "libpango_opengl",
        "libpango_glgeometry",
        "libpango_windowing",
        "libpango_display",
        "libpango_scene",
        "libpango_plot",
        "libpango_packetstream",
        "libpango_video",
        "libpango_tools",
        "libpango_python",
    ]

    selected = []
    for key in preferred:
        if key in groups:
            selected.append(min(groups[key], key=_priority))
    for key, paths in groups.items():
        if key not in preferred:
            selected.append(min(paths, key=_priority))

    for lib_path in selected:
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise ImportError(f"Failed to load Pangolin dependency: {lib_path}\n{exc}") from exc


_preload_pangolin_deps()


class Display2D(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self._alive = True

    def paint(self, img):
        if not self._alive:
            return
        # junk
        for event in pygame.event.get():
            pass

        # draw
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [2, 1, 0]])

        # RGB, not BGR (might have to switch in twitchslam)
        self.screen.blit(self.surface, (0, 0))

        # blit
        pygame.display.flip()

    def close(self):
        if not self._alive:
            return
        pygame.display.quit()
        pygame.quit()
        self._alive = False


from multiprocessing import Process, Queue
import queue
import pypangolin as pangolin
from OpenGL.GL import *
import numpy as np


class Display3D(object):
    def __init__(self, W, H, F):
        self.state = None
        self.W = W // 2
        self.H = H // 2
        self.F = F
        self.q = Queue(maxsize=5)
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()
        self.max_points = 5000

    def viewer_thread(self, q):
        self.viewer_init(self.W, self.H)
        running = True
        while running:
            try:
                while True:
                    state = q.get_nowait()
                    if state is None:
                        running = False
                        break
                    self.state = state
            except queue.Empty:
                pass

            if not running:
                break

            self._render()

        try:
            pangolin.DestroyWindowAndBind('Map Viewer')
        except Exception:
            pass

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        glEnable(GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
                pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.2, 10000),
                pangolin.ModelViewLookAt(0, -10, -8,
                                         0, 0, 0,
                                         0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach(1.0))
        self.dcam.SetHandler(self.handler)

    def draw_points(self, points, colors):
        # Check if pypangolin requires VAOs (Vertex Array Objects) and VBOs (Vertex Buffer Objects)
        # This is a simplified version and would need to be expanded with proper buffer management
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, points)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, colors)

        glDrawArrays(GL_POINTS, 0, len(points))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def draw_camera(self, camera_pose, camera_parameters=None):
        # camera_parameters should include the camera intrinsics and the image size
        # for this example, we'll assume a simple pinhole camera model

        # Define the size of the triangle
        triangle_size = 2  # 10 cm for example

        # Extract the camera position from the last column of the pose matrix
        camera_position = camera_pose[:3, 3]

        # Calculate the camera's forward direction (assuming the camera looks along the Z-axis)
        forward = camera_pose[:3, 2]

        # Calculate points for the triangle, assume the camera's up vector is the Y-axis
        p1 = camera_position
        p2 = camera_position + (camera_pose[:3, 0] * triangle_size) - (forward * triangle_size)
        p3 = camera_position - (camera_pose[:3, 0] * triangle_size) - (forward * triangle_size)

        # Now we can draw the triangle
        glBegin(GL_TRIANGLES)
        for p in [p1, p2, p3]:
            glVertex3f(*p)
        glEnd()

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:
            if len(self.state[0]) >= 2:
                # draw poses
                glColor3f(0.0, 1.0, 0.0)
                for pose in self.state[0][:-1]:
                    self.draw_camera(pose)

                # draw trajectory polyline
                glColor3f(0.2, 0.8, 1.0)
                glLineWidth(2.0)
                glBegin(GL_LINE_STRIP)
                for pose in self.state[0]:
                    center = pose[:3, 3]
                    glVertex3f(center[0], center[1], center[2])
                glEnd()

            if len(self.state[0]) >= 1:
                # draw current pose as yellow
                glColor3f(1.0, 1.0, 0.0)
                self.draw_camera(self.state[0][-1])

            if len(self.state[1]) != 0:
                # draw keypoints
                glPointSize(5)
                self.draw_points(self.state[1], self.state[2])

        pangolin.FinishFrame()

    def paint(self, mapp):
        if self.q is None:
            return

        poses, pts, colors = [], [], []
        for f in mapp.frames:
            # invert pose for display only
            poses.append(np.linalg.inv(f.pose))
        for p in mapp.points:
            pts.append(p.pt)
            colors.append(p.color)
        pts_arr = np.array(pts)
        colors_arr = np.array(colors)
        if len(pts_arr) > self.max_points:
            idx = np.linspace(0, len(pts_arr) - 1, self.max_points).astype(int)
            pts_arr = pts_arr[idx]
            colors_arr = colors_arr[idx]

        try:
            self.q.put_nowait((np.array(poses), pts_arr, colors_arr / 256.0))
        except queue.Full:
            pass

    def close(self):
        if self.q is None:
            return
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        self.vp.join(timeout=1.0)
        if self.vp.is_alive():
            self.vp.terminate()
        self.q.close()
        self.q = None
