import pygame
from pygame.locals import DOUBLEBUF


class Display2D(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, img):
        # junk
        for event in pygame.event.get():
            pass

        # draw
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [2, 1, 0]])

        # RGB, not BGR (might have to switch in twitchslam)
        self.screen.blit(self.surface, (0, 0))

        # blit
        pygame.display.flip()


from multiprocessing import Process, Queue
import pypangolin as pangolin
from OpenGL.GL import *
import numpy as np


class Display3D(object):
    def __init__(self, W, H, F):
        self.state = None
        self.W = W // 2
        self.H = H // 2
        self.F = F
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(self.W, self.H)
        while True:
            self.viewer_refresh(q)

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

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:
            if len(self.state[0]) >= 2:
                # draw poses
                glColor3f(0.0, 1.0, 0.0)
                for pose in self.state[0][:-1]:
                    self.draw_camera(pose)

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
        self.q.put((np.array(poses), np.array(pts), np.array(colors) / 256.0))
