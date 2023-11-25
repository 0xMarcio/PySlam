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
    pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [2,1,0]])

    # RGB, not BGR (might have to switch in twitchslam)
    self.screen.blit(self.surface, (0,0))

    # blit
    pygame.display.flip()

from multiprocessing import Process, Queue
import pypangolin as pangolin
from OpenGL.GL import *
import numpy as np

class Display3D(object):
    def __init__(self, W, H, F):
        self.state = None
        self.W = W//2
        self.H = H//2
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
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0, 0, 0,
                                     0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach(1.0))
        self.dcam.SetHandler(self.handler)

    # ... [rest of your Display3D class]

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

        # #if camera_parameters is None, initialize with default values
        # if camera_parameters is None:
        #     camera_parameters = {'focal_length': 500, 'image_width': 640, 'image_height': 480, 'z_near': 0.2, 'z_far': 1000}
        #
        # f = self.F
        # w = self.W
        # h = self.H
        # z_near = camera_parameters['z_near']
        # z_far = camera_parameters['z_far']
        #
        # # Calculate the necessary points for the frustum
        # aspect_ratio = w / h
        # fov = 2 * np.arctan((h / 2) / f)  # Assuming symmetric fov
        #
        # # Near plane dimensions
        # near_height = 2 * np.tan(fov / 2) * z_near
        # near_width = near_height * aspect_ratio
        #
        # # Far plane dimensions
        # far_height = 2 * np.tan(fov / 2) * z_far
        # far_width = far_height * aspect_ratio
        #
        # # Corners of the near plane
        # near_tl = np.array([-near_width / 2, near_height / 2, -z_near])
        # near_tr = np.array([near_width / 2, near_height / 2, -z_near])
        # near_bl = np.array([-near_width / 2, -near_height / 2, -z_near])
        # near_br = np.array([near_width / 2, -near_height / 2, -z_near])
        #
        # # Corners of the far plane
        # far_tl = np.array([-far_width / 2, far_height / 2, -z_far])
        # far_tr = np.array([far_width / 2, far_height / 2, -z_far])
        # far_bl = np.array([-far_width / 2, -far_height / 2, -z_far])
        # far_br = np.array([far_width / 2, -far_height / 2, -z_far])
        #
        # # Transform the points by the camera pose
        # points = [near_tl, near_tr, near_bl, near_br, far_tl, far_tr, far_bl, far_br]
        # transformed_points = [camera_pose @ np.array([x, y, z, 1.0]).T for x, y, z in points]
        #
        # # Now we can draw lines between the calculated points
        # glBegin(GL_LINES)
        #
        # # Connect near plane points
        # for i in range(4):
        #     glVertex3f(*transformed_points[i][:3])
        #     glVertex3f(*transformed_points[(i + 1) % 4][:3])
        #
        # # Connect corresponding near and far plane points
        # for i in range(4):
        #     glVertex3f(*transformed_points[i][:3])
        #     glVertex3f(*transformed_points[i + 4][:3])
        #
        # # Connect far plane points
        # for i in range(4, 8):
        #     glVertex3f(*transformed_points[i][:3])
        #     glVertex3f(*transformed_points[(i + 1 - 4) % 4 + 4][:3])
        #
        # glEnd()

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
                    # try:
                    #     self.draw_camera(pose)
                    # except Exception as e:
                    #     print(e)

            if len(self.state[0]) >= 1:
                # draw current pose as yellow
                glColor3f(1.0, 1.0, 0.0)
                self.draw_camera(self.state[0][-1])
                # try:
                #     self.draw_camera(self.state[0][-1])
                # except Exception as e:
                #     print(e)

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
        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))



# from multiprocessing import Process, Queue
# from OpenGL.GL import *
# import pypangolin as pangolin
# import numpy as np
#
# class Display3D(object):
#   def __init__(self):
#     self.state = None
#     self.q = Queue()
#     self.vp = Process(target=self.viewer_thread, args=(self.q,))
#     self.vp.daemon = False
#     self.vp.start()
#
#   def viewer_thread(self, q):
#     self.viewer_init(1024, 768)
#     while 1:
#       self.viewer_refresh(q)
#
#   def viewer_init(self, w, h):
#     pangolin.CreateWindowAndBind('Map Viewer', w, h)
#     glEnable(GL_DEPTH_TEST)
#
#     self.scam = pangolin.OpenGlRenderState(
#       pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
#       pangolin.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pangolin.AxisY))
#     self.handler = pangolin.Handler3D(self.scam)
#
#     # Create Interactive View in window
#     self.dcam = pangolin.CreateDisplay()
#     self.dcam.SetBounds(pangolin.Attach(0), pangolin.Attach(1), pangolin.Attach.Pix(0), pangolin.Attach(1), w / h)
#     self.dcam.SetHandler(self.handler)
#     # hack to avoid small Pangolin, no idea why it's *2
#     self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
#     self.dcam.Activate(self.scam)
#
#
#   def viewer_refresh(self, q):
#     while not q.empty():
#       self.state = q.get()
#
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glClearColor(0.0, 0.0, 0.0, 1.0)
#     self.dcam.Activate(self.scam)
#
#     if self.state is not None:
#
#       try:
#         if self.state[0].shape[0] >= 2:
#           # draw poses
#           glColor3f(0.0, 1.0, 0.0)
#           pangolin.glDrawColouredCube()
#           # pangolin.glDrawColouredCube(np.array((self.state[0][:-1])))
#       except Exception as e:
#         print(e)
#
#       try:
#         if self.state[0].shape[0] >= 1:
#           # draw current pose as yellow
#           # Your transformation matrix
#           transformation_matrix = np.array(self.state[0][-1:])
#
#           # Extract x and y coordinates
#           x, y, _, _ = transformation_matrix[0, :, -1]
#
#           # Define a radius for the circle
#           radius = 0.5  # Example radius
#
#           glColor3f(1.0, 1.0, 0.0)
#           pangolin.glDrawCircle(x, y, radius)
#           # pangolin.glDrawCircle(self.state[0][-1:])
#       except Exception as e:
#           print(e)
#
#       if self.state[1].shape[0] != 0:
#         # draw keypoints
#         try:
#           glPointSize(5)
#           glColor3f(1.0, 0.0, 0.0)
#           # Your points array
#           points_array = np.array(self.state[1])
#
#           # Convert your 2D array to a list of 3D column vectors
#           points_list = [np.array(point, dtype=np.float64).reshape(3, 1) for point in points_array]
#
#           pangolin.glDrawPoints(points_list)
#           # pangolin.glDrawPoints(self.state[1], self.state[2])
#         except Exception as e:
#           print(e)
#
#     pangolin.FinishFrame()
#
#   def paint(self, mapp):
#     if self.q is None:
#       return
#
#     poses, pts, colors = [], [], []
#     for f in mapp.frames:
#       # invert pose for display only
#       poses.append(np.linalg.inv(f.pose))
#     for p in mapp.points:
#       pts.append(p.pt)
#       colors.append(p.color)
#     self.q.put((np.array(poses), np.array(pts), np.array(colors)/255.0))
#
#
#
