from math import e
from pathlib import Path
import os
from re import T
import torch
import numpy as np
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from scipy.spatial.transform import Rotation as R
from typing import Literal, List
import math
import pickle
import matplotlib.pyplot as plt
import BasicTools.helpers as hp
import BasicTools.coordinate_utils as cu
import BasicTools.endpoint_sampler as es
import BasicTools.vision_helpers as vh
from BasicTools.covariance_utils import compute_cov
import cv2
import viser
import viser.transforms as tf
import trimesh

class NeRF():
    def __init__(self, config_path: Path, width:int=640, height:int=360, nerf_dir_path=None) -> None:
        # config path
        self.config_path = config_path

        # Pytorch Config
        use_cuda = torch.cuda.is_available()           
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Can programmatically modify certain key attributes but instead just change in the relevant .yml file
        # if nerf_dir_path is not None:
        #     def update_config_callback(config):
        #         config.output_dir = Path(nerf_dir_path) / config.output_dir.name
        #         config.data = Path(nerf_dir_path) / config.data.name
        #         config.pipeline.datamanager.data = Path(nerf_dir_path) / config.pipeline.datamanager.data.name
        #         # print('config.output_dir', config.output_dir)
        #         # print('config.data', config.data)
        #         # print('config.pipeline.datamanager.data', config.pipeline.datamanager.data)
        #         return config
        # else:
        #     update_config_callback = None

        update_config_callback = None

        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(self.config_path, 
                                                      None, 
                                                      "inference",
                                                      update_config_callback)

        # Get reference camera
        self.camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]

        # Render parameters
        self.channels = 3
        self.camera_out,self.width,self.height = self.generate_output_camera(width,height)

    def generate_output_camera(self,width:int,height:int):
        fx,fy = 462.956,463.002
        cx,cy = 323.076,181.184
        
        camera_out = Cameras(
            camera_to_worlds=1.0*self.camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out,width,height
    
    def render(self, xcr:np.ndarray, xpr:np.ndarray=None,
               visual_mode:Literal["static","dynamic"]="static"):
        
        if visual_mode == "static":
            image = self.static_render(xcr)
        elif visual_mode == "dynamic":
            image = self.dynamic_render(xcr,xpr)
        else:
            raise ValueError(f"Invalid visual mode: {visual_mode}")
        
        # Convert to numpy
        image = image.cpu().numpy()

        # Convert to uint8
        image = (255*image).astype(np.uint8)

        return image

    def static_render(self, state:np.ndarray) -> torch.Tensor:
        # State is of form (px, py, pz, phi, theta, psi, vx, vy, vz) in ENU

        # Extract the pose
        # state_ned = cu.ENU_to_NED(state)
        # T_c2n = pose2nerf_transform(np.hstack((state_ned[:3], state_ned[6:])))
        T_c2n = pose2nerf_transform(state)
        P_c2n = torch.tensor(T_c2n[0:3,:]).float()

        # Render from a single pose
        camera_to_world = P_c2n[None,:3, ...]
        self.camera_out.camera_to_worlds = camera_to_world

        # render outputs
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(self.camera_out, obb_box=None)

        image = outputs["rgb"]

        return image

    def dynamic_render(self, xcr:np.ndarray,xpr:np.ndarray,frames:int=10) -> torch.Tensor:
        # Get interpolated poses
        Xs = torch.zeros(xcr.shape[0],frames)
        for i in range(xcr.shape[0]):
            Xs[i,:] = torch.linspace(xcr[i], xpr[i], frames)

        # Render images
        images = []
        for i in range(frames):
            images.append(self.static_render(Xs[:,i]))

        # Average across images
        image = torch.mean(images, axis=0)

        return image
    
    def generate_point_cloud(self, bounds=None, threshold=0):
        # Get associated 3D points
        pcd_points = self.pipeline.model.means.detach().cpu().numpy()

        # colors computed from the term of order 0 in the Spherical Harmonic basis
        pcd_colors = self.pipeline.model.features_dc.detach().cpu().numpy()

        # Eliminate any falling within a threshold distance of a priori room bounds
        # bounds is a rectangle bounds = np.arra([[x_l, x_u], [y_l, y_u], [z_l, z_u]])
        if bounds is not None:
            mask = np.ones(pcd_points.shape[0], dtype=bool)  # Start with all points included
            for i, (lower, upper) in enumerate(bounds):  # Iterate through each dimension
                mask &= (pcd_points[:, i] >= lower + threshold) & (pcd_points[:, i] <= upper - threshold)
            pcd_points = pcd_points[mask]  # Keep only the points that satisfy the mask
            pcd_colors = pcd_colors[mask]

        # From INRIA
        # Base auxillary coefficient
        C0 = 0.28209479177387814

        def SH2RGB(sh):
            return sh * C0 + 0.5

        pcd_colors = SH2RGB(pcd_colors).squeeze()
        pcd_colors = np.clip(pcd_colors, 0, 1)

        return pcd_points, pcd_colors

def get_nerf(nerf_dir_path, map):
    main_dir_path = os.getcwd()

    # Update this if adding other nerfs
    maps = {
        "mid_room":"cp_1203_0"
    }

    map_folder = os.path.join(nerf_dir_path,'outputs',maps[map])
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')

    # Need to remove the nerf_dir_path portion
    nerf_cfg_path = nerf_cfg_path.split(nerf_dir_path)[1][1:]

    # print('nerf_cfg_path', nerf_cfg_path)

    abs_nerf_dir_path = os.path.join(main_dir_path, nerf_dir_path)

    # Go into NeRF data folder and get NeRF object (because the NeRF instantation
    # requires the current working directory to be the NeRF data folder)
    os.chdir(nerf_dir_path)
    nerf = NeRF(Path(nerf_cfg_path), nerf_dir_path=abs_nerf_dir_path)
    os.chdir(main_dir_path)

    return nerf

# Note regarding conventions:
# We'll use 3-2-1 (z-y'-x'') intrinsic Euler angles defined using the drone ENU frame
# These angles describe a passive rotation from the NeRF world frame to the drone ENU 
# frame. We treat the NeRF frame as the world frame. Notably, our dynamics assume the
# world frame has z pointing upwards against gravity.
# All rotations in the pos2nerf_transform function are passive

def pose2nerf_transform(pose):

    # Realsense to Drone NED Frame
    T_r2d = np.array([
        [ 0.956,-0.017, 0.291, 0.152],
        [ 0.023, 1.000,-0.018,-0.034],
        [-0.291, 0.024, 0.956,-0.043],
        [ 0.000, 0.000, 0.000, 1.000]
        ])
    
    # Drone NED to ENU
    R_ned_to_enu = np.array([[0, 1, 0], 
                             [1, 0, 0], 
                             [0, 0, -1]])
    T_ned_to_enu = np.eye(4)
    T_ned_to_enu[:3,:3] = R_ned_to_enu
    
    # Note: Flightroom Frame to NeRF world frame
    # T_f2n = np.array([
    #     [ 1.000, 0.000, 0.000, 0.000],
    #     [ 0.000,-1.000, 0.000, 0.000],
    #     [ 0.000, 0.000,-1.000, 0.000],
    #     [ 0.000, 0.000, 0.000, 1.000]
    # ])

    # Camera convention frame to realsense frame
    # JE previous version
    T_c2r = np.array([
        [ 0.0, 0.0,-1.0, 0.0],
        [ 1.0, 0.0, 0.0, 0.0],
        [ 0.0,-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ])

    # Drone ENU to NeRF world frame
    T_d2n = np.eye(4)

    # Euler angles describe passive transform from world (NeRF) to ENU drone frame
    R_n2d = cu.euler_to_rot(pose[3:6])

    # So, transpose to get passive ENU drone frame to world (NeRF) i.e., R_d2n = R_n2d.T
    # Previously did not take transpose here because believed this function used active transforms (so would transpose twice)
    R_d2n = R_n2d.T
    T_d2n[0:3,:] = np.hstack([R_d2n,pose[0:3].reshape(-1,1)])

    # Camera to realense, realsense to drone NED, drone NED to drone ENU, drone ENU to NeRF world
    T_c2n = T_d2n @ T_ned_to_enu @ T_r2d @ T_c2r

    return T_c2n

def walk_in_nerf(nerf, starting_pose=np.zeros(6), pose_increment=0.05 * np.ones(6), fx=1, fy=1):
    """Allows you to simulate maneuvering around in NeRF using keyboard."""
    
    def update_image(curr_state):
        # Convert RGB to BGR
        cv_image = nerf.render(curr_state)[:,:,::-1]
        cv_image = cv2.resize(cv_image, (0, 0), fx=fx, fy=fy)

        # Add text to show current pose
        pose_text = f"Pose: x={curr_state[0]:.2f}, y={curr_state[1]:.2f}, z={curr_state[2]:.2f}, roll={curr_state[3]:.2f}, pitch={curr_state[4]:.2f}, yaw={curr_state[5]:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.putText(cv_image, pose_text, (10, 30), font, font_scale, font_color, thickness, line_type)

        return cv_image

    curr_state = np.append(starting_pose, np.array([0.0,0.0,0.0]))

    cv_image = update_image(curr_state)

    wait_time = 30

    name = "Current Image"
    cv2.namedWindow(name)

    while True:

        cv2.imshow(name, cv_image)
        cv2.moveWindow(name, 0, 0)

        key = cv2.waitKey(wait_time) & 0xFF

        # Track each key press individually
        if key != 255:  # 255 means no key was pressed
            # Increase x
            if key == ord('q'):
                curr_state[0] += pose_increment[0]
            # Decrease x
            elif key == ord('a'):
                curr_state[0] -= pose_increment[0]
            # Increase y
            elif key == ord('w'):
                curr_state[1] += pose_increment[1]
            # Decrease y
            elif key == ord('s'):
                curr_state[1] -= pose_increment[1]
            # Increase z
            elif key == ord('e'):
                curr_state[2] += pose_increment[2]
            # Decrease z
            elif key == ord('d'):
                curr_state[2] -= pose_increment[2]
            # Increase roll
            elif key == ord('u'):
                curr_state[3] += pose_increment[3]
            # Decrease roll
            elif key == ord('j'):
                curr_state[3] -= pose_increment[3]
            # Increase pitch
            elif key == ord('i'):
                curr_state[4] += pose_increment[4]
            # Decrease pitch
            elif key == ord('k'):
                curr_state[4] -= pose_increment[4]
            # Increase yaw
            elif key == ord('o'):
                curr_state[5] += pose_increment[5]
            # Decrease yaw
            elif key == ord('l'):
                curr_state[5] -= pose_increment[5]
            # Exit
            elif key == ord('x'):
                break
            
            cv_image = update_image(curr_state)
            
    cv2.destroyAllWindows()

def vis_nerf_in_browser(nerf, bounds=None, device=None):        
    rotation = tf.SO3.from_x_radians(0.0).wxyz      # identity rotation
    server = viser.ViserServer()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    means = nerf.pipeline.model.means.detach().cpu().numpy()
    colors = nerf.pipeline.model.colors.detach().cpu().numpy()
    opacities = nerf.pipeline.model.opacities.detach().cpu().numpy()

    rots = nerf.pipeline.model.quats.detach()
    scales = nerf.pipeline.model.scales.detach().clone()
    scales = torch.exp(scales)

    # scales can be very tiny, so put a lower bound
    # scales = torch.clip(scales, 1e-5)

    covs = compute_cov(rots, scales).cpu().numpy()

    # Only visualize within some bounding box set by bounds
    if bounds is not None:
        mask = np.all((means - bounds[:, 0] >= 0) & (bounds[:, 1] - means >= 0), axis=-1)
    else:
        mask = np.ones(means.shape[0], dtype=bool)

    means = means[mask]
    covs = covs[mask]
    colors = colors[mask]
    opacities = opacities[mask]

    # Add splat to the scene
    server.scene.add_gaussian_splats(
        name="/splats",
        centers= means,
        covariances= covs,
        rgbs= colors,
        opacities= opacities,
        wxyz=rotation,
    )

    return server

def add_meshes_to_server(server, path_names, scale=1):
    # Add extra meshes to nerf vis in server if provided
    meshes = []
    for i, path_name in enumerate(path_names):
        mesh = trimesh.load_mesh(path_name)
        meshes.append(mesh)
        server.add_mesh_trimesh(
            name=f"/{path_name}",
            mesh=mesh,
            scale=scale,
            wxyz=tf.SO3.from_x_radians(0.).wxyz,
            position=(0.0, 0.0, 0.0),
            visible=True
        )
    return meshes

if __name__ == '__main__':
    NERF_NAME = "mid_room"
    nerf_dir_path = "../data/nerf_data"
    nerf = get_nerf(nerf_dir_path, NERF_NAME)

    bounds = np.array([[-8,8],[-3,3],[0,3]])

    threshold = 0.0
    point_cloud, point_colors = nerf.generate_point_cloud(bounds,threshold)
    vh.plot_point_cloud(point_cloud, bounds, view_angles=(45,-45), figsize=None, colors=point_colors)
    plt.show()

    walk_in_nerf(nerf, np.array([-6,0,1, 0,0,0]))