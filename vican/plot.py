"""
    plot.py
    Gabriel Moreira
    Sep 18, 2023
"""
import cv2 as cv
import numpy as np
import plotly.express as px

from .cam import Camera
from .geometry import SE3

from typing import Iterable


def draw_marker(im: np.ndarray, 
                marker_corners: np.ndarray,
                marker_id: str) -> np.ndarray:
    """
        Draws arUco marker on image.

        Parameters
        ----------
        im : np.ndarray
            Source image (H,W,3) with detected marker.
        marker_corners: np.ndarray
            X and Y locations of 4 corners as a (4,2) array.
        marker_id: str
            arUco marker ID.

        Returns
        -------
        im : np.ndarray 
            Image with marker drawn.
    """
    marker_corners = marker_corners.reshape((4, 2))
    top_l, top_r, bottom_r, bottom_l = marker_corners.astype(np.int32)

    cv.line(im, top_l, top_r, (0, 255, 0), 1)
    cv.line(im, top_r, bottom_r, (0, 255, 0), 1)
    cv.line(im, bottom_r, bottom_l, (0, 255, 0), 1)
    cv.line(im, bottom_l, top_l, (0, 255, 0), 1)

    if marker_id is not None:
        cv.putText(im, str(marker_id), (top_l[0], top_l[1]-5),
                cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    return im


def detect_and_draw_aruco(im_filename: str,
                    aruco: str,
                    cam: Camera,
                    brightness: int=0,
                    contrast: int=0,
                    corner_refine: str='CORNER_REFINE_SUBPIX') -> np.ndarray:
    """
        Detects and draws arUco markers on image.

        Parameters
        ----------
        im_filename : str
            Image filename.
        brightness : int
            Image preprocessing brightness correction.
        contrast : int
            Image preprocessing contrast correction.
        corner_refine: str
            Corner refinement option (see OpenCV options).

        Returns
        -------
        im : np.ndarray 
            Image with marker drawn.
    """
    
    dictionary = cv.aruco.getPredefinedDictionary(eval('cv.aruco.' + aruco))
    parameters = cv.aruco.DetectorParameters()
    
    #default params
    
    """ if corner_refine is not None:
        parameters.cornerRefinementMethod = eval('cv.aruco.' + corner_refine)
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.adaptiveThreshConstant = 10
    parameters.cornerRefinementMaxIterations = 50
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.adaptiveThreshWinSizeMax = 35 """

    im = cv.imread(im_filename)
    im = np.int16(im)

    if contrast != 0:
        im = im * (contrast/127+1) - contrast

    im += brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)
    
    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(im, dictionary, parameters=parameters)
        
    marker_points = np.array([[-1, 1, 0],
                              [1, 1, 0],
                              [1, -1, 0],
                              [-1, -1, 0]], dtype=np.float32)
    
    marker_points *= 0.087 * 0.5
    
    if len(marker_corners) > 0:
        
        marker_ids = list(map(str, marker_ids.flatten()))
        
        for corners, marker_id in zip(marker_corners, marker_ids):
            corners = corners.squeeze()

            flag, rvec, t = cv.solvePnP(marker_points,
                                        imagePoints=corners,
                                        cameraMatrix=cam.intrinsics,
                                        distCoeffs=cam.distortion,
                                        flags=cv.SOLVEPNP_IPPE_SQUARE)
            
            if flag:
                rvec, t = cv.solvePnPRefineLM(marker_points,
                                              imagePoints=corners,
                                              cameraMatrix=cam.intrinsics,
                                              distCoeffs=cam.distortion,
                                              rvec=rvec,
                                              tvec=t)
                
                print(rvec, marker_id)

            #draw frames of axis
            if flag:
                #continue
                im = cv.drawFrameAxes(im, cameraMatrix=cam.intrinsics, distCoeffs=cam.distortion, rvec=rvec, tvec=t, length=0.1)

    #im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #im = np.stack((im,im,im), axis=2)
    
    #for mc, i in zip(marker_corners, marker_ids):
    #    im = draw_marker(im, mc, i)
    #print(sorted([int(i) for i in marker_ids]))
    
    
    
    return im

def detect_and_draw_charuco(im_filename: str,
                            cam: Camera,
                            target_dict: str,
                            brightness: int=0,
                            contrast: int=0) -> np.ndarray:

    #create board in here instead of passing it as an argument, since aruco methods are not pickable.... ~1ms
    charuco_dict = dict()
    for i in range(0,target_dict["num_boards"]+1):
        
        charuco_board = cv.aruco.CharucoBoard(
            size=(target_dict[str(i)]["sizeX"], target_dict[str(i)]["sizeY"]),
            squareLength=target_dict[str(i)]["squareLength"],
            markerLength=target_dict[str(i)]["markerLength"],
            dictionary= cv.aruco.getPredefinedDictionary(target_dict[str(i)]["dictionary"]),
            ids=target_dict[str(i)]["ids"])
        
        #charuco_board.setLegacyPattern(True)
        charuco_dict[str(i)] = charuco_board        
    
   
    im = cv.imread(im_filename)
    im = np.int16(im)
    
    if contrast != 0:
        im = im * (contrast/127+1) - contrast
        
    im += brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)
    
    #extract aruco dictionary
    aruco_dict = charuco_dict["0"].getDictionary()
    
    # Detect aruco markers
    corners, ids, rejected = cv.aruco.detectMarkers(im, aruco_dict)
    
    if len(corners) == 0:
        return im
    
    #go through all the charuco boards and compute one edge for each
    for board_id, charuco_board in charuco_dict.items():
        
        if board_id == "detector":
            continue
        
        flag, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                    corners, ids, im, charuco_board)
        
        if flag:
            im = cv.aruco.drawDetectedCornersCharuco(im, charuco_corners, charuco_ids, cornerColor=(0, 255, 0))
            objPoints, imPoints = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            
            if len(objPoints) < 4:
                continue
            
            retval, rvec, tvec = cv.solvePnP(objPoints, imPoints,
                                              cam.intrinsics,
                                              cam.distortion,
                                              flags=cv.SOLVEPNP_IPPE)
            
            if retval:
                im = cv.drawFrameAxes(im, cam.intrinsics, cam.distortion, rvec, tvec, 0.1)
            
        
    return im

def plot_cams_3D(cams: Iterable[Camera],
                 pose_est: dict={},
                 scale: float=0.4,
                 output: str='output.html') -> None:
    """
        Detects and draws arUco markers on image.

        Parameters
        ----------
        cams : Iterable[Camera]
            Cameras to plot.
        scale : float
            Scale of camera axis wrt the whole scene.
        output : str
            html file path or 'browser'.
    """
    pos = np.zeros((len(cams), 3))
    axs = np.zeros((len(cams), 3, 3, 2))
    for i, cam in enumerate(cams):
        extrinsics = cam.extrinsics
        pos[i,:]     += extrinsics.t()
        axs[i,:,:,0] += extrinsics.t().reshape((-1,1))
        axs[i,:,:,1] += extrinsics.t().reshape((-1,1)) + scale*extrinsics.R()

    fig = px.scatter_3d(x=pos[:,0], y=pos[:,1], z=pos[:,2],text=[str(i) for i in range(len(cams))])
    fig.update_traces(marker_size=2, marker_color='gray')

    c = ['red', 'green', 'blue']
    for i, cam in enumerate(cams):
        # 3 axis for each camera
        for j in range(3):
            fig.add_traces(px.line_3d(x=axs[i,0,j,:],
                                      y=axs[i,1,j,:],
                                      z=axs[i,2,j,:]).update_traces(line_color=c[j]).data)
            
    if pose_est is not None:
        for cam_id, pose in pose_est.items():
            if "_" in cam_id:
                fig.add_traces(px.scatter_3d(x=[pose.t()[0]], y=[pose.t()[1]], z=[pose.t()[2]]).update_traces(marker_size=1, marker_color='black').data)
            
        
    fig.update_scenes(aspectmode='data')
    
    if output.endswith('.html'):
        fig.write_html(output)
        print("Saving to", output)
    elif output == 'browser':
        print("Opening browser")
        fig.show(renderer=output)


def plot2D(ax,
           data: dict,
           view: str,
           marker: str,
           s : float,
           c : tuple,
           invert: bool=False,
           idx: Iterable=None,
           left_gauge: SE3=None,
           right_gauge: SE3=None) -> None:
    """
        2D scatter plot of 3D rigid transformations.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Matplotlib axes.
        data : dict
            Dictionary with data[n] = Camera or data[n] = SE3.
        view : str
            Axes to plot.
            Example: "xy", "xz", "yz".
        marker : str
            Matplotlib marker.
        s : float
            Size of 2D points.
        c : tuple
            Color of 2D points.
        invert : bool
            Whether to invert the transformations.
            Default: False
        idx : Iterable
            Keys of data to plot.
            Default: None
        left_gauge : SE3
            Transform all poses via left_gauge @ pose.
            If invert flag is True, inversion happens after.
            Default: None.
        right_gauge : SE3
            Transform all poses via pose pose @ right_gauge.
            If invert flag is True, inversion happens after.
            Default: None.
    """
    if left_gauge is None:
        GL = SE3(pose=np.eye(4))
    else:
        GL = left_gauge
    if right_gauge is None:
        GR = SE3(pose=np.eye(4))
    else:
        GR = right_gauge

    if idx is None:
        idx = data.keys()

    pts = []
    for n in idx:
        item = data[n]
        if isinstance(item, Camera):
            pose = GL @ item.extrinsics @ GR
        elif isinstance(item, SE3):
            pose = GL @ item @ GR

        if invert:
            pose_xyz = pose.inv().t()
        else:
            pose_xyz = pose.t()

        if view == "xy":
            pts.append(pose_xyz[:2])
        elif view == "xz":
            pts.append(pose_xyz[0::2])
        elif view == "yz":
            pts.append(pose_xyz[1:])

    pts = np.stack(pts, axis=0)
    ax.scatter(pts[:,0], pts[:,1], s, marker=marker, c=c)