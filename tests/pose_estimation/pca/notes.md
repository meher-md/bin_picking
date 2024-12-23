## RealSense Camera Coordinate System
If the RealSense is a box with the connection on the right side and the camera facing you:

- Right side of the box is the negative X-axis (-X).
- Upward direction is the negative Y-axis (-Y).
- Toward you (camera's view direction) is the positive Z-axis (+Z).


## Sources for grasp poblem
Below are a few **open-source** projects and example codes that demonstrate how to go from a **point cloud** (acquired by a camera on the robot’s wrist) to **grasp pose generation** and eventually **pick-and-place** with a robotic arm. These range from geometry-based approaches to machine-learning solutions.

---

## 1. GPD (Grasp Pose Detection)
**Repository**: [\(\text{GPD}\)](https://github.com/atenpas/gpd)

- **Overview**: A C++ ROS-compatible library by Andreas ten Pas for detecting 6-DoF grasp poses from a point cloud.  
- **Features**: 
  - Takes a raw point cloud (e.g., from a wrist-mounted sensor).  
  - Outputs a list of candidate grasp poses \((x, y, z, \text{orientation})\).  
  - Has both ROS integration and standalone usage.

**Quick Steps**:
1. Clone and build GPD in a ROS workspace or as standalone C++.
2. Provide your point cloud topic (if using ROS) or pass the data through the GPD library interface.
3. GPD will output candidate 6-DOF grasps.  
4. Transform these grasps from **camera** frame to **robot base** frame, then plan a motion with MoveIt (or any IK solver).

A typical usage snippet (ROS-based) could look like:

```bash
# In your catkin workspace:
git clone https://github.com/atenpas/gpd.git src/gpd
catkin_make
```

Then run `roslaunch gpd tutorial.launch` or a custom launch file that subscribes to your sensor data. The code inside GPD’s `detect_grasps.cpp` / `detect_grasps_server.cpp` is a great example of how to feed point cloud data and get back grasp candidates.

---

## 2. Dex-Net
**Repository**: [\(\text{Dex-Net}\)](https://github.com/BerkeleyAutomation/dex-net)

- **Overview**: A research project from UC Berkeley’s AUTOLab that uses deep learning to generate robust grasps.  
- **Features**:  
  - Large-scale synthetic training for robust single-view grasping.  
  - Dex-Net 4.0 includes suction grasping as well.  

**Workflow**:
1. Supply a 2D/3D sensor input (depth image or point cloud) of the object.  
2. Dex-Net proposes top-ranked gripper poses.  
3. Transform to the robot’s coordinate system and execute.

Be aware that Dex-Net can be more complex to set up compared to GPD, because it involves a **trained model**. The repo includes python scripts for inference and example code for integration with ROS and real robot arms.

---

## 3. MoveIt! Pick and Place Pipeline
**Repository**: [\(\text{MoveIt}\)](https://github.com/ros-planning/moveit)

- **Overview**: ROS-based motion planning framework.  
- **Pick and Place**: MoveIt has a built-in “Pick” pipeline that, if given a **grasp pose** (position + orientation) and object info, can generate a full pick motion. 
- **Integration**:  
  - You still need to produce or supply a candidate grasp pose from your object’s point cloud.  
  - MoveIt then handles the IK, collision checking, approach/retreat motions, etc.

**Minimal Example**:
```bash
# In a ROS environment with MoveIt installed:
roslaunch moveit_setup_assistant setup_assistant.launch
```
- Create a MoveIt package for your robot.  
- Then use the **MoveIt pick/place tutorial**:  
  [MoveIt Tutorials: Pick and Place](https://ros-planning.github.io/moveit_tutorials/doc/pick_place/pick_place_tutorial.html).  
- You can feed the transformation from your wrist camera to the base plus the object’s bounding box or a known pose.

---

## 4. Python + Open3D + Custom ICP or Grasp Logic
If you want a more **hand-rolled** or minimal approach (without full frameworks), you can combine:

1. **Open3D** for point cloud processing.  
2. A **basic geometry-based** grasp approach (e.g., extracting principal axes, bounding boxes, or surface normals).  
3. A **custom IK** solver (like `ikpy` or your robot’s Python API) to place the end-effector.  

**Example**: 
- [Open3D Global Registration Example](https://github.com/isl-org/Open3D/blob/master/examples/python/Advanced/global_registration.py) – find transformations between partial and reference clouds.  
- For grasping, you’d do something like:
  1. **Segment** the object from the point cloud.  
  2. **Compute** a bounding box or principal axis.  
  3. **Define** a grasp approach vector (normal to a face or aligned with the principal axis).  
  4. **Convert** that pose from camera frame to robot base.  
  5. **Send** to robot motion planning.

---

## 5. GG-CNN (Generative Grasping CNN)
**Repository**: [\(\text{GG-CNN}\)](https://github.com/dougsm/ggcnn) (by Doug Morrison et al.)

- **Overview**: A fast, real-time convolutional neural network approach that outputs grasp poses from a depth image.  
- **Features**:
  - Outputs a **grasp map**: for each pixel in a depth image, it predicts the grasp quality, angle, and width.  
  - Very popular for pick tasks using a 2-finger gripper.

Though it mostly uses a **2D** representation (depth image), you can easily adapt or transform the result into a 3D pose in the camera frame. Then proceed with the same transformation to base frame + IK steps as before.

---

### Putting It All Together

1. **Capture** point cloud from wrist camera.  
2. **Use** an **open-source** library (like GPD, Dex-Net, or GG-CNN) to compute grasp poses from the object point cloud.  
3. **Transform** those grasp poses to your robot’s base frame (using your known hand–eye calibration or forward kinematics).  
4. **Use** MoveIt or a custom motion planner to generate a collision-free path to that grasp.  
5. **Execute** the grasp on the robot.

**All** of these libraries are open-source and have a significant user community, with examples, wikis, and active GitHub issues—making them an excellent starting point for implementing a pick-and-place pipeline with a wrist-mounted camera.




# Steps to Identify Object Orientation

## 1. Align the CAD Model to the Ground Plane

From the CAD model, determine the object's reference orientation when it is flat on the ground. For example:

- The **primary principal axis** (largest eigenvector) is aligned with the longest dimension of the object.
- The **secondary principal axis** is orthogonal to the primary axis and lies on the ground plane.
- The **tertiary principal axis** points upward (normal to the ground).

Define the ground reference frame:

\[
R_{\text{ref}} =
\begin{bmatrix}
x_{\text{ref}} & y_{\text{ref}} & z_{\text{ref}}
\end{bmatrix}
\]

where \(x_{\text{ref}}\), \(y_{\text{ref}}\), and \(z_{\text{ref}}\) are the principal axes from the CAD model.

---

## 2. Perform PCA on the Point Cloud

Perform **Principal Component Analysis (PCA)** on the segmented point cloud to extract its principal axes.

- The **eigenvectors** from PCA represent the principal orientation of the object in the current point cloud.

---

## 3. Compute Rotation from Reference Frame to Point Cloud

The transformation (rotation matrix) from the CAD reference frame to the point cloud can be computed by aligning the principal axes:

\[
R_{\text{align}} = R_{\text{cloud}} \cdot R_{\text{ref}}^{-1}
\]

where:

- \(R_{\text{cloud}}\) is the rotation matrix derived from PCA on the point cloud.
- \(R_{\text{ref}}^{-1}\) (or \(R_{\text{ref}}^T\), since it's orthonormal) is the inverse of the CAD model's ground frame orientation.

---

## 4. Compute Translation

The centroid of the CAD model (\(C_{\text{ref}}\)) and the centroid of the point cloud (\(C_{\text{cloud}}\)) provide the translation vector:

\[
T = C_{\text{cloud}} - R_{\text{align}} \cdot C_{\text{ref}}
\]

---

## 5. Final Pose

Combine the rotation and translation into a single pose matrix:

\[
T_{\text{final}} =
\begin{bmatrix}
R_{\text{align}} & T \\
0 & 1
\end{bmatrix}
\]

---

With these steps, you can compute the full pose (rotation and translation) of an object in 3D space.
