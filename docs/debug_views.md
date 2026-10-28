# Debug Views and Rendering Modes


### 1. Left vs. Right View when "substitute_vision" is Active (Toggled by 'S' Key)

When the `substitute_vision` mode is activated (by pressing 'S'), the left view changes to display debug information.

*   **RIGHT Window (Phosphene View):**
    This view never changes. It always displays the final output of the simulation: the 32x32 phosphene grid, 
    upscaled for visualization. It represents the simulated visual perception for the end-user.

*   **LEFT Window (Debug Input View):**
    Under normal operation, this view shows the camera's video feed with annotations (bounding boxes, text, etc.). 
    When 'S' is pressed, this view is **replaced** with the **exact input** being fed to the phosphene rendering engine (`PhospheneRenderer`). 
    This is a high-resolution, grayscale saliency map that combines depth-based brightness, structural contours extracted from LiDAR, 
    and the highlighted target object. It serves as a technical view to understand the information the system is using to generate 
    the final phosphene image on the right.

---

### 2. Effect of the "pure_rgb_mode" Toggle ('G' Key)

The 'G' key toggles a mode that completely ignores depth information (LiDAR).

*   'G' OFF (Normal/Hybrid Mode):
    *   The simulation is based on a hybrid fusion of 2D and 3D data.
    *   Phosphene brightness is primarily determined by the distance of objects (closer objects are brighter), derived from LiDAR depth data.
    *   Structural contours of objects, extracted from the depth map, are rendered.

*   'G' ON ("Pure RGB" Mode):
    *   All depth processing is deactivated.
    *   Phosphene brightness is determined exclusively by the luminance of the traditional grayscale video feed.
    *   No depth-based structural contours are calculated or rendered.
    *   This mode is useful for evaluating the difference between a simulation based purely on 2D video information versus the more advanced 
    simulation that leverages 3D spatial data.
