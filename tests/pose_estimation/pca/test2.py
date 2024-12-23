
import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
print("OpenCV Qt version:")
print(cv2.getBuildInformation())
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt
from OCC.Core.AIS import AIS_Point
import numpy as np

def start_opencv_window():
    """
    Start an OpenCV window to display a simple image.
    """
    window_name = "OpenCV Window"
    image = 255 * np.ones((480, 640, 3), dtype=np.uint8)  # White image
    cv2.putText(image, "Press 'q' to quit OpenCV", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.namedWindow(window_name)
    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit OpenCV window
            break

    cv2.destroyAllWindows()
    print("OpenCV window closed.")


def start_occ_window():
    """
    Start an OCC window to display a point.
    """
    display, start_display, _, _ = init_display()
    point = gp_Pnt(0, 0, 0)  # Point at origin
    ais_point = AIS_Point(point)
    display.Context.Display(ais_point, True)
    display.FitAll()
    print("OCC window started.")
    start_display()
    print("OCC window closed.")


if __name__ == "__main__":
    # Start OpenCV window
    print("Starting OpenCV window...")
    start_opencv_window()

    # Start OCC window
    print("Starting OCC window...")
    #start_occ_window()
