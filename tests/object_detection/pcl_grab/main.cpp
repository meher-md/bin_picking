#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/real_sense_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <thread>

class RealSensePointCloudViewer
{
public:
    RealSensePointCloudViewer()
        : viewer("RealSense Point Cloud Viewer")
    {
        viewer.setBackgroundColor(0, 0, 0);
        viewer.addCoordinateSystem(1.0);
        viewer.initCameraParameters();
    }

    void run()
    {
        pcl::RealSenseGrabber grabber;
        auto callback = [this](const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud) {
            if (!viewer.wasStopped()) {
                std::lock_guard<std::mutex> lock(cloud_mutex);
                latest_cloud = cloud;
            }
        };
        grabber.registerCallback(callback);

        grabber.start();
        while (!viewer.wasStopped()) {
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
            {
                std::lock_guard<std::mutex> lock(cloud_mutex);
                cloud = latest_cloud;
            }
            if (cloud) {
                if (!viewer.updatePointCloud(cloud, "cloud")) {
                    viewer.addPointCloud(cloud, "cloud");
                }
            }
            viewer.spinOnce(100);
        }
        grabber.stop();
    }

private:
    pcl::visualization::PCLVisualizer viewer;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr latest_cloud;
    std::mutex cloud_mutex;
};

int main()
{
    RealSensePointCloudViewer viewer;
    viewer.run();
    return 0;
}
