#!/bin/bash

# Update and upgrade system
echo "Updating and upgrading the system..."
sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get dist-upgrade -y


# # Install basic dependencies
sudo apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev
sudo apt-get install git wget cmake build-essential
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at


# echo "Installing basic dependencies..."
# sudo apt install v4l-utils -y
# sudo apt install git cmake libssl-dev libusb-1.0-0-dev -y

# # Install additional DRM libraries
# echo "Installing DRM libraries..."
# sudo apt-get install -y libdrm-amdgpu1 libdrm-amdgpu1-dbgsym libdrm-dev libdrm-exynos1 \
# libdrm-exynos1-dbgsym libdrm-freedreno1 libdrm-freedreno1-dbgsym libdrm-nouveau2 \
# libdrm-nouveau2-dbgsym libdrm-omap1 libdrm-omap1-dbgsym libdrm-radeon1 \
# libdrm-radeon1-dbgsym libdrm-tegra0 libdrm-tegra0-dbgsym libdrm2 libdrm2-dbgsym

# # Install OpenGL and GUI libraries
# echo "Installing OpenGL and GUI libraries..."
# sudo apt-get install -y libglu1-mesa libglu1-mesa-dev glusterfs-common libglui-dev libglui2c2
# sudo apt-get install -y mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev
# sudo apt-get install -y libglfw3 libglfw3-dev
# sudo apt install libgl1-mesa-glx libgl1-mesa-dri

# Clone librealsense repository
echo "Cloning librealsense repository..."
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense || exit

# Setup udev rules
echo "Setting up udev rules..."
sudo ./scripts/setup_udev_rules.sh
sudo ./scripts/patch-realsense-ubuntu-lts-hwe.sh

# Check the patched modules installation by examining the generated log as well as inspecting the latest entries in kernel log:
# sudo dmesg | tail -n 50
# The log should indicate that a new uvcvideo driver has been registered.

# Update and install additional dependencies for GUI applications
echo "Installing additional GUI dependencies..."
sudo apt update
sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev -y


# Build librealsense
echo "Building librealsense..."
mkdir build && cd build || exit
cmake ..
sudo make uninstall && make clean && make && sudo make install
sudo ldconfig


# Run realsense-viewer
echo "Running realsense-viewer..."
realsense-viewer
