#increase swap size
sudo vim /etc/dphys-swapfile
CONF_SWAPSIZE=2048

sudo /etc/init.d/dphys-swapfile restart swapon -s


sudo apt update && sudo apt upgrade -y
sudo apt install git cmake libssl-dev libusb-1.0-0-dev -y

sudo apt update && sudo apt upgrade -y
sudo apt install git cmake libssl-dev libusb-1.0-0-dev -y

sudo apt-get install -y libdrm-amdgpu1 libdrm-amdgpu1-dbgsym libdrm-dev libdrm-exynos1 libdrm-exynos1-dbgsym libdrm-freedreno1 libdrm-freedreno1-dbgsym libdrm-nouveau2 libdrm-nouveau2-dbgsym libdrm-omap1 libdrm-omap1-dbgsym libdrm-radeon1 libdrm-radeon1-dbgsym libdrm-tegra0 libdrm-tegra0-dbgsym libdrm2 libdrm2-dbgsym

sudo apt-get install -y libglu1-mesa libglu1-mesa-dev glusterfs-common libglu1-mesa libglu1-mesa-dev libglui-dev libglui2c2

sudo apt-get install -y libglu1-mesa libglu1-mesa-dev mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense

sudo ./scripts/setup_udev_rules.sh

sudo apt update
sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

mkdir build && cd build
cmake ../ -DBUILD_EXAMPLES=true -DFORCE_LIBUVC=true
make -j$(nproc)

realsense-viewer


# To enable mulithreading and parallelism
cd ~
wget https://github.com/PINTO0309/TBBonARMv7/raw/master/libtbb-dev_2018U2_armhf.deb
sudo dpkg -i ~/libtbb-dev_2018U2_armhf.deb
sudo ldconfig
rm libtbb-dev_2018U2_armhf.deb

# To serialise image data and transfer to python or java 
cd ~
git clone --depth=1 -b v3.5.1 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure
make -j1
sudo make install
cd python
export LD_LIBRARY_PATH=../src/.libs
python3 setup.py build --cpp_implementation 
python3 setup.py test --cpp_implementation
sudo python3 setup.py install --cpp_implementation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
sudo ldconfig
protoc --version

# To install opencv
sudo apt-get install libgl1-mesa-glx libegl1-mesa libgles2-mesa-dev

sudo apt autoremove libopencv3
wget https://github.com/mt08xx/files/raw/master/opencv-rpi/libopencv3_3.4.3-20180907.1_armhf.deb
sudo apt install -y ./libopencv3_3.4.3-20180907.1_armhf.deb
sudo ldconfig


### compilation instruction
 g++ -std=c++11 capture_pointcloud.cpp -lrealsense2 -o capture_pointcloud2