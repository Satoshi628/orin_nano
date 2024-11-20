git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build
cd build
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true
make -j4
sudo make install