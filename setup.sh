mkdir build
git submodule update --init

cmake -S . -B ./build/
cd build
make
cd ..