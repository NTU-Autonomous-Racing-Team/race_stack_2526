#!/usr/bin/env bash
set -euo pipefail

echo "[setup_dep] Refreshing apt metadata..."
apt-get update

apt-get install -y xterm gdb

echo "[setup_dep] Installing build prerequisites for range_libc..."
apt-get install -y \
	build-essential \
	python3-dev \
	python3-numpy \
	cython3

if ! python3 -c "import Cython" >/dev/null 2>&1; then
	echo "[setup_dep] Cython module still missing after apt install; installing via pip..."
	apt-get install -y python3-pip
	python3 -m pip install Cython
fi

echo "[setup_dep] Installing ROS dependencies via rosdep..."
if ! rosdep install --from-paths src --ignore-src -r -y; then
	echo "[setup_dep] rosdep failed, retrying once after apt refresh..."
	apt-get update
	rosdep install --from-paths src --ignore-src -r -y
fi

echo "[setup_dep] Building range_libc Python wrapper..."
cd src/particle_filter/range_libc/pywrapper
bash compile.sh

apt-get install --reinstall libaom3

echo "[setup_dep] Done."
