# Setup

### On your WSL terminal

Type
```bash
git clone --recursive https://github.com/NTU-Autonomous-Racing-Team/race_stack_2526 -b simracing
bash launch_in_wsl.sh
```

### After entering the docker container
*You will enter the container's terminal after the previous step finishes*
Run
```bash
bash setup_dep.sh
```
This installs all required dependencies, and to run the actual race stack

```bash
bash autodrive_devkit.sh
```

This opens the port 4567 for you to connect to in the AutoDRIVE Simulator (To be downloaded at https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/tag/2026-icra)

Please download the **practice** version so it contains the map used for qualifications.