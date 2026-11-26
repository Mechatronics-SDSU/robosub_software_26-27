# Robosub Software 2026
This is the codebase for the SDSU Mechatronics Club competing in Robosub 2026, which manages sensor data, motor control, computer vision, and mission control, among other things. 

### Outline

- Date Created: 01/01/2026
- Contributors:
    - Ryan Sundermeyer (GitHub: @rsunderr, Discord: @.kech)
    - Kai Chan (GitHub: @kchan5071, Discord: @kialli)
    - Andy Chen (GitHub: @AndyC534, Discord: @creatine4053)
    - Sam Georges (GitHub: @thesamgeorges, Discord: @smoonth.)
    - Joe Lofrese (GitHub: @Joelofrese, Discord: @croppie_luvr)
    - Juan Cota (GitHub: @Juanlo01, Discord: @chewchew0366)
    - Malaika Joiner (GitHub: @malaikajoiner, Discord: @314piekitty)
- Dependencies:
    - Python 3.11+
    - Yolov5 7.0.13
    - Argparse 1.4.0
    - Inputs 0.5
    - Python-can 3.3.3
    - cffi 1.16.0
    - onnx 1.16.1
    - PyYAML 6.0
    - Pyserial 3.5
    - Matplotlib 3.7.2
    - Zed SDK
    - NVIDIA Orin SDK

### Key Directories

- modules
    - Where the key features of the code are kept, primarily sensor handling, PIDs, and motor control
- utils
    - Miscellaneous helper or setup scripts not used in real time
- fsm
    - Where the various mode state machines are kept, which handle decision making
- display_manager
    - Display code kept here

### Key Files

- launch.py
    - The primary file of the code base, manages mission control by switching between modes and handling subprocesses
- shared_memory.py
    - How all of the different files communicate, through a list of shared variables passed through an object
- objects.yaml
    - The config file for the whole code base, lets you choose the pool at the top and customize coordinates
- btn_press.py
    - This script will read input from the button on caracara and run launch.py when green is pressed, running the sub
- pyproject.toml
    - Auto generated file tracking python dependencies

### Usage

- Edit the objects.yaml config file to choose what course to follow and target locations, read comments (IMPORTANT)
- Run "python3 btn_press.py" to run the sub after the green button is pressed
- Run "python3 launch.py" to run the sub immediately
- Run "stop" or "kill" to instantly kill the motors using an alias from the bashrc

The experimental TRAX2-based ESKF is enabled with `ROBOSUB_ENABLE_ESKF=1`.
It is disabled by default while sensor frames, timing, and pool logs are being
validated. When enabled, the depth and TRAX2 interfaces start with the
estimator, while raw DVL values remain available as the fallback path.

### Docker Development Environment

The software-only development container runs the hardware-independent tests
without requiring the Orin, ZED, DVL, TRAX2, or motor controller:

```bash
docker compose build
docker compose run --rm robosub-dev
```

The repository is mounted into the container, so source edits are available
without rebuilding. To open a shell inside the environment:

```bash
docker compose run --rm robosub-dev bash
```

This container intentionally does not expose hardware devices. Hardware
passthrough and the JetPack/ZED runtime should be added as a separate deployment
configuration after the software estimator is validated with replay data.

To replay recorded ESKF sensor data inside the container:

```bash
docker compose run --rm robosub-dev \
  python tools/replay_eskf.py recordings/sensors.csv recordings/estimated.csv
```

Replay input is CSV with timestamped TRAX2 acceleration/gyro/attitude, DVL
velocity, depth, and optional DVL position columns. The output contains the
estimated position, velocity, attitude, covariance trace, and validity flag.

### Notes

- Modes: The process of completing a single objective in the pool
- States: A specific step within a mode
- If you are a contributor, please write your code in a branch, create a pull request, and a software lead will merge after a code review

### Status

- Current status: In Progress
