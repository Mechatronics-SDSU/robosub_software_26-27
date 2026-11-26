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
- Python: 3.11 or newer
- Hardware integrations: ZED SDK and the NVIDIA JetPack/Orin environment are needed for deployment features that use those devices.

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

### Installation

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On a development computer that is not an Orin, install the project and its
non-Orin vision dependencies:

```bash
python -m pip install -e '.[non-orin]'
```

On an Orin, install the JetPack and ZED SDK versions supported by the vehicle
image, then install compatible PyTorch/torchvision builds and the project:

```bash
python -m pip install -e .
```

Do not install the `non-orin` extra on the Orin; it pins generic PyTorch wheels
that may not match the device image.

- Edit the objects.yaml config file to choose what course to follow and target locations, read comments (IMPORTANT)
- Run "python3 btn_press.py" to run the sub after the green button is pressed
- Run "python3 launch.py" to run the sub immediately
- Run "stop" or "kill" to instantly kill the motors using an alias from the bashrc

The launch process runs vision in a separate worker that publishes detections
to `SharedMemoryWrapper`. `vision_detection_count` gives the number of
detections; matching indexes in the `vision_*` arrays hold class ID,
confidence, normalized box center and size, and depth. The configurable target
tracker publishes a confirmed, smoothed detection through `vision_target_*`.
It reports image center alignment but does not steer or fire. Target-specific
aim offsets remain to be calibrated for the torpedo board.

Vision model weights are machine-specific and are not stored in Git. Put the
selected YOLO weights at `modules/vision/models/best.pt`, or edit
`modules/vision/config.yaml` to point `model.weights` at another local file.
After installing the Orin's JetPack-compatible PyTorch/torchvision and ZED SDK,
run `python -m modules.vision.run_vision` from the repository root to check
camera and inference without starting the mission or motor loop. The config's
default source is ZED; set `source: webcam` for a USB-camera check without
depth.

### Development Tests

Install the development tools in the active virtual environment:

```bash
python -m pip install -e '.[dev]'
```

Run the hardware-independent unit tests from the repository root:

```bash
python -m pytest tests/unit
```

### Notes

- Modes: The process of completing a single objective in the pool
- States: A specific step within a mode
- If you are a contributor, please write your code in a branch, create a pull request, and a software lead will merge after a code review

### Status

- Current status: In Progress
