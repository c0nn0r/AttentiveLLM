# AttentiveLLM

_Attention-aware look-to-talk for voice LLMs._

AttentiveLLM links a webcam to a voice-mode LLM device (e.g., an iPad). When you **look at** the device (detected via head-pose “facing camera”), a Raspberry Pi **clicks** a hacked USB mouse to **unmute** the mic. When you **look away**, it clicks again to **mute**. This enables natural, multi-agent turn-taking with multiple cameras/devices.

---

## How it works (high level)
```
┌──────────┐   video    ┌────────────────┐    TCP: play/pause    ┌──────────────┐   GPIO sink   ┌──────────┐
│ Webcam 1 │───────────▶│ head_pose.py   │──────────────────────▶│ button2.py   │──────────────▶│ Mouse SW │
└──────────┘            │  (MediaPipe)   │                       │ (TCP server) │               └────┬─────┘
                        └────────────────┘                       └──────────────┘                    │
                                                                                                     │ click
                                                                                                     ▼
                                                                                               iPad (mute/unmute)
```



Multi-device is the same pattern (different camera index, TCP port, and GPIO pin) using `*_cam2.py` for the second pipeline.

---

## Repo layout

- `head_pose.py` — webcam → head-pose estimation → sends `play`/`pause` over TCP (default port **5000**) to the Pi mouse controller. Includes calibration and adjustable thresholds.
- `head_pose_cam2.py` — same as above for a second webcam, defaults to TCP port **5001**.
- `button2.py` — Raspberry Pi TCP server on port **5000** that “clicks” the hacked mouse via a GPIO sink.
- `button2_cam2.py` — same as above for port **5001** (second device).
- `requirements.txt` — OpenCV, MediaPipe, NumPy.

---

## Hardware wiring (mouse hack)

Inside the mouse, identify the two pads of the left-click switch:

- one pad tied to battery negative → **GND pad**  
- the other pad “floating high” via pull-up → **SIGNAL pad**

Wire to the Pi:

- **GND pad** → any Pi **GND** header pin  
- **SIGNAL pad** → chosen **GPIO** (defaults: `button2.py` = **BCM 18**; `button2_cam2.py` = **BCM 13**)

Electrical behavior:

- “Released” = GPIO configured **INPUT** (high-Z)  
- “Pressed”  = GPIO configured **OUTPUT LOW** (current sink)  
- The Pi never drives HIGH into the mouse; it only sinks a small current to emulate a press.

---

## Software prerequisites

- Raspberry Pi OS (Bullseye/Bookworm recommended)
- Python 3.9+  
- Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running (single camera + single device)

Terminal A (GPIO/mouse TCP server):
```bash
# for camera/device 1
python button2.py
```
• Listens on TCP :5000 and clicks GPIO 18 on play/pause.

Terminal B (head-pose detector):
```bash
python head_pose.py
```
• Defaults to webcam index 0 and connects to TCP localhost:5000.

• Shows a preview with FPS and “Looking at Camera” status, plus hotkeys to tweak thresholds and calibration.


## Running (two cameras + two devices)

Terminal A (first device):
```bash
python button2.py
```
• Port 5000, GPIO 18.

Terminal B (second device):
```bash
python button2_cam2.py
```
• Port 5001, GPIO 13.

Terminal C (head-pose for camera 1):
```bash
python head_pose.py
```
• Uses camera index 0, connects to 5000.

Terminal D (head-pose for camera 2):
```bash
python head_pose_cam2.py
```
• Uses camera index 2, connects to 5001.

## Calibration & thresholds (runtime hotkeys)

While the head-pose window is focused:

- `R` — reset calibration  
- `1` / `2` — decrease / increase **left yaw** threshold  
- `3` / `4` — decrease / increase **right yaw** threshold  
- `5` / `6` — decrease / increase **pitch** threshold  
- `T` — run built-in test cases  
- `D` — debug current values  
- `ESC` — quit  

These are logged/overlaid in the OpenCV window.  
**Defaults:** left/right = `6°`, pitch = `8°`.

## Head-pose details (what “looking at camera” means)

• MediaPipe Face Mesh landmarks → 2D/3D sets → solvePnP to estimate yaw/pitch.

• If yaw ∈ (−left_thresh, +right_thresh) and pitch ∈ (−thr, +thr), we treat it as FORWARD (eye-contact proxy). Otherwise LEFT/RIGHT/UP/DOWN.

• More landmarks are used for robustness; focal length uses the larger image dimension.


## Troubleshooting

• No click: verify TCP connectivity and ports (5000/5001) and that the head-pose script prints “Connected to TCP server …”.

• GPIO not doing anything: confirm correct BCM pin (18/13 by default) and that the mouse pads are wired to GND and the selected GPIO.

• False triggers: increase yaw/pitch thresholds (1..6 keys) or re-run calibration
