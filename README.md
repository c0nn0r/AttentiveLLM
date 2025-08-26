# AttentiveLLM

_Attention-aware push-to-talk for voice LLMs._

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

