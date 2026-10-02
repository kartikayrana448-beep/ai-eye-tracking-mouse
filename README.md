# AI Eye Tracking Mouse 👁️🖱️

An AI-powered hands-free mouse controller that uses eye gaze and blink detection to control the computer.

## Features

- 👁️ Eye gaze controls the mouse cursor
- 👀 Blink → Left Click
- 👀 Double Blink → Right Click
- ⬆️ Look at the top of the screen → Scroll Up
- ⬇️ Look at the bottom of the screen → Scroll Down
- 🎯 9-point gaze calibration
- 📷 Real-time webcam-based face and eye tracking
- 🤖 MediaPipe Face Landmarker
- 🖱️ Real mouse control using PyAutoGUI

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- PyInstaller

## How It Works

The webcam tracks facial and eye landmarks using MediaPipe. The system extracts eye-gaze and head-position features and maps them to screen coordinates.

Blink detection is used for mouse clicks:

- Single blink → Left click
- Double blink → Right click

Looking toward the top or bottom of the screen activates scrolling.

## Running the Project

### Python Version

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1