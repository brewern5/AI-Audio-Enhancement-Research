# VISQoL Audio Processing

This guide explains how to set up and run the VISQoL backend using a Python virtual environment (venv) on Windows, macOS, and Linux.

---

## 1. Clone the Repository

```
git clone https://github.com/brewern5/AI-Audio-Enhancement-Research.git
cd AI-Audio-Enhancement-Research/backend/VISQoL
```

---

## 2. Create a Virtual Environment

### Windows
```
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```
Or, if you don't have a `requirements.txt` file:
```
pip install librosa ffmpeg-python
```

---

## 4. Run the Main VISQoL File

### Windows

```
python main.py
```
Or use the batch file:
```
run.bat
```

### macOS/Linux
```
python3 main.py
```

---

## 5. Deactivate the Virtual Environment

```
deactivate
```

---

## Notes
- Make sure your audio files are in the `assets/` directory.
- Edit `main.py` to specify the correct audio file path if needed.
- For troubleshooting, check that Python and pip are installed and available in your PATH.

---

## Example Directory Structure
```
VISQoL/
├── assets/
├── lib/
├── main.py
├── run.bat
├── requirements.txt
└── README.md
```


# Additional Information

Creators:
- Nate Brewer 
- Isha Nepal

## Lib documentation
[Librosa](https://librosa.org/doc/latest/install.html) - Audio File I/O and processing

[VISQOL](https://www.mathworks.com/help/audio/ref/visqol.html#description) - Evaluation Metrics

