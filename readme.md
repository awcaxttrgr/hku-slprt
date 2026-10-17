# SLPRT: Sign Language Praat Project

## Live handshape demo

On Windows, double-click [`run_demo.bat`](./run_demo.bat). If Conda is unavailable, it downloads and installs Miniconda for the current Windows user. When creating the `slprt` environment, the launcher accepts Conda Terms of Service for Anaconda's `main`, `r`, and `msys2` channels. It then creates the environment with Python 3.10.11 when needed, installs missing requirements, and launches `live_detect.py`. Internet access is required on the first run. The demo uses the webcam when available and falls back to the bundled video. Press `Q` in the display window to stop.

Handshape images belong in `hs_char` and must be named `<three-digit decimal Unicode codepoint>.png` (for example, `A` uses `065.png`). Characters remain the internal prediction labels.

## How to run through the codes
1. run paths.py
2. put the videos under src/vid/xxx/
3. run lm_detect.py
4. run graph_extract.py
5. run extract_plot.py / run model_dataset.py


## How to train the model
1. run paths.py
2. put the videos under src/vid/xxx/
3. run lm_detect.py
4. run graph_extract.py
5. run random_split.py: randomly split the data (defaultly named as "Cynthia_full") into 0.8:0.2 train:validation sets. 
6. run model_dataset.py