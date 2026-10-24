# SLPRT: Sign Language Praat Project

## Live handshape demo

On Windows, double-click [`run_demo.bat`](./run_demo.bat). If Conda is unavailable, it downloads and installs Miniconda for the current Windows user. When creating the `slprt` environment, the launcher accepts Conda Terms of Service for Anaconda's `main`, `r`, and `msys2` channels. It then creates the environment with Python 3.10.11 when needed, installs missing requirements, and launches `live_detect.py`. Internet access is required on the first run. The demo tries webcam indexes `-1`, `0`, `1`, `2`, and `-2`; if none opens, it plays the bundled video and displays that fallback reason in the render window. On an unexpected error, the traceback is printed and the console waits for Enter before exiting. Press `Q` in the display window to stop.

Handshape images belong in `hs_char` and must be named `<three-digit decimal Unicode codepoint>.png` (for example, `A` uses `065.png`). Characters remain the internal prediction labels.

On a newly recognized handshape, the left sidebar immediately loads and shows matching videos. Each video starts playing as soon as decoded frames are available, loops those buffered frames while the rest load in the background, then loops the complete video. Matches come from the condensed `database/data.lookup.json` index. On launch, the index is generated if missing, outdated, or schema-mismatched; regenerate it after editing `database/data.json` with `python database/build_handshape_lookup.py`. The index precomputes up to 60 ranked WebP filenames per handshape combination and the full match count for every two-hand and single-hand query; the sidebar footer displays that full count. Low-end mode shows the first 12; standard mode randomly samples 12 from the indexed results. Once either of the final two initial videos begins playing, a background lookup ranks every remaining database match and appends it to the playlist. Detection runs on a dedicated worker while the render loop stays responsive; WebP decoding uses three workers and downsizes decoded frames to at most 240x135 pixels. Decoded animations remain in a 512 MiB LRU cache across handshape changes; only uncached videos are decoded again when cache capacity is exceeded. Hidden-activation charts refresh on new detection results rather than every render. Missing-hand indicators use `000.png`; indicators sit side by side above the videos, left hand on the left and right hand on the right. Videos loop; the three-entry window advances after each lead video completes a loop. Low-end mode loads videos in batches of 6; standard mode uses batches of 12. Press `1` or `2` while the render window is active to switch modes; the active mode is marked in the footer. Startup defaults to low-end; `--performance-mode standard` selects standard mode initially. All rendered text uses bundled Public Sans and GenKiGothic2TW fonts; Cantonese captions are larger, IDs smaller, and digits use tabular spacing.

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