# 🏓 TT Track – Table Tennis Match Analyzer for TT Cup

## Overview
This app automates the analysis of table tennis TT cup tournaments. It downloads the entire live stream containing all matches in the tournament, then automatically cuts it into individual matches. For each match, it generates graphs, basic analysis, and visualizations such as ball tracking and heatmaps of ball placements.

![til](./data/output.gif)

## Requirements

### Python packages

All required Python packages are listed in the `requirements.txt` file. Install them with:

```bash
pip install -r requirements.txt
````

### System dependencies

You also need to have the following system packages installed:

* [FFmpeg](https://ffmpeg.org/) – for video processing
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) – for text recognition (OCR)

#### Debian/Ubuntu install with:

```bash
sudo apt update
sudo apt install ffmpeg tesseract-ocr
```
## Usage

This app includes an easy-to-use interface implemented with [Streamlit](https://streamlit.io/).

To run the app, simply use:

```bash
streamlit run app.py
```
## Tests
All tests are located in the `./tests` directory.

**Note:** Some tests require internet access to download videos from the web. Other tests use pre-downloaded sample files available in `./tests/data`.

To run the tests, use:

```bash
pytest ./tests
```
## Reminders

When downloading a large number of videos, you may get flagged as a bot. In such cases, you must import your own cookies.  
See the tutorial: [How do I pass cookies to yt-dlp?](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)

The whole program may take a while to finish, with YOLO being the most time-consuming part.
## Datasets
While working on this project, I needed a labeled dataset, so I created my own with around 2,000 entries.  
If you're interested, you can access it here: [View dataset on Google Drive](https://drive.google.com/drive/folders/1lDObkqGvZvd6wKxJ2YXjZrZCehVE5kq4?usp=sharing)
