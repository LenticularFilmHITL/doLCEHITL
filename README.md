# HitL Color Reconstruction of Lenticular Films

Official implementation of the paper **"A Human-in-the-Loop Deep Learning Framework for Color Reconstruction of Lenticular Films"** ([arXiv:2608.02835](https://arxiv.org/abs/2608.02835)).

## Abstract

Historical lenticular films, such as those created with the Kodacolor process, encode color information in a distinctive spatial format. This structure requires specialized techniques for accurate color reconstruction. While recent signal processing approaches like doLCE and deep learning methods like deep-doLCE have advanced automated color recovery, they often fail with cases such as curved lenticules, low-contrast, or badly captured regions. We propose a human-in-the-loop (HITL) deep learning framework which is designed for color reconstruction in lenticular films. Our approach introduces an editable, vector-based representation of lenticule boundaries, allowing experts to interactively refine boundary positions before color extraction and demosaicing. This decoupled architecture enables targeted corrections and iterative fine-tuning, embedding expert knowledge into the detection model and improving robustness across challenging frames. To preserve image details using information solely present in the original silver emulsion, we merge the reconstructed chrominance with the original film scan’s luminance. We evaluate our pipeline on a challenging lenticular film sequence where previous automated approaches fail and the reconstructed colors are not suitable for exhibition. In contrast, our HITL approach successfully produces high-quality, exhibitable color reconstructions with preserved texture. This work is the first to combine expert guidance, editable intermediate representations, and texture-preserving post-processing for lenticular film color reconstruction, advancing the state of the art in this field.

## Contents

- **`app.py`**: Gradio app implementing a human-in-the-loop workflow\
  (detection → interactive correction → dataset creation → fine-tuning).\
  Started automatically by the container.
- **`train.sh`** / **`colorize_dataset.sh`**: Wrappers for running training and colorization from the command line, outside the app.
- **`preprocess_dataset.py`**: Preprocesses raw scans into dataset by cropping, grayscale-converting and flipping (only needed for BGR encoded lenticules).
- **`finish_colorized_frame.py`**: Merges reconstructed chrominance back into the full-resolution scan to improve detail.


### How to pull the checkpoints and images

Run the following commands to download the checkpoints

1. **Install Git LFS** (if you haven't already on your system):
   ```bash
   git lfs install
   ```

2. **Pull the actual LFS files**:
   ```bash
   git lfs pull
   ```

## Installation

Clone the repository and place your imageset in `data/images/raw/<YOUR_SET>/`:

```text
data/
├── images/
│   ├── raw/<YOUR_SET>/
│   │   ├── inputgrayIMGs/         # 8-bit grayscale frames
│   │   └── test_images_list.json  # frames to process
│   └── colorized/                 # outputs from  manual colorization
├── checkpoints/                   # model checkpoints 
└── datasets/                      # datasets created by the app
```

When starting from raw scans use `preprocess_dataset.py` to produce this layout. To exclude frames, remove them before pre-processing or drop them from the JSON list.

## Docker
### 1. Build the docker image

```bash
docker build -t <IMAGE_NAME> .
```

### 1. Run the docker container
Run the app (open <http://localhost:7860>):

```bash
docker run --rm -p 7860:7860 --gpus all --shm-size=1g \
  -v "$(pwd)/data/images:/images" \
  -v "$(pwd)/data/checkpoints:/checkpoints" \
  -v "$(pwd)/data/datasets:/datasets" \
  -v "$(pwd):/home" \
  <IMAGE_NAME>
```

The container launches `app.py` on startup, so override the command with `bash` and add `-it` for an interactive shell (needed to run scripts etc.):

```bash
docker run --rm -it -p 7860:7860 --gpus all --shm-size=1g \
  -v "$(pwd)/data/images:/images" \
  -v "$(pwd)/data/checkpoints:/checkpoints" \
  -v "$(pwd)/data/datasets:/datasets" \
  -v "$(pwd):/home" \
  <IMAGE_NAME> bash
```

## Usage
### Gradio app

The app walks through five steps:

1. **Mode**: start fresh from the shipped detection checkpoint, or continue from a previously fine-tuned checkpoint.
2. **Checkpoint**: (Only on continue) pick tag → experiment → checkpoint.
3. **Images**: pick an image set from `/images/raw/`.
4. **Selection**: frames are processed in pages of 100. Work can start on the complete colorization of the first page.
   - Select the wanted frame
   - If needed change **shift** and **tilt** for improved coloring.
   - **Apply to page** copies the current **shift**/**tilt** to all frames on the current page (neighbouring frames often need simmilar corrections).
   - Use the **crop** sliders to cut off badly colorized regions.
   - **Take image** selects the current frame for the new dataset, **remove** deselects it again.
   - If the page is empty or no usable frames remain, move on using the **page** slider.
   - When done enter a dataset name at the bottom of the page. On confirmation the dataset is created and written to `/datasets/<name>/`.
5. **Training**: Adjust parameters if needed (empty cell = default) and press **train**.
   - On completion either start another training with different parameters on the same dataset, or reload the page and start over from step 1 with the new checkpoint.

## Demo

Demo of using **app.py**

<details>
<summary>Preview as GIF / local file</summary>


Full resolution: 

</details>

## Authors and acknowledgment

### Main collaborators

- Saptarshi Neil Sinha — Project administration and conceptualization
- Tiago Kleist
- Giorgio Trumpy


## License
This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## Citation

If you use this repository or our pipeline in your research, please cite our paper:

~~~bibtex
@misc{sinha2026humanintheloopdeeplearningframework,
      title={A Human-in-the-Loop Deep Learning Framework for Color Reconstruction of Lenticular Films}, 
      author={Saptarshi Neil Sinha and Tiago Kleist and Giorgio Trumpy},
      year={2026},
      eprint={2608.02835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.02835}, 
}
~~~