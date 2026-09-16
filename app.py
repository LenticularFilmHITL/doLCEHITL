import os
import gradio as gr
import json
import numpy as np
import colorize_suite as cs
import developing_suite as ds
from torchvision import transforms
from PIL import Image
import shutil
from tqdm import tqdm
import pandas as pd
from functools import lru_cache
import atexit

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import app_utils.state as state
import app_utils.config as config

os.makedirs(config.CACHE_DIR, exist_ok=True)
atexit.register(lambda: shutil.rmtree(config.CACHE_DIR, ignore_errors=True))


def clear_cache():
    shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)


def load_scalars(
    logdir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()
    rows_train = [
        {"step": e.step, "value": e.value, "tag": tag}
        for tag in acc.Tags()["scalars"]
        for e in acc.Scalars(tag)
        if str(tag).startswith("train")
    ]
    rows_val = [
        {"step": e.step, "value": e.value, "tag": tag}
        for tag in acc.Tags()["scalars"]
        for e in acc.Scalars(tag)
        if str(tag).startswith("val")
    ]
    return pd.DataFrame(rows_train), pd.DataFrame(rows_val)


@lru_cache(maxsize=1)
def create_colorize_suite(img_dir, ckpt_path):
    args = cs.parser.parse_args(
        [
            "--model=UResNet",
            "--model_2=UResNetColorize",
            f"--resume={ckpt_path}",
            "--resume_2=/checkpoints/model_colorize.pth.tar",
            "--lambda1=0.5",
            "--lambda2=5",
            f"--datafolder={img_dir}",
            "--save-only-color",
        ]
    )

    return cs.ColorizeSuite(args)


def colorize_image(caption, state, shift, tilt, img_dir, ckpt_path):
    if caption == "":
        return (gr.skip(),) * 2

    lenticules_location_top, lenticules_location_bottom = state[
        caption
    ].shift_tilt_lenticules(shift, tilt)

    colorize_suite = create_colorize_suite(img_dir, ckpt_path)

    image_path = os.path.join(img_dir, "inputgrayIMGs", caption)
    raw_image = Image.open(image_path)
    image_tensor = prepare_image(image=raw_image)

    colorized_image_tensor, _, _, _, _, _ = colorize_suite.colorize_image(
        image_tensor,
        True,
        lenticules_location_top,
        lenticules_location_bottom,
    )

    colorized_image = Image.fromarray(
        (255 * colorized_image_tensor)
        .squeeze()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype("uint8"),
        "RGB",
    )

    state[caption].image_data = colorized_image
    state[caption].cache_webp()

    return state, state[caption].overlay


def apply_crop(caption, state, left_crop, right_crop):
    if caption == "":
        return (gr.skip(),) * 4

    state[caption].left = left_crop
    state[caption].right = right_crop
    left_crop_max, right_crop_min, left_scale, right_scale = state[
        caption
    ].crop_slider_limits

    return (
        state,
        gr.update(maximum=left_crop_max, scale=left_scale, value=left_crop),
        gr.update(minimum=right_crop_min, scale=right_scale, value=right_crop),
        state[caption].overlay,
    )


def init_images(img_dir, ckpt_path, progress=gr.Progress(track_tqdm=False)):
    colorize_suite = create_colorize_suite(img_dir, ckpt_path)
    page: state.GalleryState = {}
    dataset = colorize_suite.dataloaders["test"].dataset

    max_iter = len(dataset)

    bar = tqdm(
        dataset,
        bar_format="{n_fmt}/{total_fmt} - {elapsed}<{remaining} - {rate_fmt}",
        disable=False,
    )
    for idx, sample in enumerate(bar):
        progress(progress=(idx + 1, max_iter), desc=str(bar))

        x = sample["x"]
        img_name = sample["name"]

        y, _, _, bottom, top, _ = colorize_suite.colorize_image(x)

        colorized_img = Image.fromarray(
            (255 * y).squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8"),
            "RGB",
        )

        page[img_name] = state.ImageState(
            name=img_name,
            image_data=colorized_img,
            raw_width=x.size(dim=2),
            lenticules_top=top,
            lenticules_bottom=bottom,
        )

        if ((idx + 1) % config.PAGE_SIZE == 0) or idx >= (max_iter - 1):
            current_page = (idx) // config.PAGE_SIZE
            yield (
                page,
                gr.update(maximum=max(1, current_page), visible=current_page > 0),
            )
            page = {}


def shift_lenticules(caption, page, state, ckpt_path, img_dir):
    if caption == "":
        return gr.skip()
    else:
        current_state = state[caption]
        colorize_suite = create_colorize_suite(img_dir, ckpt_path)
        shift = current_state.lenticules.shift
        tilt = current_state.lenticules.tilt
        items = list(state.items())[
            page * config.PAGE_SIZE : (page + 1) * config.PAGE_SIZE
        ]
        for name, item in items:
            if not (item.selected or name == caption):
                state[name].lenticules.shift = shift
                state[name].lenticules.tilt = tilt
                lenticules_location_top = state[name].lenticules.shifted_top
                lenticules_location_bottom = state[name].lenticules.shifted_bottom

                image_path = os.path.join(img_dir, "inputgrayIMGs", name)
                raw_image = Image.open(image_path)
                image_tensor = prepare_image(image=raw_image)

                colorized_image_tensor, _, _, _, _, _ = colorize_suite.colorize_image(
                    image_tensor,
                    True,
                    lenticules_location_top,
                    lenticules_location_bottom,
                )

                colorized_image = Image.fromarray(
                    (255 * colorized_image_tensor)
                    .squeeze()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype("uint8"),
                    "RGB",
                )

                state[name].image_data = colorized_image
                state[name].cache_webp()

        return state


def update_datasets(state, page=0):
    if state == {}:
        return gr.skip(), gr.skip()
    items = list(state.items())[page * config.PAGE_SIZE : (page + 1) * config.PAGE_SIZE]
    old_ds = []
    new_ds = []
    for _, item in items:
        if item.selected:
            new_ds.append((item.preview_path, item.name))
        else:
            old_ds.append((item.preview_path, item.name))
    return old_ds, new_ds


def train(params, ds_name, progress=gr.Progress(track_tqdm=True)):
    arg_list = [
        "--save-dir=/checkpoints/",
        "--save-model=best",
        "--val-every-n-epochs=10",
        "--resume=/checkpoints/model_lenticule_detection.pth.tar",
        "--mode=train",
        "--data=lenticular_square_patch",
        f"--datafolder=/datasets/{ds_name}/",
        "--optimizer=adam",
        "--model=UResNet",
    ]
    params_dict = params.to_dict("index")
    for _, param in params_dict.items():
        value = param["value"] if param["value"] else param["default"]
        arg_list.append(f"--{param['param']}={value}")

    args = ds.parser.parse_args(arg_list)
    developingSuite = ds.DevelopingSuite(args)
    logdir = developingSuite.writer.get_logdir()

    yield gr.skip(), gr.update(visible=True), gr.update(visible=True), gr.update(
        active=True
    ), logdir

    developingSuite.train_and_eval()

    developingSuite.writer.close()
    yield "Done", *load_scalars(logdir), gr.update(active=False), gr.skip()


def prepare_image(image):
    if np.max(image) > 255:
        image = np.array(image)
        image = image / (256 * 256 - 1)
        image = image.astype(float)

    image = image.convert("L")
    tensor_transform = transforms.ToTensor()
    image_tensor = tensor_transform(image)

    image_tensor = image_tensor.float()

    return image_tensor


def concat_path(base, new):
    return os.path.join(base, new)


def dropdown_listdir(dir, start, end):
    choices = sorted(
        choice
        for choice in os.listdir(dir)
        if choice.startswith(start) and choice.endswith(end)
    )

    return gr.Dropdown(choices=choices, interactive=True)


def new_set_selected(caption, data_state, selected):
    if caption == "":
        return gr.skip()
    else:
        data_state[caption].set_selected(selected)

    return data_state


def create_dataset(name, state, img_dir, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Starting")
    dataset_path = os.path.join(config.DATASET_PATH, name)
    gray_path = os.path.join(dataset_path, config.GRAY_PATH)
    truth_path = os.path.join(dataset_path, config.TRUTH_PATH)
    os.mkdir(dataset_path)
    os.mkdir(gray_path)
    os.mkdir(truth_path)

    final_state = dict(state)

    image_list = [caption for caption, item in final_state.items() if item.selected]
    for caption in tqdm(image_list, desc="Building dataset"):
        item = final_state[caption]
        image_path = os.path.join(img_dir, config.GRAY_PATH, caption)
        lenticules = item.lenticules
        filename, extension = os.path.splitext(caption)
        lent_image_name = "debugImage_" + filename + "_truth" + extension
        raw_image = Image.open(image_path)
        w, h = raw_image.size
        lent_array = cs.reconstruct_boundaries(
            lenticules_location_bottom=lenticules.shifted_bottom,
            lenticules_location_top=lenticules.shifted_top,
            size=(h, w),
        )
        lent_image = Image.fromarray((255 * lent_array).astype("uint8"), "L")

        if item.is_cropped:
            raw_image = raw_image.crop((w * item.left, 0, w * item.right, h))
            lent_image = lent_image.crop((w * item.left, 0, w * item.right, h))

        raw_image.save(os.path.join(gray_path, caption))
        lent_image.save(os.path.join(truth_path, lent_image_name))
    with open(
        os.path.join(dataset_path, "train_val_images_list.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(image_list, file)
    return gr.update(interactive=False)


def build_image(caption, state):
    if caption == "":
        return None

    return state[caption].overlay


def create_training_param_frame(img_dir):
    param_frame = pd.DataFrame(
        [
            {
                "param": "device",
                "desc": "Training device",
                "default": "cuda",
                "value": "",
            },
            {
                "param": "tag",
                "desc": "Tag for training",
                "default": img_dir,
                "value": "",
            },
            {
                "param": "batch-size",
                "desc": "Batch size used for training/validation",
                "default": "4",
                "value": "",
            },
            {
                "param": "train-val-ratio",
                "desc": "Ratio of trainingdata to validationdata",
                "default": "0.8",
                "value": "",
            },
            {
                "param": "epochs",
                "desc": "Number of epochs for training",
                "default": "125",
                "value": "",
            },
            {
                "param": "lr",
                "desc": "Learningrate for training",
                "default": "1e-05",
                "value": "",
            },
            {
                "param": "logstep-train",
                "desc": "Log training every x steps",
                "default": "25",
                "value": "",
            },
        ]
    )
    return gr.DataFrame(
        value=param_frame,
        type="pandas",
        datatype="str",
        interactive=True,
        buttons=[],
        elem_id="param_frame",
        show_search="none",
        static_columns=[0, 1, 2],
    )


page_selection_slider = gr.Slider(
    minimum=0, maximum=1, value=0, step=1, label="page", visible=False
)

with gr.Blocks() as iface:
    timer = gr.Timer(value=10, active=False)
    img_pth_state = gr.State("/images/raw/")
    ckpt_pth_state = gr.State("/checkpoints/")
    logdir_state = gr.State("")
    selected_image_old = gr.Textbox(interactive=False, value="", visible=False)
    selected_image_new = gr.Textbox(interactive=False, value="", visible=False)
    data_state = gr.State(state.GalleryState)
    loading_page_state = gr.State(state.GalleryState)
    with gr.Walkthrough(selected=0) as walkthrough:
        with gr.Step("Mode", id=0):
            with gr.Row():
                new_model_button = gr.Button(value="start fresh")
                checkpoint_model_button = gr.Button(
                    value="start from previous checkpoint"
                )
        with gr.Step("Checkpoint", id=1):
            with gr.Row():
                ckpt_tag = gr.Dropdown(interactive=True)
                exp_number = gr.Dropdown(interactive=False)
                exp_ckpt = gr.Dropdown(interactive=False)
        with gr.Step("Images", id=2):
            img_dir = gr.Dropdown()
        with gr.Step("Selection", id=3):
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        crop_slider_left = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0,
                            step=0.001,
                            label="left crop",
                        )
                        crop_slider_right = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=1,
                            step=0.001,
                            label="right crop",
                        )
                    current_image = gr.Image(
                        value=None,
                        format="png",
                        image_mode="RGBA",
                        type="pil",
                        interactive=False,
                    )
                with gr.Group():
                    with gr.Row():
                        old_dataset = gr.Gallery(
                            value=[],
                            format="webp",
                            type="filepath",
                            elem_id="old_ds",
                            object_fit="contain",
                            buttons=[],
                            interactive=True,
                            allow_preview=False,
                            file_types=["image"],
                            sources=list(),
                        )
                        with gr.Column():
                            shift_slider = gr.Slider(
                                minimum=-13,
                                maximum=13,
                                value=0,
                                step=0.01,
                                label="shift",
                            )
                            tilt_slider = gr.Slider(
                                minimum=-13,
                                maximum=13,
                                value=0,
                                step=0.01,
                                label="tilt",
                            )
                            apply_page_button = gr.Button(value="Apply to page")
                            take_button = gr.Button(value="Take image")
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1):
                            page_selection_slider.render()
                        with gr.Column(scale=1):
                            loading_placeholder_text = gr.Text(
                                interactive=False,
                                value="",
                                show_label=False,
                                visible=True,
                            )
                with gr.Group():
                    with gr.Row():
                        new_dataset = gr.Gallery(
                            value=[],
                            format="webp",
                            type="filepath",
                            elem_id="new_ds",
                            object_fit="contain",
                            buttons=[],
                            interactive=True,
                            allow_preview=False,
                            file_types=["image"],
                            sources=list(),
                        )
                        with gr.Column():
                            remove_button = gr.Button(value="remove")
                    create_dataset_name = gr.Text(
                        type="text", label="Create a new dataset", submit_btn=True
                    )
        with gr.Step("Training", id=4):
            with gr.Row():
                train_progress_plot = gr.LinePlot(
                    x="step",
                    y="value",
                    color="tag",
                    visible=False,
                    x_title="iteration",
                    x_axis_format="d",
                    y_title="value",
                    y_axis_format=".2f",
                )
                val_progress_plot = gr.LinePlot(
                    x="step",
                    y="value",
                    color="tag",
                    visible=False,
                    x_title="epoch",
                    x_axis_format="d",
                    y_title="value",
                    y_axis_format=".2f",
                )
            train_params_frame = gr.DataFrame()
            train_progress = gr.Textbox(show_label=False, value="", interactive=False)
            train_button = gr.Button(value="train")

    def load_page_to_state(page_state, data_state):
        data_state.update(page_state)
        return data_state

    loading_page_state.change(
        fn=load_page_to_state,
        inputs=[loading_page_state, data_state],
        outputs=data_state,
    )

    train_button.click(
        fn=train,
        inputs=[train_params_frame, create_dataset_name],
        outputs=[
            train_progress,
            train_progress_plot,
            val_progress_plot,
            timer,
            logdir_state,
        ],
        show_progress_on=train_progress,
    )

    timer.tick(
        load_scalars,
        inputs=logdir_state,
        outputs=[train_progress_plot, val_progress_plot],
        concurrency_limit=None,
    )

    create_dataset_name.submit(
        fn=create_dataset,
        inputs=[create_dataset_name, data_state, img_pth_state],
        outputs=[create_dataset_name],
    ).success(
        fn=create_training_param_frame,
        inputs=img_dir,
        outputs=train_params_frame,
    ).success(
        lambda: gr.Walkthrough(selected=4), outputs=walkthrough
    )

    take_button.click(
        fn=lambda caption, data_state: new_set_selected(
            caption,
            data_state,
            True,
        ),
        inputs=[
            selected_image_old,
            data_state,
        ],
        outputs=[data_state],
        js="(name, ...args) => window.myOldSelectedFn(name, ...args)",
    )

    remove_button.click(
        fn=lambda caption, data_state: new_set_selected(
            caption,
            data_state,
            False,
        ),
        inputs=[
            selected_image_new,
            data_state,
        ],
        outputs=[data_state],
        js="(name, ...args) => window.myNewSelectedFn(name, ...args)",
    )

    gr.on(
        trigger_mode="always_last",
        triggers=[shift_slider.release, tilt_slider.release],
        fn=colorize_image,
        inputs=[
            selected_image_old,
            data_state,
            shift_slider,
            tilt_slider,
            img_pth_state,
            ckpt_pth_state,
        ],
        outputs=[data_state, current_image],
        js="(name, ...args) => window.myOldSelectedFn(name, ...args)",
    )

    gr.on(
        trigger_mode="always_last",
        triggers=[crop_slider_left.release, crop_slider_right.release],
        fn=apply_crop,
        inputs=[
            selected_image_old,
            data_state,
            crop_slider_left,
            crop_slider_right,
        ],
        outputs=[data_state, crop_slider_left, crop_slider_right, current_image],
        js="(name, ...args) => window.myOldSelectedFn(name, ...args)",
    )

    apply_page_button.click(
        fn=shift_lenticules,
        inputs=[
            selected_image_old,
            page_selection_slider,
            data_state,
            ckpt_pth_state,
            img_pth_state,
        ],
        outputs=[data_state],
        js="(name, ...args) => window.myOldSelectedFn(name, ...args)",
    )

    def select_image(caption, state):
        if caption == "":
            return (
                None,
                gr.update(maximum=1, scale=1, value=0),
                gr.update(minimum=0, scale=1, value=1),
                0,
                0,
            )

        else:
            current_state = state[caption]
            left_crop_max, right_crop_min, left_scale, right_scale = (
                current_state.crop_slider_limits
            )
            return (
                current_state.overlay,
                gr.update(
                    maximum=left_crop_max, scale=left_scale, value=current_state.left
                ),
                gr.update(
                    minimum=right_crop_min, scale=right_scale, value=current_state.right
                ),
                current_state.lenticules.shift,
                current_state.lenticules.tilt,
            )

    old_dataset.select(
        fn=lambda x: x,
        inputs=[selected_image_old],
        outputs=[selected_image_old],
        js="(name, ...args) => window.myOldSelectedFn(name, ...args)",
    )

    gr.on(
        trigger_mode="always_last",
        triggers=[data_state.change, page_selection_slider.release],
        fn=update_datasets,
        inputs=[data_state, page_selection_slider],
        outputs=[old_dataset, new_dataset],
    ).then(
        trigger_mode="always_last",
        fn=lambda x: x,
        inputs=[selected_image_old],
        outputs=[selected_image_old],
        js="(name, ...args) => window.myOldSelectedFn(name, ...args)",
    )

    gr.on(
        triggers=[selected_image_old.change],
        fn=select_image,
        inputs=[selected_image_old, data_state],
        outputs=[
            current_image,
            crop_slider_left,
            crop_slider_right,
            shift_slider,
            tilt_slider,
        ],
    )

    new_model_button.click(
        fn=lambda x: concat_path(x, "model_lenticule_detection.pth.tar"),
        inputs=ckpt_pth_state,
        outputs=ckpt_pth_state,
    ).success(
        lambda img_pth_state: dropdown_listdir(img_pth_state, "", ""),
        inputs=[img_pth_state],
        outputs=[img_dir],
    ).success(
        lambda: gr.Walkthrough(selected=2), outputs=walkthrough
    )

    checkpoint_model_button.click(
        fn=lambda x: concat_path(x, "lenticular_square_patch"),
        inputs=ckpt_pth_state,
        outputs=ckpt_pth_state,
    ).success(
        lambda x: dropdown_listdir(x, "UResNet", ""),
        inputs=ckpt_pth_state,
        outputs=ckpt_tag,
    ).success(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    )

    ckpt_tag.select(
        fn=concat_path, inputs=[ckpt_pth_state, ckpt_tag], outputs=ckpt_pth_state
    ).success(
        fn=lambda x: dropdown_listdir(x, "experiment", ""),
        inputs=ckpt_pth_state,
        outputs=exp_number,
    ).success(
        fn=lambda: gr.update(interactive=False),
        outputs=ckpt_tag,
    )

    exp_number.select(
        fn=concat_path, inputs=[ckpt_pth_state, exp_number], outputs=ckpt_pth_state
    ).success(
        fn=lambda x: dropdown_listdir(x, "", ".pth.tar"),
        inputs=ckpt_pth_state,
        outputs=exp_ckpt,
    ).success(
        fn=lambda: gr.update(interactive=False),
        outputs=exp_number,
    )

    exp_ckpt.select(
        fn=concat_path, inputs=[ckpt_pth_state, exp_ckpt], outputs=ckpt_pth_state
    ).success(
        fn=lambda: gr.update(interactive=False),
        outputs=exp_ckpt,
    ).success(
        lambda img_pth_state: dropdown_listdir(img_pth_state, "", ""),
        inputs=[img_pth_state],
        outputs=[img_dir],
    ).success(
        lambda: gr.Walkthrough(selected=2), outputs=walkthrough
    )

    img_dir.select(
        fn=concat_path, inputs=[img_pth_state, img_dir], outputs=img_pth_state
    ).success(
        fn=lambda: gr.update(interactive=False),
        outputs=img_dir,
    ).success(
        lambda: gr.Walkthrough(selected=3), outputs=walkthrough
    ).success(
        fn=init_images,
        inputs=[img_pth_state, ckpt_pth_state],
        outputs=[loading_page_state, page_selection_slider],
        show_progress_on=[loading_placeholder_text],
    ).success(
        fn=lambda: gr.update(visible=False), outputs=[loading_placeholder_text]
    )

    iface.unload(clear_cache)

iface.launch(
    css_paths=["styles.css"],
    head_paths=["js_functions.html"],
    allowed_paths=[config.CACHE_DIR],
)
