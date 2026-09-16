import sys, argparse, os
import numpy as np

import torch
import torch.nn.functional as F

if "ipykernel" in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from models import get_model
from utils.other import new_log
from dataloading import get_dataloaders
from dataloading.destriping_dataset import get_interpolation_sets_extended
from utils.plot_utils import *
from models.color_restoration import ColorRestoration
from models.interpolate_stripes import interpolateStripes
from models.lenticules_vectorization import *

import json
from scipy.ndimage import median_filter

parser = argparse.ArgumentParser(description="colorizing script")

#### general parameters #####################################################
parser.add_argument("--tag", default="__", type=str)
parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
parser.add_argument(
    "--save-dir", help="Path to directory where models and logs should be saved saved"
)
parser.add_argument(
    "--mode",
    default="test",
    type=str,
    choices=["None", "train", "test", "train_and_test"],
    help="mode to be run",
)
parser.add_argument(
    "--save-only-color",
    action="store_true",
    default=False,
    help="avoids non useful calculation if only interested in obtaining the colored output",
)


#### data parameters ##########################################################
parser.add_argument(
    "--data", default="lenticular_full_image", type=str, help="dataset selection"
)
parser.add_argument(
    "--datafolder",
    default="/scratch/code/data_dolce",
    type=str,
    help="root directory of the dataset",
)
parser.add_argument(
    "--workers", type=int, default=4, metavar="N", help="dataloader threads"
)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument(
    "--normalize-patch", action="store_true", default=False, help="normalize patches"
)
parser.add_argument(
    "--train-val-ratio",
    type=float,
    default=1.0,
    help="ratio of the dataset to use for training",
)

#### model parameters #####################################################
parser.add_argument("--model", type=str, help="model to run for lenticules detection")
parser.add_argument(
    "--model_2", type=str, default=None, help="model to run for colorization"
)
parser.add_argument(
    "--resume", type=str, default=None, help="path to resume the model if needed"
)
parser.add_argument(
    "--resume_2", type=str, default=None, help="path to resume the model if needed"
)
parser.add_argument(
    "--style",
    type=str,
    default="new_dolce",
    choices=["new_dolce", "old_dolce", "interpolate"],
    help="type of colorization",
)
parser.add_argument(
    "--order",
    type=str,
    default="nope",
    choices=["nearest", "linear", "cubic", "nope"],
    help="type of interpolation",
)
parser.add_argument(
    "--lambda1", type=float, default=1.0, help="width variation regularization"
)
parser.add_argument(
    "--lambda2", type=float, default=10.0, help="width absolute value regularization"
)


class ColorizeSuite(object):

    def __init__(self, args):

        self.args = args

        if args.datafolder is None:
            self.data_stats = {
                "input_channels": 1,
                "output_channels": 1,
            }
        else:
            self.dataloaders, self.data_stats = get_dataloaders(args)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )

        self.model = get_model(args, self.data_stats)
        if args.resume is not None:
            self.resume(self.model, path=args.resume)
            self.model.to(self.device)

        if args.model_2 is not None:
            self.model_color = get_model(args, self.data_stats, secondary_model=True)
            if args.resume is not None:
                self.resume(self.model_color, path=args.resume_2)
            self.model_color.to(self.device)

    def resume(self, model, path):
        if not os.path.isfile(path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(path))

        model.load_state_dict(torch.load(path))

        print("model loaded.")

        return

    def to_device(self, sample, device=None):
        if device is None:
            device = self.device
        sampleout = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                sampleout[key] = val.to(device)
            elif isinstance(val, list):
                new_val = []
                for e in val:
                    if isinstance(e, torch.Tensor):
                        new_val.append(e.to(device))
                    else:
                        new_val.append(val)
                sampleout[key] = new_val
            else:
                sampleout[key] = val
        return sampleout

    def colorize_dataset(self, dataset_name="test"):
        self.experiment_folder = new_log(
            args.save_dir, args.model + "_" + args.tag, args=args
        )
        if not os.path.exists(os.path.join(self.experiment_folder, "color/")):
            os.mkdir(os.path.join(self.experiment_folder, "color/"))
        if not self.args.save_only_color:
            if not os.path.exists(os.path.join(self.experiment_folder, "lenticules/")):
                os.mkdir(os.path.join(self.experiment_folder, "lenticules/"))
            if not os.path.exists(
                os.path.join(self.experiment_folder, "lenticules_raw/")
            ):
                os.mkdir(os.path.join(self.experiment_folder, "lenticules_raw/"))
            if not os.path.exists(
                os.path.join(self.experiment_folder, "lenticules_combo/")
            ):
                os.mkdir(os.path.join(self.experiment_folder, "lenticules_combo/"))

        data_stats = []
        for sample in tqdm(self.dataloaders[dataset_name].dataset, leave=False):

            x = sample["x"]

            y, z, z_rough, _, _, stat_entry = self.colorize_image(x)

            if not stat_entry == None:
                stat_entry["name"] = sample["name"]
                data_stats.append(stat_entry)

            if not self.args.save_only_color:
                combo_z = torch.zeros((3, z.shape[-2], z.shape[-1]))
                combo_z[0:1, :, :] = z_rough
                combo_z[1:2, :, :] = z

                img = Image.fromarray(
                    (255 * z).cpu().numpy().astype("uint8").squeeze(), "L"
                )

                img.save(
                    os.path.join(
                        self.experiment_folder,
                        "lenticules/",
                        "lenticules_" + sample["name"],
                    )
                )

                img = Image.fromarray(
                    (255 * z_rough).cpu().numpy().astype("uint8").squeeze(), "L"
                )

                img.save(
                    os.path.join(
                        self.experiment_folder,
                        "lenticules_raw/",
                        "lenticules_raw_" + sample["name"],
                    )
                )

                img = Image.fromarray(
                    (255 * combo_z)
                    .squeeze()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype("uint8")
                    .squeeze(),
                    "RGB",
                )

                img.save(
                    os.path.join(
                        self.experiment_folder,
                        "lenticules_combo/",
                        "lenticules_combo_" + sample["name"],
                    )
                )

            img = Image.fromarray(
                (255 * y).squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8"),
                "RGB",
            )
            img.save(
                os.path.join(self.experiment_folder, "color/", "color" + sample["name"])
            )

        with open(os.path.join(self.experiment_folder, "stats.json"), mode="w") as file:
            json.dump(data_stats, file, indent=4)

    def colorize_image(
        self,
        raw_img,
        is_modified=False,
        lenticules_location_top=None,
        lenticules_location_bottom=None,
    ):

        while len(raw_img.shape) < 4:
            raw_img = raw_img.unsqueeze(0)

        return_device = raw_img.device

        raw_img = raw_img.to(self.device)
        if not is_modified:
            raw_pred = self.process_image_lenticules(raw_img)
            lenticules_location_bottom, lenticules_location_top, z_raster = (
                self.predict_lenticule_locations(raw_pred)
            )
        else:
            z_raster = None
            raw_pred = None

        x_stripe, stat_entry = self.extract_stripes(
            lenticules_location_bottom, lenticules_location_top, raw_img
        )

        if self.args.style == "old_dolce":
            _, Pxx_den = sg.periodogram(
                np.sum(raw_pred.cpu().numpy().squeeze(), axis=0)
            )
            w = raw_pred.squeeze().shape[1] / (50 + np.argmax(Pxx_den[50:350]))

            min_lenticule_width = int(np.floor(w) - 1)
            max_lenticule_width = int(np.ceil(w) + 1)

            z_raster = reconstruct_boundaries(
                lenticules_location_bottom,
                lenticules_location_top,
                size=raw_img.squeeze().shape,
                lenticule_min=min_lenticule_width,
                lenticule_max=max_lenticule_width,
            )
            z_raster = (
                torch.from_numpy(z_raster)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            colorize = ColorRestoration(max_lenticule_width).to(self.device)
            out_dict = colorize(raw_img, z_raster)

            if self.args.save_only_color:
                z_raster = None

            y = out_dict["y"].squeeze()
        elif self.args.style == "new_dolce":
            y = self.destriping(x_stripe)
        elif self.args.style == "interpolate":
            y = torch.from_numpy(
                interpolateStripes(
                    x_stripe.cpu().numpy().squeeze(), order=self.args.order
                )
            ).float()
        else:
            raise NotImplementedError

        gain_matrix = torch.tensor(
            [[0.789, 0.154, 0.057], [-0.286, 1.195, 0.060], [-0.049, 0.035, 1.035]]
        )
        gain_matrix = gain_matrix.unsqueeze(2).unsqueeze(2).float().to(y.device)
        y = torch.clamp(torch.sum(y.unsqueeze(0) * gain_matrix, 1), 0, 1)

        return (
            y.detach().to(return_device),
            None if z_raster is None else z_raster.detach().to(return_device),
            (
                None
                if raw_pred is None
                else raw_pred.detach().to(return_device).squeeze(0)
            ),
            lenticules_location_bottom,
            lenticules_location_top,
            stat_entry if self.args.style == "new_dolce" else None,
        )

    def process_image_lenticules(self, x):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x.to(self.device))

            if "y_pred_sigmoid" in out:
                y = out["y_pred_sigmoid"].to(x.device)
            else:
                y = out["y_pred"].to(x.device)

        return y.to(x.device).detach()

    def predict_lenticule_locations(self, model_prediction):
        _, Pxx_den = sg.periodogram(
            np.sum(model_prediction.cpu().numpy().squeeze(), axis=0)
        )
        w = model_prediction.squeeze().shape[1] / (50 + np.argmax(Pxx_den[50:350]))

        delta_max = 18
        min_lenticule_width = int(np.floor(w) - 1)
        max_lenticule_width = int(np.ceil(w) + 1)

        score_matrix = build_score_matrix(
            1 - model_prediction.cpu().numpy().squeeze(), delta_max
        )

        lenticules_location_bottom, lenticules_location_top = optimize_locations(
            score_matrix,
            delta_max,
            min_lenticule_width,
            max_lenticule_width,
            w,
            lambda1=self.args.lambda1,
            lambda2=self.args.lambda2,
        )

        if not self.args.save_only_color:
            z_raster = reconstruct_boundaries(
                lenticules_location_bottom,
                lenticules_location_top,
                size=model_prediction.squeeze().shape,
            )

            z_raster = (
                torch.from_numpy(z_raster)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(self.device)
                .squeeze(0)
            )
        else:
            z_raster = None

        return lenticules_location_bottom, lenticules_location_top, z_raster

    def extract_stripes(
        self, lenticules_location_bottom, lenticules_location_top, raw_img
    ):
        raw_img = raw_img.to(self.device)
        raw_img_np = raw_img.cpu().numpy().squeeze()
        H, W = raw_img_np.shape

        first = np.where(
            (lenticules_location_top > 0) & (lenticules_location_bottom > 0)
        )[0][0]
        last = np.where(
            (lenticules_location_top <= W - 1) & (lenticules_location_bottom <= W - 1)
        )[0][-1]

        n = np.arange(first, last)
        N = last - first
        idx_row = np.arange(H)

        curr_bottom = lenticules_location_bottom[n].astype(float)
        curr_top = lenticules_location_top[n].astype(float)
        next_bottom = lenticules_location_bottom[n + 1].astype(float)
        next_top = lenticules_location_top[n + 1].astype(float)

        curr_delta = curr_top - curr_bottom
        next_delta = next_top - next_bottom

        idx_curr_col = curr_bottom[None, :] + idx_row[:, None] * curr_delta[None, :] / H
        idx_next_col = next_bottom[None, :] + idx_row[:, None] * next_delta[None, :] / H
        width = idx_next_col - idx_curr_col

        ratios = (5 / 19, 9 / 19, 14 / 19)
        x_stripe = np.zeros((3, H, 3 * N), dtype=float)

        row_idx = idx_row[:, None]
        lent_count = n - first

        for c, r in enumerate(ratios):
            idx_col_color = idx_curr_col + width * r
            col_pre = np.floor(idx_col_color).astype(int)
            col_post = np.ceil(idx_col_color).astype(int)

            w_pre = 1 - (idx_col_color - col_pre)
            w_post = 1 - (col_post - idx_col_color)
            w_pre = w_pre / (w_pre + w_post)
            w_post = 1 - w_pre

            extr = (
                w_pre * raw_img_np[row_idx, col_pre]
                + w_post * raw_img_np[row_idx, col_post]
            )

            extr = median_filter(extr, size=(13, 1), mode="nearest")

            x_stripe[c][:, 3 * lent_count + c] = extr

        x_stripe = torch.from_numpy(x_stripe).float().unsqueeze(0)

        bottom_width = (
            lenticules_location_bottom[last] - lenticules_location_bottom[first] + 1
        )
        height_scaling = 3 * N / bottom_width

        x_stripe = F.interpolate(
            x_stripe,
            None,
            scale_factor=(height_scaling, 1),
            recompute_scale_factor=False,
        )

        resulting_height = x_stripe.shape[2]
        missing_bottom_rows_count = (
            height_scaling * H - resulting_height
        ) / height_scaling

        needed_height = np.round(H - missing_bottom_rows_count)
        needed_offset_left = lenticules_location_bottom[first]
        needed_offset_right = W - 1 - lenticules_location_bottom[last]

        stat_entry = {
            "name": "",
            "height": needed_height,
            "missing_floor": missing_bottom_rows_count,
            "left": needed_offset_left,
            "right": needed_offset_right,
            "scale": height_scaling,
        }

        return x_stripe, stat_entry

    def destriping(self, x_stripe):

        network_input_size = 256

        B, C, H, W = x_stripe.shape

        H_large, W_large = (H // network_input_size + 1) * network_input_size, (
            W // network_input_size + 1
        ) * network_input_size
        x_padded = F.pad(
            x_stripe,
            (
                0,
                W_large - W,
                0,
                H_large - H,
            ),
        )

        u_dict = get_interpolation_sets_extended(x_padded.squeeze())
        u_dict["u_r"] = u_dict["u_r"].unsqueeze(0).to(self.device)
        u_dict["u_g"] = u_dict["u_g"].unsqueeze(0).to(self.device)
        u_dict["u_b"] = u_dict["u_b"].unsqueeze(0).to(self.device)
        sample = {
            "x": x_padded.to(self.device),
            "mask": (x_padded > 0.0).float().to(self.device),
        }

        sample = {**sample, **u_dict}

        self.model_color.eval()
        with torch.no_grad():
            out_dict = self.model_color(sample)

        y_padded = out_dict["y_pred"]
        y = y_padded[:, :, 0:H, 0:W]

        return y.squeeze()


if __name__ == "__main__":

    args = parser.parse_args()
    developingSuite = ColorizeSuite(args)

    developingSuite.colorize_dataset()
