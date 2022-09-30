import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
# import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 8]
import torch
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist
from ltr.data.image_loader import opencv_loader
from ltr.data.loader import ltr_collate
from pytracking.features.net_wrappers import NetWithBackbone
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

dataset_name = 'otb'
batch_size = 10


def compute_best_appearance_score():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = get_dataset(dataset_name)
    net = NetWithBackbone(net_path='dimp50.pth', use_gpu=torch.cuda.is_available())
    net.initialize()
    prroi_pool = PrRoIPool2D(3, 3, 1 / 16)

    sequence_ids = ...

    # variable for re-use
    batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(device)

    for seq, seq_i in zip((dataset), range(len(dataset))):
        if os.path.isfile(f"./results/best_appearance/{seq.name}_corr_scores.pt"):
            continue
        print(f"Running ({seq_i + 1}/{len(dataset)}): {seq.name} ")
        num_frames = len(seq.frames)
        if not os.path.isfile(f"./tmp/{seq.name}_b{batch_size}_{0}.pt"):
            for batch_start in tqdm(range(0, num_frames, batch_size)):
                end = min(batch_start + batch_size, num_frames)
                frames = ltr_collate([opencv_loader(path) for path in seq.frames[batch_start:end]]).permute(0, 3, 1, 2)
                bboxes = ltr_collate(seq.ground_truth_rect[batch_start:end]).float().to(device)
                bboxes = bboxes.clone()
                bboxes[:, 2:4] = bboxes[:, 0:2] + bboxes[:, 2:4]
                rois = torch.cat((batch_index[:bboxes.shape[0]], bboxes), dim=1)
                # extract features
                feat = net.extract_backbone(frames)
                # extract ROI Features
                pooled_feat = prroi_pool(feat['layer3'], rois)
                # pooled_feat = feat['layer3'][:,:,:3,:3]
                torch.save(pooled_feat, f"./tmp/{seq.name}_b{batch_size}_{batch_start}.pt")

        all_pooled, product = load_all_pooled(seq, batch_size)
        max_until_frame = []
        for i in range(1, num_frames):
            max_until_frame.append(product[i, :i].max())

        torch.save(torch.tensor(max_until_frame), f"./results/best_appearance/{seq.name}_corr_scores.pt")
        print(torch.load(f"./results/best_appearance/{seq.name}_corr_scores.pt")[:5])


def load_all_pooled(seq, batch_size):
    num_frames = len(seq.frames)
    all_pooled = None
    for batch_start in tqdm(range(0, num_frames, batch_size)):
        pooled_feat = torch.load(f"./tmp/{seq.name}_b{batch_size}_{batch_start}.pt")
        all_pooled = torch.cat((all_pooled, pooled_feat), 0) if all_pooled is not None else pooled_feat
        # print(pooled_feat.shape, all_pooled.shape, all_pooled.device, pooled_feat.device)

    all_pooled = all_pooled.flatten(1)
    # print(all_pooled.shape)
    product = torch.matmul(all_pooled, all_pooled.T)
    product = product.cpu().detach().numpy()
    return all_pooled, product


def compute_best_prev_hist():
    dataset = get_dataset(dataset_name)
    for seq, seq_i in zip((dataset), range(len(dataset))):
        all_pooled, product = load_all_pooled(seq, batch_size)
        # max_until_frame = []
        best_prev_dist = []
        for i in range(1, len(seq.frames)):
            max_i = product[i, :i].argmax()
            best_prev_dist.append(int(i - max_i))
        plt.hist(best_prev_dist, density=False, bins=len(seq.frames))  # density=False would make counts
        plt.ylabel('Num frames')
        plt.xlabel('Dist form similar looking prev frame')
        plt.savefig(f'result_plots/{dataset_name}/{seq.name}_best_prev_hist.jpg')

def compute_avg_iou_per_frame():
    trackers = []
    # trackers.extend(trackerlist('atom', 'default', range(0,5), 'ATOM'))
    trackers.extend(trackerlist('dimp', 'dimp18', range(0, 1), 'DiMP18'))
    trackers.extend(trackerlist('dimp', 'dimp50', range(0, 1), 'DiMP50'))
    # trackers.extend(trackerlist('dimp', 'prdimp18', range(0,5), 'PrDiMP18'))
    # trackers.extend(trackerlist('dimp', 'prdimp50', range(0,5), 'PrDiMP50'))

    dataset = get_dataset(dataset_name)

    for seq_id, seq in enumerate(tqdm(dataset)):
        best_app_arr = torch.load(f"./results/best_appearance/{seq.name}_corr_scores.pt").numpy()
        best_app_arr = np.append(0, best_app_arr)
        plot_dict = {}
        print(best_app_arr.shape)
        for trk_id, trk in enumerate(trackers):
            iou_arr = [np.array(a) for a in torch.load(f"results/iou_data/{trk.parameter_name}_{seq.name}")]
            iou_arr = np.vstack(iou_arr)
            iou_arr = iou_arr.mean(0)
            plot_dict[trk.parameter_name] = iou_arr
            print(iou_arr.shape)
        x = np.arange(best_app_arr.shape[0])
        # print(x)
        fig, ax1 = plt.subplots()
        fig.suptitle(f'Seq{seq.name}', fontsize=20)
        ax2 = ax1.twinx()
        ax1.plot(x, best_app_arr, 'black')
        for k in plot_dict.keys():
            ax2.plot(x, plot_dict[k])
        ax1.legend('best_app')
        ax2.legend(plot_dict.keys())
        fig.savefig(f'result_plots/{dataset_name}/{seq.name}.jpg')


if __name__ == "__main__":
    # compute_best_appearance_score()
    # compute_avg_iou_per_frame()
    compute_best_prev_hist()
