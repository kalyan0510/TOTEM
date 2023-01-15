import os

from pytracking.analysis.playback_results import playback_results

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
# import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import pickle
from itertools import groupby

plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist
import torch
import numpy as np
from tqdm import tqdm
from pytracking.evaluation.environment import env_settings


def plot_per_seq_iou(trackers, dataset):
    for tracker_param_name in set([t.parameter_name for t in trackers]):
        # print(tracker.run_id)
        for seq in tqdm(dataset):
            iou_arr = [np.array(a) for a in torch.load(f"results/iou_data/{tracker_param_name}_{seq.name}")]
            iou_arr = np.vstack(iou_arr)
            # CAREFUL, this is averaging out IOU which does not have any non-statistical meaning
            iou_arr = iou_arr.mean(0)

            # print(iou_arr)
            blocks = max(1, len(iou_arr) // 500)
            fig, axs = plt.subplots(blocks, 1, figsize=(50, blocks * 7))  # figsize=(10*len(iou_arr)//
            if blocks==1:
                axs = [axs]
            # ax1 = ax1[0]
            fig.suptitle(f'Tracker: {tracker_param_name}, Seq: {seq.dataset}/{seq.name}', fontsize=20)
            # ax2 = ax1.twinx()
            x = np.arange(iou_arr.shape[0])
            for b in range(blocks):
                ax1 = axs[b]
                x_ = x[b * 500:min(b * 500 + 500, len(iou_arr))]
                iou_arr_neg = iou_arr[b * 500:min(b * 500 + 500, len(iou_arr))].copy()
                iou_arr_neg[iou_arr_neg >= 0] = np.nan
                iou_arr_neg[iou_arr_neg < 0] = 0

                iou_arr_pos = iou_arr[b * 500:min(b * 500 + 500, len(iou_arr))].copy()
                # iou_arr_pos[iou_arr<0]=0
                iou_arr_pos[iou_arr_pos < 0] = np.nan

                ax1.plot(x_, iou_arr_pos, 'black', label="avg iou")
                ax1.plot(x_, iou_arr_neg, 'red', linewidth=10, label=None)
                ax1.set_xlabel('frames')
                ax1.set_ylabel('iou')
                ax1.set_ylim([0, 1])

                # for k in plot_dict.keys():
                #     ax2.plot(x, plot_dict[k])
                ax1.legend()
            fig.savefig(f'result_plots/{tracker_param_name}_{seq.dataset}/{seq.name}.jpg')
            plt.close(fig)
            # break
            # plot_dict[trk.parameter_name] = iou_arr


def get_eval_data(report_name):
    settings = env_settings()
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path):
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
            return eval_data
    else:
        raise Exception("no file found")


def pull_auc_from_processed_data(eval_data):
    sequences = eval_data['sequences']
    trackers = eval_data['trackers']
    valid_sequence = np.array(eval_data['valid_sequence'])
    ave_success_rate_plot_overlap = np.array(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = np.array(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = np.array(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = np.array(eval_data['avg_overlap_all'])
    threshold_set_overlap = np.array(eval_data['threshold_set_overlap'])
    threshold_set_center = np.array(eval_data['threshold_set_center'])
    threshold_set_center_norm = np.array(eval_data['threshold_set_center_norm'])
    sequnece_num_frames = np.array(eval_data['sequnece_num_frames'])
    # print(ave_success_rate_plot_overlap.shape)
    # print(threshold_set_overlap)
    # print(ave_success_rate_plot_overlap[0, 0, :])
    #
    sequence_stats = []
    for seq_i, seq in enumerate(sequences):
        for tracker_gp in [list(g[1]) for g in groupby(enumerate(trackers), lambda x: x[1]['param'])]:
            tracker_indices = [t[0] for t in tracker_gp]
            auc_avg = ave_success_rate_plot_overlap[seq_i, tracker_indices, :].mean(0)
            auc_weighted_avg = auc_avg * sequnece_num_frames[seq_i]
            neg_auc_impact = sequnece_num_frames[seq_i] * auc_avg.shape[0] - auc_weighted_avg.sum()
            # print(f"Seq: {seq} (#f {sequnece_num_frames[seq_i]}) = {auc_weighted_avg.sum()}")
            sequence_stats.append(
                {'seq': seq, 'neg_auc_impact': neg_auc_impact, 'seq_i': seq_i, 'auc_weighted_avg': auc_weighted_avg})

    processed_seq_stats = list(map(lambda x: [x['seq'], sequnece_num_frames[x['seq_i']], np.round(x['neg_auc_impact'])],
                                   sorted(sequence_stats, reverse=True, key=lambda x: x['neg_auc_impact'])))
    for seq_stat in processed_seq_stats:
        print("xx {:<15} {:<6} {:<6}".format(*seq_stat))

    return processed_seq_stats


def plot_per_seq_auc(report_name):
    eval_data = get_eval_data(report_name)
    seq_stats = pull_auc_from_processed_data(eval_data)
    print(seq_stats)
    class_neg = {}
    for s in seq_stats:
        f, neg = class_neg.get(s[0].split('_')[0], (0,0))
        class_neg[s[0].split('_')[0]] = (f+s[1], neg +s[2])
    print(seq_stats)
    for k,v in sorted(class_neg.items(), key=lambda x: x[1][1]):
        print("-->",k,v)

def get_seq(dataset, seq_name):
    return next(seq for seq in dataset if seq.name == seq_name)


def iou_stats(trackers, dataset):
    iou_f100_frames = []
    for tracker_param_name in set([t.parameter_name for t in trackers]):
        # print(tracker.run_id)
        for seq in dataset:
            iou_arr = [np.array(a) for a in torch.load(f"results/iou_data/{tracker_param_name}_{seq.name}")]

            iou_arr = np.vstack(iou_arr)
            print(iou_arr.shape)
            # CAREFUL, this is averaging out IOU which does not have any non-statistical meaning
            iou_arr = iou_arr.mean(0)
            # iou_f100_frames.append(iou_arr[:20].mean())
            print(f"{seq.name} : {iou_arr[:100].mean()} ")
    # print(np.mean(iou_f100_frames))


def seq_iou_stats(trackers, dataset):
    iou_per_class = {}
    for tracker_param_name in set([t.parameter_name for t in trackers]):
        # print(tracker.run_id)
        for seq in dataset:
            iou_arr = [np.array(a) for a in torch.load(f"results/iou_data/{tracker_param_name}_{seq.name}")]

            iou_arr = np.vstack(iou_arr)
            print(iou_arr.shape)
            # CAREFUL, this is averaging out IOU which does not have any non-statistical meaning
            iou_arr = iou_arr.mean(0)
            # iou_f100_frames.append(iou_arr[:20].mean())
            print(f"{seq.name} : {iou_arr.mean()} ")
            (n, iou) = iou_per_class.get(seq.name.split('_')[0], (0, 0))
            iou_per_class[seq.name.split('_')[0]] = (n+len(iou_arr), iou+iou_arr.sum())
    for k in iou_per_class.keys():
        n, iou = iou_per_class[k]
        iou_per_class[k]= (n, iou/n)
    x = list(iou_per_class.items())
    x = sorted(x, key=lambda a:a[1][1])
    for i in x:
        print(i)

def plot_first100frame_successplot(report_name):
    eval_data = get_eval_data(report_name + "100f")
    avg_overall_overlap = np.array(eval_data['ave_success_rate_plot_overlap'])
    avg_overall_overlap = avg_overall_overlap.mean(0)
    avg_overall_overlap = avg_overall_overlap.mean(0)
    print(avg_overall_overlap.shape)
    plt.plot(eval_data['threshold_set_overlap'], avg_overall_overlap)
    plt.show()


if __name__ == "__main__":
    trackers = []
    # trackers.extend(trackerlist('tomp', 'tomp50_totblasot', None, 'Original Tomp'))
    # trackers.extend(trackerlist('tomp', 'retrained_tomp50onTOTB', None, 'Fine Tuned ToMP'))
    # trackers.extend(trackerlist('transtomp', 'tomp50wFusion_onLasotTOTB', None, 'Fusion Module on LaSOT + TOTB'))
    # trackers.extend(trackerlist('tomp', 'tomp50_totb__objwise_f1', None, 'Fine Tuned ToMP Obj'))

    # trackers.extend(trackerlist('transtomp', 'tomp50wFusion__objwise_f1', None, 'Fusion Module Obj'))
    # trackers.extend(trackerlist('transtomp', 'tomp50wFusion__objwise_f1_two_step_all_train', None, 'Fusion Module 2 Step All train'))
    # trackers.extend(trackerlist('gt', 'groundtruth', None, 'Ground Truth'))
    trackers.extend(trackerlist('tomp', 'tomp50_300', None, 'TOMP(Baseline)'))
    trackers.extend(trackerlist('transtomp', 'tomp50wFusion__objwise_f1_two_step', None, 'T3Net(Ours)'))
    # trackers.extend(trackerlist('transtomp', 'tomp50wFusion__objwise_f1_two_step_all_train', None, 'xXXX(Ours)'))
    # trackers.extend(trackerlist('tomp', 'tomp50_totb__objwise_f1_second', None, 'Fine Tuned ToMP Obj'))
    # trackers.extend(trackerlist('atom', 'transatom', None, 'TransATOM'))


    dataset = get_dataset('totb')
    # dataset = dataset[1:]
    # print(dataset)
    report_name = 'FusionVsFineTuning'
    print_results(trackers, dataset, report_name, force_evaluation=True)
    # iou_stats(trackers, dataset)
    # seq_iou_stats(trackers, dataset)

    # plot_per_seq_iou(trackers, dataset)
    # MAINMAIN
    # plot_per_seq_auc(report_name)

    playback_results(trackers, get_seq(dataset, 'Custom_1'))
    # RUN THIS BEFORE
    # plot_per_seq_auc(report_name)


# trackers.extend(trackerlist('rts', 'rts50', None, 'RTS'))
# trackers.extend(trackerlist('atom', 'default', None, 'TransAtom myrun'))
"""
    # print(len(dataset[0].ground_truth_rect))
    # print((dataset[0].ground_truth_rect))
    # print(len(dataset[0].frames))
    # print(dataset[0].frames)
    # exit()
    # plot_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec'),
    #              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

    # dataset = get_dataset('otb')
    # print_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

    # print_per_sequence_results(trackers, dataset, 'OTB', merge_results=True, filter_criteria=None,
    #                            force_evaluation=False)
"""

















