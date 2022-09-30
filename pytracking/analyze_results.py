import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

if __name__ == '__main__':
    trackers = []
    trackers.extend(trackerlist('tomp', 'tomp50', None, 'Original Tomp'))
    trackers.extend(trackerlist('temptomp', 'tomp_temp', None, 'My Tomp with 10 temporal frame features'))
    trackers.extend(trackerlist('temptomp', 'tomp_temp_3t', None, 'My Tomp with 3 temporal frame features'))
    trackers.extend(trackerlist('temptomp', 'tomp_temp_1t', None, 'My Tomp with 1 temporal frame features'))
    trackers.extend(trackerlist('temptomp', 'tomp_temp_3tp1th', None, 'My Tomp with 3 temporal frame features and low (.1) threshold on selecting feat '))
    trackers.extend(trackerlist('temptomp', 'tomp_temp_1tp1th', None, 'My Tomp with 1 temporal frame features and low (.1) threshold on selecting feat '))
    trackers.extend(trackerlist('temptomp', 'tomp_temp_use_processed_s', None, 'My Tomp with 10 temporal frame features but feature selection over processed score maps'))
    trackers.extend(trackerlist('temptomp', 'tomp_temp_misplaced_trainframepos', None, 'My Tomp with misplaced train frames (ablate study?)'))

    dataset = get_dataset('lasot')
    print_results(trackers, dataset, 'LaSOT TempTomp', merge_results=True, plot_types=('success', 'prec'),
                 skip_missing_seq=True, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
