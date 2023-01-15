from pytracking.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    # Set your local paths here.
    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.totb_path = '/home/kalyan/trans/TOTB'
    settings.lasot_path = '/home/kalyan/data/LaSOTBenchmark/'
    settings.network_path = '/home/kalyan/desk/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/nfs/bigtoken.cs.stonybrook.edu/add_disk0/tracking/otb2015'
    settings.oxuva_path = ''
    settings.result_plot_path = '/home/kalyan/desk/pytracking/pytracking/result_plots/'
    settings.results_path = '/home/kalyan/desk/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/kalyan/desk/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    return settings

