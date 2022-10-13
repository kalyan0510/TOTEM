class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './checkpoints'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = '/home/kalyan/data/LaSOTBenchmark'
        self.got10k_dir = '/nfs/bigtoken.cs.stonybrook.edu/add_disk0/tracking/got10k/train'
        self.trackingnet_dir = '/nfs/bigtoken.cs.stonybrook.edu/add_disk0/tracking/TrackingNet'
        self.coco_dir = '/nfs/bigtoken.cs.stonybrook.edu/add_disk0/tracking/coco'
        self.totb_dir = '/home/kalyan/trans/TOTB'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
