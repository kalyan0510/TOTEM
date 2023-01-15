class RunCommandMaker:

    def __init__(self, total_tests, gpu_ids, tracker_name, param_names, conda, pkg_path, dataset):
        self.gpu_ids = gpu_ids
        self.total_tests = total_tests
        self.tracker_name = tracker_name
        self.param_names = param_names
        self.conda = conda
        self.pkg_path = pkg_path
        self.dataset = dataset

    @staticmethod
    def chunks(n, num_chks):
        """Yield successive n-sized chunks from lst. (start inclusive & end non-inclusive)"""
        # took = 0
        if num_chks > n:
            raise Exception('num_chks > n')
        st = 0
        for i in range(num_chks - 1):
            yield st, st + (n // num_chks)
            st += (n // num_chks)
        yield st, n + 1

    def make_cmd(self, st, ed, gpu_id, param_name):
        return f'conda activate {self.conda} ; cd {self.pkg_path} ; CUDA_VISIBLE_DEVICES={gpu_id} python run_tracker.py {self.tracker_name} {param_name} --dataset_name {self.dataset} --sequence {st}:{ed}'

    def make_cmds(self):
        # print(self.tracker_name)
        # print(self.param_names)
        allgpus = []
        for server in self.gpu_ids.keys():
            allgpus.extend(self.gpu_ids[server])
        first_gpu_id_per_server = [0]
        start_id = 0
        for server in list(self.gpu_ids.keys())[:-1]:
            start_id += len(self.gpu_ids[server])
            first_gpu_id_per_server.append(start_id)

        if isinstance(self.param_names, list):
            gpus_per_param = []
            for (st, ed), param in zip(self.chunks(len(allgpus), len(self.param_names)), self.param_names):
                gpus_per_param.append(allgpus[st:ed])
        else:
            gpus_per_param = [allgpus]
            self.param_names = [self.param_names]

        print(gpus_per_param)
        print(self.param_names)

        task = 0
        for gpu_ids, param_name in zip(gpus_per_param, self.param_names):
            for (st, ed), gpu_id in zip(self.chunks(total_tests, len(gpu_ids)), gpu_ids):
                # print(f'screen -dmS track_{st}_{ed} bash -c "{make_cmd(st, ed, gpu_id)}"')
                if task in first_gpu_id_per_server:
                    print(('='*70) + '    ' + list(self.gpu_ids.keys())[first_gpu_id_per_server.index(task)].upper() +  '    ' + ('='*70))
                print(f'screen -R task{task}')
                print(self.make_cmd(st, ed, gpu_id, param_name))
                task+=1


if __name__ == '__main__':
    gpu_ids = {
        'token': [0, 1, 2, 3],
        # 'net': [0,1,2,3],
        # 'query': [1, 2, 3, 4, 6, 7],
        # 'tensor': [5],
    }
    total_tests = 225
    conda='t3'
    dataset = 'totb'
    pkgpath='~/desk/pytracking/pytracking'

    tracker_name = 'transtomp'
    # param_names = ['tomp_temp_1t', 'tomp_temp_3t', 'tomp_temp_1tp1th', 'tomp_temp_3tp1th',
    #                'tomp_temp_use_processed_s']
    # param_names = ['tomp_temp_misplaced_trainframepos','tomp_temp_0t',  'tomp_temp_correct_pos_emb.py']
    param_names = ['tomp50wFusion__objwise_f1_two_step']
    RunCommandMaker(total_tests, gpu_ids, tracker_name, param_names, conda, pkgpath, dataset).make_cmds()
