import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2 as cv
import numpy as np
# def load(path):
from pytracking.analysis.playback_results import Display
from pytracking.analysis.plot_results import get_plot_draw_styles
import matplotlib.pyplot as plt

np.random.seed(2)


def display_images(file_name, viewing_batch=0):
    base_dir = 'W:\\CV\\observations\\tomp_input_data\\'
    path = base_dir + file_name
    i = 0
    data_i = torch.load(path)
    print(data_i.keys())
    print(data_i['train_images'].shape)
    print(data_i['train_anno'].shape)
    print(data_i['test_images'].shape)
    print(data_i['test_anno'].shape)
    print(data_i['train_label'].shape)
    print(data_i['test_label'].shape)
    print(data_i['train_ltrb_target'].shape)
    print(data_i['test_ltrb_target'].shape)
    print(data_i['train_sample_region'].shape)
    print(data_i['test_sample_region'].shape)

    train_images = data_i['train_images'].numpy()
    test_images = data_i['test_images'].numpy()
    train_labels = data_i['train_label'].numpy()
    test_labels = data_i['test_label'].numpy()
    test_ltrb_target = data_i['test_ltrb_target'].numpy()

    # np.
    # cv.imshow('Yo', train_images[0,1,...].transpose(1,2,0)[:,:,::-1])
    # cv.waitKey(0)
    def ltrb_to_bb(ltrb_s, idx):
        shifts = np.arange(0, 288, step=16) + 8
        # shifts_y = np.arange(0, 288, step=16) + 8
        # shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
        # shift_x = shift_x.reshape(-1)
        # shift_y = shift_y.reshape(-1)
        # locations = np.stack((shift_x, shift_y), dim=1) + 16//2
        # xs, ys = locations[:, 0], locations[:, 1]
        c_i = shifts[idx[0]]
        c_j = shifts[idx[1]]
        l = c_j - ltrb_s[0]
        r = c_j + ltrb_s[2]
        t = c_i - ltrb_s[1]
        b = c_i + ltrb_s[3]
        return (l, t, r - l, b - t)

    # viewing_batch = 9
    ims_batch = test_images
    labels_batch = test_labels
    display = Display(ims_batch.shape[0] * 2, get_plot_draw_styles(), 'train_input')

    frame_i = 0
    while display.active and frame_i < ims_batch.shape[0]:
        # frame_i = frame_i + 1
        frame_number = display.frame_number
        # print(ims_batch[frame_i,viewing_batch ,...].shape)
        image = ims_batch[frame_number, viewing_batch, ...].transpose(1, 2, 0)[:, :, :]
        label = cv.resize(labels_batch[frame_number, viewing_batch, ...], (288, 288), interpolation=cv.INTER_CUBIC)
        print(label.shape)
        # frame_i = frame_i+1
        max_idx = np.unravel_index(np.argmax(labels_batch[frame_number, viewing_batch, ...], axis=None),
                                   labels_batch[frame_number, viewing_batch, ...].shape)
        ltrb = test_ltrb_target[frame_number, viewing_batch, :, max_idx[0], max_idx[1]]
        ltrb = ltrb * 288
        # print(ltrb*288)
        print(image.max(), image.min(), display.frame_number)
        image = (image - image.min()) / (image.max() - image.min())
        # display.show(label, [], [])
        # time.sleep(0.001)
        # w, h = -ltrb[2]+ltrb[0], -ltrb[3]+ltrb[1]
        # print([(int(ltrb[1])+h//2, int(ltrb[1])+w//2, h, w)])
        print(ltrb_to_bb(ltrb, max_idx))
        display.show(image, [ltrb_to_bb(ltrb, max_idx)], ['box'])

        time.sleep(0.2)
        if display.pause_mode and display.frame_number == frame_number:
            time.sleep(0.1)
        elif not display.pause_mode:
            display.step()


def display_input_output(file_name, viewing_batch=0):
    base_dir = 'W:\\CV\\observations\\stats\\mytomp50p_loss_is_mean\\data_inp\\'
    path = base_dir + file_name
    alldata = torch.load(path, map_location=torch.device('cpu'))
    # print()
    print(alldata.keys())
    data_i = alldata['data']
    print(alldata['output_bbpreds'][20].shape)
    print(data_i.keys())
    print('train_images', data_i['train_images'].shape)
    print('train_anno', data_i['train_anno'].shape)
    print('test_images', data_i['test_images'].shape)
    print('test_anno', data_i['test_anno'].shape)
    print('train_label', data_i['train_label'].shape)
    print('test_label', data_i['test_label'].shape)
    print('train_ltrb_target', data_i['train_ltrb_target'].shape)
    print('test_ltrb_target', data_i['test_ltrb_target'].shape)
    print('train_sample_region', data_i['train_sample_region'].shape)
    print('test_sample_region', data_i['test_sample_region'].shape)

    train_images = data_i['train_images'].numpy()
    test_images = data_i['test_images'].numpy()
    train_labels = data_i['train_label'].numpy()
    test_labels = data_i['test_label'].numpy()
    test_ltrb_target = data_i['test_ltrb_target'].numpy()
    test_sample_region = data_i['test_sample_region'].numpy()

    train_label = train_labels[0,0]
    print('train_labels', train_label.min(), train_label.max(), train_label.mean(), train_label.std())

    n_test_ims, b, _, _, _ = test_images.shape
    output_tscores = np.array([x.numpy() for x in alldata['output_tscores']]).reshape((n_test_ims, b, 18,18) )
    print("output_tscores", output_tscores.shape)
    output_bbpreds = np.array([x.numpy() for x in alldata['output_bbpreds']]).reshape((n_test_ims, b, 4, 18,18) )

    # np.
    # cv.imshow('Yo', train_images[0,1,...].transpose(1,2,0)[:,:,::-1])
    # cv.waitKey(0)
    def ltrb_to_bb(ltrb_s, idx):
        shifts = np.arange(0, 288, step=16) + 8
        c_i = shifts[idx[0]]
        c_j = shifts[idx[1]]
        l = c_j - ltrb_s[0]
        r = c_j + ltrb_s[2]
        t = c_i - ltrb_s[1]
        b = c_i + ltrb_s[3]
        return (l, t, r - l, b - t)

    def res_to_bbox(scores, ltrb_matrix):
        max_idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        ltrb = ltrb_matrix[:, max_idx[0], max_idx[1]]
        ltrb = ltrb * 288
        # print(ltrb, max_idx)
        return ltrb_to_bb(ltrb, max_idx)

    # viewing_batch = 9
    ims_batch = test_images
    labels_batch = test_labels
    samples_batch = test_sample_region
    scores_batch = output_tscores
    preds_batch = output_bbpreds
    display = Display(ims_batch.shape[0], get_plot_draw_styles(), 'train_input')

    show = {'label': False, 'sample': False, 'scores': False, 'preds': True}
    frame_i = 0
    while display.active and frame_i < ims_batch.shape[0]:
        # frame_i = frame_i + 1
        frame_number = display.frame_number
        # print(ims_batch[frame_i,viewing_batch ,...].shape)
        image = ims_batch[frame_number, viewing_batch, ...].transpose(1, 2, 0)[:, :, :]
        label = cv.resize(labels_batch[frame_number, viewing_batch, ...], (288, 288), interpolation=cv.INTER_CUBIC)
        # print(samples_batch[frame_number, viewing_batch, 0, ...].shape)
        sample = cv.resize(samples_batch[frame_number, viewing_batch, 0, ...]*1.0, (288, 288), interpolation=cv.INTER_CUBIC)
        scores = cv.resize(scores_batch[frame_number, viewing_batch, ...], (288, 288), interpolation=cv.INTER_CUBIC)
        # frame_i = frame_i+1
        max_idx = np.unravel_index(np.argmax(labels_batch[frame_number, viewing_batch, ...], axis=None),
                                   labels_batch[frame_number, viewing_batch, ...].shape)
        ltrb = test_ltrb_target[frame_number, viewing_batch, :, max_idx[0], max_idx[1]]
        ltrb = ltrb * 288
        max_idx_pred = np.unravel_index(np.argmax(scores_batch[frame_number, viewing_batch, ...], axis=None),
                                   scores_batch[frame_number, viewing_batch, ...].shape)
        ltrb_pred = preds_batch[frame_number, viewing_batch, :, max_idx[0], max_idx[1]]
        ltrb_pred = ltrb_pred * 288
        print(image.max(), image.min(), display.frame_number)
        image = (image - image.min()) / (image.max() - image.min())
        if show['label']:
            display.show(label, [], [])
            time.sleep(0.001)
        if show['sample']:
            display.show(sample, [], [])
            time.sleep(0.001)
        if show['scores']:
            display.show(scores, [], [])
            time.sleep(0.5)
        if show['preds']:
            bboxes = [res_to_bbox(labels_batch[frame_number, viewing_batch, ...], test_ltrb_target[frame_number, viewing_batch, :]),
                      res_to_bbox(scores_batch[frame_number, viewing_batch, ...], preds_batch[frame_number, viewing_batch, :]),
                      res_to_bbox(labels_batch[frame_number, viewing_batch, ...], preds_batch[frame_number, viewing_batch, :]),
                      res_to_bbox(scores_batch[frame_number, viewing_batch, ...], 0.001+test_ltrb_target[frame_number, viewing_batch, :])]
            names = ['ground', 'both preds', 'ltrb pred', 'cls pred']
        else:
            bboxes = [res_to_bbox(labels_batch[frame_number, viewing_batch, ...], test_ltrb_target[frame_number, viewing_batch, :])]
            names = ['ground']
        # w, h = -ltrb[2]+ltrb[0], -ltrb[3]+ltrb[1]
        # print([(int(ltrb[1])+h//2, int(ltrb[1])+w//2, h, w)])
        print(ltrb_to_bb(ltrb, max_idx))
        display.show(image,bboxes,names)

        time.sleep(0.2)
        if display.pause_mode and display.frame_number == frame_number:
            time.sleep(0.1)
        elif not display.pause_mode:
            display.step()


def random_momentum(num_id):
    print(torch.exp(torch.tensor(-1)), torch.exp(torch.tensor(-0.5)), torch.exp(torch.tensor(0)),
          torch.exp(torch.tensor(0.5)), torch.exp(torch.tensor(1)))
    print(torch.randn(5).mean())
    x = []
    y = []

    # momentum_sz = 0
    # rand = torch.randn(1)
    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(np.pad(y, ((box_pts) // 2, (box_pts - 1) // 2), mode='edge'), box, mode='valid')
        return y_smooth

    cur_rand = 0
    i = 0
    while i < 50:
        target = np.random.normal(0, 0.5, (1))[0]
        num_steps = int(max(3, np.random.normal(10, 10, size=(1, 1))))
        print("steps:", num_steps, (target - cur_rand) / num_steps)
        for step in [(target - cur_rand) / num_steps] * num_steps:
            print(cur_rand, step)
            cur_scale = np.exp(cur_rand * 0.5)
            x.append(i)
            y.append(cur_scale)
            cur_rand += step
            i = i + 1

        # momentum_sz =  max(min(torch.randn(1), torch.randn(1)+momentum_sz), torch.randn(1))
        # rand = rand/10 + momentum_sz
        # rand =  max(min(torch.randn(1),rand), torch.randn(1))
        # scale = 1 * torch.exp(torch.exp(rand) * .5)
        # x.append(i)
        # y.append(scale.numpy()[0])
        # print(scale, rand)
    # How to simulate smooth randomized peaks;
    # understand that and implement that

    print(x)
    print(y)
    print(len(y), smooth(y, 10).shape)
    plt.cla()
    plt.xlim(0, 50)
    plt.ylim(0, 3)
    plt.plot(x, smooth(y, 10))
    plt.savefig(f'W:\\CV\\observations\\motion_graphs\\{num_id}.png')
    # plt.pause(5.0)


if __name__ == '__main__':
    # display_images('tomps_default_jittering_data_1.pt')
    # display_images('tomp_with_synth_motion_data_1.pt')
    # display_images('mytomp_with_prev_box_for_jittering.pt')
    # display_images('mytomp_with_prev_box_for_jittering_firstframe_fixed.pt')
    # display_images('mytomp_with_smooth_scale_motion_2.pt', 10 )
    display_input_output('data_50-inp-op.pt', 1)
    # for i in range(10):
    #     random_momentum(i)
