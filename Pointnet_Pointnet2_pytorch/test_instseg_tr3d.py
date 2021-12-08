"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.TR3DRoofsDataLoader import TR3DRoofsDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Roof': [0,1,2,3,4,5,6,7,8,9,10,11]}
seg_label_to_cat = {}  # {0:Roof, 1:Roof, ...11:Roof}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

inst_label_to_sem = { 0:0, 1: 0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4, 10:4, 11:4 }
sem_label_to_inst = { 0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9,10,11]}

plane_classes = { 'Rectangular': [0,1], 'Isosceles trapezoid': [2,3], 'Triangular': [4,5], 'Parallelogram': [6,7], 'Ladder shaped': [8,9,10,11]}
plane_label_to_cat = {}  # {0:Rectangular, 1:Rectangular, ...11:Ladder shaped}
for cat in plane_classes.keys():
    for label in plane_classes[cat]:
        plane_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/tr3d_roof_segmented_dataset/'

    TEST_DATASET = TR3DRoofsDataset(root=root, npoints=args.num_point, split='test', seg_type='inst')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_sem = 5
    num_inst = 11

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_inst, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
            test_metrics = {} # keep
            total_correct = 0 
            total_seen = 0
            total_seen_inst = [0 for _ in range(num_inst)]
            total_correct_inst = [0 for _ in range(num_inst)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}

            # precision & recall
            total_gt_ins = np.zeros(num_sem)
            at = 0.5
            tpsins = [[] for _ in range(num_sem)]
            fpsins = [[] for _ in range(num_sem)]
            # mucov and mwcov
            all_mean_cov = [[] for _ in range(num_sem)]
            all_mean_weighted_cov = [[] for _ in range(num_sem)]

            classifier = classifier.eval()

            # TODO: check this file for inspiration to calculate mCov, mWCov, mPrec and mRec
            # https://github.com/dlinzhao/JSNet/blob/master/models/JISS/eval_iou_accuracy.py from line 64
            # Can use this file as this part is equal to the test/eval part.
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()

                # Values to cuda
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                # Predict on points
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)

                # Get target values
                target = target.cpu().data.numpy()

                # For each batch
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_inst):
                    # This will be total correct based on instance label 
                    total_seen_inst[l] += np.sum(target == l)
                    total_correct_inst[l] += (np.sum((cur_pred_val == l) & (target == l)))

                # instance mucov & mwcov
                # TODO: Make this a correct cov calculation
                # NB: I write this as if there is only one class. Not generally for several object classes as we only have roofs.
                # Must combine all the cov's at the end to get a mCov

                # OR should we measure mCov based on only the roof class, and not the different semantic labels??
                for i in range(cur_batch_size):
                    pred_inst = cur_pred_val[i, :]                              # segmentation predictions
                    pred_sem  = np.vectorize(inst_label_to_sem.get)(pred_inst)  # map pred_inst to pred_sem
                    gt_inst = target[i, :]                                      # segmentation ground truth labels
                    gt_sem  = np.vectorize(inst_label_to_sem.get)(gt_inst)      # map label_inst to label_sem
                    # cat = seg_label_to_cat[label_inst[0]]

                    un = np.unique(pred_inst)
                    pts_in_pred = [[] for _ in range(num_sem)]
                    for ig, g in enumerate(un): # each object in prediction
                        if g == -1:
                            continue
                        tmp = (pred_inst == g) # the predicted group of points in this instance
                        sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
                        pts_in_pred[sem_seg_i] += [tmp]
                    
                    un = np.unique(gt_inst)
                    pts_in_gt = [[] for _ in range(num_sem)]
                    for ig, g in enumerate(un):
                        tmp = (gt_inst == g) # the true group of points in this instance
                        sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
                        pts_in_gt[sem_seg_i] += [tmp]

                    # instance mucov & mwcov
                    for i_sem in range(num_sem):
                        sum_cov = 0
                        mean_cov = 0
                        mean_weighted_cov = 0
                        num_gt_point = 0
                        for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                            ovmax = 0.
                            num_ins_gt_point = np.sum(ins_gt)
                            num_gt_point += num_ins_gt_point
                            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                                union = (ins_pred | ins_gt)
                                intersect = (ins_pred & ins_gt)
                                iou = float(np.sum(intersect)) / np.sum(union)

                                if iou > ovmax:
                                    ovmax = iou
                                    ipmax = ip

                            sum_cov += ovmax
                            mean_weighted_cov += ovmax * num_ins_gt_point

                        if len(pts_in_gt[i_sem]) != 0:
                            mean_cov = sum_cov / len(pts_in_gt[i_sem])
                            all_mean_cov[i_sem].append(mean_cov)

                            mean_weighted_cov /= num_gt_point
                            all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

                    
                    # instance precision & recall
                    for i_sem in range(num_sem):
                        tp = [0.] * len(pts_in_pred[i_sem])
                        fp = [0.] * len(pts_in_pred[i_sem])
                        gtflag = np.zeros(len(pts_in_gt[i_sem]))
                        total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

                        for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                            ovmax = -1.

                            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                                union = (ins_pred | ins_gt)
                                intersect = (ins_pred & ins_gt)
                                iou = float(np.sum(intersect)) / np.sum(union)

                                if iou > ovmax:
                                    ovmax = iou
                                    igmax = ig

                            if ovmax >= at:
                                tp[ip] = 1  # true
                            else:
                                fp[ip] = 1  # false positive

                        tpsins[i_sem] += tp
                        fpsins[i_sem] += fp

            # Should be done for all classes, but we only have Roof so don't bother
            Cov_sem = np.zeros(num_sem)
            WCov_sem = np.zeros(num_sem) 
            for i_sem in range(num_sem):
                Cov_sem[i_sem] = np.mean(all_mean_cov[i_sem])
                WCov_sem[i_sem] = np.mean(all_mean_weighted_cov[i_sem])
            
            # Mean for all the instances of Roof, so not mCov as mCov is over all classes and we only have one class Roof
            Cov = np.mean(Cov_sem)
            WCov = np.mean(WCov_sem)
            
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['inst_avg_accuracy'] = np.mean(
                np.array(total_correct_inst) / np.array(total_seen_inst, dtype=np.float))

            # Log Cov for each semantic label
            for i_sem in range(num_sem):
                inst_label = sem_label_to_inst[i_sem][0]
                plane_label = plane_label_to_cat[inst_label]
                log_string('eval sem  Cov of %s %f' % (plane_label + ' ' * (14 - len(plane_label)), Cov_sem[i_sem]))

            # Log WCov for each semantic class
            for i_sem in range(num_sem):
                inst_label = sem_label_to_inst[i_sem][0]
                plane_label = plane_label_to_cat[inst_label]
                log_string('eval sem WCov of %s %f' % (plane_label + ' ' * (14 - len(plane_label)), WCov_sem[i_sem]))
            
            # Log Cov and WCov for the roof class
            log_string('eval  Cov of %s %f' % ('Roof' + ' ' * (14 - len('Roof')), Cov))
            log_string('eval WCov of %s %f' % ('Roof' + ' ' * (14 - len('Roof')), WCov))
            test_metrics['cov'] = Cov
            test_metrics['wcov'] = WCov

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Instance avg accuracy is: %.5f' % test_metrics['inst_avg_accuracy'])
    log_string('Cov is: %.5f' % test_metrics['cov'])
    log_string('WCov is: %.5f' % test_metrics['wcov'])

    root = 'data/tr3d_roof_segmented_dataset/'

    # TODO: Check that the instances in viz dataset come from test dataset and not train dataset
    VIZ_DATASET = TR3DRoofsDataset(root=root, npoints=args.num_point, split='viz', seg_type='inst')
    vizDataLoader = torch.utils.data.DataLoader(VIZ_DATASET, batch_size=1, shuffle=False, num_workers=4)
    save_path = os.path.join(experiment_dir, "results/")
    # log_string("The number of test data is: %d" % len(TEST_DATASET))

    with torch.no_grad():
        classifier = classifier.eval()
        file_index = 0
        for batch_id, (points, label, target) in tqdm(enumerate(vizDataLoader), total=len(vizDataLoader),
                                                      smoothing=0.9):
            # print('batch_id:', batch_id)
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_inst).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                
                xyz = points.cpu().data.numpy()
                seg_values = cur_pred_val[i, :]
                VIZ_DATASET.store_segmented_roof(file_index, xyz, seg_values, save_path)
                file_index += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
