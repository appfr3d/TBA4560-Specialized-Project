"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from scipy import stats

from pathlib import Path
from tqdm import tqdm
from data_utils.TR3DRoofsDataLoader import TR3DRoofsDataset

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

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('inst_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/tr3d_roof_segmented_dataset/'

    TRAIN_DATASET = TR3DRoofsDataset(root=root, npoints=args.npoint, split='trainval')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = TR3DRoofsDataset(root=root, npoints=args.npoint, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # TODO: check if num classes is correct, because what about combinations of roof types...
    num_classes = 1 # classes, only roofs
    num_sem = 5     # roof types
    num_inst = 12   # roof plane shapes

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_inst, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size


    best_acc = 0
    global_epoch = 0
    # best_class_avg_iou = 0
    # best_inctance_avg_iou = 0
    best_cov = 0
    best_wcov = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # Augmention by scaling and shifting points
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_inst)
            target = target.view(-1, 1)[:, 0]


            # TODO: This should also be based on samantic label, not instance label...
            # But how should it be able to differentiate between label 0 and 1... it could be all 0 and all 1 and get full score...
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))


            # Map instance seg to semantic
            pred_inst_all = seg_pred.data.numpy()
            pred_sem_all = np.vectorize(inst_label_to_sem.get)(pred_inst_all)

            gt_inst_all = target.data.numpy()
            gt_sem_all = np.vectorize(inst_label_to_sem.get)(gt_inst_all)
            
            # Find which pred_inst group is covering the most of each corresponding instance label.
            # Say we have two groups for one samantic label
            # look at each group with the corresponding gt_inst labels and choose instance label based on which covers most of each...

            # Say two groups have inst label 0 and 1. 
            # But the group with instance label 1 covers most of the points for inst label 0, and vice versa
            # Then we want to swap the instance labels of the two groups.
            cur_batch_size, NUM_POINT, _ = points.size()
            for i in range(cur_batch_size):
                pred_inst = pred_inst_all[i]
                gt_inst = gt_inst_all[i]
                pred_sem = pred_sem_all[i]
                gt_sem = gt_sem_all[i]

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

                # len_in_gt = [[len(x) for x in lst] for lst in pts_in_gt]

                for ig, g in enumerate(pts_in_gt):
                    # ig is now the same as seg_sem_i
                    # sorted_i based on length in g
                    sorted_i = sorted(range(len(g)), key=lambda k: len(g[k]))
                    for 

            pred_sem = torch.Tensor(pred_sem).float().cuda()
            gt_sem = torch.Tensor(gt_sem).float().cuda()



            # loss = criterion(seg_pred, target, trans_feat)
            loss = criterion(pred_sem, gt_sem, trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)


        # TODO: Start to change here!
        # Everything in the with-block below should be placed in a test_instseg.py file as well.
        # only exception is that test_instseg.py should also have a voting_pool (see line 108 in test_partseg.py)
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
                    '''
                    # no need...
                    for l in seg_classes[cat]: # only one seg_class...

                        sem_l = inst_label_to_sem[l]
                        inst_covs = [0.0 for _ in range(len(seg_classes[cat]))]

                        

                        sum_cov = 0
                        mean_cov = 0
                        mean_weighted_cov = 0
                        num_gt_points = 0
                        
                        if (np.sum(seg_label == l) == 0) and (
                                np.sum(seg_pred == l) == 0):  # inst is not present, no prediction as well
                            inst_covs[l - seg_classes[cat][0]] = 1.0
                        else:
                            # This mus be corrected.
                            inst_covs[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    '''
                '''
                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))
                '''

            # Should be done for all classes, but we only have Roof so don't bother
            Cov_sem = np.zeros(num_sem)
            WCov_sem = np.zeros(num_sem) 
            for i_sem in range(num_sem):
                Cov_sem[i_sem] = np.mean(all_mean_cov[i_sem])
                WCov_sem[i_sem] = np.mean(all_mean_weighted_cov[i_sem])
            
            # Mean for all the instances of Roof, so not mCov as mCov is over all classes and we only have one class Roof
            Cov = np.mean(Cov_sem)
            WCov = np.mean(WCov_sem)
            

            '''
            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            '''
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['inst_avg_accuracy'] = np.mean(
                np.array(total_correct_inst) / np.array(total_seen_inst, dtype=np.float))
            
            '''
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            '''

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

        log_string('Epoch %d test Accuracy: %f  Cov: %f   WCov: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['cov'], test_metrics['wcov']))
        
        
        # TODO: Check if shit is appropriate...
        if (test_metrics['cov'] >= best_cov):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                # 'class_avg_iou': test_metrics['class_avg_iou'],
                'cov': test_metrics['cov'],
                'wcov': test_metrics['wcov'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['cov'] > best_cov:
            best_cov = test_metrics['cov']
        if test_metrics['wcov'] > best_wcov:
            best_wcov = test_metrics['wcov']

        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best Cov is: %.5f' % best_cov)
        log_string('Best WCov is: %.5f' % WCov)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
