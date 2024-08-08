import glob
import os
import logging, sys
import itertools
from typing import List, Dict
import time
from dataclasses import dataclass, field

from uncos import UncOS
import cv2
import numpy as np
import open3d as o3d
from skimage.morphology import disk
import pickle
import torch
import random
from scipy.optimize import linear_sum_assignment

N_SEG_HYPOTHESES_TRIAL = 5  # 12
N_TRIAL_PER_QUERY_POINT = 3
SAVE_DIR = "eval"
OCID_DIR = "/data2/xiaolinf/sam_probing/OCID-dataset/"

@dataclass
class SegResultLogger:
    all_fscore: List = field(default_factory=list)
    all_fscore_osn: List = field(default_factory=list)
    all_prec: List = field(default_factory=list)
    all_prec_osn: List = field(default_factory=list)
    all_recall: List = field(default_factory=list)
    all_recall_osn: List = field(default_factory=list)
    all_pred_num: List = field(default_factory=list)
    all_gt_num: List = field(default_factory=list)
    all_boundary_fscore: List = field(default_factory=list)

    all_boundary_fscore_osn: List = field(default_factory=list)
    all_boundary_prec: List = field(default_factory=list)
    all_boundary_prec_osn: List = field(default_factory=list)
    all_boundary_recall: List = field(default_factory=list)
    all_boundary_recall_osn: List = field(default_factory=list)
    all_correct_num: List = field(default_factory=list)
    all_correct_ratio: List = field(default_factory=list)

    def add_result(self, eval_results: Dict):
        self.all_fscore.append(eval_results['fscore'])
        self.all_fscore_osn.append(eval_results['fscore_osn'])
        self.all_prec.append(eval_results['precision'])
        self.all_prec_osn.append(eval_results['precision_osn'])
        self.all_recall.append(eval_results['recall'])
        self.all_recall_osn.append(eval_results['recall_osn'])
        self.all_pred_num.append(eval_results['pred_object_num'])
        self.all_gt_num.append(eval_results['gt_object_num'])

        self.all_correct_num.append(eval_results['correct_object_num'])
        self.all_correct_ratio.append(
            eval_results['correct_object_num'] / max(eval_results['gt_object_num'], eval_results['pred_object_num'])
            if eval_results['gt_object_num'] > 0 else 0)

        self.all_boundary_fscore.append(eval_results['boundary_fscore'])
        self.all_boundary_fscore_osn.append(eval_results['boundary_fscore_osn'])
        self.all_boundary_prec.append(eval_results['boundary_precision'])
        self.all_boundary_prec_osn.append(eval_results['boundary_precision_osn'])
        self.all_boundary_recall.append(eval_results['boundary_recall'])
        self.all_boundary_recall_osn.append(eval_results['boundary_recall_osn'])

        logging.info(f"\t\t\t pred #{eval_results['pred_object_num']} \t\t gt #{eval_results['gt_object_num']}")

    def __len__(self):
        return len(self.all_fscore)

    def print(self, header=None):
        logging.info(f"{header} cycle {self.__len__()} : ")
        logging.info(
            f'\t\tavg fscore is {np.mean(self.all_fscore)}\t\t avg fscore_osn is {np.mean(self.all_fscore_osn)}\n'
            f'\t\tcorrect num is {np.mean(self.all_correct_num)}\t\tcorrect obj ratio is {np.mean(self.all_correct_ratio)}\n'
            f'\t\tavg precision is {np.mean(self.all_prec)}\t\t avg precision_osn is {np.mean(self.all_prec_osn)}\n'
            f'\t\tavg recall is {np.mean(self.all_recall)}\t\t avg recall_osn is {np.mean(self.all_recall_osn)}\n'
            f'\t\tavg boundary fscore is {np.mean(self.all_boundary_fscore)}\t\t avg boundary fscore_osn is {np.mean(self.all_boundary_fscore_osn)}\n'
            f'\t\tavg boundary precision is {np.mean(self.all_boundary_prec)}\t\t avg boundary precision_osn is {np.mean(self.all_boundary_prec_osn)}\n'
            f'\t\tavg boundary recall is {np.mean(self.all_boundary_recall)}\t\t avg boundary recall_osn is {np.mean(self.all_boundary_recall_osn)}\n'
            f'\t\tavg pred num is #{np.mean(self.all_pred_num)}\t\t avg gt num is #{np.mean(self.all_gt_num)}')


# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    # from IPython import embed; embed()

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + np.floor((y - 1) + height / h)
                    i = 1 + np.floor((x - 1) + width / h)
                    bmap[j, i] = 1;

    return bmap


def get_boundary(mask, bound_th=0.003):
    mask_boundary = seg2bmap(mask)
    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(mask_boundary.shape))
    dilated_boundary = cv2.dilate(mask_boundary.astype(np.uint8), disk(bound_pix), iterations=1)
    # import matplotlib.pyplot as plt;plt.subplot(121);plt.imshow(mask);plt.subplot(122);plt.imshow(dilated_boundary);plt.show()
    return mask_boundary, dilated_boundary


# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def boundary_overlap(predicted_mask, gt_mask, bound_th=0.003):
    """
    Compute true positives of overlapped masks, using dilated boundaries

    Arguments:
        predicted_mask  (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        overlap (float): IoU overlap of boundaries
    """
    assert np.atleast_3d(predicted_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(predicted_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(predicted_mask)
    gt_boundary = seg2bmap(gt_mask)

    from skimage.morphology import disk

    # Dilate segmentation boundaries
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix), iterations=1)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix), iterations=1)

    # Get the intersection (true positives). Calculate true positives differently for
    #   precision and recall since we have to dilated the boundaries
    fg_match = np.logical_and(fg_boundary, gt_dil)
    gt_match = np.logical_and(gt_boundary, fg_dil)

    # Return precision_tps, recall_tps (tps = true positives)
    return np.sum(fg_match), np.sum(gt_match), np.sum(fg_boundary), np.sum(gt_boundary)


def eval_pred_to_gt(maskpred, maskgt, osn=True, correct_thr=.75):
    """
    Args:
        maskpred:       K1 x H x W array bool
        maskgt:         K2 x H x W array bool
        true_positive:
        fscore:
        pred_sum:
        gt_sum:
        overlap:
        osn:
        use_sameordering_iou:

    Returns:

    """
    return_dict = {}
    num_pred, num_gt = len(maskpred), len(maskgt)
    if num_pred == 0 and num_gt > 0:
        return {'fscore': 0.,
                'fscore_osn': 0,
                'precision': 1,
                'recall': 0,
                'precision_osn': 1,
                'recall_osn': 0,
                'pred_object_num': num_pred,
                'gt_object_num': num_gt,
                'boundary_fscore': 0.,
                'boundary_precision': 1.,
                'boundary_recall': 0.,
                'boundary_fscore_osn': 0.,
                'boundary_precision_osn': 1.,
                'boundary_recall_osn': 0.,
                # 'fscore_list' :np.array([0 for _ in range(num_gt)]),
                # 'precision_list' :np.array([1 for _ in range(num_gt)]),
                # 'recall_list' :np.array([0 for _ in range(num_gt)]),
                'fscore_list': np.array([]),
                'precision_list': np.array([]),
                'recall_list': np.array([]),
                'correct_object_num': 0,
                'match_mat': [[], []]
                }

    elif num_pred > 0 and num_gt == 0:  # all false negatives
        return {'fscore': 0.,
                'fscore_osn': 0,
                'precision': 0,
                'recall': 1,
                'precision_osn': 0,
                'recall_osn': 1,
                'pred_object_num': num_pred,
                'gt_object_num': num_gt,
                'boundary_fscore': 0.,
                'boundary_precision': 0.,
                'boundary_recall': 1.,
                'boundary_fscore_osn': 0.,
                'boundary_precision_osn': 0.,
                'boundary_recall_osn': 1.,
                # 'fscore_list' :np.array([0 for _ in range(num_pred)]),
                # 'precision_list' :np.array([0 for _ in range(num_pred)]),
                # 'recall_list' :np.array([1 for _ in range(num_pred)]),
                'fscore_list': np.array([]),
                'precision_list': np.array([]),
                'recall_list': np.array([]),
                'correct_object_num': 0,
                'match_mat': [[], []]
                }
    elif (num_pred == 0 and num_gt == 0):  # correctly predicted nothing
        return {'fscore': 1.,
                'fscore_osn': 1.,
                'precision': 1,
                'recall': 1,
                'precision_osn': 1,
                'recall_osn': 1,
                'pred_object_num': num_pred,
                'gt_object_num': num_gt,
                'Objects Recall': 1.,
                'boundary_fscore': 1.,
                'boundary_precision': 1.,
                'boundary_recall': 1.,
                'boundary_fscore_osn': 1.,
                'boundary_precision_osn': 1.,
                'boundary_recall_osn': 1.,
                'fscore_list': np.array([]),
                'precision_list': np.array([]),
                'recall_list': np.array([]),
                'correct_object_num': 0,
                'match_mat': [[], []]
                }

    pred_sum_2d = maskpred.sum(-1).sum(-1)  # K1
    gt_sum_2d = maskgt.sum(-1).sum(-1)  # K2

    expanded_maskpred = np.repeat(maskpred[:, np.newaxis, :, :], num_gt, axis=1).astype(bool)  # K1xK2xHxW
    expanded_maskgt = np.repeat(maskgt[np.newaxis, ...], num_pred, axis=0).astype(bool)  # K1xK2xHxW
    pred_gt_intersection = np.logical_and(expanded_maskpred, expanded_maskgt).sum(-1).sum(-1)  # K1 x K2
    # pred_gt_union = np.logical_or(expanded_maskpred,expanded_maskgt).sum(-1).sum(-1)  # K1 x K2
    pred_gt_precision = pred_gt_intersection / pred_sum_2d[..., np.newaxis]
    pred_gt_recall = pred_gt_intersection / gt_sum_2d
    pred_gt_fscore = (2 * pred_gt_precision * pred_gt_recall) / (pred_gt_precision + pred_gt_recall)

    pred_gt_fscore[np.isnan(pred_gt_fscore)] = 0.
    best_match_r, best_match_c = linear_sum_assignment(pred_gt_fscore,
                                                       maximize=True)  # munkres to find max assignment of fscore

    return_dict['match_mat'] = [best_match_r, best_match_c]
    precision = pred_gt_intersection[best_match_r, best_match_c].sum() / pred_sum_2d.sum()  # precision
    recall = pred_gt_intersection[best_match_r, best_match_c].sum() / gt_sum_2d.sum()  # recall
    fscore_final = (2 * precision * recall) / (precision + recall)
    if np.isnan(fscore_final):
        fscore_final = 0.
    if osn:
        obj_P_osn = pred_gt_precision[best_match_r, best_match_c].sum() / pred_gt_precision.shape[0]
        obj_R_osn = pred_gt_recall[best_match_r, best_match_c].sum() / pred_gt_precision.shape[1]
        obj_F_osn = pred_gt_fscore[best_match_r, best_match_c].sum() / max(pred_gt_precision.shape)
        return_dict['precision_osn'] = obj_P_osn
        return_dict['recall_osn'] = obj_R_osn
        return_dict['fscore_osn'] = obj_F_osn
        fscore_list = pred_gt_fscore[best_match_r, best_match_c]
        precision_list = pred_gt_precision[best_match_r, best_match_c]
        recall_list = pred_gt_recall[best_match_r, best_match_c]
        assert num_pred >= len(best_match_c)
        return_dict['fscore_list'] = fscore_list
        return_dict['precision_list'] = precision_list
        return_dict['recall_list'] = recall_list

    ### Compute the number of "detected objects" ###
    correct_obj_detected = 0
    for fscore_of_matched_obj_i in pred_gt_fscore[best_match_r, best_match_c]:
        if fscore_of_matched_obj_i > correct_thr:
            correct_obj_detected += 1
    return_dict['precision'] = precision
    return_dict['recall'] = recall
    return_dict['fscore'] = fscore_final
    return_dict.update({
        'pred_object_num': num_pred,
        'gt_object_num': num_gt,
        'correct_object_num': correct_obj_detected
    })

    # boundary
    # Preprocess boundary counts
    boundary_overlap_tp = np.zeros((num_pred, num_gt, 2))
    boundary_map_maskpreds, dilated_boundary_map_maskpreds = [], []
    for pred_mask in maskpred:
        boundary_map, dilated_boundary_map = get_boundary(pred_mask)
        boundary_map_maskpreds.append(boundary_map)
        dilated_boundary_map_maskpreds.append(dilated_boundary_map)
    boundary_map_maskgts, dilated_boundary_map_maskgts = [], []
    for gt_mask in maskgt:
        boundary_map, dilated_boundary_map = get_boundary(gt_mask)
        boundary_map_maskgts.append(boundary_map)
        dilated_boundary_map_maskgts.append(dilated_boundary_map)
    boundary_map_maskpreds, dilated_boundary_map_maskpreds, boundary_map_maskgts, dilated_boundary_map_maskgts = \
        map(np.array, [boundary_map_maskpreds, dilated_boundary_map_maskpreds, boundary_map_maskgts,
                       dilated_boundary_map_maskgts])

    boundary_sum_pred, boundary_sum_gt = np.array([x.sum() for x in boundary_map_maskpreds]), \
                                         np.array([x.sum() for x in boundary_map_maskgts])

    for i in range(num_pred):
        for j in range(num_gt):
            # Get the intersection (true positives). Calculate true positives differently for
            #   precision and recall since we have to dilated the boundaries
            p2gt = np.logical_and(boundary_map_maskpreds[i], dilated_boundary_map_maskgts[j]).sum()
            gt2p = np.logical_and(boundary_map_maskgts[j], dilated_boundary_map_maskpreds[i]).sum()
            boundary_overlap_tp[i, j] = p2gt, gt2p

    # Boundary measures
    boundary_precision = np.sum(boundary_overlap_tp[best_match_r, best_match_c][:, 0]) / boundary_sum_pred.sum()
    boundary_recall = np.sum(boundary_overlap_tp[best_match_r, best_match_c][:, 1]) / boundary_sum_gt.sum()
    boundary_F_measure = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
    if np.isnan(boundary_F_measure):  # b/c/ precision = recall = 0
        boundary_F_measure = 0
    return_dict['boundary_precision'] = boundary_precision
    return_dict['boundary_recall'] = boundary_recall
    return_dict['boundary_fscore'] = boundary_F_measure

    boundary_pred_gt_precision = boundary_overlap_tp[..., 0] / boundary_sum_pred[..., np.newaxis]
    boundary_pred_gt_recall = boundary_overlap_tp[..., 1] / boundary_sum_gt
    boundary_pred_gt_fscore = (2 * boundary_pred_gt_precision * boundary_pred_gt_recall) / (
                boundary_pred_gt_precision + boundary_pred_gt_recall)
    boundary_pred_gt_fscore[np.isnan(boundary_pred_gt_fscore)] = 0.
    boundary_P_osn = boundary_pred_gt_precision[best_match_r, best_match_c].sum() / num_pred
    boundary_R_osn = boundary_pred_gt_recall[best_match_r, best_match_c].sum() / num_gt
    boundary_F_osn = boundary_pred_gt_fscore[best_match_r, best_match_c].sum() / max([num_pred, num_gt])
    return_dict['boundary_precision_osn'] = boundary_P_osn
    return_dict['boundary_recall_osn'] = boundary_R_osn
    return_dict['boundary_fscore_osn'] = boundary_F_osn

    return return_dict


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(log_path):
    """
        Print to both stdout and file handler.
    """
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_path, mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)


def read_ocid_rgbd_gtmask(rgb_path):
    rgb_im = cv2.imread(rgb_path)[..., ::-1]

    label_path = rgb_path.replace('rgb', 'label')
    pcd_path = rgb_path.replace('rgb', 'pcd')[:-3] + 'pcd'

    gt_im = cv2.imread(label_path, -1)
    background_label_id = [0, 1] if 'table' not in label_path else [0, 1, 2]

    gt_masks_individual = []
    pointcloud_o3d = o3d.io.read_point_cloud(pcd_path)
    pointcloud = np.asarray(pointcloud_o3d.points)
    for objid in np.unique(gt_im):
        if objid not in background_label_id:
            gt_masks_individual.append(gt_im == objid)

    im_h, im_w = gt_im.shape
    pointcloud = pointcloud.reshape((im_h, im_w, 3))
    pointcloud[np.isnan(pointcloud.reshape(-1, 3)).any(axis=1).reshape(im_h, im_w)] = 0
    return rgb_im, pointcloud, gt_masks_individual


def main():
    save_dir = SAVE_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.system(f"cp {os.path.join(os.path.dirname(__file__),'../uncos/uncos.py')} {os.path.join(save_dir,'uncos.py')}")
        os.system(f"cp {os.path.join(os.path.dirname(__file__), 'benchmarking.py')} {os.path.join(save_dir, 'benchmarking.py')}")
        os.system(f"cp {os.path.join(os.path.dirname(__file__),'../uncos/groundedsam_wrapper.py')} {os.path.join(save_dir,'groundedsam_wrapper.py')}")
        os.system(f"cp {os.path.join(os.path.dirname(__file__),'../uncos/uncos_utils.py')} {os.path.join(save_dir,'uncos_utils.py')}")
    logname = 'log'
    log_file_name = os.path.join(save_dir, logname)
    setup_logging(log_file_name)
    logging.info(f'saving to {save_dir}')

    ocid_directory = OCID_DIR
    rgb_paths = glob.glob(ocid_directory + '**/rgb/**png', recursive=True)

    uncos = UncOS(initialize_tracker=False)

    logging.info(f'{len(rgb_paths)} images to be evaluated.')

    best_hyp_result_logger, worst_hyp_result_logger, ml_hyp_result_logger = SegResultLogger(), SegResultLogger(), SegResultLogger()

    for i, rgb_path in enumerate(rgb_paths):
        rgb_im, pointcloud, gt_masks_individual = read_ocid_rgbd_gtmask(rgb_path)

        st = time.time()
        uncos.set_image(rgb_im.copy(), pointcloud=pointcloud)
        st1 = time.time()
        table_mask = uncos.get_table_or_background_mask(pointcloud, include_background=True, fast_inference=False)
        logging.info(f'TIME get table {time.time() - st1}')

        masks_all_allhypotheses: List[List[np.ndarray]] = []
        masks_all_certain, hypotheses = uncos.segment_scene(rgb_im.copy(), pointcloud=pointcloud, return_most_likely_only=False,
                                                          table_or_background_mask=table_mask, debug=False,  # DEBUG,
                                                          n_seg_hypotheses_trial=N_SEG_HYPOTHESES_TRIAL,
                                                          n_trial_per_query_point=N_TRIAL_PER_QUERY_POINT)
        logging.info(f'TIME segment scene all {time.time() - st}')
        with open(os.path.join(save_dir, f'ocid_{i:04d}_res.pkl'), 'wb') as f:
            pickle.dump([rgb_path, masks_all_certain, hypotheses], f)

        for one_hypothesis_of_alluncertain in itertools.product(
                *[hypothesis.get_region_hypotheses_masks() for hypothesis in hypotheses]):
            masks = masks_all_certain + list(itertools.chain(*one_hypothesis_of_alluncertain))
            masks_all_allhypotheses.append(masks)

        ################################## evaluation ##################################

        logging.info(f'{len(hypotheses)} uncertain regions. {len(masks_all_allhypotheses)} hypotheses.')
        # ############# fscore #############
        # all_hypotheses_eval_results = []
        # for masks in masks_all_allhypotheses:
        #     eval_results = eval_pred_to_gt(np.array(masks), np.array(gt_masks_individual))
        #     all_hypotheses_eval_results.append(eval_results)
        most_likely_hypothesis = itertools.chain(
            *[hypothesis.get_most_likely_hypothesis() for hypothesis in hypotheses])
        ml_masks = list(most_likely_hypothesis) + masks_all_certain
        ml_eval_result = eval_pred_to_gt(np.array(ml_masks), np.array(gt_masks_individual))

        # ############# sort and print ###########
        # hypotheses_result_ranking_worsttobest_idx = np.argsort([x['fscore'] for x in all_hypotheses_eval_results])
        # best_idx, worst_idx = hypotheses_result_ranking_worsttobest_idx[-1], hypotheses_result_ranking_worsttobest_idx[
        #     0]
        # logging.info(f'best idx {best_idx} worst {worst_idx}')
        # best_hyp_result, worst_hyp_result = all_hypotheses_eval_results[best_idx], \
        #                                     all_hypotheses_eval_results[worst_idx]
        # best_hyp_result_logger.add_result(best_hyp_result)
        # worst_hyp_result_logger.add_result(worst_hyp_result)
        ml_hyp_result_logger.add_result(ml_eval_result)

        # best_hyp_result_logger.print(header='best hypothesis')
        # worst_hyp_result_logger.print(header='worst hypothesis')
        ml_hyp_result_logger.print(header='most likely')

        ############ save and visualize ###############
        if (len(ml_hyp_result_logger) - 1) % 50 == 0:
            save_path = os.path.join(save_dir, f'ocid_{i:04d}_ml.png')
            uncos.visualize_confident_uncertain(ml_masks, [], show=False, save_path=save_path,
                                              plot_anno=f"score {ml_eval_result['fscore']:.2f}|osn {ml_eval_result['fscore_osn']:.2f}"
                                                        f"|prec {ml_eval_result['precision']:.2f}|rec {ml_eval_result['recall']:.2f}"
                                                        f"|pred #{ml_eval_result['pred_object_num']}. actual #{ml_eval_result['gt_object_num']}")
            # save_path = os.path.join(save_dir, f'ocid_{i:04d}_worst.png')
            # uncos.visualize_confident_uncertain(masks_all_allhypotheses[worst_idx], [], show=False, save_path=save_path,
            #                                   plot_anno=f"score {worst_hyp_result['fscore']:.2f}|osn {worst_hyp_result['fscore_osn']:.2f}"
            #                                             f"|prec {worst_hyp_result['precision']:.2f}|rec {worst_hyp_result['recall']:.2f}"
            #                                             f"|pred #{worst_hyp_result['pred_object_num']}. actual #{worst_hyp_result['gt_object_num']}")
            logging.info(f'save to {save_path}')


if __name__ == '__main__':
    fix_randseed(5678)
    main()
