import glob
import os
import logging, sys
import itertools
from typing import List, Dict
from dataclasses import dataclass, field

from uncos import UncOS
import cv2
import numpy as np
import open3d as o3d
import pickle
import torch
import random
from scipy.optimize import linear_sum_assignment

SAVE_DIR = "eval"
OCID_DIR = "data/OCID-dataset/"


@dataclass
class SegResultLogger:
    all_fscore_pix: List = field(default_factory=list)
    all_fscore_osn: List = field(default_factory=list)
    all_prec_pix: List = field(default_factory=list)
    all_prec_osn: List = field(default_factory=list)
    all_recall_pix: List = field(default_factory=list)
    all_recall_osn: List = field(default_factory=list)
    all_pred_num: List = field(default_factory=list)
    all_gt_num: List = field(default_factory=list)

    all_correct_num: List = field(default_factory=list)
    all_correct_ratio: List = field(default_factory=list)

    def add_result(self, eval_results: Dict):
        self.all_fscore_osn.append(eval_results['fscore_osn'])
        self.all_prec_osn.append(eval_results['precision_osn'])
        self.all_recall_osn.append(eval_results['recall_osn'])

        self.all_fscore_pix.append(eval_results['fscore_pix'])
        self.all_prec_pix.append(eval_results['precision_pix'])
        self.all_recall_pix.append(eval_results['recall_pix'])

        self.all_pred_num.append(eval_results['pred_object_num'])
        self.all_gt_num.append(eval_results['gt_object_num'])

        self.all_correct_num.append(eval_results['correct_object_num'])
        self.all_correct_ratio.append(
            eval_results['correct_object_num'] / max(eval_results['gt_object_num'], eval_results['pred_object_num'])
            if eval_results['gt_object_num'] > 0 else 0)
        logging.info(f"\t\t\t pred #{eval_results['pred_object_num']} \t\t gt #{eval_results['gt_object_num']}")

    def __len__(self):
        return len(self.all_fscore_pix)

    def print(self, header=None):
        logging.info(f"{header} cycle {self.__len__()} : ")
        logging.info(
            f'\t\tavg fscore_osn is {np.mean(self.all_fscore_osn)}\t\t avg precision_osn is {np.mean(self.all_prec_osn)}\n'
            f'\t\tavg recall_osn is {np.mean(self.all_recall_osn)}\t\t \n'
            f'\t\tavg avg fscore_pix is {np.mean(self.all_fscore_pix)}\t\t avg precision_osn is {np.mean(self.all_prec_pix)}\n'
            f'\t\tavg recall_osn is {np.mean(self.all_recall_pix)}\t\t \n'
            f'\t\tcorrect num is {np.mean(self.all_correct_num)}\t\t correct obj ratio is {np.mean(self.all_correct_ratio)}\n'
            f'\t\tavg pred num is #{np.mean(self.all_pred_num)}\t\t avg gt num is #{np.mean(self.all_gt_num)}')


def eval_pred_to_gt(maskpred, maskgt, osn=True, correct_thr=.75):
    """
    Args:
        maskpred:       K1 x H x W array bool
        maskgt:         K2 x H x W array bool
        osn:            Compute OSN score or no
        correct_thr:    IoU cutoff for an object to be considered 'correctly segmented'
    Returns:

    """
    return_dict = {}
    num_pred, num_gt = len(maskpred), len(maskgt)
    if num_pred == 0 and num_gt > 0:
        return {'fscore_osn': 0,
                'precision_osn': 1,
                'recall_osn': 0,
                'fscore_pix': 0.,
                'precision_pix': 1,
                'recall_pix': 0,
                'pred_object_num': num_pred,
                'gt_object_num': num_gt,
                'fscore_list': np.array([]),
                'precision_list': np.array([]),
                'recall_list': np.array([]),
                'correct_object_num': 0,
                'match_mat': [[], []]
                }

    elif num_pred > 0 and num_gt == 0:  # all false negatives
        return {'fscore_osn': 0,
                'precision_osn': 0,
                'recall_osn': 1,
                'fscore_pix': 0.,
                'precision_pix': 0,
                'recall_pix': 1,
                'pred_object_num': num_pred,
                'gt_object_num': num_gt,
                'fscore_list': np.array([]),
                'precision_list': np.array([]),
                'recall_list': np.array([]),
                'correct_object_num': 0,
                'match_mat': [[], []]
                }
    elif (num_pred == 0 and num_gt == 0):  # correctly predicted nothing
        return {'fscore_osn': 1.,
                'precision_osn': 1,
                'recall_osn': 1,
                'fscore_pix': 1.,
                'precision_pix': 1,
                'recall_pix': 1,
                'pred_object_num': num_pred,
                'gt_object_num': num_gt,
                'Objects Recall': 1.,
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
    return_dict['precision_pix'] = precision
    return_dict['recall_pix'] = recall
    return_dict['fscore_pix'] = fscore_final
    return_dict.update({
        'pred_object_num': num_pred,
        'gt_object_num': num_gt,
        'correct_object_num': correct_obj_detected
    })
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
        os.system(
            f"cp {os.path.join(os.path.dirname(__file__), '../uncos/uncos.py')} {os.path.join(save_dir, 'uncos.py')}")
        os.system(
            f"cp {os.path.join(os.path.dirname(__file__), 'benchmarking.py')} {os.path.join(save_dir, 'benchmarking.py')}")
        os.system(
            f"cp {os.path.join(os.path.dirname(__file__), '../uncos/groundedsam_wrapper.py')} {os.path.join(save_dir, 'groundedsam_wrapper.py')}")
        os.system(
            f"cp {os.path.join(os.path.dirname(__file__), '../uncos/uncos_utils.py')} {os.path.join(save_dir, 'uncos_utils.py')}")
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

        masks_all_allhypotheses: List[List[np.ndarray]] = []
        uncos.set_image(rgb_im.copy(), pointcloud=pointcloud)
        table_mask = uncos.get_table_or_background_mask(pointcloud, include_background=True, fast_inference=False)
        masks_all_certain, hypotheses = uncos.segment_scene(rgb_im.copy(), pointcloud=pointcloud,
                                                            return_most_likely_only=False,
                                                            table_or_background_mask=table_mask, debug=False)
        with open(os.path.join(save_dir, f'ocid_{i:04d}_res.pkl'), 'wb') as f:
            pickle.dump([rgb_path, masks_all_certain, hypotheses], f)

        for one_hypothesis_of_alluncertain in itertools.product(
                *[hypothesis.get_region_hypotheses_masks() for hypothesis in hypotheses]):
            masks = masks_all_certain + list(itertools.chain(*one_hypothesis_of_alluncertain))
            masks_all_allhypotheses.append(masks)

        ################################## evaluation ##################################

        logging.info(f'{len(hypotheses)} uncertain regions. {len(masks_all_allhypotheses)} hypotheses.')
        most_likely_hypothesis = itertools.chain(
            *[hypothesis.get_most_likely_hypothesis() for hypothesis in hypotheses])
        ml_masks = list(most_likely_hypothesis) + masks_all_certain
        ml_eval_result = eval_pred_to_gt(np.array(ml_masks), np.array(gt_masks_individual))
        ml_hyp_result_logger.add_result(ml_eval_result)
        ml_hyp_result_logger.print(header='most likely')

        ############ save and visualize ###############
        if (len(ml_hyp_result_logger) - 1) % 50 == 0:
            save_path = os.path.join(save_dir, f'ocid_{i:04d}_ml.png')
            uncos.visualize_confident_uncertain(ml_masks, [], show=False, save_path=save_path,
                                                plot_anno=f"fscore_osn {ml_eval_result['fscore_osn']:.2f}"
                                                          f"|pred obj num #{ml_eval_result['pred_object_num']}. "
                                                          f"actual #{ml_eval_result['gt_object_num']}")
            logging.info(f'save to {save_path}')


if __name__ == '__main__':
    fix_randseed(5678)
    main()
