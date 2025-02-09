import os
import numpy as np
import numpy.ma as ma
import scipy.ndimage
import itertools
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Union
from functools import reduce

import torch
import trimesh
import open3d as o3d

from scipy.optimize import linear_sum_assignment
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from .uncos_utils import iou_fn, iom_fn, intersection_fn, bfs_cluster, crop, overlay_masks, visualize_pointcloud, \
    overlay_mask_simple, is_degenerated_mask, MaskWrapper, SegHypothesis, RegionHypothesis
from .groundedsam_wrapper import GroundedSAM
from .config import IOU_THRES, SAM_CONF_SCORE_THR, MAX_DEPTH, INTERSECTION_THRES, TABLE_INLIER_THR, MIN_AREA_PERCENTAGE, \
                    USE_SAM2, UNCOS_CKPT_DIR_PATH, SAM2_CKPT_PATH

class UncOS:
    def __init__(self, add_topdown=True, initialize_tracker=False, device=None, take_union_if_uncertain=False):
        self.rgb_im = None
        self.pcd = None
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cache_dir = os.path.expanduser(UNCOS_CKPT_DIR_PATH)
        sam_ckpt_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")
        if not os.path.exists(sam_ckpt_path):
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Downloading SAM checkpoint to {sam_ckpt_path}')
            torch.hub.download_url_to_file(
                'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                sam_ckpt_path)

        if not USE_SAM2:
            sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            # 1024 x 1024 is the input size for SAM pretrained model
            self.mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=int(MIN_AREA_PERCENTAGE * 1024 * 1024))
        else:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", SAM2_CKPT_PATH)
            sam = sam2
            self.predictor = SAM2ImagePredictor(sam2)
            # 1024 x 1024 is the input size for SAM pretrained model
            self.mask_generator = SAM2AutomaticMaskGenerator(sam2, min_mask_region_area=int(MIN_AREA_PERCENTAGE * 1024 * 1024))

        if initialize_tracker:
            raise NotImplementedError
            # # TODO. remove from the minimal uncos repo (if not doing EOS). use submodule.
            # track_anything_dir = os.path.join(os.path.dirname(__file__),'../Track-Anything')
            # sys.path.append(track_anything_dir)
            # from tracker.base_tracker import BaseTracker
            # self.matcher = BaseTracker(os.path.join(track_anything_dir,'checkpoints/XMem-s012.pth'), device=device)
        self.device = torch.device(device)

        self.add_topdown_highprecision_masks = add_topdown
        if add_topdown:
            self.grounded_sam_wrapper = GroundedSAM(box_thr=.1, text_thr=.05, loaded_sam=sam)
        # logging.getLogger().setLevel(logging.INFO if not VERBOSE_DEBUG else logging.DEBUG)
        self.take_union_if_uncertain = take_union_if_uncertain

    def set_image(self, rgb_im, pointcloud):
        self.rgb_im = rgb_im
        self.predictor.set_image(rgb_im)
        self.pcd = pointcloud

    def filter_mask_depth(self, masks, depth, threshold=0.3, max_depth=MAX_DEPTH):
        """
            Remove predicted masks that contain objects far from the camera.
        """
        filtered_masks = []
        for i, mask in enumerate(masks):
            roi_depth = depth[mask().astype(bool)][:, 2]
            too_far_percentage = (np.sum(roi_depth > max_depth) + np.sum(roi_depth == 0)) / np.sum(mask())
            if too_far_percentage < threshold:
                filtered_masks.append(mask)
        return filtered_masks

    def filter_mask_excludemask(self, masks: List[MaskWrapper], exclude_mask: np.ndarray, io_exclude_threshold=0.6) -> \
            List[MaskWrapper]:
        """
        Remove predicted mask if intersection over excluded mask > threshold.
        :param masks: mask in MaskWrapper
        :param exclude_mask: bool type np array
        :param io_exclude_threshold:
        :return:
        """
        filtered_masks = []
        for i, mask in enumerate(masks):
            io_exclude = (mask().astype(bool) & exclude_mask).astype(np.float32).sum() / mask().sum()
            if io_exclude < io_exclude_threshold:
                filtered_masks.append(mask)
        return filtered_masks

    def get_uncertain_areas(self, masks_raw: List[MaskWrapper], iom_thershold=IOU_THRES, added_masks=None) -> Tuple[
        List[np.ndarray], List[List]]:
        """

        Args:
            masks: list of masks ( in MaskWrapper )

        Returns:
            1) A list of confident masks. np.ndarray
            2) A list of uncertain regions. An uncertain region contains a list of np.ndarray masks.
        """
        if added_masks is None:
            added_masks = []
        masks = masks_raw + added_masks
        iom_matrix = np.eye(len(masks))
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                iom = iom_fn(masks[i](), masks[j]())
                iom_matrix[i, j] = iom
                iom_matrix[j, i] = iom
        clusters = bfs_cluster(iom_matrix, iom_thershold)

        all_confident_masks, uncertain_regions = [], []
        for cluster in clusters:
            ucr = [masks[i]() for i in cluster]
            if self.take_union_if_uncertain:
                # baseline: take union of mask if overlap. Works well if objects are isolated.
                all_confident_masks.append(self.get_union_mask(ucr))
                continue
            if len(cluster) == 1:
                if list(cluster)[0] < len(masks_raw):
                    # is a predicted mask
                    confident_mask, is_certain = self.check_is_confident(ucr[0])
                else:
                    # is a topdown-high-precision mask
                    confident_mask, is_certain = ucr[0], True
                if is_certain:
                    all_confident_masks.append(confident_mask)
                    continue
            elif is_degenerated_mask(self.get_union_mask(ucr))[1]:
                print(f'AREA too small. {self.get_union_mask(ucr).sum() / self.get_union_mask(ucr).shape[0] / self.get_union_mask(ucr).shape[1]} ')
                continue
            uncertain_regions.append(ucr)
        return all_confident_masks, uncertain_regions

    def check_is_confident(self, mask: np.ndarray, n_checking_points=5):
        """
        Verify confident area. Issue N randomly sampled point queries and check if the returned masks are consistent.
        :param mask:
        :param n_checking_points:
        :return:
        """
        sampled_points = self.sample_points_in_mask(mask, size=n_checking_points)
        is_confident = True
        all_masks = []
        for sampled_point in sampled_points:
            masks = self.masks_from_point_query(np.array([sampled_point]), return_masknum=3)
            valid_masks = [maskx() for maskx in masks if maskx.score > SAM_CONF_SCORE_THR]
            all_masks.extend(valid_masks)
        iou_matrix = np.array(
            list(itertools.chain(*[[iou_fn(mask1, mask2) for (i, mask1) in enumerate(all_masks) if i > j]
                                   for (j, mask2) in enumerate(all_masks)])))

        if any(iou_matrix < IOU_THRES):
            is_confident = False
        if len(all_masks) == 0:
            all_masks = [mask]
        if not is_confident:
            non_repeat_masks = []
            for maskx in all_masks:
                mask2d_iou_to_prevmasks = [iou_fn(maskx, mask_prev) for mask_prev in non_repeat_masks]
                if len(mask2d_iou_to_prevmasks) == 0 or max(mask2d_iou_to_prevmasks) <= .95:
                    non_repeat_masks.append(maskx)
                    continue
            return non_repeat_masks, is_confident
        return reduce(np.logical_or, all_masks), True

    def get_union_mask(self, masks: List[np.ndarray]) -> np.ndarray:
        return reduce(np.logical_or, [mask.astype(bool) for mask in masks])

    def is_same_segmentation(self, hyp1: SegHypothesis, hyp2: SegHypothesis, iou_threshold=.8):
        """
        Check if two partitions of the same region are equivalent.
        Args:
            hyp1: SegHypothesis.
            hyp2: SegHypothesis.

        Returns:
            bool. Are they the same segmentation

        """
        if hyp1.mask_num != hyp2.mask_num:
            return False
        masks_1 = hyp1.masks
        masks_2 = hyp2.masks
        iou_matrix = np.array([[iou_fn(s_i, s_j) for s_j in masks_2] for s_i in masks_1])
        if len(iou_matrix.shape) != 2:
            breakpoint()
        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
        mean_iou = iou_matrix[row_ind, col_ind].mean()
        if mean_iou > iou_threshold:
            return True
        return False

    def make_point_query(self, uncertain_area, pointcloud, hypothesis, sampled_points, sampled_points_labels, split=False, debug=False):
        """
        Return one valid segmentation mask from a set of sampled (positive and negative) prompt points
        Return None if there is no valid mask from the set of point queries
        Args:
            sampled_points
        Returns:
            queried_mask :          A binary mask for the point queries. H x W
            queried_mask_score :    A scalar score (predicted IoU from SAM) for the returned mask
            if no valid mask, return (None, None)
        """
        queried_masks_all = self.masks_from_point_query(sampled_points, input_label=sampled_points_labels, vis=False, return_masknum=3)
        queried_masks = [(mask, mask.score) for mask in queried_masks_all if mask.score > SAM_CONF_SCORE_THR]
        if len(queried_masks) == 0:
            if debug:
                plt.figure(figsize=(18, 6))
                for i, mask in enumerate(queried_masks_all):
                    plt.subplot(1, len(queried_masks_all), i + 1)
                    m1 = overlay_mask_simple(self.rgb_im, mask())
                    for sampled_point in sampled_points:
                        cv2.circle(m1, tuple(sampled_point), radius=5, color=(0, 0, 1.))
                    plt.imshow(m1)
                    plt.title(mask.score)
                plt.show()
                # plt.savefig(f'{time.time()}.png')

                # logging.info(f'no valid queried masks above score {SAM_CONF_SCORE_THR}. skip')
            return None, None
        queried_masks.sort(key=lambda x: x[1], reverse=True)
        queried_mask_, queried_mask_score = queried_masks[0]
        queried_mask = queried_mask_()

        if is_degenerated_mask(queried_mask, pointcloud)[1]:
            return None, None
        if (queried_mask & uncertain_area).sum() / queried_mask.sum() < IOU_THRES:
            if debug:
                print('outside of uncertain region')
            return None, None
        if hypothesis.mask_num >= 1:
            iom = np.array([iom_fn(queried_mask, x) for x in hypothesis.masks])
            overlapped = np.where(iom > IOU_THRES)[0]
            if len(overlapped) > 0:
                if split and len(
                        np.where(np.array([iou_fn(queried_mask, x) for x in hypothesis.masks]) > IOU_THRES)[0]) == 0:
                    for overlap_mask_idx in overlapped:
                        is_query_smaller = queried_mask.sum() < hypothesis.masks[overlap_mask_idx].sum()
                        if not is_query_smaller:
                            # TODO: set-based subtraction
                            queried_mask = queried_mask & ~hypothesis.masks[overlap_mask_idx]
                            remaining_area_subtractedmask, is_mask_degenerated_subtractedmask = is_degenerated_mask(
                                queried_mask,
                                pointcloud=pointcloud)
                            if is_mask_degenerated_subtractedmask:
                                return None, None
                            # else:
                            #     break
                        else:
                            hypothesis.masks[overlap_mask_idx] = hypothesis.masks[overlap_mask_idx] & ~queried_mask
                            remaining_area_prevmask, is_mask_degenerated_prevmask = is_degenerated_mask(hypothesis.masks[overlap_mask_idx],
                                                                                       pointcloud=pointcloud)
                            if is_mask_degenerated_prevmask:
                                return None, None
                else:
                    return None, None
            intersection = [intersection_fn(queried_mask, x) for x in hypothesis.masks]
            if not split:
                if any(np.array(intersection) > INTERSECTION_THRES):
                    return None, None
        return queried_mask, queried_mask_score


    def generate_a_hypothesis(self, uncertain_area, pointcloud,
                              n_trial_per_query_point,
                              ori_uncertain_area,
                              added_mask=None, debug=False,
                              num_neg_points_base=0,
                              sampled_points_bg=None,
                              split=False) -> Union[SegHypothesis, None]:
        """
        Generate a segmentation hypothesis for an uncertain region.
        """
        num_neg_points = num_neg_points_base
        remaining_area = uncertain_area.copy()
        hypothesis = SegHypothesis()
        remaining_area, is_mask_degenerated = is_degenerated_mask(remaining_area, pointcloud=pointcloud)
        if is_mask_degenerated:
            hypothesis.add_part(ori_uncertain_area.copy())
            return hypothesis
        loop_i = 0
        if added_mask is not None:
            hypothesis.add_part(added_mask)
            remaining_area = remaining_area & ~added_mask
            remaining_area, is_mask_degenerated = is_degenerated_mask(remaining_area,
                                                                      pointcloud=pointcloud)
        while not is_mask_degenerated:  # or iou_fn(hypothesis.area_mask,uncertain_area)<IOU_THRES:
            if hypothesis.mask_num > 0 and iou_fn(hypothesis.area_mask, ori_uncertain_area) >= .9:
                return hypothesis
            if loop_i > 30:
                wholearea_hypothesis = SegHypothesis()
                wholearea_hypothesis.add_part(ori_uncertain_area.copy())
                del hypothesis
                hypothesis = wholearea_hypothesis
                return hypothesis
            loop_i += 1
            if hypothesis.mask_num > 10:
                breakpoint()
            try:
                sampled_point = self.sample_points_in_mask(remaining_area, 1)[0]
            except Exception as e:
                print(e)
                sampled_point = self.sample_points_in_mask(remaining_area, 1, avoid_boundary=False)[0]
            sampled_points_fg = np.array([sampled_point])
            sampled_points_labels = np.array([0 for _ in range(len(sampled_points_bg))] + [1])
            sampled_points = np.concatenate((sampled_points_bg, sampled_points_fg), axis=0)
            if len(sampled_points) == 0:
                return None

            if hypothesis.mask_num > 0:
                sampled_points_negative = self.sample_points_in_mask(hypothesis.area_mask, num_neg_points)
                sampled_points_labels = np.concatenate((sampled_points_labels, np.zeros(num_neg_points)), axis=0)
                assert len(sampled_points_labels.shape) == 1
                sampled_points = np.concatenate((sampled_points, sampled_points_negative), axis=0)

            for _ in range(n_trial_per_query_point):
                queried_mask, queried_mask_score = self.make_point_query(uncertain_area, pointcloud, hypothesis, sampled_points,
                                                                         sampled_points_labels, split=split, debug=debug)
                if queried_mask is None:
                    continue
                hypothesis.add_part(queried_mask, queried_mask_score)
                remaining_area = remaining_area & ~queried_mask
                remaining_area_, is_mask_degenerated = is_degenerated_mask(remaining_area, pointcloud=pointcloud)
                if remaining_area_ is None:
                    assert is_mask_degenerated
                    hypothesis.add_part(remaining_area)
                remaining_area = remaining_area_
                break
            else:
               return None
        if (not is_mask_degenerated) or remaining_area is None:
            return None
        return hypothesis

    def get_hypotheses_of_region(self, uncertain_area, n_result=5, pointcloud=None,
                                 remove_repeat_hypothesis=True, n_trial_per_query_point=3, debug=False,
                                 added_masks=None) -> RegionHypothesis:
        """
        Generate X segmentation hypotheses for an uncertain region.
        Args:
            uncertain_area: np.ndarray. bool mask of uncertain area.
            n_result: number of times to query SAM for hypothesis. If the hypothesis is similar to a previous one,
                the result gets merged. The returning hypotheses may be smaller than `n_result`
            pointcloud: H x W x 3
        Returns:
            RegionHypothesis which contains X hypotheses.
        """
        hypotheses = []
        whole_area_mask = uncertain_area.copy()
        num_bg_points = 0  # 15
        num_neg_points_base = 0  # 3
        sampled_points_bg = self.sample_points_in_mask(~whole_area_mask, num_bg_points)

        if added_masks is None:
            added_masks = []

        ori_uncertain_area = uncertain_area

        while len(added_masks) > n_result:
            n_result += 2

        for nri in range(n_result):
            added_mask = None
            if len(added_masks) > 0:
                added_mask = added_masks[0]
            split_overlapping_mask = True if np.random.rand()<0.5 else False
            hypothesis = self.generate_a_hypothesis(uncertain_area, pointcloud, n_trial_per_query_point,
                                                    ori_uncertain_area, added_mask, debug, num_neg_points_base,
                                                    sampled_points_bg, split=split_overlapping_mask)
            hypotheses.append(hypothesis)
            if len(added_masks) > 0 and \
                    (hypothesis is not None or (len(hypotheses) > 0 and all([x is None for x in hypotheses[-3:]]))):
                added_masks.pop(0)

        hypotheses = [x for x in hypotheses if x is not None]
        if remove_repeat_hypothesis:
            filtered_hypotheses = []
            for hypothesis in hypotheses:
                similarity_to_prev_hyp = [self.is_same_segmentation(hypothesis, hypothesis_prev) for hypothesis_prev in
                                          filtered_hypotheses]
                is_the_same_hyp = any(similarity_to_prev_hyp)
                if is_the_same_hyp:
                    same_hyp_ids = np.where(np.array(similarity_to_prev_hyp))[0]
                    for same_hyp_id in same_hyp_ids:
                        filtered_hypotheses[same_hyp_id].increment_confidence()
                else:
                    filtered_hypotheses.append(hypothesis)
            hypotheses = filtered_hypotheses

        if len(hypotheses) == 0:
            wholearea_hypothesis = SegHypothesis()
            wholearea_hypothesis.add_part(ori_uncertain_area.copy())
            hypotheses.append(wholearea_hypothesis)

        ret = RegionHypothesis(hypotheses)
        return ret

    def re_generate_hypothesis_for_degenerated_region(self, mask: MaskWrapper, pointcloud,
                                                      query_trial_num=3) -> MaskWrapper:
        """
        For all the flat regions that are not overlapped with any other masks, query new mask for them.
        Args:
            mask: A single mask
            pointcloud:

        Returns:

        """
        for _ in range(query_trial_num):
            points_sampled_from_mask = self.sample_points_in_mask(mask(), avoid_boundary=False)
            queried_masks = self.masks_from_point_query(np.array([points_sampled_from_mask]))
            for queried_mask in queried_masks:
                if not is_degenerated_mask(queried_mask(), pointcloud)[1]:
                    return queried_mask
        # add points from neighboring area
        kernel_size = 3
        for _ in range(query_trial_num):
            dilated = self.dilate(mask(), kernel_size=kernel_size)
            edge = dilated ^ mask()
            point_sampled_from_mask = self.sample_points_in_mask(mask(), avoid_boundary=False)
            point_sampled_from_edge = self.sample_points_in_mask(edge, avoid_boundary=False)
            queried_mask = self.masks_from_point_query(np.array([point_sampled_from_mask, point_sampled_from_edge]))[0]
            if not is_degenerated_mask(queried_mask(), pointcloud)[1]:
                return queried_mask
            kernel_size += 1
        return queried_mask

    def propose_new_masks_for_degenerated_regions(self, masks, pointcloud):
        ## handle degenerated areas
        added_masks = []
        if pointcloud is None:
            return added_masks
        for i, mask in enumerate(masks):
            if not is_degenerated_mask(mask(), pointcloud)[1]:
                continue

            overlap_with_nondegenerated = False
            for j, mask_other in enumerate(masks + added_masks):
                if i == j:
                    continue
                iom = iom_fn(mask(), mask_other())
                if iom > IOU_THRES and not is_degenerated_mask(mask_other(), pointcloud)[1]:
                    overlap_with_nondegenerated = True
                    break
            if overlap_with_nondegenerated:
                continue
            requeried_mask = self.re_generate_hypothesis_for_degenerated_region(mask, pointcloud)
            added_masks.append(requeried_mask)
        return added_masks

    def get_table_or_background_mask(self, point_cloud, include_background=True, table_inlier_thr=TABLE_INLIER_THR,
                                     far=3, near=.03, fast_inference=True, pointcloud_frame=None):
        """
        Return the mask of table/background/non-foreground area.
        """
        print(f'point cloud pre-processing...')
        pred = np.zeros((point_cloud.shape[0], point_cloud.shape[1]))
        if pred.shape[1] == 3:
            raise RuntimeError('point_cloud needs to have the same shape as rgb (H W 3)')

        point_cloud_n3 = point_cloud.reshape(-1,3)
        if pointcloud_frame is None:
            valid_pt_below_zero_ratio = np.logical_and((point_cloud_n3 != 0).any(axis=1), point_cloud_n3[..., 2] < near).sum() / (point_cloud_n3 != 0).any(axis=1).sum()
            if valid_pt_below_zero_ratio > 0.25:
                pointcloud_frame = 'world'
            else:
                pointcloud_frame = 'camera'
        if pointcloud_frame == 'world':
            valid_cond = point_cloud.reshape(-1, 3)[..., 2] > -0.1
        elif pointcloud_frame == 'camera':
            valid_cond = np.logical_and(point_cloud.reshape(-1, 3)[..., 2] > near, point_cloud.reshape(-1, 3)[..., 2] < far)
        else:
            raise RuntimeError('pointcloud_frame should be [world/camera]')

        print(f'point cloud in {pointcloud_frame} frame')
        valid_cond = np.logical_and(valid_cond, ~(np.isnan(point_cloud.reshape(-1, 3)).any(axis=1)))
        valid_idx = np.where(valid_cond)[0]  # remove points too close to the camera

        valid_cloud = point_cloud.reshape(-1, 3)[valid_idx]
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_cloud))
        # visualize_pointcloud(valid_cloud, np.full(valid_cloud.shape[0], fill_value=False, dtype=bool))

        inliers_idx_in_valid_cloud = []
        while len(inliers_idx_in_valid_cloud) == 0 and table_inlier_thr < 0.1:
            plane_model, inliers_idx_in_valid_cloud = cloud_o3d.segment_plane(distance_threshold=table_inlier_thr,
                                                                              ransac_n=3,
                                                                              num_iterations=500)
            table_inlier_thr += 0.02
        if table_inlier_thr >= 0.1:
            raise RuntimeError(f'Error in getting table plane')
        inlier_bool_mask = np.full(valid_cloud.shape[0], fill_value=False, dtype=bool)
        inlier_bool_mask[inliers_idx_in_valid_cloud] = True
        # visualize_pointcloud(valid_cloud, inlier_bool_mask)

        if not fast_inference:
            # remove points that are not connected(outside of table) e.g. the wall at the back, that is on the plane, but not on the table
            table_plane_cloud = cloud_o3d.select_by_index(inliers_idx_in_valid_cloud)
            table_plane_and_other_indices = np.array(table_plane_cloud.cluster_dbscan(eps=.02, min_points=3))
            table_plane_and_other_indices[table_plane_and_other_indices == -1] = table_plane_and_other_indices.max() + 1
            largest_part_label = np.bincount(table_plane_and_other_indices).argmax()
            inliners_to_delete = []
            for group_id in np.unique(table_plane_and_other_indices):
                if group_id != largest_part_label and (table_plane_and_other_indices == group_id).sum() < 10000:
                    inliners_to_delete.extend(np.where(table_plane_and_other_indices == group_id)[0].tolist())
            inliers_idx_in_valid_cloud = np.delete(inliers_idx_in_valid_cloud, inliners_to_delete).astype(np.int64)
            inlier_bool_mask = np.full(valid_cloud.shape[0], fill_value=False, dtype=bool)
            inlier_bool_mask[inliers_idx_in_valid_cloud] = True
            # visualize_pointcloud(valid_cloud, inlier_bool_mask)

        self.last_table_cloud = valid_cloud[inlier_bool_mask]
        self.last_plane_model = plane_model
        if include_background:
            a, b, c, d = plane_model
            plane_normal = np.array([a, b, c])

            if pointcloud_frame == 'camera' and np.dot(plane_normal, np.array([0, 0, 1])) > 0:
                # revert if pointing 'into' the table
                plane_normal *= -1
            if pointcloud_frame == 'world' and np.dot(plane_normal, np.array([0, 0, -1])) > 0:
                # revert if pointing to the floor
                plane_normal *= -1

            # transform table to align with world xy plane
            plane_origin = np.array([0, 0, -d / c])
            rotation_matrix_align_plan_with_z = trimesh.points.plane_transform(plane_origin, plane_normal)
            rotated_valid_cloud = rotation_matrix_align_plan_with_z.dot(
                np.concatenate((valid_cloud.T, np.ones((1, valid_cloud.shape[0])))))
            rotated_valid_cloud = (rotated_valid_cloud / rotated_valid_cloud[-1])[:3].T
            # visualize_pointcloud(rotated_valid_cloud, return_trimesh_obj=True)

            rotated_trimeshpcd = trimesh.points.PointCloud(rotated_valid_cloud)
            to_origin_transform2d, _ = trimesh.bounds.oriented_bounds(rotated_trimeshpcd[:, :2], ordered=False)
            to_origin_transform = np.eye(4)
            to_origin_transform[:2, :2] = to_origin_transform2d[:2, :2]
            to_origin_transform[:2, 3] = to_origin_transform2d[:2, 2]

            rotated_valid_cloud2_xyaligned = to_origin_transform.dot(
                np.concatenate((rotated_valid_cloud.T, np.ones((1, rotated_valid_cloud.shape[0])))))
            rotated_valid_cloud2_xyaligned = (rotated_valid_cloud2_xyaligned / rotated_valid_cloud2_xyaligned[-1])[:3].T
            # visualize_pointcloud(rotated_valid_cloud2_xyaligned)

            # remove points outside of the table x y limit
            table_bound_x = rotated_valid_cloud2_xyaligned[inliers_idx_in_valid_cloud, 0].min(), \
                            rotated_valid_cloud2_xyaligned[inliers_idx_in_valid_cloud, 0].max()
            table_bound_y = rotated_valid_cloud2_xyaligned[inliers_idx_in_valid_cloud, 1].min(), \
                            rotated_valid_cloud2_xyaligned[inliers_idx_in_valid_cloud, 1].max()
            out_of_plane_cond = [rotated_valid_cloud2_xyaligned[:, 0] < table_bound_x[0],
                                 rotated_valid_cloud2_xyaligned[:, 0] > table_bound_x[1],
                                 rotated_valid_cloud2_xyaligned[:, 1] < table_bound_y[0],
                                 rotated_valid_cloud2_xyaligned[:, 1] > table_bound_y[1],
                                 rotated_valid_cloud2_xyaligned[:, 2] < 0]
            out_of_bound_idx_in_valid = np.where(reduce(np.logical_or, out_of_plane_cond))[0]

            # out of bound is background. mark them as inliers if include_background in returned table mask
            out_of_bound_mask = np.zeros((point_cloud.shape[0], point_cloud.shape[1]))
            out_of_bound_mask = out_of_bound_mask.reshape(-1)
            out_of_bound_mask[valid_idx[out_of_bound_idx_in_valid.tolist()]] = 1

            inliers_idx_in_valid_cloud = list(set(out_of_bound_idx_in_valid.tolist() + list(inliers_idx_in_valid_cloud)))

        pred = pred.reshape(-1)
        pred[valid_idx[inliers_idx_in_valid_cloud]] = 1
        if include_background:
            pred[np.where(~valid_cond)[0]] = 1
        pred_hwshape = pred.reshape(*point_cloud.shape[:2]).astype(bool)
        print(f'point cloud pre-processing done.')
        # visualize_pointcloud(point_cloud.reshape(-1,3), pred_hwshape.reshape(-1))
        return pred_hwshape

    def get_table_or_background_mask_coloronly(self):
        table_mask = self.grounded_sam_wrapper.process_image(self.rgb_im, text_prompt='table surface,wall,floor')
        return reduce(np.logical_or, [x() for x in table_mask])

    def get_topdown_masks(self, rgb_im, exclude_background_mask, text_prompt):
        masks_all_certain_raw_sammask = self.grounded_sam_wrapper.process_image(rgb_im, text_prompt=text_prompt)
        masks_all_certain_sammask = [mask for mask in masks_all_certain_raw_sammask if
                                     np.logical_and(mask(), exclude_background_mask).sum() / mask().sum() < 0.8]
        filtered_mask = [x for x in masks_all_certain_sammask if x.score > SAM_CONF_SCORE_THR]
        masks_all_certain_sammask = filtered_mask
        return masks_all_certain_sammask

    def segment_scene(
        self, rgb_im, pointcloud, table_or_background_mask=None, return_most_likely_only=False,
        debug=False,
        pointcloud_frame=None,
        n_seg_hypotheses_trial=5,
        n_trial_per_query_point=3,
        visualize_hypotheses=False,
        fast_inference=True
    ) -> Tuple[List[np.ndarray], List[RegionHypothesis]]:
        self.set_image(rgb_im, pointcloud)
        im_h, im_w, _ = rgb_im.shape

        if table_or_background_mask is None:
            if pointcloud is not None:
                table_or_background_mask = self.get_table_or_background_mask(pointcloud, fast_inference=fast_inference, pointcloud_frame=pointcloud_frame)
            else:
                table_or_background_mask = np.zeros(rgb_im.shape[:2]).astype(bool)

        if debug:
            print(f'visualizing table and background mask.')
            plt.imshow(table_or_background_mask)
            plt.show()
            plt.close()

        if self.add_topdown_highprecision_masks:
            topdown_highprecision_masks = [MaskWrapper({'segmentation': x(), 'predicted_iou': x.score})
                                           for x in self.get_topdown_masks(rgb_im, table_or_background_mask,
                                                                           text_prompt='A rigid object.')]
        else:
            topdown_highprecision_masks = []

        with torch.no_grad():
            masks = [MaskWrapper(m) for m in self.mask_generator.generate(rgb_im)]
        if pointcloud is not None:
            masks = self.filter_mask_depth(masks, pointcloud)
        masks = self.filter_mask_excludemask(masks, exclude_mask=table_or_background_mask.astype(bool))
        if pointcloud is not None:
            masks = self.remove_degenerate(masks, pointcloud)
        added_masks_from_flat_region = self.propose_new_masks_for_degenerated_regions(masks, pointcloud)
        masks = masks + added_masks_from_flat_region

        confident_masks, uncertain_areas = self.get_uncertain_areas(masks,
                                                                    added_masks=topdown_highprecision_masks)
        if debug:
            print(f'visualizing the confident area')
            self.visualize_confident_uncertain(confident_masks, [], plot_anno='confident area')

        hypotheses = []  # {}

        hypotheses_for_each_area = []
        for i, uncertain_area in enumerate(uncertain_areas):
            uncertain_area_areamask = self.get_union_mask(uncertain_area)
            hypotheses_for_area_i = self.get_hypotheses_of_region(uncertain_area_areamask,
                                                                  n_result=n_seg_hypotheses_trial,
                                                                  pointcloud=pointcloud,
                                                                  remove_repeat_hypothesis=True,
                                                                  n_trial_per_query_point=n_trial_per_query_point,
                                                                  debug=debug,
                                                                  added_masks=[x() for x in topdown_highprecision_masks
                                                                               if iom_fn(x(),
                                                                                         uncertain_area_areamask) > IOU_THRES])
            hypotheses_for_each_area.append(hypotheses_for_area_i)

        for hypotheses_for_area_i in hypotheses_for_each_area:
            if hypotheses_for_area_i.hypotheses_num > 1:
                hypotheses.append(hypotheses_for_area_i)
            else:
                # add it back to confident_masks if there is only one hypothesis
                confident_masks.extend(hypotheses_for_area_i.masks_union)
        if visualize_hypotheses:
            self.visualize_confident_uncertain(confident_masks, hypotheses)
        if return_most_likely_only:
            most_likely_hypothesis = itertools.chain(
                *[hypothesis.get_most_likely_hypothesis() for hypothesis in hypotheses])
            confident_masks += most_likely_hypothesis
            return confident_masks, []

        return confident_masks, hypotheses

    def remove_degenerate(self, masks: List[MaskWrapper], pointcloud) -> List[MaskWrapper]:
        if pointcloud is None:
            return masks
        volumes = []
        for i, mask in enumerate(masks):
            if is_degenerated_mask(mask(), pointcloud)[1]:
                volumes.append(0)
                continue
            volumes.append(.01)  # could compute volume but skipped for efficiency
        non_degenerated_masks = []
        threshold = 0.00001  # volume threshold - masks under this will be merged w others
        small_regions_id = [index for index, val in enumerate(volumes) if val < threshold]
        for i in range(len(masks)):
            if i in small_regions_id and not any(
                    j != i and volumes[j] > volumes[i] and iom_fn(masks[i](), masks[j]()) > IOU_THRES for j in
                    range(len(masks))):
                continue
            else:
                non_degenerated_masks.append(masks[i])
        return non_degenerated_masks

    def dilate(self, mask, kernel_size=3):
        return scipy.ndimage.binary_dilation(mask, structure=np.ones((kernel_size, kernel_size)))

    def sample_points_in_mask(self, mask, size=None, avoid_boundary=True, edge_kernel_size=3):
        """
        Return points from mask (Row, Col)
        Args:
            mask:
            size:
            edge_kernel_size:
        Returns:
        """
        if avoid_boundary:
            mask = scipy.ndimage.binary_erosion(mask, structure=np.ones((edge_kernel_size, edge_kernel_size)))
        pts = np.transpose(ma.nonzero(mask))
        if (len(pts) == 0): return np.zeros((0, 2))
        pts[:, [0, 1]] = pts[:, [1, 0]]
        idx = np.random.randint(len(pts), size=size)
        return (pts[idx, :])

    def masks_from_point_query(self, input_point, input_label=None, return_masknum=1, vis=False) -> List[MaskWrapper]:
        """
        Query SAM with point prompts.
        """
        if input_label is None:
            input_label = np.ones(len(input_point))
        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        if vis:
            plt.figure(figsize=(18, 6))
            for i, mask in enumerate(masks):
                plt.subplot(1, len(masks), i + 1)
                overlayed = overlay_mask_simple(self.rgb_im, mask)
                for sampled_point in input_point:
                    cv2.circle(overlayed, tuple(sampled_point), radius=5, color=(0, 0, 1.))
                plt.imshow(overlayed)
                plt.title(scores[i])
            plt.show()

        mask_list = []
        for (mask, score) in zip(masks, scores):
            mask_list.append(MaskWrapper({'segmentation': mask, 'predicted_iou': score}))
        mask_list.sort(key=lambda x: x.score, reverse=True)
        return mask_list[:return_masknum]

    def visualize_confident_uncertain(self, confident_masks: List[np.ndarray],
                                      region_hypotheses: List[RegionHypothesis],
                                      show=True, save_path=None, plot_anno=None):
        all_visualize = []
        confident_parts_union = reduce(np.logical_or, confident_masks) if len(confident_masks) > 0 \
            else np.zeros_like(self.rgb_im)[..., 0].astype(bool)
        overlayed_certain_union = overlay_mask_simple(self.rgb_im, confident_parts_union)
        overlayed_certain_separate = overlay_masks(self.rgb_im, confident_masks,  # colors=(0,255,0),
                                                   contour_width=4, paint_area=True)
        all_visualize.append(
            [['Confident regions', overlayed_certain_union], [], [['confident masks', overlayed_certain_separate, 0]]])
        if self.pcd is not None:
            vis_dep = self.pcd[..., 2]
            vis_dep[vis_dep > 3] = 0
            all_visualize[-1][-1].append(['dep', vis_dep, 0])

        for r_i, region_hypothesis in enumerate(region_hypotheses):
            union_mask = region_hypothesis.area_mask
            overlayed_union = overlay_mask_simple(crop(self.rgb_im, union_mask), crop(union_mask, union_mask))
            # cv2.putText(overlayed_union, f'region {r_i}', (0, 20), fontScale=.3, color=(1, 1, 1), thickness=1,
            #                 lineType=cv2.LINE_AA,
            #                 fontFace=cv2.FONT_HERSHEY_COMPLEX)
            region_hyp = [[f'Uncertain region {r_i}', overlayed_union]]
            distinct_masks = []
            for distinct_mask_i, mask_distinct in enumerate(region_hypothesis.masks_union):
                overlayed_distinct_i = overlay_mask_simple(crop(self.rgb_im, union_mask),
                                                           crop(mask_distinct, union_mask))
                # cv2.putText(overlayed_distinct_i, f'{distinct_mask_i}', (0, 20), fontScale=.3, color=(1, 1, 1), thickness=1,
                #             lineType=cv2.LINE_AA,
                #             fontFace=cv2.FONT_HERSHEY_COMPLEX)
                overlayed_distinct_i = [f'Piece {distinct_mask_i}', overlayed_distinct_i]
                distinct_masks.append(overlayed_distinct_i)
            region_hyp.append(distinct_masks)
            overlayed_hyp = []
            for (hyp_i_mask_ids, hyp_i_score) in zip(region_hypothesis.region_hypotheses_2d_corresponding_masks,
                                                     region_hypothesis.region_hypotheses_2d_corresponding_scores):
                masks = [region_hypothesis.masks_union[i] for i in hyp_i_mask_ids]
                overlayed_separate = overlay_masks(crop(self.rgb_im, union_mask), [crop(x, union_mask) for x in masks],
                                                   plot_anno=None)
                overlayed_hyp.append(
                    [
                        f'Combo piece {[int(k) for k in hyp_i_mask_ids]}\n'
                        f'Conf {hyp_i_score / sum(region_hypothesis.region_hypotheses_2d_corresponding_scores):.2f}',
                        overlayed_separate,
                        hyp_i_score])
            overlayed_hyp.sort(key=lambda x: x[2], reverse=True)
            region_hyp.append(overlayed_hyp)
            all_visualize.append(region_hyp)

        total_row = len(all_visualize)
        total_maskunion = max([len(row_x[1]) for row_x in all_visualize])
        total_overlayed_hyp = max([len(row_x[2]) for row_x in all_visualize])
        total_col = 1 + total_maskunion + total_overlayed_hyp

        row_col_to_n = lambda row_0index, col_0index: row_0index * total_col + col_0index + 1

        plt.figure(figsize=(6.4 * 3, 4.8 * 3))
        for row_i, visualize_rowi in enumerate(all_visualize):
            plt.subplot(total_row, total_col, row_col_to_n(row_i, 0))
            plt.imshow(visualize_rowi[0][1])
            plt.title(visualize_rowi[0][0])
            plt.axis('off')

            visualize_rowi_maskunion = visualize_rowi[1]
            for col_j, fig_ij in enumerate(visualize_rowi_maskunion):
                plt.subplot(total_row, total_col, row_col_to_n(row_i, col_j + 1))
                plt.imshow(fig_ij[1])
                plt.title(fig_ij[0])
                plt.axis('off')

            visualize_rowi_overlayedhyp = visualize_rowi[2]
            for col_j, fig_ij in enumerate(visualize_rowi_overlayedhyp):
                plt.subplot(total_row, total_col, row_col_to_n(row_i, col_j + 1 + total_maskunion))
                plt.imshow(fig_ij[1])
                plt.title(fig_ij[0])
                plt.axis('off')
        if plot_anno is not None:
            plt.suptitle(plot_anno)
        plt.tight_layout(pad=0.01)

        if show:
            plt.show()
        else:
            if save_path is None:
                save_path = 'uncos_result.png'
            plt.savefig(save_path)
