import glob
import os
from uncos import UncOS
from uncos.uncos_utils import load_data_npy

def main():
    test_most_likely = True

    demo_files_list = glob.glob(os.path.join(os.path.dirname(__file__), f"../demo_files/**.npy"))
    uncos = UncOS()
    for demo_file_path in demo_files_list:
        print(f"Testing {demo_file_path}")
        rgb_im, pcd = load_data_npy(demo_file_path)
        pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(rgb_im, pcd,
                                                                         return_most_likely_only=test_most_likely,
                                                                         n_seg_hypotheses_trial=5)
        uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)
        # uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses, show=False,
        #                                     save_path=f'{demo_file_path[:-4]}_result.png')


if __name__ == "__main__":
    main()
