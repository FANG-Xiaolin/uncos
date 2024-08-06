from uncos import UncOS
from uncos_utils import load_data

def main():
    test_most_likely = False

    rgb_im, pcd = load_data("./demo_files/test_01.json")
    uncos = UncOS()
    pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(rgb_im, pcd,
                                     return_most_likely_only=test_most_likely, n_seg_hypotheses_trial=12)
    if test_most_likely:
        assert len(uncertain_hypotheses)==0
    uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)

if __name__ == "__main__":
    main()
