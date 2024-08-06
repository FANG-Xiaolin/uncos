# Embodied Uncertainty-Aware Object Segmentation

**[Embodied Uncertainty-Aware Object Segmentation](https://sites.google.com/view/embodied-uncertain-seg)**
<br />
[Xiaolin Fang](https://fang-xiaolin.github.io/), 
[Leslie Pack Kaelbling](https://people.csail.mit.edu/lpk/), and
[Tomás Lozano-Pérez](https://people.csail.mit.edu/tlp/)
<br />
International Conference on Intelligent Robots and Systems (IROS) 2024
<br />
[[Paper]]()
[[Website]](https://sites.google.com/view/embodied-uncertain-seg)
```
@inproceedings{Fang2024Uncos,
  title={{Embodied Uncertainty-Aware Object Segmentation}},
  author={Xiaolin Fang and Leslie Pack Kaelbing and Tomas Lozano-Perez},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024}
}
```

Uncos is an active prompting strategy for combining promptable top-down 
and bottom-up pre-trained object instance segmentation methods to obtain 
a distribution over image-segmentation hypotheses.

Embodied object segmentation code to be added soon. Stayed tuned!

## Installation

The code is tested with python 3.10, CUDA 12.2, pytorch 2.4.0.

Install UncOS directly:

```
pip install git+https://github.com/FANG-Xiaolin/uncos.git
```

or clone the repository locally and install with

```
git clone git@github.com:FANG-Xiaolin/uncos.git
cd uncos; pip install -e .
```

## Usage

To retrieve the most likely hypothesis from uncos
``` 
# rgb_image:        rgb image in 0-255. H x W x 3
# pcd:              point cloud in camera frame or world frame. H x W x 3
# pred_masks_boolarray:  A list of predicted binary masks.

from uncos import UncOS
uncos = UncOS()
pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(
    rgb_im, pcd,
    return_most_likely_only=True, n_seg_hypotheses_trial=5
)
```

Set `return_most_likely_only` to `False` for multiple hypotheses.

To visualize the result
``` 
# visualize the result
uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)
# # or save to file
# uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses, show=False, save_path='demo_result.png')
```
