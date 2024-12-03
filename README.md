# Embodied Uncertainty-Aware Object Segmentation

**[Embodied Uncertainty-Aware Object Segmentation](https://sites.google.com/view/embodied-uncertain-seg)**
<br />
[Xiaolin Fang](https://fang-xiaolin.github.io/),
[Leslie Pack Kaelbling](https://people.csail.mit.edu/lpk/), and
[Tomás Lozano-Pérez](https://people.csail.mit.edu/tlp/)
<br />
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024
<br />
[[Paper]](https://arxiv.org/abs/2408.04760)
[[Website]](https://sites.google.com/view/embodied-uncertain-seg)

```
@inproceedings{Fang2024Uncos,
  title={{Embodied Uncertainty-Aware Object Segmentation}},
  author={Xiaolin Fang and Leslie Pack Kaelbing and Tomás Lozano-Pérez},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024}
}
```

![UncOS EOS diagram](assets/eos_uncos_diagram.png?raw=true)

UncOS is an active prompting strategy for combining promptable top-down
and bottom-up pre-trained large models to obtain
a distribution over image-segmentation hypotheses. We can either query
the most likely hypothesis from UncOS or use all the
hypotheses as a basis for downstream applications, such as embodied
object segmentation / active information gathering.

![UncOS diagram](assets/uncos_diagram.png?raw=true)

**[News]** 
Support for the [SAM2](https://github.com/facebookresearch/sam2) backbone has been added! 
Please specify the backbone option and provide the path to the checkpoint in `config.py` accordingly.

## Installation

The code is tested with Python 3.10, CUDA 12.2, and PyTorch 2.4.0. CPU-only mode is supported.
<details>
<summary>(Optional) Create conda environment</summary>

```
conda create --name uncos_env python=3.10
conda activate uncos_env
```

</details>

Install UncOS directly

```
pip install git+https://github.com/FANG-Xiaolin/uncos.git
```

or clone the repository locally and install with

```
git clone git@github.com:FANG-Xiaolin/uncos.git
cd uncos; pip install -e .
```

## Usage

We've provided a few testing examples in `demo_files`. Add `-v`
to visualize the results, `-m` to return the most likely segmentation
hypothesis.

```
python scripts/demo.py -v
```

To retrieve the most likely hypothesis from UncOS.

``` 
# rgb_image:        rgb image in 0-255. (H x W x 3) np.uint8.
# pcd:              point cloud in camera frame or world frame. (H x W x 3) np.float32.

from uncos import UncOS
uncos = UncOS()
pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(rgb_im, pcd, return_most_likely_only=True)
```

`pred_masks_boolarray` is a list of predicted binary masks. `[(H x W)] bool`

Set `return_most_likely_only` to `False` for multiple hypotheses.

To visualize the result

``` 
# visualize the result
uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)
# # or save to file
# uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses, show=False, save_path='demo_result.png')
```

### Benchmarking

<details>
<summary>Download the OCID dataset</summary>

- Download the file from [here](https://researchdata.tuwien.at/records/pcbjd-4wa12) and unzip it.

- You may need to manually correct a folder name

    ```
    cd OCID-dataset/ARID10/floor/bottom/fruits/seq37/
    mv pd pcd
    ```

</details>

```
python scripts/benchmarking.py
```
