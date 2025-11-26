## Requirements for dance generation
* We recommend Linux for performance and compatibility reasons. Windows will probably work, but is not officially supported.
* 64-bit Python 3.7+
* PyTorch 1.12.1
* At least 16 GB RAM per GPU
* 1&ndash;8 high-end NVIDIA GPUs with at least 16 GB of GPU memory, NVIDIA drivers, CUDA 11.6 toolkit.

The example build this repo was validated on:
* Debian 10
* 64-bit Python 3.7.12
* PyTorch 1.12.1
* 16 GB RAM
* 1 x NVIDIA T4, CUDA 11.6 toolkit

## Requirements for Reinforcement learning
* Python 3.11
* Pythorch 2.8.0

This repository additionally depends on the following libraries, which may require special installation procedures:
* [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)
* [pytorch3d](https://github.com/facebookresearch/pytorch3d)
* [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index)
	* Note: after installation, don't forget to run `accelerate config` . We use fp16.
* [wine](https://www.winehq.org) (Optional, for import to Blender only)
## Getting started

### Load custom music
You can test the model on custom music by downloading them as `.wav` files into a directory, e.g. `custom_music/` and running
```.bash
python test.py --music_dir custom_music/ --save_motions
```
This process may take a while, since the script will extract all the Jukebox representations for the specified music in memory. The representations can also be saved and reused to improve speed with the `--cache_features` and `--use_cached_features` arguments. See `args.py` for more detail.
Note: make sure file names are regularized, e.g. `Britney Spears - Toxic (Official HD Video).wav` may cause unpredictable behavior due to the spaces and parentheses, but `toxic.wav` will behave as expected. See how the demo notebook achieves this using the `youtube-dl --output` flag.
this will save the generated dance sequence motion in mentioned directory in pkl file format.

## Refining The generated Dance motions
download the PPO model file, check the model file name before running
then keep the input pkl file in the input folder and then run the inference file
```.bash
python infer5.py --model model_file.zip --input input.pkl --output final_out.pkl --audio_dir inference_folder
```

## Blender 3D rendering
In order to render generated dances in 3D, we convert them into FBX files to be used in Blender. We provide a sample rig, `SMPL-to-FBX/ybot.fbx`.
After generating dances with the `--save-motions` flag enabled, move the relevant saved `.pkl` files to a folder, e.g. `smpl_samples`
Run
```.bash
python SMPL-to-FBX/Convert.py --input_dir SMPL-to-FBX/smpl_samples/ --output_dir SMPL-to-FBX/fbx_out
```
to convert motions into FBX files, which can be imported into Blender and retargeted onto different rigs, 
## Development
This is a research implementation and, in general, will not be regularly updated or maintained long after release.
## Citation
```
@article{tseng2022edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C Karen},
  journal={arXiv preprint arXiv:2211.10658},
  year={2022}
}
```
## Acknowledgements
We would like to thank [lucidrains](https://github.com/lucidrains) for the [Adan](https://github.com/lucidrains/Adan-pytorch) and [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) repos, [softcat477](https://github.com/softcat477) for their [SMPL to FBX](https://github.com/softcat477/SMPL-to-FBX) library, and [BobbyAnguelov](https://github.com/BobbyAnguelov) for their [FBX Converter tool](https://github.com/BobbyAnguelov/FbxFormatConverter).
