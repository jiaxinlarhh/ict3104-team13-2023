<div align="center">
<h2><font color="red"> 🕺 Follow Your Pose (ver. ict3104-team13-2023) 💃 </font></center> <br> <center>Pose-Guided Text-to-Video Generation using Pose-Free Videos</h2>

## 💃💃💃 Abstract

<b>TL;DR: We tune a text-to-image model to create videos of characters from the [Charades Project](https://prior.allenai.org/projects/charades) based on their poses and textual descriptions.</b>

<details><summary>CLICK for full abstract</summary>

> We are leveraging the Charades Project's existing video dataset alongside the open-source MMpose framework and the original Follow Your Pose project to generate character videos. These videos are crafted by interpreting both the characters' poses and textual descriptions, resulting in a dynamic fusion of visual and textual information.

</details>

## 🍻🍻🍻 Setup Environment

Our method is trained using cuda11, accelerator and xformers on 8 A100.

```
conda create -n fupose python=3.8
conda activate fupose

pip install -r requirements.txt
```

`xformers` is recommended for A100 GPU to save memory and running time.

<details><summary>Click for xformers installation </summary>

We find its installation not stable. You may try the following wheel:

```bash
wget https://github.com/ShivamShrirao/xformers-wheels/releases/download/4c06c79/xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
pip install xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
```

</details>

Our environment is similar to Tune-A-video ([official](https://github.com/showlab/Tune-A-Video), [unofficial](https://github.com/bryandlee/Tune-A-Video)). You may check them for more details.

## 💃💃💃 Training

The original FollowYourPose has fixed the bug in Tune-a-video and finetune stable diffusion-1.4 on 8 A100.
To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
    --multi_gpu --num_processes=8 --gpu_ids '0,1,2,3,4,5,6,7' \
    train_followyourpose.py \
    --config="configs/pose_train.yaml"
```

## 🕺🕺🕺 Inference

Once the training is done, run inference:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
    --gpu_ids '0' \
    txt2video.py \
    --config="configs/pose_sample.yaml" \
    --skeleton_path="./pose_example/vis_ikun_pose2.mov"
```

You could make the pose video by [mmpose](https://github.com/open-mmlab/mmpose) , we detect the skeleton by [HRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019). You just need to run the video demo to obtain the pose video. Remember to replace the background with black.

## 💃💃💃 Local Gradio Demo

You could run the gradio demo locally, only need a `A100/3090`.

```bash
python app.py
```

then the demo is running on local URL: `http://0.0.0.0:Port`

## 🕺🕺🕺 Weight

[Stable Diffusion] [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4))

[FollowYourPose] We also provide our pretrained checkpoints in [Huggingface](https://huggingface.co/YueMafighting/FollowYourPose_v1/tree/main). you could download them and put them into `checkpoints` folder to inference our models.

```bash
FollowYourPose
├── checkpoints
│   ├── followyourpose_checkpoint-1000
│   │   ├──...
│   ├── stable-diffusion-v1-4
│   │   ├──...
│   └── pose_encoder.pth
```

## 🎼🎼🎼 Citation

```bibtex
@article{ma2023follow,
  title={Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos},
  author={Ma, Yue and He, Yingqing and Cun, Xiaodong and Wang, Xintao and Shan, Ying and Li, Xiu and Chen, Qifeng},
  journal={arXiv preprint arXiv:2304.01186},
  year={2023}
}
```

## 👯👯👯 Acknowledgements

This repository draws significant inspiration from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [FateZero](https://github.com/ChenyangQiQi/FateZero) and [FollowyourPose](https://github.com/mayuelala/FollowYourPose). We extend our gratitude to the authors for generously sharing their code and models.
