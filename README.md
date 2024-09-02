# Spectron: Target Speaker Extraction using Conditional Transformer

## Abstract
<div style="text-align: justfy">Recently, attention-based transformers have become a defacto standard in many deep learning applications including natural language processing, computer vision, signal processing, etc.. In this paper, we propose a transformer-based end-to-end model to extract a target speaker’s speech from a monaural multi-speaker mixed audio signal. Unlike existing speaker extraction methods, we introduce two additional objectives to impose speaker embedding consistency and waveform encoder invertibility and jointly train both speaker encoder and speech separator to better capture the speaker conditional embedding. Furthermore, we leverage a multiscale discriminator to refine the perceptual quality of the extracted speech. Our experiments show that the use of a dual path transformer in the separator backbone along with proposed training paradigm improves the CNN baseline by 3.12 dB points. Finally, we compare our approach with recent state-of-the-arts and show that our model outperforms existing methods by 4.1 dB points on an average without creating additional data dependency</div>
![image](https://github.com/user-attachments/assets/bb422ebd-df12-49cf-8411-797d2b0ca9f0) <br>
Project Page: [https://tatban.github.io/spec-res/](https://tatban.github.io/spec-res/) 
## Dataset
We assume the dataset is same as [VoiceFilter](https://google.github.io/speaker-id/publications/VoiceFilter/) paper. Data paths must be updated in the corresponding .csv files in ``data`` folder
## Training
- train spectron full model: ``python train_spectron_msd.py``
- train spectron without adversarial refinement: ``python train.py``
- train spectron with pretrained transformer (without adv. refinement): ``python train_with_pretrained_DPT.py``
## Inference
- change the ``OUT_DIR`` and ``weights_path`` as per the training choice as above
- run: ``python test.py``

## Results
![image](https://github.com/user-attachments/assets/b5d94c15-2198-46f6-8d28-3f7635f8d2d3)
