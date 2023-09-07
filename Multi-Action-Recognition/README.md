Hi, In sign language many complex actions combine into a word or a sentence.
Therefore, video classification will be used for this.
#### Enviroment
```
pip install -r requirements.txt
```
#### Dataset
```
dataset
|── label_1
|   |── video_1.mp4
|   └── video_2.mp4
|   └── ...
|── label_2
|── ...
```
#### Training
```
python train.py --root your_root_dataset
```
#### Inference
```
python inference.py --root your_root_dataset --model_ckpt ckpt.pt --video_path your_video.mp4
```
