import av
import torch
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="Dinh/videomae-small-finetuned-kinetics-finetuned-action",
        help="The model ckpt in Hugging Face.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def inference(video, labels):
    image_processor = AutoImageProcessor.from_pretrained("args.model_ckpt")
    model = VideoMAEForVideoClassification.from_pretrained("args.model_ckpt")
    inputs = image_processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    
    print(labels[predicted_label])

if __name__ == '__main__':
    args = parse_arguments()
    container = av.open(args.video_path)
    # sample 16 frames
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    labels = os.listdir(args.root)
    video = inference(video, labels)