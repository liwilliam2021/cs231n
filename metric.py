import torch
import os
from PIL import Image
import numpy as np

def load_scene(scene_path):
    images = []
    for file_path in os.listdir(scene_path):
        if file_path == '.DS_Store':
            continue
        try:
            image_path = os.path.join(scene_path, file_path)
            images.append(Image.open(image_path))
        except:
            print(f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail.")
    return images

def calculate_scene_score(scene, model, extractor):
    processed_imgs = extractor.preprocess(scene)
    score = torch.tensor(np.array(processed_imgs['pixel_values']))
    score = model.forward(score).pooler_output
    l2 = score.diff(dim=0).norm().data
    cosine = torch.nn.CosineSimilarity(dim=0)
    sim = cosine(score[0], score[1]).data
    return l2, sim, processed_imgs

def deep_sim_metric(scenes_dir, model, processor):
    scores = {'l2': [], 'sim': []}

    for scene_path in os.listdir(scenes_dir):
        if scene_path == '.DS_Store':
            continue
        path = os.path.join(scenes_dir, scene_path)
        scene = load_scene(path)
        l2, sim, imgs= calculate_scene_score(scene, model, processor)
        scores['l2'].append(l2)
        scores['sim'].append(sim)
        if scene_path == 'good' or scene_path == 'bad':
            print(sim, l2)
    return scores, imgs

if __name__ == "__main__":
    deep_sim_metric('good_scenes')