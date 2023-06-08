import torch
import os
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import spacy

# perhaps run python3 -m spacy download en_core_web_sm, python3 -m spacy download en_core_web_lg

def get_verb (caption, nlp):
  for token in nlp (caption):
    if token.pos_ == 'VERB':
      return token

def get_verb_class (caption):
  nlp = spacy.load("en_core_web_sm")
  cap_verb = get_verb (caption, nlp).lemma_

  with open ('util/action_classification/verb_classes.txt', 'r') as file:
    verbs = [nlp(line.rstrip()) for line in file]
  similarity_list = [nlp(cap_verb).similarity(verb) for verb in verbs]
  return verbs[np.argmax(similarity_list)]

def get_image_caption (PREFIX, PATH):
  return Image.open(PREFIX + PATH), PATH[0: PATH.find('.')]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_clip_pred (PREFIX, PATH):
  image, caption = get_image_caption (PREFIX, PATH)

  true_class = get_verb_class (caption)

  with open ('util/action_classification/verb_classes.txt', 'r') as file:
    labels = [line.rstrip() for line in file]

  clip_labels = [f"a photo of a {label}" for label in labels]

  image = processor(
      text=None,
      images=image,
      return_tensors='pt'
  )['pixel_values'].to(device)
  img_emb = model.get_image_features(image)
  img_emb = img_emb.detach().cpu().numpy()

    # create label tokens
  label_tokens = processor(
      text=clip_labels,
      padding=True,
      images=None,
      return_tensors='pt'
  ).to(device)
  label_emb = model.get_text_features(**label_tokens)
  label_emb = label_emb.detach().cpu().numpy()
  label_emb = label_emb / np.linalg.norm(label_emb, axis=0)

  scores = np.dot(img_emb, label_emb.T)
  pred = labels[np.argmax(scores)]
  print ("predicted ", pred)
  print ("true ", true_class)

PREFIX = "util/action_classification/action_pics/"
PATH = "Girl plays soccer.jpeg"


get_clip_pred (PREFIX, PATH)

