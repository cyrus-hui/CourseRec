from tqdm import tqdm
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from time import time
import matplotlib.pyplot as plt
import heapq


model = SentenceTransformer('all-MiniLM-L6-v2');

course_list = np.load('course.npy')
embeddings = np.load('embeddings.npy')

original_topic = '(user input)'
embedding_origin = model.encode(original_topic)

score = {}

for embedding, course in zip(embeddings, course_list):
    similarity = util.pytorch_cos_sim(embedding_origin, embedding)
    score[course] = similarity
    
top_courses = heapq.nlargest(10, score, key=score.get)


for course in top_courses:
    print("Course: ", course)
    print("Score: ", score.get(course))
    print("")
