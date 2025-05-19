import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from time import time

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2');

#Our sentences we like to encode
sentences = ['The plant world is essential for human life, and shapes human culture. Plants are food, fuel and raw materials. They transform and sustain the soil, air and water of our ecosystems. They produce molecules that are the active ingredients in herbal medicine, modern pharmacology and psychoactive drugs. Humans alter plants using breeding and biotechnology, and use them to enhance their environments and their cultural activities. Using introductory concepts from the life sciences, this course explores these vital relationships between humans and plants.',
    'This course complements SC/CHEM 1000 3.00 - with emphasis on chemical change and equilibrium. Topics include chemical kinetics; chemical equilibrium; entropy and free energy as driving forces for chemical change; electrochemistry; frontiers in chemistry.',
    'An intensive introduction for aspiring screenwriters to the subtle but encompassing problems they may expect to encounter when writing for series television. Students will study the form and format of half-hour and one hour episodic comedies and dramas intended to be encompassed as part of a television series. They will also undertake the pitching, outlining and drafting of a single episode; the creation and development of a series proposal; the make up and function of a story department; plus an overview of the industry as a whole. Long form drama including television movies and mini-series will also be examined.',
    'Topics include spherical and cylindrical coordinates in Euclidean 3-space, general matrix algebra, determinants, vector space concepts for Euclidean n-space (e.g. linear dependence and independence, basis, dimension, linear transformations etc.), an introduction to eigenvalues and eigenvectors.',
    'Covers the fundamentals of marketing theory, concepts and management as applied to marketing\'s strategic role in meeting customer needs, including product (goods and services), price, promotion, distribution, consumer, segmentation, positioning, ethics, research. Includes the creation of an actual marketing plan.',
    'A study of cell biology and aspects of related biochemistry. Topics include membranes, the endomembrane system, the cytoskeleton, cellular motility, the extracellular matrix, intercellular communication and intracellular regulation.',
    'An examination of the components and principles of fitness and health with particular attention to the evaluation and modification of fitness and health status.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

courses = ['NATS 1565', 'CHEM 1001', 'FILM 5122', 'MATH 1025', 'ADMS 2200', 'BIOL 2030', 'KINE 1020']

similarity = util.pytorch_cos_sim(embeddings[0, :], embeddings[1, :])
print('sentance 0 and 1:', similarity)

similarity = util.pytorch_cos_sim(embeddings[0, :], embeddings[2, :])
print('sentance 0 and 2:', similarity)

similarity = util.pytorch_cos_sim(embeddings[1, :], embeddings[2, :])
print('sentance 1 and 2:', similarity)

original_topic = 'computer science'
embedding_origin = model.encode(original_topic)

score = []

for sentence, embedding, course in zip(sentences, embeddings, courses):
    similarity = util.pytorch_cos_sim(embedding_origin, embedding)
    
    print('Course: ', course)
    print("Sentence:", sentence[:120], '...')
    print("similarity score:", similarity)
    print("")
    score.append(similarity)
