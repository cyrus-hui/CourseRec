from tqdm import tqdm
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

#folder of json files with course description
folder_path = "/"
value_course = "course"

desc_list = []
course_list = []


for filename in os.listdir(folder_path):
  if filename.endswith(".json"):
    file_path = os.path.join(folder_path, filename)
    with open(file_path) as json_file:
      try:
        data = json.load(json_file)
        dept_course = data.get("courses")
        for course in dept_course:
          key = course.get("key")
          name = course.get("name")
          desc = course.get("desc")
          if len(desc) < 15:
            continue
          faculty = key.get("faculty").upper()
          dept = key.get("dept").upper()
          code = key.get("code")
          level = int(code[0])
          if level >= 5:
            continue
          credit = str(key.get("credit"))
          course_name = f"{faculty}/{dept} {code} {name} {credit}"
          course_list.append(course_name)
          desc_list.append(desc)
      except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {filename}: {e}")
      except KeyError:
        print(f"not found")

# print(len(course_list))
# print(len(desc_list))

#SentenceTransformer model might differs
model = SentenceTransformer('all-MiniLM-L6-v2');

np.save("course.npy", course_list)
#Course desc are encoded by calling model.encode()
embeddings = model.encode(desc_list)
np.save("embeddings.npy", embeddings)
