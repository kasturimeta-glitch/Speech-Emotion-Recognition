import os
dataset_path = "dataset"
count = 0
for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)
    if os.path.isdir(actor_path):
        files = os.listdir(actor_path)  
        for file in files:
            if file.endswith(".wav"):
                count += 1
print("Total audio files:", count)
