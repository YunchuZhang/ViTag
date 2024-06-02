import glob, random
folder_path = '/home/yunchuz/deeptag/dataset_org/'

# Get all the image files in the folder (jpg and png)
files_list = glob.glob(f"{folder_path}/**/*", recursive=True)
files_list = [f for f in files_list if f.endswith(('jpg', 'png'))]
# Shuffle the list to ensure random splitting
random.shuffle(files_list)

# Determine the split index for 80/20 split
split_idx = int(len(files_list) * 0.8)

# Split the list into train and test sets
train_files = files_list[:split_idx]
test_files = files_list[split_idx:]

# Save the file paths to train.txt and test.txt
with open('dataset/train.txt', 'w') as train_file:
    for file in train_files:
        train_file.write(f"{file}\n")

with open('dataset/test.txt', 'w') as test_file:
    for file in test_files:
        test_file.write(f"{file}\n")

# Print the number of files in each set
print(f"Total files: {len(files_list)}")
print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")

folder_path = '/home/yunchuz/deeptag/dataset_org_noise/'

# Get all the image files in the folder (jpg and png)
files_list_new = glob.glob(f"{folder_path}/**/*", recursive=True)
files_list_new = [f for f in files_list_new if f.endswith(('jpg', 'png'))]
files_list += files_list_new
# Shuffle the list to ensure random splitting
random.shuffle(files_list)

# Determine the split index for 80/20 split
split_idx = int(len(files_list) * 0.8)

# Split the list into train and test sets
train_files = files_list[:split_idx]
test_files = files_list[split_idx:]

# Save the file paths to train.txt and test.txt
with open('dataset/train_with_noise.txt', 'w') as train_file:
    for file in train_files:
        train_file.write(f"{file}\n")

with open('dataset/test_with_noise.txt', 'w') as test_file:
    for file in test_files:
        test_file.write(f"{file}\n")

# Print the number of files in each set
print(f"Total files: {len(files_list)}")
print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")