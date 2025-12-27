import os
import random
import shutil
import kagglehub

dataset_path = kagglehub.dataset_download(
    "xhlulu/140k-real-and-fake-faces",
    force_download=False
)

def find_class_dirs(root):
    real_dirs, fake_dirs = [], []
    for r, d, _ in os.walk(root):
        for name in d:
            lname = name.lower()
            if lname == "real":
                real_dirs.append(os.path.join(r, name))
            if lname == "fake":
                fake_dirs.append(os.path.join(r, name))
    if not real_dirs or not fake_dirs:
        raise RuntimeError("Could not locate real/fake directories")

    real_dirs.sort(key=lambda x: len(os.listdir(x)), reverse=True)
    fake_dirs.sort(key=lambda x: len(os.listdir(x)), reverse=True)
    return real_dirs[0], fake_dirs[0]


real_dir, fake_dir = find_class_dirs(dataset_path)

TOTAL_IMAGES = 10_000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGES_PER_CLASS = TOTAL_IMAGES // 2
SEED = 42
random.seed(SEED)

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(dir_path):
    return [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f.lower())[1] in VALID_EXTS
    ]

def split(images):
    random.shuffle(images)
    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    return images[:train_end], images[train_end:val_end], images[val_end:]

def copy_files(files, src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

real_images = list_images(real_dir)
fake_images = list_images(fake_dir)

if len(real_images) < IMAGES_PER_CLASS or len(fake_images) < IMAGES_PER_CLASS:
    raise RuntimeError("Not enough images")

real_images = random.sample(real_images, IMAGES_PER_CLASS)
fake_images = random.sample(fake_images, IMAGES_PER_CLASS)

real_train, real_val, real_test = split(real_images)
fake_train, fake_val, fake_test = split(fake_images)

base_out = "./data"

copy_files(real_train, real_dir, f"{base_out}/train/real")
copy_files(real_val,   real_dir, f"{base_out}/val/real")
copy_files(real_test,  real_dir, f"{base_out}/test/real")

copy_files(fake_train, fake_dir, f"{base_out}/train/fake")
copy_files(fake_val,   fake_dir, f"{base_out}/val/fake")
copy_files(fake_test,  fake_dir, f"{base_out}/test/fake")
