import os
import sys
import shutil
import random
import tensorflow as tf
from keras import layers, models

# Directories
original_dir_target = "rendered_images"
original_dir_other = "rendered_images_other"

data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Classes
classes = ["object_class", "other_objects"]

# Create directories if not exist
for subset in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(data_dir, subset, cls), exist_ok=True)

# Gather all png images from rendered_images
all_target_images = [f for f in os.listdir(original_dir_target) if f.lower().endswith(".png")]
all_other_images = [f for f in os.listdir(original_dir_other) if f.lower().endswith(".png")]

# Simple heuristic for class assignment:
# If filename starts with "img_", it's object_class, else other_objects
object_images = [img for img in all_target_images if img.startswith("img_")]
other_images = [img for img in all_other_images if not img.startswith("img_")]

# Split into train/val/test sets
def split_data(images, train_ratio=0.7, val_ratio=0.2):
    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    return train_imgs, val_imgs, test_imgs

train_obj, val_obj, test_obj = split_data(object_images)
train_oth, val_oth, test_oth = split_data(other_images)

# Function to copy images to target directory
def copy_target_images(image_list, target_dir, cls):
    for img in image_list:
        src = os.path.join(original_dir_target, img)
        dst = os.path.join(target_dir, cls, img)
        shutil.copyfile(src, dst)

# Function to copy images to other directory
def copy_other_images(image_list, target_dir, cls):
    for img in image_list:
        src = os.path.join(original_dir_other, img)
        dst = os.path.join(target_dir, cls, img)
        shutil.copyfile(src, dst)

# Copy images to respective directories
copy_target_images(train_obj, train_dir, "object_class")
copy_target_images(val_obj, val_dir, "object_class")
copy_target_images(test_obj, test_dir, "object_class")

copy_other_images(train_oth, train_dir, "other_objects")
copy_other_images(val_oth, val_dir, "other_objects")
copy_other_images(test_oth, test_dir, "other_objects")

# Now we have data in:
# data/train/object_class, data/train/other_objects, ...
# data/val/object_class, data/val/other_objects, ...
# data/test/object_class, data/test/other_objects, ...

##############################################
# Training code (similar to previous, grayscale and scale invariance)
##############################################

batch_size = 32
img_height = 224
img_width = 224

# Create training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',      # use grayscale
    shuffle=True
)

# Create validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "val"),
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True
)

class_names = train_ds.class_names
print("Class Names:", class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# If test directory exists with data
test_path = os.path.join(data_dir, "test")
has_test = len(os.listdir(test_path)) > 0
if has_test:
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=False
    )
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
else:
    test_ds = None

# Data augmentation for scale invariance
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3))
])

num_classes = len(class_names)

# Simple CNN for grayscale images
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Evaluate on the test set if available
if test_ds is not None:
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test accuracy:", test_acc)

# Save the model
model.save("object_classifier_scale_invariant_grayscale.keras")

# Example prediction on a single grayscale image from test set
test_image_path = os.path.join(data_dir, "test", "object_class", random.choice(os.listdir(os.path.join(data_dir, "test", "object_class"))))
img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(img_height, img_width), color_mode='grayscale')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.expand_dims(x, axis=0)  # model expects a batch
predictions = model.predict(x)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[tf.argmax(score)]
confidence = 100 * tf.reduce_max(score)
print("Predicted class:", predicted_class, "with confidence:", confidence.numpy(), "%")
