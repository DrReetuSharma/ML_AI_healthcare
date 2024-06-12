# Training images
train_dict = train_gen.class_indices

# Extracting class names ----> train_dict dictionary
classes = list(train_dict.keys())

# next batch ---> of images and labels from train_gen
images, labels = next(train_gen)

# Visualization----> parameters
num_images_to_display = 6
images_per_row = 3
num_rows = num_images_to_display // images_per_row

# P---> images
plt.figure(figsize=(20, 20))

for i in range(num_images_to_display):
    plt.subplot(num_rows, images_per_row, i + 1)
    image = images[i] / 255
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name, color='green', fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.show()
