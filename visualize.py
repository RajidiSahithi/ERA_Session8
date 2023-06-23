import matplotlib.pyplot as plt
import numpy as np

def visualize_data(sample_images, fig, axes,dataset,classes):
    # visualize training image for each class
    
    #sample_images = [dataset.data[np.asarray(dataset.targets) == label][0] for label in range(10)]
    # show images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    i = 0
    for row in axes:
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(classes[i], fontsize=15)
            axis.imshow(sample_images[i])
            i += 1