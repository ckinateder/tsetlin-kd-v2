from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from pickle import dump, load
from time import time
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os

def visualize_activation_maps(teacher_model, student_model, distilled_model, samples, true_labels, image_shape, output_filepath):
    """
    Creates a visualization of activation maps for teacher, student, and distilled models.
    
    Args:
        teacher_model: The teacher TsetlinMachine model
        student_model: The student TsetlinMachine model
        distilled_model: The distilled TsetlinMachine model
        samples: List of input samples to visualize
        true_labels: List of true labels for the samples
        image_shape: Tuple with image dimensions (height, width)
        output_filepath: Path where to save the output image
        class_idx: Specific class index to visualize. If None, uses the predicted class.
    """
    # Set font to Times New Roman for all plots
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.serif'] = ['CMU Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'  # For math text
    plt.rcParams['font.size'] = 14

    correct_color = "lightgray"#'green'
    incorrect_color = "lightgray"#'red'
    line_width = 1

    # invert the orignal image

    # Get number of samples
    num_samples = len(samples)
    
    # Create figure with num_samples x 4 layout
    fig, axes = plt.subplots(num_samples, 4, figsize=(8, 2 * num_samples))
    
    # Handle single sample case (make axes indexable)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Get prediction from teacher model
        teacher_output = teacher_model.predict(sample.reshape(1, -1))
        sample_class_idx = int(teacher_output[0])
        correct = sample_class_idx == true_labels[i]

        # Display original image
        axes[i, 0].imshow((sample.reshape(image_shape)), cmap='gray')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(line_width)
        
        # Generate and display teacher activation map
        teacher_activation = teacher_model.get_activation_map(sample, class_idx=sample_class_idx, image_shape=image_shape)
        axes[i, 1].imshow(teacher_activation)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        for spine in axes[i, 1].spines.values():
            spine.set_edgecolor(correct_color) if correct else spine.set_edgecolor(incorrect_color)
            spine.set_linewidth(line_width)
        
        # Generate and display student activation map
        student_activation = student_model.get_activation_map(sample, class_idx=sample_class_idx, image_shape=image_shape)
        axes[i, 2].imshow(student_activation)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        for spine in axes[i, 2].spines.values():
            spine.set_edgecolor(correct_color) if correct else spine.set_edgecolor(incorrect_color)
            spine.set_linewidth(line_width)
        
        # Generate and display distilled activation map
        distilled_activation = distilled_model.get_activation_map(sample, class_idx=sample_class_idx, image_shape=image_shape)
        axes[i, 3].imshow(distilled_activation)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        for spine in axes[i, 3].spines.values():
            spine.set_edgecolor(correct_color) if correct else spine.set_edgecolor(incorrect_color)
            spine.set_linewidth(line_width)

        # add titles to top of each column
        if i == 0:
            axes[i, 0].set_title("Original Sample")
            axes[i, 1].set_title("Teacher Features")
            axes[i, 2].set_title("Student Features")
            axes[i, 3].set_title("Distilled Features")
    
    #plt.suptitle(f"Activation Maps Comparison", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save figure
    plt.savefig(output_filepath, dpi=150)
    plt.close()
    
    print(f"Activation maps comparison saved to {output_filepath}")
