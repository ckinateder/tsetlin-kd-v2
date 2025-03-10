import numpy as np
from scipy.special import softmax, kl_div
import warnings

_OFFSET = 1e-12

# Normalize voting scores to probabilities
def normalize(scores):
    # normalize between 0 and 1; input is a list of votes that may be negative
    votes = np.array(scores)
    votes = votes - np.min(votes)
    if np.sum(votes) == 0:
        return np.ones_like(votes) / len(votes)
    return votes / np.sum(votes)

def softmax(scores):
    # softmax function
    return np.exp(scores) / np.sum(np.exp(scores))

def kl_divergence(teacher_probs, student_probs):
    """
    Calculate KL divergence between teacher and student probability distributions
    
    Parameters:
    -----------
    teacher_probs : numpy.ndarray
        Array of shape (n_samples, n_classes) containing teacher probability distributions
    student_probs : numpy.ndarray
        Array of shape (n_samples, n_classes) containing student probability distributions
    
    Returns:
    --------
    float
        Average KL divergence across all samples
    """
    # Add small offset to avoid log(0)
    teacher_probs = teacher_probs + _OFFSET
    student_probs = student_probs + _OFFSET
    
    # Normalize to ensure probabilities sum to 1
    teacher_probs = teacher_probs / np.sum(teacher_probs, axis=1, keepdims=True)
    student_probs = student_probs / np.sum(student_probs, axis=1, keepdims=True)
    
    n_samples = teacher_probs.shape[0]
    mi = 0.0
    
    # Process samples one at a time to be memory efficient
    for i in range(n_samples):
        p_x = teacher_probs[i]
        p_y = student_probs[i]
        
        # Calculate MI using the formula: MI = sum(p(x) * log(p(x)/p(y)))
        # This is equivalent to KL divergence when comparing distributions
        mi += np.sum(p_x * np.log(p_x / p_y))
    
    return mi / n_samples  # Return average MI across samples

def mutual_information(teacher_probs, student_probs):
    """
    Calculate mutual information between teacher and student probability distributions
    
    Parameters:
    -----------
    teacher_probs : numpy.ndarray
        Array of shape (n_samples, n_classes) containing teacher probability distributions
    student_probs : numpy.ndarray
        Array of shape (n_samples, n_classes) containing student probability distributions
    
    Returns:
    --------
    float
        Average mutual information across all samples
    """
    # Add small offset to avoid log(0)
    teacher_probs = teacher_probs + _OFFSET
    student_probs = student_probs + _OFFSET
    
    # Normalize to ensure probabilities sum to 1
    teacher_probs = teacher_probs / np.sum(teacher_probs, axis=1, keepdims=True)
    student_probs = student_probs / np.sum(student_probs, axis=1, keepdims=True)
    
    n_samples = teacher_probs.shape[0]
    mi = 0.0
    
    # Process samples one at a time to be memory efficient
    for i in range(n_samples):
        # compute marginal distributions
        p_x = teacher_probs[i]
        p_y = student_probs[i]
        
        # compute joint distribution
        joint_distribution = np.outer(p_x, p_y)

        # sum over x any y: P_{XY}(x,y) * log(P_{XY}(x,y) / (P_X(x) * P_Y(y)))
        mi_i = np.sum(joint_distribution * np.log(joint_distribution / (p_x * p_y)))
        mi += mi_i

    return mi / n_samples  # Return average MI across samples

from datasets import Dataset
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

def info_theory_experiment(output: dict, dataset: Dataset, distilled_model: MultiClassTsetlinMachine, teacher_model: MultiClassTsetlinMachine, student_model: MultiClassTsetlinMachine):
    print(output["experiment_name"])

    # get training data
    X_train = dataset.X_train
    Y_train = dataset.Y_train

    # get test data
    X_test = dataset.X_test

    # get soft labels
    soft_labels_d = distilled_model.get_soft_labels(X_test)
    soft_labels_t = teacher_model.get_soft_labels(X_test)
    soft_labels_s = student_model.get_soft_labels(X_test)

    print(soft_labels_d)
    print(soft_labels_t)
    print(soft_labels_s)

    # get mutual information
    mi_d_t = mutual_information(soft_labels_d, soft_labels_t)
    mi_d_s = mutual_information(soft_labels_d, soft_labels_s)
    mi_t_s = mutual_information(soft_labels_t, soft_labels_s)

    print(f"MI(D,T): {mi_d_t}")
    print(f"MI(D,S): {mi_d_s}")
    print(f"MI(T,S): {mi_t_s}")
    