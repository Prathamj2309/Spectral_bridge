Overview

This repository contains a PyTorch-based machine learning pipeline designed to reconstruct and predict missing values in time-series signal data. The model takes partially observed signals (context points) and uses a Transformer architecture with Fourier time embeddings to impute the continuous signal across all time steps.



Dependencies

Ensure you have the following libraries installed:



torch (PyTorch)



pandas



numpy



scikit-learn



matplotlib



scipy



Dataset Structure

The pipeline expects time-series data in a CSV format with the following columns:



Sample\_ID: Unique identifier for each time-series sample.



Time\_ms: The time step (in milliseconds) for the signal.



Value: The actual signal value.



Is\_Context: A binary mask (0 or 1) indicating whether the point is a known "context" point (1) or a missing point that needs to be predicted (0).



Model Architecture: SpectralTransformer

The core model is a custom Transformer encoder (SpectralTransformer) designed specifically for time-series reconstruction:



Time Embedding: Uses Fourier feature embeddings (sine/cosine waves) to encode the continuous Time\_ms values into a 128-dimensional space.



Value Projection: A linear layer that projects the signal values into the same 128-dimensional space.



Mask Embedding: An embedding layer that encodes whether a point is a known context point or a target point.



Transformer Encoder: The embeddings are summed and passed through a 4-layer Transformer Encoder (4 attention heads, 256 feed-forward dimension).



Output Head: A final linear layer outputs the predicted continuous signal value for every time step.



Training Methodology

Context Splitting

During training, the split\_context function takes the known context points (Is\_Context == 1) and randomly hides a subset of them (the validation set). The model is then forced to reconstruct these hidden points using only the remaining visible context points.



Loss Functions

The model optimizes a combined loss function with three specific goals:



Reconstruction Loss (Weight = 1.0): Mean Squared Error (MSE) between the model's predictions and the actual values of the artificially hidden validation points.



Smoothness Loss (Weight = 0.15): A second-derivative penalty that discourages jagged, noisy predictions and forces the reconstructed signal to be smooth.



Anchor Loss (Weight = 0.6): MSE on the visible context points to ensure the model's output tightly fits the known anchor points.



Pipeline Workflow

Data Loading: Reads the training data, groups it by Sample\_ID, and sorts it chronologically by Time\_ms.



Dataset \& DataLoader: Wraps the data in a custom PyTorch Dataset (SpectralDataset) and batches it using DataLoader.



Training Loop: Trains the SpectralTransformer for 25 epochs using the Adam optimizer (learning rate = 1e-3).



Visualization: \* Includes functions to plot the raw predictions against the context points.



Includes an advanced plotting function (plot\_sample\_spline) that uses SciPy's CubicSpline to visualize a perfectly smooth curve interpolated through the model's predictions.



Model Saving: Saves the trained weights to best\_model\_spectral\_bridge.pth.



Inference \& Submission: \* Loads the test dataset (test\_features\_spectral.csv).



Runs the saved model in evaluation mode (model.eval()) to generate predictions.



Formats the outputs into a standard Pandas DataFrame and exports them to submission.csv.

# Spectral_bridge
