# RTL Timing Violation Predictor

AI algorithm to predict combinational complexity/depth of signals to quickly identify timing violations in RTL designs using synthetic datasets.

## Problem Overview

Timing analysis is crucial in complex IP/SoC design, but timing reports are only generated after synthesis, which is time-consuming. This project creates an AI algorithm to predict combinational logic depth of signals in behavioral RTL to identify potential timing violations early in the design process.

## Key Concepts

- **Combinational Complexity/Logic-depth**: Number of basic gates (AND/OR/NOT/NAND etc.) required to generate a signal following the longest path
- **Timing Violation**: Indicates when combinational logic depth exceeds what's supported at a given frequency

## Features

- Synthetic RTL dataset generation
- Feature extraction from RTL
- Timing violation prediction
- Model training and evaluation

## Project Structure