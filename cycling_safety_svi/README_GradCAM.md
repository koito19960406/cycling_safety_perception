# Grad-CAM Visualizations for Cycling Safety Model

This directory contains scripts to generate Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations for the cycling safety perception model. These visualizations help understand what parts of street images the model focuses on when making safety predictions.

## Files

- `generate_gradcam_visualizations.py` - Main Python script for generating Grad-CAM visualizations
- `run_gradcam.sh` - Bash script with example usage
- `README_GradCAM.md` - This documentation file

## What is Grad-CAM?

Grad-CAM is a technique that produces visual explanations for CNN-based models by highlighting the regions of input images that are most important for the model's predictions. In the context of cycling safety, this shows which parts of street scenes the model considers when determining safety levels.

## Features

- **Attention Heatmaps**: Raw visualization of model attention as heatmaps
- **Overlay Images**: Original images with attention heatmaps overlaid (50% alpha blending as requested)
- **Multiple Backbones**: Supports VGG, ResNet, AlexNet, and DenseNet architectures
- **Batch Processing**: Efficient processing of multiple images
- **Statistics**: Summary statistics about attention patterns

## Output Structure

The script creates the following directory structure in the specified output directory:

```
/home/kiito/cycling_safety_svi/data/processed/gradcam_visualizations/
├── heatmaps/           # Raw attention heatmaps (colormap: jet)
│   ├── heatmap_image1.jpg
│   ├── heatmap_image2.jpg
│   └── ...
├── overlays/           # Original images with attention overlays
│   ├── overlay_image1.jpg
│   ├── overlay_image2.jpg
│   └── ...
└── gradcam_results.csv # Summary statistics for all processed images
```

## Usage

### Quick Start

Run the provided bash script:

```bash
./run_gradcam.sh
```

### Custom Usage

Run the Python script directly with custom parameters:

```bash
python3 generate_gradcam_visualizations.py \
    --images_dir /path/to/your/images \
    --model_path /path/to/model.pt \
    --output_dir /path/to/output \
    --backbone vgg \
    --model_type rsscnn \
    --batch_size 8 \
    --max_images 100 \
    --device cuda
```

### Parameters

- `--images_dir`: Directory containing images to process
- `--model_path`: Path to the trained model (.pt file)
- `--output_dir`: Directory to save visualizations (default: `/home/kiito/cycling_safety_svi/data/processed`)
- `--backbone`: CNN backbone (alex, vgg, dense, resnet)
- `--model_type`: Model type (rcnn, sscnn, rsscnn)
- `--batch_size`: Batch size for processing (default: 8)
- `--max_images`: Maximum number of images to process (useful for testing)
- `--device`: Device to use (auto, cpu, cuda)

## Technical Details

### Alpha Blending

As requested, the overlay images use:
- **50% opacity** for the original image
- **50% opacity** for the attention heatmap

This is implemented in the `overlay_heatmap()` function with `alpha=0.5`.

### Target Layers

The script automatically selects appropriate target layers for Grad-CAM based on the backbone:

- **VGG19**: `backbone.features.35` (last conv layer)
- **ResNet50**: `backbone.layer4.2.conv2` 
- **AlexNet**: `backbone.features.12`
- **DenseNet121**: `backbone.features.denseblock4.denselayer16.conv2`

### Model Compatibility

The script is designed to work with the cycling safety model architecture from the `cycling_safety_subjective_learning_pairwise` submodule, specifically:

- Uses self-pairing for ranking models (passes the same image as both left and right inputs)
- Handles different output formats from the CNN class
- Compatible with the same image preprocessing pipeline as the original prediction script

## Requirements

The script requires the following Python packages:

- torch
- torchvision  
- pandas
- numpy
- opencv-python (cv2)
- matplotlib
- PIL (Pillow)
- tqdm

## Examples

### Basic Usage

Process 50 images with VGG backbone:

```bash
python3 generate_gradcam_visualizations.py \
    --max_images 50 \
    --backbone vgg
```

### High-Quality Processing

Process all images with higher resolution:

```bash
python3 generate_gradcam_visualizations.py \
    --batch_size 4 \
    --max_images 1000
```

## Output Analysis

The `gradcam_results.csv` file contains statistics for each processed image:

- `image_name`: Name of the processed image
- `cam_mean`: Average attention intensity
- `cam_std`: Standard deviation of attention
- `cam_max`: Maximum attention value
- `cam_min`: Minimum attention value
- `heatmap_path`: Path to the saved heatmap
- `overlay_path`: Path to the saved overlay

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model file exists at the specified path
2. **No images found**: Check that the images directory contains supported formats (jpg, png, etc.)
3. **CUDA out of memory**: Reduce batch size or use CPU device
4. **Layer not found**: The script will automatically fall back to the last convolutional layer

### Performance Tips

- Use smaller batch sizes (4-8) for Grad-CAM to avoid memory issues
- Start with a small number of test images (`--max_images 10`) to verify setup
- Use GPU for faster processing if available

## Citation

If you use these visualizations in research, please cite the original Grad-CAM paper:

```
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).
Grad-cam: Visual explanations from deep networks via gradient-based localization.
In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
```