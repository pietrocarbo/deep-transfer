# deep-transfer

This is a Pytorch implementation of the "Universal Style Transfer via Feature Trasforms" NIPS17 [paper](https://arxiv.org/abs/1705.08086).

## Requirements 
+ Python environment with the following packages
    + PyTorch
    + torchvision
    

## Usage
main.py [-h] [--content CONTENT] [--style STYLE] [--synthesis]
    [--stylePair STYLEPAIR] [--mask MASK]
    [--contentSize CONTENTSIZE] [--styleSize STYLESIZE]
    [--outDir OUTDIR] [--outPrefix OUTPREFIX] [--alpha ALPHA]
    [--beta BETA] [--no-cuda] [--single-level]

+  -h, --help            show this help message and exit
+  --content CONTENT     Path of the content image (or a directory containing
                        images) to be trasformed
+  --style STYLE         Path of the style image (or a directory containing
                        images) to use
+  --synthesis Flag to syntesize a new texture. Must provide a
                        texture style image
+  --stylePair STYLEPAIR  Path of two style images (separated by ",") to use in
                        combination
+  --mask MASK           Path of the binary mask image (white on black) to
                        trasfer the style pair in the corrisponding areas
+  --contentSize CONTENTSIZE
                        Reshape content image to have the new specified
                        maximum size (keeping aspect ratio)
+  --styleSize STYLESIZE
                        Reshape style image to have the new specified maximum
                        size (keeping aspect ratio)
+  --outDir OUTDIR       Path of the directory where stylized results will be
                        saved
+  --outPrefix OUTPREFIX
                        Name prefixed in the saved stylized images
+  --alpha ALPHA         Hyperparameter balancing the blending between original
                        content features and WCT-transformed features
+  --beta BETA           Hyperparameter balancing the interpolation between the
                        two images in the stylePair
+  --no-cuda             Flag to enables GPU (CUDA) accelerated computations
+  --single-level        Flag to switch to single level stylization

Supported image file formats are: jpg, jpeg, png
