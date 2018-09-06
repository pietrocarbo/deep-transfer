# deep-transfer

Pytorch implementation of arbitrary style transfer via CNN features WCT trasform.

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


### TODOs

##### code for grid 
images = []
for im_fn in sorted(os.listdir("outputs/test")):
    images.append(load_img(os.path.join("outputs/test", im_fn), None))
save_image(images, "outputs/grid.jpg", nrow=9)

#### Result analysis
* cubic/bilinear/nearest upsampling
* covariance matrix [regularization](https://github.com/sunshineatnoon/PytorchWCT/issues/7)
* differents whitening [methods](http://joelouismarino.github.io/blog_posts/blog_whitening.html)
* adain transform vs WCT ?

#### Applications
* satellite images style transfer
* artworks pair combinations:
    * figure 1 at https://arxiv.org/pdf/1705.04058.pdf
