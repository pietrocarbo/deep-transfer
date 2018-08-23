# deep-transfer


#### Features to test/add
* resize content/style input imgs based on cli args

* torchvision.transforms.LinearTransformation(transformation_matrix) for whitening [here](torchvision.transforms.LinearTransformation(transformation_matrix))
* covariance matrix [regularization](https://github.com/sunshineatnoon/PytorchWCT/issues/7)

* missing first convolution in decoder?

* confront nn.upsamplingbilinear2d
* MISSING RELUs in encoders

* optimize memory usage with [pinned memory buffers](https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers)

#### TODOs
* logging statements
* comments/code documentation

#### Possible applications
* satellite images style transfer
* photorealistic style transfer
* artworks pair combinations:
    * figure 1 at https://arxiv.org/pdf/1705.04058.pdf
