# deep-transfer


#### TODOs
...

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
* photorealistic style transfer
* artworks pair combinations:
    * figure 1 at https://arxiv.org/pdf/1705.04058.pdf
