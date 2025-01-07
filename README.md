# GAN-latent-space-explorer

There's still value in using GANs to explore materials. This repo is my attempt to recreate the latent space explorer that once allowed people to visualize materials (and the space between!) materials at the MET;

![](https://mmlsparkdemo.blob.core.windows.net/met/assets/gen_studio.gif)

This was the interface for the Met Museum’s ‘Gen Studio’ which no longer exists, though the source code is at https://github.com/microsoft/GenStudio.


This is my version.

1. Use the jupyter notebook to train a GAN
2. clone this repo.
3. use requirements.txt to install everything you need
4. drop the saved generator_model_final.keras file into this same folder on your machine
5. make sure your training images are also in this folder in a 'training_images' subfolder
6. python app.py
7. open browser at http://127.0.0.1:5000/ . Click on the points of the diamond to select the images. Then, when you move the mouse in the area between them, you will see an image from the latent space between those points.

For my intial experiments, I used the oslomini image dataset. These can be retrieved with:

```python
pip install image_datasets
import image_datasets
image_datasets.oslomini.download()
```

though they'll be nested a bit.
