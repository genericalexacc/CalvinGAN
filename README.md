# CalvinGAN

## Conda environment
```console
foo@bar:~$ conda env create --name calvin -f calvin.yaml
foo@bar:~$ conda activate calvin
```

## To just try inference
- Download weights bellow
```console
foo@bar:~$ python calvin_gan.py generate 10
```

## Preprocessing and training (REQUIRES CUDA GPU)
```console
foo@bar:~$ python pdf_to_png.py
foo@bar:~$ python split_page_to_panels.py
foo@bar:~$ python calvin_gan.py train
```

For metrics and merging text and images look at the notebooks
```console
foo@bar:~$ jupyter notebook
```

## Download links
Download Comic https://www.pdfdrive.com/calvin-and-hobbes-sunday-pages-1985-1995-e156743420.html
Download Weights (folder "weights"): https://drive.google.com/file/d/1SelVENIDKvm3So1G1iTiAdFfkLZftkiG/view?usp=sharing
Download Real Calvin panels (folder "real"): https://drive.google.com/file/d/1Vq8qqhshJELoLFlvNOcN1ThJIi8DSGkE/view?usp=sharing
Download Fake Calvin panels (folder "image_bank"): https://drive.google.com/file/d/1DmE01n0fsuMIrSBiiDGyDVWswFnaGAID/view?usp=sharing
