# Vintage-Film-Enhancer
Colorizes black and white footage and reduces background static

Colorizer:
Utilizes a GAN to train a discriminator and generator model on black-and-white to colored picture conversions
Each frame of the video is separated and colorized, and spliced back together for final video

Audio Refinement:
Signal Processing libraries create a mask of all frequencies, identifies the frequency belonging to background static, and cuts it out of the video 
