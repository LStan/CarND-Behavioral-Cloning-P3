# Project 3. Behavioral Cloning

## Model architecture

The final model consists of the following layers:

1. Cropping2D - crops 60 px from the top of the image and 20 px from the bottom. Output - 80x320
2. AveragePooling2D with 5x5 pool size - resizes the image to 16x64
3. Lambda(lambda x: x/127.5 - 1.) - normalization layer
4. 2D convolution with kernel size (1,1), depth 3, same padding - acts as a space converter (thanks Vivek Yadav for the idea)
5. 2D convolution with kernel size (3,3), depth 4, valid padding and elu activation.
6. Max pooling layer with kernel size (2,2) and valid padding.
7. Dropout to prevent overfitting
8. 2D convolution with kernel size (3,3), depth 4, valid padding and elu activation.
9. Max pooling layer with kernel size (2,2) and valid padding.
10. Dropout to prevent overfitting
11. Flatten layer
12. Dense layer with 1 neuron - outputs the steering angle.

##  Dataset

I tried to gather appropriate data by myself with a keyboard but it was not an easy task so I ended up using udacity data
(https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
Some examples of images are in IMG folder
I used center, left and right images. For left/right images I also modified the steering angle (data_left.steering = data_left.steering * steering_mult + steering_add and data_right.steering = data_right.steering * steering_mult - steering_add, where steering_add = 0.27 and steering_mult = 1.65).
Also I added flipped version of every image and used random brightness ajustment during training (in the generator)


## Training 

All data was split: 80% - training set, 20%  - validation set. 
The model was trained with a batch size of 128 and for 10 epochs. Using more than 10 epochs almost did not decrease the validation loss (actually, I discovered that loss in not a good measure here because I had non-working models with low loss and working ones with high loss).  An Adam optimizer was used. Input size of the model is 160x320 (full image) so I used a generator.  Testing was done in the simulator.

## The approach taken for deriving and designing the model

I started with the NVIDIA model and gradually removed layers and reduced parameters until I arrived to a model that worked on track 1, but not on track 2. I continued to reducing parameters to make track2 work, added random brightness adjustment, tweek the parameters and finally got the model described above that works on both tracks (although the car started to wiggle a little bit on track 1).

Video on Youtube: https://youtu.be/EO8R9KtEvp4