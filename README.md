

This repo includes a bunch of panoramas labeled with solar elevation. the labelling was done manually so mybe you can do a better job then me?
The labeles are in "sun_elevation_labels.csv", and the dataset is included in a google drive link. you migh have to mess around with the paths in
the csv, but the image names are correct.

Once you have the labeled panoramas, you gotta make "normal" images out of them.
Use the python script inside of "pano-convert" to do so, there are some settings to change the properties of the images generated, like FOV, pitch, etc.

Once you have the normal images from the panos, you should have 12x more labeled images that are not panoramas.

To train the model, run "net.py" in "spa". make sure the paths are all correct.
There is also a pretrained model if you dont care about retraining (pretrained is pretty good already, but note that the domain is street view images exclusively)

In "spa", you can then run "infer.py" to run inference with the model giving it the path to some image. make sure the path to the model is correct.
This will output the predicted solar elevation for the image you passed.

You can then do whatever you like with that elevation.

I have not included the algoritm that estimated location of images based on the timestamp and the elevation (infered from the model), because...
I have no idea where on my computer i put it and i am out of time to submit the code. sorry...

It is pretty simple though. You can just take the intersection of multiple arcs around the globe (see the paper for more).
My implementation used a grid search and I defined a metric for how "good" the intersection is at some point given mutiple circles.
Then it can find the best point on the globe according to that metric.

My bad once again for losing it.

The hardest part about this project is messing around with all the different csv files, youll prob have to write a few scripts that convert one
csv header format to another to get stuff working.



