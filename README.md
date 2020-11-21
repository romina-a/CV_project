# CV_project
# For creating virtual environment using VirtualEnv:
virtualenv -p python3 ./
source ./bin/activate

# For installing requirements
pip3 install -r requirements.txt


# To run the video (can find the defaults in the code)
# Optional arguments:
# -fd <face detection method> , -> DNN or ViolaJones 
# -cp <path to the pretrained mask classifier>
python3 video.py 
#or
python3 video.py -fd 'DNN'


# To run the image 
# Optional arguments: (can find the defaults in the code)
# -fd <face detection method> , -> 'DNN' or 'ViolaJones'
# -cp <path to the pretrained mask classifier>
# -sp <path to save the result> (example: './sampleoutput/name.jpg')
# -im <path to the image to be marked> 
# if sp is not provided the result will not be saved
python3 video.py 
#or
python3 image.py -fd 'DNN' -im './data/test_images/16.jpg' -sp './sampleoutput/testimage.jpg'