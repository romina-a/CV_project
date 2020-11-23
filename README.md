# CV_project

# Running with pretrained models

Creating virtual environment using VirtualEnv:
```bash 
virtualenv -p python3 ./
source ./bin/activate
 ```

Installing requirements

```bash
pip3 install -r requirements.txt
```

**To Run the video** (can find the defaults in the code)

Optional arguments:  
-fd [face detection method] -->'DNN' or 'ViolaJones'   
-cp [path to the pretrained mask classifier]   
```bash
python3 video.py 
``` 

an example with arguments:
```bash
python3 video.py -fd 'DNN' -cp "experiments/classifier5/classifier5.model"
```

**To run the image annotation**

 Optional arguments: (can find the defaults in the code)  
 -fd [face detection method] , -> 'DNN' or 'ViolaJones'  
 -cp [path to the pretrained mask classifier]   
 -sp [path to save the result] (example: './sampleoutput/name.jpg')  
 -im [path to the image to be marked]  
 if sp is not provided the result will not be saved
 ```bash
python3 image.py 
```

an example with arguments:
```bash 
python3 image.py -fd 'DNN' -im './data/test_images/with_mask/1.jpg' -sp './testimage_result.jpg'
```

Current pretrained classifiers are:   
"experiments/classifier3/classifier3.model" (trained with dataset3)  
"experiments/classifier5/classifier5.model" (trained with dataset5)  

# Training Models

TODO...