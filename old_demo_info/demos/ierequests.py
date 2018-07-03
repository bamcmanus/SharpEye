#!/usr/bin/python

##################################################
## Author:      Vijay Arputharaj
## Description: Client side HTTP request script
##################################################

import requests
import json
import base64
from PIL import Image
from PIL import ImageFile

DEBUG = 0

def send_notification(child_data):
   
    if 'ID' in child_data:
    	child_data['StudentId']  = child_data['ID']   

    # Base64Encode Image
    else:
    	image_in        = open(child_data['ImagePath'], 'rb')
    	image_read      = image_in.read()
    	image_64_encode = base64.encodestring(image_read)

    	if DEBUG:
        	ImageFile.LOAD_TRUNCATED_IMAGES = True       
        	image_64_decode = base64.decodestring(image_64_encode)
        	image_out       = open('decode.png', 'wb')
        	image_out.write(image_64_decode)
        	img = Image.open('decode.png')
        	img.show()

    	# Create Json
    	child_data['ImageType'] = 'png'
    	child_data['Image']     =  image_64_encode

   
    json_child_data = json.dumps(child_data)

    if DEBUG:   
        print json_child_data


    headers = {'x-api-key': '2WlcSIVKET8lEu0R9sHEA4xf88mCYhgD7pCIqpJQ'}
    if 'ID' in child_data:
        r = requests.post("https://jigr0vbfih.execute-api.us-west-2.amazonaws.com/dev/bus/ingress/notification", data=json_child_data, headers=headers)
        print r.text
    else:
        r = requests.post("https://jigr0vbfih.execute-api.us-west-2.amazonaws.com/dev/bus/ingress/unknown", data=json_child_data, headers=headers)
        print r.text




