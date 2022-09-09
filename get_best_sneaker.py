import json
import numpy as np
import os
import sys

def classify_deployed(file_name, classes):
    payload = None
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)

    deployed_endpoint.content_type = 'application/x-image'
    result = json.loads(deployed_endpoint.predict(payload))
    best_prob_index = np.argmax(result)
    
    print("Classes are: "+classes);
    print("Best Prediction Should Be: "+classes[best_prob_index])
    
    return (classes[best_prob_index], result[best_prob_index])
