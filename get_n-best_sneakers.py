import json
import numpy as np
import os

def classify_deployed(file_name, classes, k):
    payload = None
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)

    deployed_endpoint.content_type = 'application/x-image'
    result = json.loads(deployed_endpoint.predict(payload))
    best_prob_indexes = np.partition(arr,-k)[-k:]
    
    print("Classes are: "+classes);
    print("Best Predictions Indexes Should Be: "+best_prob_indexes)
    
    return (classes, result, best_prob_indexes)
