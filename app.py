import json

from joblib import load

# Initialize models_loaded flag
models_loaded = False

# Load the models outside of the Lambda handler function
try:
    presence_classifier = load('presence_classifier.joblib')
    presence_vect = load('presence_vectorizer.joblib')
    category_classifier = load('category_classifier.joblib')
    category_vect = load('category_vectorizer.joblib')
    models_loaded = True
except Exception as e:
    print("Error loading models:", e)

def lambda_handler(event, context):
    # Check if models failed to load
    if not models_loaded:
        response = {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({ 'error': 'Internal Server Error: Models failed to load' })
        }
        return response
    
    try:
        # Extract data from the event
        data = json.loads(event['body']) if 'body' in event else {}

        # Process the data using the loaded models
        output = []
        for token in data.get('tokens', []):
            result = presence_classifier.predict(presence_vect.transform([token]))
            if result == 'Dark':
                cat = category_classifier.predict(category_vect.transform([token]))
                output.append(cat[0])
            else:
                output.append(result[0])

        # Prepare the response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({ 'result': output })
        }

    except Exception as e:
        # Log the error
        print("Error:", e)
        
        # Prepare an error response
        response = {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({ 'error': 'Internal Server Error' })
        }

    return response
