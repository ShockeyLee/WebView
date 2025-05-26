import requests
import base64
import json
import os

class QwenVLClient:
    """Client for communicating with the Qwen-VL API server"""
    
    def __init__(self, api_url="http://localhost:8000/generate"):
        """Initialize the client with API endpoint URL"""
        self.api_url = api_url
    
    def analyze_image(self, image_path, prompt=None, temperature=0.7, max_tokens=512):
        """
        Send an image to the Qwen-VL API for analysis
        
        Args:
            image_path (str): Path to the image file
            prompt (str, optional): Text prompt to guide the analysis
            temperature (float): Temperature parameter for generation
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: API response or None if request failed
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at {image_path}")
                return None
                
            # Open image file
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                
                # Prepare data payload
                data = {
                    "prompt": prompt if prompt else "Please analyze this image.",
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Send request
                response = requests.post(self.api_url, files=files, data=data)
                
                # Check response
                if response.status_code != 200:
                    print(f"API request failed: {response.text}")
                    return None
                    
                return response.json()
                
        except Exception as e:
            print(f"Error in API request: {str(e)}")
            return None
    
    def extract_json_from_response(self, response):
        """
        Extract JSON data from the model's response
        
        Args:
            response (dict): The API response
            
        Returns:
            dict: Extracted JSON data or None if extraction failed
        """
        try:
            # Get content from assistant message
            if not response or 'choices' not in response:
                return None
                
            content = response['choices'][0]['message']['content']
            
            # Extract JSON content
            json_start = content.find('```json\n')
            if json_start == -1:
                json_start = content.find('```json')
                if json_start == -1:
                    # Try to find any JSON object
                    json_start = content.find('{')
                    if json_start == -1:
                        return None
                    
                    # Find the matching closing bracket
                    bracket_count = 0
                    for i in range(json_start, len(content)):
                        if content[i] == '{':
                            bracket_count += 1
                        elif content[i] == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_str = content[json_start:i+1]
                                break
                    else:
                        return None
                else:
                    json_start += 7
                    json_end = content.find('```', json_start)
                    json_str = content[json_start:json_end].strip()
            else:
                json_start += 8
                json_end = content.find('\n```', json_start)
                json_str = content[json_start:json_end].strip()
            
            # Parse JSON
            return json.loads(json_str)
            
        except Exception as e:
            print(f"Error extracting JSON: {str(e)}")
            return None
