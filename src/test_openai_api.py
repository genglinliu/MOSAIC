#!/usr/bin/env python3
"""
A simple script to test the OpenAI API.
This script demonstrates how to make basic API calls to OpenAI.
"""

import openai
from openai import OpenAI
import sys
from keys import OPENAI_API_KEY

def test_openai_api():
    """Test the OpenAI API by making a simple completion request."""
    try:
        # Check if API key is set
        api_key = OPENAI_API_KEY
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is not set.")
            print("Please set your API key using:")
            print("export OPENAI_API_KEY='your-api-key'")
            return False

        # Initialize the client
        client = OpenAI(api_key=api_key)
        
        # Make a simple test request
        print("Making a test request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me? This is a test of the OpenAI API."}
            ],
            max_tokens=50
        )
        
        # Print the response
        print("\nAPI Response:")
        print(f"Model used: {response.model}")
        print(f"Response content: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Usage - prompt tokens: {response.usage.prompt_tokens}")
        print(f"Usage - completion tokens: {response.usage.completion_tokens}")
        print(f"Usage - total tokens: {response.usage.total_tokens}")
        
        print("\nAPI test completed successfully!")
        return True
        
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
    except openai.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
    except openai.AuthenticationError as e:
        print(f"Authentication error: {e}")
        print("Please check your API key.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return False

if __name__ == "__main__":
    print("OpenAI API Test Script")
    print("======================")
    
    success = test_openai_api()
    
    if success:
        sys.exit(0)
    else:
        print("\nAPI test failed. Please check the error messages above.")
        sys.exit(1)
