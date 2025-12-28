import os
import openai


def test_openai_api():
    """Test the OpenAI API connection"""
    try:
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not found in environment variables")
            return False

        print(f"Using OpenAI API key: {api_key[:5]}...")

        # Configure OpenAI client
        openai.api_key = api_key

        # Simple test request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using a cheaper model for testing
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, are you working?"},
            ],
            max_tokens=50,
            temperature=0.7,
        )

        # Extract and print the response
        ai_response = response.choices[0].message.content
        print(f"OpenAI API test successful. Response: {ai_response}")
        return True

    except Exception as e:
        print(f"OpenAI API test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_openai_api()
