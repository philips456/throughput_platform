import os
import sys


def set_openai_api_key():
    """Set the OpenAI API key as an environment variable"""
    if len(sys.argv) < 2:
        print("Usage: python set_env.py YOUR_OPENAI_API_KEY")
        return

    api_key = sys.argv[1]
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"OPENAI_API_KEY environment variable set to: {api_key[:5]}...")

    # Now run your Django server or test script
    if len(sys.argv) > 2 and sys.argv[2] == "test":
        # Run the test script
        import test_openai

        test_openai.test_openai_api()
    else:
        print("You can now run your Django server with:")
        print("python manage.py runserver")


if __name__ == "__main__":
    set_openai_api_key()
