import requests
import json


def test_api_endpoint():
    """
    Tests the /schedule/next-week endpoint to inspect its response.
    """
    url = "http://127.0.0.1:8000/schedule/next-week"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        print("API Response:")
        print(json.dumps(data, indent=2))

        if isinstance(data, list):
            print("\n✅ The response is a list (Array).")
        elif isinstance(data, dict):
            print(
                "\n⚠️ The response is an object (Dictionary), not a list. This is the likely cause of the 'F.map is not a function' error in the frontend."
            )
        else:
            print(f"\n❓ The response is of type: {type(data)}")

    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")


if __name__ == "__main__":
    test_api_endpoint()
