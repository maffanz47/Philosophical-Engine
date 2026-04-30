import httpx
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Infer dominant philosophy from text")
    parser.add_argument("text", type=str, help="Text to classify")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="API URL")
    args = parser.parse_args()

    try:
        response = httpx.post(f"{args.url}/predict", json={"text": args.text})
        response.raise_for_status()
        data = response.json()
        print(f"Consensus Theme: {data['consensus_theme']}\n")
        
        print("--- Ensemble Predictions ---")
        for model_name, result in data['ensemble_results'].items():
            print(f"\n{model_name.replace('_', ' ')}:")
            print(f"  Dominant: {result['dominant_theme']} (Confidence: {result['confidence']})")
            
        print("\n--- Suggested Reading Sources ---")
        for source in data['suggested_reading']:
            print(f"  - {source}")
            
    except httpx.RequestError as e:
        print(f"Failed to connect to API at {args.url}. Is the server running? Error: {e}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"API returned an error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
