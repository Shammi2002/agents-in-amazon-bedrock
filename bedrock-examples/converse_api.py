import boto3 # python SDK in aws.lets you call ,connect with aws services using python code
import json

def use_converse_api():
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_id = "us.amazon.nova-lite-v1:0"

    # Define a system prompt to set model behavior
    system_prompt = [
        {
            "text": "You are a helpful technical assistant who explains concepts clearly and concisely."
        }
    ]

    # User message
    user_message = "What is serverless computing?"

    print("Using Bedrock Converse API")
    print("=" * 60)
    print(f"System Prompt: {system_prompt[0]['text']}")
    print(f"User Message: {user_message}\n")

    try:
        # Use the Converse API
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=system_prompt,
            messages=[                           # messages array is about how you structure the conversation that to be send
                {
                    "role": "user",              # send list of messages as context each with role and content
                    "content": [{"text": user_message}]
                }                                # as model is stateless the complete conversation should send as contet as each turn during conversation
            ],
            inferenceConfig={
                "temperature": 0.7,             # how random, creative model response is
                "maxTokens": 2000
            }
        )

        # Extract the response
        output_text = response['output']['message']['content'][0]['text']

        print("Assistant Response:")
        print(output_text)

        # Display token usage
        usage = response.get('usage', {})
        print(f"\nToken Usage:")                  #token is small chunk of text
        print(f"  Input tokens: {usage.get('inputTokens', 'N/A')}")
        print(f"  Output tokens: {usage.get('outputTokens', 'N/A')}")
        print(f"  Total tokens: {usage.get('totalTokens', 'N/A')}")

        print(f"\nStop Reason: {response['stopReason']}")

        return response

    except Exception as e:
        print(f"Error using Converse API: {e}")
        raise

if __name__ == "__main__":
    use_converse_api()
