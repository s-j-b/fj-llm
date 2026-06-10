from fj_llm import LLMClient

def test_all_llms():
    client = LLMClient()

    prompt = 'count to 10'
    
    PAIRS = [
        ('gpt-best', prompt),
        ('gpt-reasoner', prompt),
        ('gpt-light', prompt),
        ('deepseek-light', prompt),
        ('deepseek-reasoner', prompt)
    ]
    for p in PAIRS:
        print("\n\n-----------\nQUERYING: ", p[0])
        response = client.query(p[0], p[1])
        if response.success:
            print(f"\nResponse ({response.input_tokens} in + {response.output_tokens} out = {response.total_tokens} tokens, ${response.cost:.6f}):")
            print("-" * 50)
            print(response.content)
        else:
            print(f"Error: {response.error}")

if __name__ == '__main__':
    test_all_llms()
