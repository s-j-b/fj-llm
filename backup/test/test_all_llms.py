from 
def test_all_llms():
    client = LLMClient()
    model_alias = sys.argv[1]
    prompt = " ".join(sys.argv[2:])
    
    print(f"Querying {model_alias}...")
    response = client.query(model_alias, prompt)





if __name__ == '__main__':
    test_all_llms()
