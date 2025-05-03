from transformers import pipeline
generation = pipeline('translation_en_to_fr')
prompt = "Hello how are you. Hope you are doing well."
result = generation(prompt)
print(result)