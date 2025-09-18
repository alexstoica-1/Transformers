import tiktoken 
#import tiktoken._educational

enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("Tiktoken"), enc.n_vocab)

# Train a BPE tokeniser on a small amount of text
#enc = tiktoken._educational.train_simple_encoding()

# Visualise how the GPT-4 encoder encodes text
#enc = tiktoken._educational.SimpleBytePairEncoding.from_tiktoken("cl100k_base")
#enc.encode("hello world aaaaaaaaaaaa")

print(enc.encode("Hy there!"))
print(enc.decode([35576, 1354, 0]))

def compare_encodings(example_string: str) -> None:
    """Prints a comparison of three string encodings."""

    print(f'\nExample string: "{example_string}"')
    # for each encoding, print the # of tokens, the token integers, and the token bytes
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
        print()
        print(f"{encoding_name}: {num_tokens} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")

compare_encodings("Alexiscool.")