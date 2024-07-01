import time
import sentencepiece as spm


def train_tokenizer(corpus_file, model_name="tiny-llm"):
    start_time = time.time()
    print("Start Training Tokenizer...")
    
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_name,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        pad_id=3,
        model_type="BPE",
        vocab_size=60000,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc"
    )

    end_time = time.time()
    print(f"Finish Training, Cost: {end_time - start_time}s")


if __name__ == "__main__":
    train_tokenizer(corpus_file="/root/autodl-tmp/Corpus/corpus_for_train_tokenizer.txt", model_name="tiny-llm")