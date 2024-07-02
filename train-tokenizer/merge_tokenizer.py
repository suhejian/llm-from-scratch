import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm


def merge_tokenizer(raw_hf_tokenier, add_sp_tokenizer, merged_sp_dir, merged_hf_dir, model_name="merge.model"):
    """
    将开源的Tokenizer和自己训练的Tokenizer合并
    开源的Tokenizer是hugging face上下载的，自己的Tokenizer是用sentencepiece训练得到的
    """
    raw_spm = sp_pb2_model.ModelProto()
    raw_spm.ParseFromString(raw_hf_tokenier.sp_model.serialized_model_proto())
    add_spm = sp_pb2_model.ModelProto()
    add_spm.ParseFromString(add_sp_tokenizer.serialized_model_proto())

    # Add tokens to raw tokenizer
    raw_spm_tokens_set = set(p.piece for p in raw_spm.pieces)
    for p in add_spm.pieces:
        piece = p.piece
        if piece not in raw_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            raw_spm.pieces.append(new_p)  # 追加新的token到之前的模型

    # save
    os.makedirs(merged_sp_dir, exist_ok=True)
    with open(os.path.join(merged_sp_dir, model_name), 'wb') as f:
        f.write(raw_spm.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=os.path.join(merged_sp_dir, model_name))
    tokenizer.save_pretrained(merged_hf_dir)
    print("Merge Successfully")


if __name__ == "__main__":
    LLAMA_MODEL_DIR = "./llama/llama-7b"
    llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_DIR)  # 原生LLaMA分词模型

    our_sp_model = spm.SentencePieceProcessor()
    our_sp_model.Load("./save_models/tiny-llm.model")

    merge_tokenizer(raw_hf_tokenier=llama_tokenizer,
                    add_sp_tokenizer=our_sp_model,
                    merged_sp_dir="./llama_add_chinese_sp",
                    merged_hf_dir="./llama_add_chinese_hf")

    print("-----------------Test----------------------")
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained("./llama_add_chinese_hf")
    text = "苟利国家生死以，岂因祸福避趋之"
    print(text)
    print("LLaMA: ")
    print(f"num of tokens: {len(llama_tokenizer)}")
    print(llama_tokenizer.tokenize(text))
    print(len(llama_tokenizer.tokenize(text)))
    print(f"Chinese LLaMA: ")
    print(f"num of tokens: {len(chinese_llama_tokenizer)}")
    print(chinese_llama_tokenizer.tokenize(text))
    print(len(chinese_llama_tokenizer.tokenize(text)))
