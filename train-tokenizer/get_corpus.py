import os
import glob
from tqdm import tqdm
import json
from zhconv import convert


RAW_CORPUS_DIR = "/root/autodl-tmp/Corpus/tokenizer"

CLUECORPUSSMALL_DATASET_NAMES = ["comment2019zh_corpus", "news2016zh_corpus", "webText2019zh_corpus", "wiki2019zh_corpus"]

OUTPUT_DIR = "/root/autodl-tmp/train_tokenizer/data"

MERGED_DIR = "/root/autodl-tmp/train_tokenizer/data"

MERGED_PATH = "/root/autodl-tmp/train_tokenizer/corpus_for_train_tokenizer.txt"


def process_clue(dataset_dir, dataset_name_list, output_dir):
    """处理CLUECorpusSmall数据集
    
    该数据集有多个子数据集，对于每个子数据集，需要将多个文件合并到一个文件中
    
    数据地址：https://aistudio.baidu.com/datasetdetail/194775

    Args:
        dataset_dir: 数据集目录
        dataset_name_list: 数据集名称
        output_dir: 合并之后的文件保存目录
    """
    
    def _process_clue(dataset_dir, dataset_name, output_path):
        corpus_clue = open(output_path, "w", encoding="utf-8")
        
        input_path = os.path.join(dataset_dir, dataset_name) + "/*.txt"
        cnt = 0
        for file in tqdm(glob.glob(input_path)):
            with open(file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if len(line.strip()) > 50:
                        corpus_clue.write(line.strip() + "\n")
                        cnt += 1
        
        corpus_clue.close()
        print(f"{dataset_name}数据集大小为{cnt}")

    for dataset in dataset_name_list:
        save_path = os.path.join(output_dir, "tokenizer_" + dataset + ".txt")
        _process_clue(dataset_dir, dataset, save_path)
        
        
def process_cls(dataset_dir, output_path):
    """处理CLS数据集
    
    数据地址：https://huggingface.co/datasets/neuclir/csl/tree/main

    Args:
        dataset_dir: 数据集存放目录
        output_path: 合并文件位置
    """
    pass


def process_lcsts(dataset_dir, output_path):
    """处理LCSTS数据集
    
    需要将语料合并到一个文件中
    数据地址：
        https://bj.bcebos.com/paddlenlp/datasets/LCSTS_new/train.json
        https://bj.bcebos.com/paddlenlp/datasets/LCSTS_new/dev.json

    Args:
        dataset_dir: 数据集目录
        output_path: 合并文件位置
    """
    corpus_lcsts = open(output_path, "w", encoding="utf-8")
    
    cnt = 0
    train_path = os.path.join(dataset_dir, "train.json")
    dev_path = os.path.join(dataset_dir, "dev.json")
    lines = open(train_path, "r", encoding="utf-8").readlines() + open(dev_path, "r", encoding="utf-8").readlines()
    for line in lines:
        sample = json.loads(line.strip())
        corpus_lcsts.write(sample["summary"] + " " + sample["content"] + "\n")
        cnt += 1
    
    corpus_lcsts.close()
    print(f"lcsts数据集大小为{cnt}")
    
    
def process_zhwiki_data(dataset_dir, output_path):
    """处理中文维基百科数据
    
    数据地址：https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
    抽取wiki data:
        
        git clone https://github.com/attardi/wikiextractor
        cd https://github.com/attardi/wikiextractor
        cd python setup.py install

        wikiextractor -o [output] --process 2 -b 1024K --json [input]

    Args:
        dataset_dir: 维基百科数据集目录
        output_path: 合并文件路径
    """
    corpus_zhwiki = open(output_path, "w", encoding="utf-8")
    
    cnt = 0
    for wiki_doc in tqdm(os.listdir(dataset_dir)):
        with open(os.path.join(dataset_dir, wiki_doc), "r", encoding="utf-8") as f:
            for line in tqdm(f, leave=False, desc=""):
                sample = json.loads(line.strip())
                sample["title"] = convert(sample["title"], "zh-cn")
                sample["text"] = convert(sample["text"], "zh-cn")
                text = sample["title"] + " " + sample["text"]
                corpus_zhwiki.write("".join(text.split("\n")) + "\n")
                cnt += 1
        
    print(f"zhwiki文档个数：{cnt}")
    corpus_zhwiki.close()         
    
    
def merge_corpus(merge_data_dir, merge_path):
    """将目录下的文件合并到一个文件中

    Args:
        merge_data_dir: 数据目录
        merge_path: 合并文件路径
    """
    corpus = open(merge_path, "w", encoding="utf-8")
    
    cnt = 0
    for file in glob.glob(merge_data_dir + "/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                if len(line.strip()) > 100:
                    corpus.write(line.strip() + "\n")
                    cnt += 1
    
    corpus.close()
    print("Corpus For Training Tokenizer has {cnt} lines")


if __name__ == "__main__":
    print("Process CLUECorpusSmall Dataset")
    process_clue(dataset_dir=RAW_CORPUS_DIR, 
                 dataset_name_list=CLUECORPUSSMALL_DATASET_NAMES, 
                 output_dir=OUTPUT_DIR)
    print("------------------------------------")
    
    print("Process LCSTS Dataset")
    lcsts_dir = os.path.join(RAW_CORPUS_DIR, "LCSTS")
    lcsts_file = os.path.join(OUTPUT_DIR, "tokenizer_lcsts.txt")
    process_lcsts(dataset_dir=lcsts_dir, output_path=lcsts_file)
    print("------------------------------------")
    
    print("Process ZH Wiki Dataset")
    zhwiki_dir = os.path.join(RAW_CORPUS_DIR, "zhwiki-data/AA") 
    zhwiki_file = os.path.join(OUTPUT_DIR, "tokenizer_zhwiki.txt")
    process_zhwiki_data(dataset_dir=zhwiki_dir, output_path=zhwiki_file)
    print("-------------------------------------")
    
    print("Merge Corpus For Training Tokenizer")
    merge_corpus(merge_data_dir=MERGED_DIR, merge_path=MERGED_PATH)
    
    