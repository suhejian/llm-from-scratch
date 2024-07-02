# Train Tokenizer

当前主流的LLM都采用`BPE`子词切分算法，不过即使是`BPE`也分为字符级别的`BPE`和字节级别的`BPE`（也称为`BBPE`，即Byte-level BPE）。目前有两个主流的Python包`sentencepiece`和`tiktoken`，前者主要支持字符级别的BPE，而后者主要支持字节级别的BPE。

`sentencepiece`将句子视为Unicode字符序列，独立于语言，同时空格也作为普通符号（`_`）处理。

## BPE与BBPE

**Byte-level BPE**

文本的UTF-8编码可以将每个Unicode字符编码成1-4个字节，这允许我们将句子建模成**字节序列**，而**不是字符序列**。文本的字节序列表示通常比字符序列表示长得多（高达4倍），所以如果原样使用字节计算成本是很高的。因此BBPE考虑将字节序列分割成可变长度的`n-gram`，即字节级的"subwords"。

BBPE具有与BPE相当的性能，而其大小仅为BPE的1/8。在多语言设置中，BBPE最大限度地共享多种语言的词汇并实现更好的翻译质量。实验表明，BBPE可以在具有非重叠字符集的语言之间实现可迁移的模型。


BBPE和BPE的比较：

优点：
1. 效果和BPE相当，但是词表减小
2. 可以在多语言之间通过字节级别的子词实现更好的共享
3. 即使字符集不重叠，也可以通过字节层面的共享实现良好的迁移


缺点：
1. 编码序列时，长度可能会略长于BPE，计算成本更高
2. BBPE字符可能是某个字符的一部分，也可能是一些完整的字符或者不完整字符的组合，由字节解码时，可能会遇到歧义，需要通过上下文信息和动态规划来进行解码


---

UTF-8编码中，表示一个英文字符需要一个字节，表示一个中文字符需要三个字节

```python
chinese_text = "苟利国家生死以"
print(f"length of chinese text: {len(chinese_text)}")
print(f"length of chinese ids: {len(list(chinese_text.encode('utf-8')))}")

english_text = "hello"
print(f"length of english text: {len(english_text)}")
print(f"length of english ids: {len(list(english_text.encode('utf-8')))}")

# length of chinese text: 7
# length of chinese ids: 21
# length of english text: 5
# length of english ids: 5
```

## 开源模型的Tokenizer

### BPE

LLaMA开源模型采用`sentencepiece`训练tokenizer，不过由于采用的训练语料大多数是英文，因此它的词表也主要是英文，对中文的支持不太友好。这会导致：1）有些中文字符模型无法识别；2）即使可以通过`UTF-8`编码识别，也耗费更多的token

以[llama-hf-7b](https://modelscope.cn/models/skyline2006/llama-7b)为例：

```python
from transformers import LlamaTokenizer

LLAMA_MODEL_DIR = "llama/llama-7b"
llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_DIR)       # 原生LLaMA分词模型
text = "苟利国家生死以，岂因祸福避趋之"

print(f"num of tokens: {len(llama_tokenizer)}")
print(llama_tokenizer.tokenize(text))
```

输出结果如下：
```text
num of tokens: 32000
['▁', '<0xE8>', '<0x8B>', '<0x9F>', '利', '国', '家', '生', '死', '以', '，', '<0xE5>', '<0xB2>', '<0x82>', '因', '<0xE7>', '<0xA5>', '<0xB8>', '福', '<0xE9>', '<0x81>', '<0xBF>', '<0xE8>', '<0xB6>', '<0x8B>', '之']
```

我们可以看到，该模型的词表大小是32000，很多中文字符也不在词表中，视作Unicode字符后被`UTF-8`编码为三个字节。例如，`b'\xe8\x8b\x9f'.decode("utf-8")`得到`苟`。

----

MiniCPM模型作为通用模型，具备英文、中文、中国古文、代码、表情符号，其他语言等多方面能力，因此词表相对较大，大小为122753。该词表构建于大量综合语料上，使用`sentencepiece`库进行BPE，添加了包括繁体中文、罕见字、emoji、希腊字母、俄文字母等等特殊符号。

```python
from transformers import AutoTokenizer

MINICPM_MODEL_DIR = "openbmb/MiniCPM-2B-sft-bf16"
minicpm_tokenizer = AutoTokenizer.from_pretrained(MINICPM_MODEL_DIR)

print(f"num of tokens: {len(minicpm_tokenizer)}")
print(minicpm_tokenizer.tokenize(text))
```

输出结果如下：
```text
num of tokens: 122753
['▁', '苟', '利', '国家', '生死', '以', '，', '岂', '因', '祸', '福', '避', '趋', '之']
```

我们可以看到，该模型的词表大小是122753，同时对中文的支持更加友好。

### BBPE

Qwen-7B采用了`UTF-8`字节级别的BPE tokenization方式，并依靠`tiktoken`执行分词。它的词表有两类：1）源于BPE、`bytes`类型的普通token；2）特殊指定、`str`类型的特殊token。

尽管基于字节序列的方式保证了所有文本均可被tokenize，且没有未登录token问题，但是在处理罕见文本时可能回退到字节级别的编码。由于从字节序列解码为文本时，`errors`参数设为`replace`，处理不完整的token序列可能会遇到`UTF-8`解码错误，表象是生成中包含“替换字符”(�)。这一行为可以通过将`errors`参数设为`ignore`来规避。

```python
from transformers import AutoTokenizer

QWEN_MODEL_DIR = "Qwen/Qwen-7B-Chat"
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)

print(f"num of tokens: {len(qwen_tokenizer)}")
print(qwen_tokenizer.tokenize(text))
```

输出结果如下：
```text
num of tokens: 151851
[b'\xe8\x8b\x9f', b'\xe5\x88\xa9', b'\xe5\x9b\xbd\xe5\xae\xb6', b'\xe7\x94\x9f\xe6\xad\xbb', b'\xe4\xbb\xa5', b'\xef\xbc\x8c', b'\xe5\xb2\x82', b'\xe5\x9b\xa0', b'\xe7\xa5\xb8', b'\xe7\xa6\x8f', b'\xe9\x81\xbf', b'\xe8\xb6\x8b', b'\xe4\xb9\x8b']
```

我们可以看到词表大小是151851，token都是字节序列，将这些字节序列解码可以看到实际的分词结果是这样的`['苟', '利', '国家', '生死', '以', '，', '岂', '因', '祸', '福', '避', '趋', '之']`。


总结一下，针对文本“苟利国家生死以，岂因祸福避趋之”，LLaMA模型需要耗费26个token，MiniCPM模型需要耗费14个token，而Qwen模型需要耗费13个token。Tokenizer和LLM是紧密相关的，高效的Tokenizer可以使得LLM支持更长的上下文。

## 训练自己的Tokenizer

1. 准备训练语料
   详见`get_corpus.py`文件

2. 使用`sentencepiece`进行训练
   详见`spm_train_tokenizer.py`文件，可自行修改超参数

3. 测试分词效果
   依旧是以“苟利国家生死以，岂因祸福避趋之”为例，我们的tokenizer的分词结果如下：
   > ['▁', '苟', '利', '国家', '生死', '以', ',', '岂', '因', '祸', '福', '避', '趋', '之']
   需要14个token，而且可以将“国家”作为一个完整的token。

## 合并词表

如果我们还是想用LLaMA模型，只是需要扩充中文词表，那么可以将LLaMA原生的词表和我们在中文语料上训练得到的词表进行合并，详见`merge_tokenizer.py`文件。

合并词表前后的分词结果如下：

```text
苟利国家生死以，岂因祸福避趋之
LLaMA:
num of tokens: 32000
['▁', '<0xE8>', '<0x8B>', '<0x9F>', '利', '国', '家', '生', '死', '以', '，', '<0xE5>', '<0xB2>', '<0x82>', '因', '<0xE7>', '<0xA5>', '<0xB8>', '福', '<0xE9>', '<0x81>', '<0xBF>', '<0xE8>', '<0xB6>', '<0x8B>', '之']
26

Chinese LLaMA:
num of tokens: 89985
['▁', '苟', '利', '国家', '生死', '以', '，', '岂', '因', '祸', '福', '避', '趋', '之']
14
```

如何使用修改之后的词表也很重要。如果重新从头开始训练模型，那么只需要调用`model.resize_token_embeddings(len(tokenizer))`修改`token embedding`层的维度即可。但是如果想要保留原始模型`token embedding`层的参数，那么可以这样做：

1. 找到新词表和旧词表之间的映射关系
2. 如果某个token既出现在新词表中，又出现在旧词表中，那么就用原始模型的embedding替换
3. 如果某个token只出现在新词表中，而没有出现在旧词表中，那么就进行相应的初始化再进行赋值

具体可以参考[LLMPruner](https://github.com/yangjianxin1/LLMPruner)


【参考】
1. [how-to-train-tokenizer](https://github.com/yanqiangmiffy/how-to-train-tokenizer/tree/main)
2. [minbpe](https://github.com/karpathy/minbpe)
3. [MiniCPM：揭示端侧大语言模型的无限潜力](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a)
4. [Qwen-tokenization_note_zh.md](https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md)
5. [怎么让英文大语言模型支持中文？](https://zhuanlan.zhihu.com/p/639144223)
