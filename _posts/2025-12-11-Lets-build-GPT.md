# Lets Build GPT

In this blog, I followed Andrey Carpathy course on youtube to build a GPT which here is a character based generative transformer model. We are not going to use the whole internet to build this model but instead we will use what is called tiny_shekspier 
as an input. We name it input.txt. What we are going to do is to create a model that is able to generate charcater sequences that look like this text. The dataset is **Tiny Shekspear**. 

This dataset jas 40000 lines and 65 characters. It is possible to build a character-level tokenizer that can tokenize a sentence to number at the character level.

## Tokenization
```
#string to integer
stoi = {ch:i for i, ch in enumerate(chars)}
#integer to string
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l) 

print(encode("hii there"))
print(decode(encode('hii there')))
```

After implementing encoder and decoder, it is possible to tokenize the whole tiny-Shakespear text to tokens and save it into a torch tensor.

## Train and validation split
We then take 90% of the text for training and the remaining 10% for validation. But we can not feed the whole text at once to the model for training. We need to send chunks of text at a time.


## Chunking the input data
let's assume the chunks have length of 8 character. In that block, there are 8 different examples for the model to train on. 

```
block_size = 8
train_data[: block_size + 1]

output:
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
```

This means when 18 goes in, the model should predict 47, when [18, 47] goes in, the model should predict 56. When [18, 47, 56] goes in, the model should predict 57 and so on!

```
When input is tensor([18]), the target is: 47
When input is tensor([18, 47]), the target is: 56
When input is tensor([18, 47, 56]), the target is: 57
When input is tensor([18, 47, 56, 57]), the target is: 58
When input is tensor([18, 47, 56, 57, 58]), the target is: 1
When input is tensor([18, 47, 56, 57, 58,  1]), the target is: 15
When input is tensor([18, 47, 56, 57, 58,  1, 15]), the target is: 47
When input is tensor([18, 47, 56, 57, 58,  1, 15, 47]), the target is: 58

```
 
