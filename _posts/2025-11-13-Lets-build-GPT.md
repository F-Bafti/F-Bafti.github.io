# Lets Build GPT

In this blog, I follow Andrej Karpathy’s course on YouTube to build a GPT—specifically, a character-based generative transformer model. We won’t use the entire internet to train this model; instead, we’ll use a small dataset called **Tiny Shakespeare**, which we save as `input.txt`. The goal is to create a model that can generate character sequences resembling this text.  

This dataset has **40,000 lines** and **65 unique characters**, making it possible to build a character-level tokenizer that converts each character into a number.

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

After implementing the encoder and decoder, we can tokenize the entire Tiny Shakespeare text and save it as a Torch tensor.

## Train and validation split
We use 90% of the text for training and the remaining 10% for validation. We can’t feed the entire text to the model at once; instead, we process it in chunks.


## Chunking the input data
Assume chunks of length 8 characters. Each chunk provides multiple training examples for the model.

```python
block_size = 8
train_data[: block_size + 1]
```
```markdown
output:
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
```

This means when 18 goes in, the model should predict 47, when [18, 47] goes in, the model should predict 56. When [18, 47, 56] goes in, the model should predict 57 and so on!

```markdown
When input is tensor([18]), the target is: 47
When input is tensor([18, 47]), the target is: 56
When input is tensor([18, 47, 56]), the target is: 57
When input is tensor([18, 47, 56, 57]), the target is: 58
When input is tensor([18, 47, 56, 57, 58]), the target is: 1
When input is tensor([18, 47, 56, 57, 58,  1]), the target is: 15
When input is tensor([18, 47, 56, 57, 58,  1, 15]), the target is: 47
When input is tensor([18, 47, 56, 57, 58,  1, 15, 47]), the target is: 58

```
This approach allows the transformer to see input sequences ranging from a single character up to the full block size.

 ## Batching
We also generate batches of input because we cannot feed the entire dataset at once. Here’s a function to get a batch:

```python
batch_size = 4 # how many independent sequence will be process in parallel?
block_size = 8 # what is the max context length for predictions?

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i+ block_size + 1] for i in ix])
    return x, y

inputs:
torch.Size([4, 8])
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
targets:
torch.Size([4, 8])
tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])
```
```markdown
When input is: tensor([24]), target is: 43
When input is: tensor([24, 43]), target is: 58
When input is: tensor([24, 43, 58]), target is: 5
When input is: tensor([24, 43, 58,  5]), target is: 57
When input is: tensor([24, 43, 58,  5, 57]), target is: 1
When input is: tensor([24, 43, 58,  5, 57,  1]), target is: 46
When input is: tensor([24, 43, 58,  5, 57,  1, 46]), target is: 43
When input is: tensor([24, 43, 58,  5, 57,  1, 46, 43]), target is: 39
When input is: tensor([44]), target is: 53
When input is: tensor([44, 53]), target is: 56
When input is: tensor([44, 53, 56]), target is: 1
When input is: tensor([44, 53, 56,  1]), target ....
```

## Bigram Language Model
Now that the input data is ready we can start building the language model. The model we gonna build is called Bigram and in the following we will discuss details of the model.
The only layer the model has for now is just an embedding layer which is a look up table with the size of our vocabulary and since in Tiny-Shakespear we have 65 characters, the vovab size is 65. The model in the forward pass takes the input which is of the size of (batch_size , block_size) or (B, T) and it will look into the embedding table and for each numebr takes that row and print it out as an output. So the output is of the size of (Batch_size* Block_size , Vocab_size) or (B, T, C). For the loss we use cross entropy loss.

```
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Takes input idx os shape B*T and returns a shape of B*T*vocab_size or we say (B*T*C)
        logits = self.token_embedding_table(idx)
        # For the cross entropy loss we shpuld first change the shape from (B*T*C) to (C*T*B)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B*T) array
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim= -1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the current sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
```

The generation block takes the input sequence for each batch so the size is (B*T), then do the forward pass to obtain the logits. Now in order to obtain the next character we should compute the probability of next char by looking at the final character in the sequence. Therefore for the logits with dimension of (B,T,C), we focus on the last time step and therefore reduce the size to (B, C) and then we sample from the computed probs to guess what is the next character and we add it to our current sequence. 

Now lets see what our model can generate before any training: 

```
idx = torch.zeros((1,1), dtype=torch.long)
idx_seq = m.generate(idx, max_new_tokens=100)[0].tolist()
print(idx_seq)
print(decode(idx_seq))

output:
SKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHjgChzbQ?u!3bLIgwevmyFJGUGp
wnYWmnxKWWev-tDqXErVKLgJ
```

Then we use an optimizer and perform a training and bring the loss down from 4.7 to 2.4, and try to generate again. here is the houtput: 

```
FRI tiriddirtuce m, s nthy, es?
Ances my:
Lol is,
Faurqu by ot, omyoveanouree an, y NCERENG wicos ury,'s ts yofooowe,
AR: lleris mmeanse y ht?

RUK:
NICIUMy pen my hossoond:
IUS:
IXI per tow,
I hath tund me, y Y: metosishyoco wit wo y me ald t s mpithelveigne.
LYe EOu avemecedernildoreig
WI burn stche bye d or.
```

It looks better, it is still non-sense but defenitely had changed from our first trial a lot. **But the tokens are not talking to each other and we are looking only at the last token to generate new ones.** Here we are going to start talking about transformers where tokens will talk to each other.**


# The mathematical trick in self-attention

The token in nth location should not talk to the tokens comeing after that, it only should talk to the tokens before. So the information flows from the previous times. One way is to get the AVG information from the past tokens the AVG is not a good way but it can be okay for now. If we want to implement the attention in a for loop, it will look like this:

```
xbow = torch.zeros(B,T,C)
for b in range(B):
    for t in range(T):
        x_prev = x[b, : t+1] # (t, C)
        # print(x_prev.shape)
        xbow[b, t] = torch.mean(x[b, : t+1], 0)
```
however we can use matrix multiplication to implement this for loop. 

```
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
```

And there is yet another way that we can do this using softmax which is equal o what we did before:

```
tril = torch.tril(torch.ones((T,T)))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
```

# Self Attention
So far with this matrix multiplication, we are doing a simple average over all the previous tokens. But we dont want it to be like this, different tokens might get info from specific tokens. In this method, each token will have a query, key and a value matrix. Then we perform a dot product between the key of one token with respect to the query of all the other tokens.

Query vector roughly speaking is : What am I looking for!

Key vector roughly speaking is: What do I contain!

value vector roughly speaking is : What I will communicate to you (in this specific head)!

When we do a dot product, then dot product will be come the "wei" matrix. Now if the query and key dot product results in a high value, then it means that those two tokens are attending to each other.


```
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn( B, T, C)

# Lets see a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) #(B, T, 16) 
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones((T,T)))
# wei = torch.zeros((T, T)) So instead of this, wei is coming from product of k and q
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ x
```

and therefore
```
wei:
tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
         [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
         [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
         [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1687, 0.8313, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2477, 0.0514, 0.7008, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4410, 0.0957, 0.3747, 0.0887, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0069, 0.0456, 0.0300, 0.7748, 0.1427, 0.0000, 0.0000, 0.0000],
         [0.0660, 0.0892, 0.0413, 0.6316, 0.1649, 0.0069, 0.0000, 0.0000],
         [0.0396, 0.2288, 0.0090, 0.2000, 0.2061, 0.1949, 0.1217, 0.0000],
         [0.3650, 0.0474, 0.0767, 0.0293, 0.3084, 0.0784, 0.0455, 0.0493]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4820, 0.5180, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1705, 0.4550, 0.3745, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0074, 0.7444, 0.0477, 0.2005, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.8359, 0.0416, 0.0525, 0.0580, 0.0119, 0.0000, 0.0000, 0.0000],
         [0.1195, 0.2061, 0.1019, 0.1153, 0.1814, 0.2758, 0.0000, 0.0000],
         [0.0065, 0.0589, 0.0372, 0.3063, 0.1325, 0.3209, 0.1378, 0.0000],
         [0.1416, 0.1519, 0.0384, 0.1643, 0.1207, 0.1254, 0.0169, 0.2408]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.6369, 0.3631, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2586, 0.7376, 0.0038, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4692, 0.3440, 0.1237, 0.0631, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1865, 0.4680, 0.0353, 0.1854, 0.1248, 0.0000, 0.0000, 0.0000],
         [0.0828, 0.7479, 0.0017, 0.0735, 0.0712, 0.0228, 0.0000, 0.0000],
         [0.0522, 0.0517, 0.0961, 0.0375, 0.1024, 0.5730, 0.0872, 0.0000],
         [0.0306, 0.2728, 0.0333, 0.1409, 0.1414, 0.0582, 0.0825, 0.2402]]],
```

So wei is now different for every batch and therefore it is data dependent. What does this suggesting is as the following. For example look at the first wei matrix on the top. on the 8th row, we have 0.2297 for the token 4 and 0.2391 for token 8th. This means that they have a high affinity and when we multiply x by this matrix, then those two tokens get more information from each other instead of receiving avg info from all the token. in that specific channel. 

So wei is now different for every batch and therefore it is data dependent. there is one more thing about self attention. We do not multiply x by wei but instead we multiply somehting called value. so we have another matrix v similar to q and k. and we compute wei @ v instead of wei @ x

```
value = nn.Linear(C, head_size, bias=False)
v = value(x)
out = wei @ v
```


**Where does name self-attention** is coming from? since key, query and value matrices are all generated using x, it is called self-attention.

one more thing we need to do before finishing the self-attention is to divide it by the square root of the head_size like it is explained in the original paper called "Attention is all you need". If we dont do that then when we multiply k and q, numbers become large and when we apply softmax on those numbers the representation becomes like one-hot vectors instead of being diffuse numbers. and what does that mean, it means that each token will receive attention from only one other token instead of receiving info from other as well in a more diffucsive way.

one more thing we need to do before finishing the self-attention is to divide it by the square root of the head_size like it is explained in the original paper called "Attention is all you need". If we dont do that then when we multiply k and q, numbers become large and when we apply softmax on those numbers the representation becomes like one-hot vectors instead of being diffuse numbers. and what does that mean, it means that each token will receive attention from only one other token instead of receiving info from other as well in a more diffucsive way.

```
#therefore:
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```

After implementing the single-head attention layer to the model and train the model this is one example of output:
```
K:
NGey

Letnrad wineam:
Kicou hitipteavimancraby whet muthe hus darge.

Wind!
IRD: Ind, tind spoof om and f.
Sy stllalevere here me honouen fot in,
So and, vist orby?
Thar hous mat deest she rd?

Wowin wof t, ath th ay miligiryouchth-orto mou tenges, ald pors banebe y prothetack aklel I veriplansnidierd avit for,
KI thit ndist allll perd the:
Acu Empoouthant, I to
Ten mar.

S:
Bugh the I hy nd meis moh h!


AThamen es ty I has.

MI ithe thensterat blo gaar,
A d muts ed ronur wiend tl-ou,
Therim
```

So it looks better but still we have a long way to improve this.
Instead of only one-attnetion head we can have multiple attention heads and when we do this and run the model again, the loss become lower which shows the model is improving and the output looks like this:

```
ITIS
Way'm hat no It bot dich hose, onowea pavish;
I tand tpes; he to IOnd rius dick chatthtamill vil and all of nernd ing, gold?
Wour
Ast; hell thore it gest nosin.

WYo lour sow gone iks jepry lo em, Arow trowimad foreme wit no the
Whours, serveardy'e wide huightalgom I st; bye
Anerwetery into overe my yous le atilpjack thean so if hougirse youne TICANG PEveemaks to bre hand Th's of. Werxever! pargyfrooroke me Owa ther do you isty 'lid Mevesesve thich han doieanty moa pwar sind wow sad aly,.
```

Which is still not good. the thing is that we have collected info from all other tokens BUT the model did not have enough time to process that data. When went directly to create logits and outputs. So here is the place that we can add more computations to the model.
For that we add some linear layer and then some non-linearity.

```
class FeedFroward(nn.Module):
    """ a simple linear layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.net(x)
```

# The communication and computation blocks in transformer
We create another class which implements feedforward layers and multiheaded attentions blocks and then we can apply that multiple times to have a deeper network. 

```
class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #communication
        self.ffwd = FeedForward(n_embd) #computation

    def forward(self,x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
```

However the results did not improve much:

```
BOSGGETR:
Tae yo mate figcth bikles minl was ththen;
Bev mash neeti sheru,
Noath,
Tinge onnd 'rsates odary neat I paneushane:
Tow iy nthoke fouds hede temmte wo shoac at thum my sit weoa watl, thuere?
Neut uneth ce whe

AOrusthoashct mis walslefisesh at, norl:
Tat dhou biped, at lan bowe; lrid ird eoaI.

HLESINTACDCFT:
Te,ndne
Wit Rlaen pasl'-to mimty Hom,
And in wav.

Sikdsp ondte
Mler Relobe'?
Niy do math thas nellal Cived,
Qit mly san Aye thore it we ir it thub threr:
Nate thoat yonttenst'r a
```

# Residual Block
There is yet another block that we can add to the model to make the results better. it is called residual block. so we change out block implemenation as below:
```
class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #communication
        self.ffwd = FeedForward(n_embd) #computation

    def forward(self,x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
```

We add Layernorm layers and dropouts to the model in order to make the model better and at the end, we are able to get something which we think is good enough taking into account our input and the task that we have. Also pay attention that our model is a letter generation and not word or larger than letter tokens. 
This model that we implemented here, is actually an encoder only model like chatgpt. We dont need an encoder to encode the text and then generate output from it. 

```
Where the house unfold these lawful blame.

KING RICHARD III:
Go, Grood Warwick's our ancest; and,
Who nowoful light England's roye, and thereof,
With Bolingbrod and Deck's Xan of Walion,
Hath of Signion Buckingham's shall lie.

WARWICK:
Rightor Northumberland, Ely weep; cobscept thy fast;
Which, he lodg me no found, tooking in the them;
And that will practish that for Richard warm
And all his spreading with Angelo,
For back, I hay given me hence'll so,
Is he maked their charter: which I prizent
```

