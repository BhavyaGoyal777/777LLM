#!/usr/bin/env python
# coding: utf-8

# In[39]:


get_ipython().system('pip install torch')


# In[40]:


import urllib.request 
url=("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)


# In[41]:


with open ("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
print("Total number of character:",len(raw_text))
print(raw_text[:99])    


# In[42]:


import re 
text="HELLO, I AM BHAVYA GOYAL !"
result=re.split(r'(\s)',text)
result


# In[43]:


result = re.split(r'([,.]|\s)', text)
print(result)


# In[44]:


result=[item for item in result if item.strip()]
result


# In[45]:


text="hello my name is bhavya goyal I am intested in deep generative models "
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)


# In[46]:


preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed=[item for item in preprocessed if item.strip()]
print(len(preprocessed))
len(set(preprocessed))


# In[47]:


import torch

if torch.backends.mps.is_available():
    print("MPS is available!")
else:
    print("MPS is not available.")


# In[48]:


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


# In[49]:


all_words=sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)


# In[50]:


vocab={token:integer for integer,token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    print(item)
    if i>50:
        break


# In[51]:


import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self._int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        # Using 'text' instead of 'raw_text'
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        # Using '_int_to_str' instead of 'int_to_str'
        text = " ".join([self._int_to_str[i] for i in ids])
        print(text)
        text = re.sub(r'\s+([,.?!"()\'\\])', r'\1', text)
        return text


# In[52]:


tokenizer=SimpleTokenizerV1(vocab)


# In[53]:


tokenizer.encode("my,is ")


# In[54]:


tokenizer.decode([697, 5,584])


# In[55]:


tokenizer.encode("hi my name is bhavya ")


# In[56]:


all_tokens=sorted(list(set(preprocessed)))
all_tokens.extend(['<|unk|>','<|endoftext|>'])
vocab={token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))


# In[57]:


for i , item in enumerate(list(vocab.items())[-5:]):
    print(item)


# In[62]:


import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!()\'"])', r'\1', text)
        return text



# In[66]:


test="hi my name is bhavya goyal my biggest regret is pursuing btech"
tokenizer=SimpleTokenizerV2(vocab)
print(tokenizer.encode(test))
print(tokenizer.decode(tokenizer.encode(test)))


# In[79]:


text1="My name is BHAVYA GOYAL I am currently a sophomore."
text2="My dog's name is BRUNO."
concatenated_text=" <|endoftext|> ".join([text1,text2])
print(concatenated_text)


# In[80]:


print(tokenizer.encode(concatenated_text))


# In[81]:


print(tokenizer.decode(tokenizer.encode(concatenated_text)))


# In[82]:


get_ipython().system('pip install tiktoken')


# In[83]:


from importlib.metadata import version
import tiktoken
print('TIKTOKEN VERISON:',version('tiktoken'))


# In[87]:


bpe_tokenizer=tiktoken.get_encoding('gpt2')


# In[92]:


text1="My name is BHAVYA GOYAL I am currently a sophomore."
text2="My dog's name is BRUNO."
concatenated_text=" <|endoftext|> ".join([text1,text2])
print(bpe_tokenizer.encode(concatenated_text,allowed_special={"<|endoftext|>"}))
print(tokenizer.decode(bpe_tokenizer.encode(concatenated_text,allowed_special={"<|endoftext|>"})))


# In[95]:


with open('the-verdict.txt','r') as f:
    content=f.read()
encoded_text=tokenizer.encode(content)  
print(len(encoded_text))


# In[97]:


print(tokenizer.decode(encoded_text[:10]))


# In[108]:


context_size=5
x=encoded_text[:5]
y=encoded_text[1:6]
print(f'input to the model:{x}')
print(f'labels for the model: {y}')
print(tokenizer.decode(encoded_text[:5]))
print(tokenizer.decode(encoded_text[1:6]))




# In[123]:


for i in range(1,context_size+1):
    context=encoded_text[:i]
    desired=encoded_text[i]
    print(f'{context}---->{desired}')


# In[126]:


for i in range(1,context_size+1):
    context=encoded_text[:i]
    
    
    desired=encoded_text[i]
    
    print(f'{bpe_tokenizer.decode(context)}---->{bpe_tokenizer.decode([desired])}')


# In[133]:


import torch
from torch.utils.data import DataLoader,Dataset
class GPTDatasetV1(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        # super().__init_()
        self.input_ids=[]
        self.target_ids=[]

        token_ids=bpe_tokenizer.encode(text)


        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]
            





# In[136]:


def create_dataloader_v1(txt,batch_size=4,max_length=267,
                         stride=128,shuffle=True,drop_last=True,
                         num_workers=0):
    bpe_tokenizer=tiktoken.get_encoding('gpt2')
    dataset=GPTDatasetV1(txt,bpe_tokenizer,max_length,stride)
    dataloader=DataLoader(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    
    return dataloader


# In[150]:


torch.manual_seed(2000)
with open("the-verdict.txt",'r',encoding='utf-8') as f:
    raw_text=f.read()
dataloader=create_dataloader_v1(raw_text,batch_size=4,max_length=4
                                ,stride=3
                                ,shuffle=False)

data_iter=iter(dataloader)

first_batch=next(data_iter)
print(first_batch)


# In[155]:


torch.manual_seed(2000)
with open("the-verdict.txt",'r',encoding='utf-8') as f:
    raw_text=f.read()
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=4
                                ,stride=4
                                ,shuffle=False)

data_iter=iter(dataloader)

input_batch,target_batch=next(data_iter)
print(f'input batch----->\n{input_batch}')
print(f'targer batch----->\n{target_batch}')


# In[156]:


input_ids=torch.tensor([0,1,2,3])


# In[157]:


vocab_size=6
output_dim=3


# In[158]:


torch.manual_seed(100)
embedding_layer=torch.nn.Embedding(vocab_size,output_dim)
print(embedding_layer.weight)


# 

# In[159]:


embedding_layer(input_ids)


# In[162]:


vocab_size=50257
output_dim=256
token_embedding_layer=torch.nn.Embedding(vocab_size,output_dim)
print(token_embedding_layer)


# In[167]:


max_lenth=4
dataloader=create_dataloader_v1(
    raw_text,batch_size=8,stride=max_lenth,max_length=max_lenth,
    shuffle=False
)
data_iter=iter(dataloader)
input_batch,target_batch=next(data_iter)
print(f'input batch:{input_batch.shape}')


# In[170]:


token_embeddings=token_embedding_layer(input_batch)
token_embeddings.shape


# In[184]:


pos_embedding=torch.nn.Embedding(max_lenth,output_dim)
pos_embedding=pos_embedding(torch.arange(0,max_lenth))
pos_embedding.shape


# In[186]:


input_embeddings=token_embeddings+pos_embedding
print(input_embeddings.shape)


# In[ ]:




