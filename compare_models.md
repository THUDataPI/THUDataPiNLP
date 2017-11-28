## 一条命令实现情感分析

  网上有人使用一行代码“实现”人脸识别，听起来很有吸引力。估且不论它是否可以被称做“实现”，对于想尝试一下的同学来说，这无疑是最简单的一种方式。事实上，大多数机器学习模型都可以用一个命令直接调用，连一行代码都不用写。
  
  废话不多说，先来个例子：
  
```shell
curl -X POST -d '{
  "document": "I really like Algorithmia!"
}' -H 'Content-Type: application/json' -H 'Authorization: Simple your_api_key_here https://api.algorithmia.com/v1/algo/nlp/SentimentAnalysis/1.0.4
```
  输入一句话"I really like Algorithmia!"，返回就是它的情感分析得分。我测试的结果是0.474，这个分数可能会变化。运行时需要把`your_api_key`替换为自己的API_KEY。`API_KEY`可以自己去网站上[Algorithmia](http://algorithmia.com) 上注册。
  
  常见的机器学习模型都可以在这里找得到，有免费调用的，也有收费的。有了这个模型库，许多事情都变得非常方便。比如面对一个问题，不知道选择哪个模型好，那就用测试数据在多个模型上跑一下，结果自然就清楚了。或者自己写出来的模型效果不理想，不确定实现的对不对，那就与模型库里的版本比较一下。 

## 中文情感分析

  目前模型库里的情感分析只支持英文，我自己实现一个针对中文情感分析接口，使用方法如下：
```shell
  curl -X POST -d '{
  "document": "这个电影一点也不好看"
}' -H 'Content-Type: application/json' -H 'Authorization: Simple your_api_key_here https://api.algorithmia.com/v1/algo/threefoldo/ChineseSentimentAnalysis/0.3.0
```

  返回内容如下：
```shell
  {
    "result":
      {
        "sentiment":"negative"
      },
    "metadata":
     {
       "content_type":"json",
       "duration":0.009943533
     }
  }
```
  简单的模型只能区分"positive"和"negative"，比较复杂的模型会返回一个概率值。
  
### 选择不同的情感分析模型

  目前只实现了一个最基本的词典查找，后续准备添加多种实现，包括贝叶期分类器、KNN，以及LSTM。接口将保持不变，通过"model"来选择模型。
  
  没有指定模型时，会使用词典查找法，统计正面词汇和负面词汇出现的次数，结合否定词、副词做一定的调整，计算出一个数值。大于0则为正面，小于0为负面。这种简单的模型虽然准确率低些，但是适用范围广，可以作为测试的基线。更详细的介绍参考这里：[文本情感分类（一）：传统模型](http://kexue.fm/archives/3360/)

  
### LSTM模型

#### 调用方式

```shell
  curl -X POST -d '{
  "model": "lstm",
  "document": "这个电影一点也不好看"
}' -H 'Content-Type: application/json' -H 'Authorization: Simple your_api_key_here https://api.algorithmia.com/v1/algo/threefoldo/ChineseSentimentAnalysis/0.3.0
```

#### 模型结构

  这是个非常简单的模型，embedding + LSTM，先生成字向量，再用LSTM生成句子向量。在网上有大量的文章介绍它，这里就不再重复。
  
  模型的输入是句子，输出是标签。用Keras实现只需要几行代码：
  
```python
    model = Sequential()
    model.add(Embedding(len(data['chars']), 256, input_length=maxlen))
    model.add(LSTM(128)) 
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
```

  训练LSTM需要人工标注的句子，不再需要词典。这在实际应用中是个好事，因为网络上很容易找到带有正负面标签的评论。
  
  另外，虽然这个模型的实现很简单，但很难通过算法上的优化进一步提升准确率。不服气的可以尝试一下，如果能有5%的提升，那就很厉害了。在算法上花时间，远不如在训练数据花时间收益大。一般来说，训练数据与实际数据越接近，最终的效果就越好。

#### 组合使用

  有多个模型除了方便比较外，还有一个好处就是组合使用。很少存在一个模型在所有的情况下都是最优的。简单的模型虽然准确率低，但往往速度快。对响应时间要求高的情况，可以使用简单模型。对响应时间不敏感的情况，则调用复杂模型。
  
