# RAG (Retrieval Augmented Generation) with local models

![header](/docs/header.png?raw=true "header")

This repo wants to be an easy way to showcase how to implement RAG (Retrieval Augemented Generation) with only local models (no need for OpenAI api keys).
For this example we leverage MongoDB [Atlas Search](https://www.mongodb.com/docs/atlas/atlas-search/) as Vector Store, [Hugging Face](https://huggingface.co/) transformers to compute the embeddings, [Langchain](https://python.langchain.com/docs/get_started/introduction) as LLM framework and a quantized version of Llama2-7B from [TheBloke's](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF).

> **Note**
> You are free to experiment with different models, the main goal of this work was to get something that can run entirely on CPU with the least amount of memory possible. Bigger models will give you way better results.

<a id="AtlasCluster"></a>

## Load sample data

In this example we are using the `sample_mflix.movies` collection, part of the sample dataset available in Atlas.
Once you have uploaded the sample dataset in you Atlas Cluster, you can run encoder.py to compute vectors out of your MongoDB documents.

```console
python3 encoder.py
```

## Download the quantized Llama2 LLM locally from Huggingface

Download the model from hugging face by following this [documentation](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#how-to-download-gguf-files)

## Define Atlas Search Index

Create the following search index on the `sample_mflix.movies` collection:

```json
{
  "mappings": {
    "fields": {
      "vector": [
        {
          "dimensions": 384,
          "similarity": "cosine",
          "type": "knnVector"
        }
      ]
    }
  }
}
```

<a id="test1"></a>

## Run the streamlit app to query the LLM

Run the streamlit app with `streamlit run main.py`

We can then ask the LLM something like:

> give me some movies about racing
