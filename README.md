# 10kGNAD Classification
In this repository I worked on a Text Classification task using the [Ten Thousand German News Articles Dataset](https://tblock.github.io/10kGNAD/).

## Dataset visualization
First, I have done some data exploration and visualization to better understand the dataset and its different classes.
This step is done in a Jupyter Notebook: [visualization.ipynb](./visualization.ipynb)

## Model training
I first chose the [german BERT pretrained model by Deepset.ai](https://www.deepset.ai/german-bert) on HuggingFace hub and trained it on a subset of the training set (splitted as 90/10 with validation). I performed the fine-tuning on Colab using Google GPUs, you can find the details in the Jupyter Notebook: [training.ipynb](./training.ipynb). Then I pushed the fine-tuned model on the [Hugging Face Model Hub](https://huggingface.co/Mathking/bert-base-german-cased-gnad10).

## Model evaluation
I performed the evaluation using accuracy, precision, recall and f1-score and plotted the classification results in a confusion matrix. The results of this step are done in the [evaluation.ipynb Jupyter notebook](./evaluation.ipynb).
The model achieved 90.6% accuracy on the test set (as splitted by the authors of the dataset).

## Demo Application
I implemented a Gradio demo app (in the app folder) and pushed it on the brand new Hugging Face Spaces hub, you can [try the demo here](https://huggingface.co/spaces/Mathking/german-news-classification). The requirements.txt file contains only the libraries needed for the demo app.

There is also an option to build this app with Docker. To build the image you must start from the [app directory](./app/)
```
cd app
docker build --tag gradio-german-news-classify .
```
Then you can run the image locally and access it on http://localhost:7860/:
```
docker run --publish 7860:7860 gradio-german-news-classify
```
