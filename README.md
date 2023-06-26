# CalcGPT
Calculator from GPT 

**Method for Encoding Strings**

The encoding method used in the project is byte pair encoding (BPE). BPE is a subword tokenization algorithm that breaks down rare or unknown words into subwords that are already known. After tokenization, the input is converted into a high-dimensional vector representation. Below are the encodings for the input prompts used in the project:

**Baseline:**
The baseline input is a simple summation of two numbers: "1+2 =". It is tokenized into ['1', '+', '2', '='].

**Few-Shot Learning:**
For few-shot learning, the model is trained with a limited number of labeled examples. In this project, two labeled examples are provided to the model: "1+2 = 3" and "3+2 = 5". The tokenization of these examples is shown in the input.

**Method for Generating Text**

The language model used is "EleutherAI/gpt-neo-1.3B," which has over 1.3 billion parameters. It is trained as a masked autoregressive model with cross-entropy loss. The model takes a string input and predicts the next token.

The following parameters are used for text generation:
- `do_sample`: True, which ensures that the next token is selected based on assigned probabilities.
- `temperature`: 0.1, 0.01, 0.001 (experiments conducted with different temperature values).
- `max_new_tokens`: 30, representing the maximum number of tokens in the output sequence without including the tokens in the prompt.
- `min_new_tokens`: 10, representing the minimum number of new tokens in the output sequence.

The `attention_mask` is used to specify which tokens the model should attend to instead of considering all tokens in the input, including padded tokens. A batch size of 25 is used to speed up the text generation process.

**Method for Decoding Strings**

Decoding is the process of converting the generated output strings into meaningful representations. The output for the baseline input ("1+0 =") shows incoherent generated text. These output strings are decoded to obtain an integer value.

For the few-shot learning prompts, the decoded output strings are clipped to remove any additional tokens beyond the expected format (e.g., "0, 1+1=1, 1+2=2, ...").

**Results**

**Baseline:**
The baseline model, with a simple input like "1+2 =", performed poorly. Out of the 2500 inputs, which included number combinations ranging from 0 to 50, the model provided correct answers for only 25 input equations. The accuracy of the baseline model was around 1%.

**Few-Shot Learning:**
In contrast, few-shot learning, which involved providing two labeled examples in the prompt, improved the model's performance. With the parameter values mentioned above, the model achieved an accuracy of over 9.88% compared to the 1% accuracy of the baseline model.

**K-Mean Clustering Algorithm:**
To analyze the results, the K-means clustering algorithm was used. This classifier achieved an 82% accuracy on the few-shot learning prompts. K-means clustering initializes k centroids, assigns each datapoint to the closest centroid based on the Euclidean distance, and clusters correct and incorrectly predicted values. It was found to be the best fit for the scatter plot.

**Experiments:**
- Different temperature values (0.1, 0.01, 0.001) were tested. The accuracy of the baseline model did not change significantly, but for few-shot learning, the accuracy ranged from 6.3% to 9.88%.
- Including more labeled examples in the input prompt

 did not significantly improve the accuracy. In fact, when including an input outside the range (0,50), the accuracy was only around 7.29%.
- Other arithmetic operations were experimented with, although the details are not provided.
- Fine-tuning of the gpt2 model was explored, achieving an accuracy of over 75.6% on the test dataset. The fine-tuned model showed a significant improvement compared to the non-fine-tuned version.

**Observations:**
- Without fine-tuning, the results were incoherent with "gpt2".
- The fine-tuned "gpt2" model performed better than "EleutherAI/gpt-neo-1.3B" for the provided dataset, but struggled with data points outside the range (0,50).
- It is believed that the limited size of the dataset used for fine-tuning (1600 labeled data points) could be a reason for the model's performance limitations on out-of-range data.
- Fine-tuning "EleutherAI/gpt-neo-1.3B" with a larger dataset is expected to yield better results.
- The concentration of correct answers near x1 or x2 = 0 in few-shot learning prompts is noted, but the reason behind this observation and the model's learning process are not fully understood.

**AI Collaborators and References:**
- The OpenAI chat (https://chat.openai.com/) was used for doubts, documentation, and code validation.
- A YouTube video (https://www.youtube.com/watch?v=elUCn_TFdQc&t=1458s) and an article on fine-tuning GPT-2 for text generation (https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272) were referenced.
- A Google Colab notebook (https://colab.research.google.com/drive/1KlfIFHGj7crizwDMiEqUcj4cLPw1XwW9?usp=sharing) was used for fine-tuning the gpt2 model.
