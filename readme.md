# SmallLaMo

Implements a
Small Language Model for Efficient Inference.

Use a existing Java Bibliothek to implement a small language model that can be trained on a small set of characters.

Tokens are characters build by 8 numbers representing the UTF-8 Codes.

The Model is trained by reading text and predict the next character in the text.
We start with some simple:
ae aa ea aa in random Order.

The model has only 8 Inputs and 8 Outputs.
The context is stored internally by allowing backward connections in the network to build an internal context-memory.

If this is working well, we can add more characters to the training set:
aeiou

