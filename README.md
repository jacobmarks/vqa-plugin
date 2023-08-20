## Visual Question Answering Plugin

This plugin is a Python plugin that allows you to answer visual questions about images in your dataset!

It demonstrates how to do the following:

- Run a local model on your dataset
- Use `ctx.selected` to operate on only selected samples
- Pass data from `execute()` to `resolve_output()`

### Supported Models

This version of the plugin supports the default Vision Language Transformer (ViLT) model used in the [Visual Question Answering pipeline](https://huggingface.co/tasks/visual-question-answering) in the Hugging Face transformers library.

Feel free to fork this plugin and add support for other models!

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/vqa-plugin
```

Install the Hugging Face transformers library:

```shell
pip install transformers
```

## Operators

### `answer_visual_question`

- Generates an image from a text prompt and adds it to the dataset
