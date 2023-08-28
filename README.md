## Visual Question Answering Plugin

This plugin is a Python plugin that allows you to answer visual questions about
images in your dataset!

It demonstrates how to do the following:

- Run a local model on your dataset
- Use `ctx.selected` to operate on only selected samples
- Pass data from `execute()` to `resolve_output()`

### Supported Models

This version of the plugin supports the default Vision Language Transformer
(ViLT) model used in the [Visual Question Answering pipeline](https://huggingface.co/tasks/visual-question-answering)
in the Hugging Face transformers library, as well as Salesforce's
[BLIPv2](https://replicate.com/andreasjansson/blip-2) via
[Replicate](https://replicate.com/).

Feel free to fork this plugin and add support for other models!

## Installation

### Pre-requisites

1. If you plan to use it, install the Hugging Face transformers library:

```shell
pip install transformers
```

2. If you plan to use it, install the Replicate library:

```shell
pip install replicate
```

And add your Replicate API key to your environment:

```shell
export REPLICATE_API_TOKEN=<your-api-token>
```

### Install the plugin

```shell
fiftyone plugins download https://github.com/jacobmarks/vqa-plugin
```

## Operators

### `answer_visual_question`

- Applies the selected visual question answering model to the selected sample in
  your dataset and outputs the answer.
