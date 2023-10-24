## Visual Question Answering Plugin

![vqa_updated](https://github.com/jacobmarks/vqa-plugin/assets/12500356/67819454-19e3-4b4a-861f-afed465f4866)

### Updates

- **2023-10-24**: Added support for Llava-13b and Fuyu-8b models from
  [Replicate](https://replicate.com/).

## Plugin Overview

This plugin is a Python plugin that allows you to answer visual questions about
images in your dataset!

It demonstrates how to do the following:

- Run a local model on your dataset
- Use `ctx.selected` to operate on only selected samples
- Pass data from `execute()` to `resolve_output()`

### Supported Models

This version of the plugin supports the following models:

- [Fuyu-8b](https://replicate.com/lucataco/fuyu-8b/) from Adept AI (via [Replicate](https://replicate.com/))
- [Llava-13b](https://replicate.com/yorickvp/llava-13b) (via [Replicate](https://replicate.com/))
- [ViLT](https://huggingface.co/transformers/model_doc/vilt.html) (default Vision Language Transformer used in the [Visual Question Answering pipeline](https://huggingface.co/tasks/visual-question-answering))
- [BLIPv2](https://replicate.com/andreasjansson/blip-2) (via [Replicate](https://replicate.com/))

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

## Usage

To use this plugin, you must have exactly one sample selected in your dataset.
Here are the messages you will see when you have zero, one, or more than one
sample selected:

![vqa_selected](https://github.com/jacobmarks/vqa-plugin/assets/12500356/73b1f2c6-eedd-4534-85c6-df1349ec6c58)
