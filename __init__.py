"""VQA plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
import os
from PIL import Image

import fiftyone as fo
import fiftyone.core.utils as fou
import fiftyone.operators as foo
from fiftyone.operators import types


transformers = fou.lazy_import("transformers")
replicate = fou.lazy_import("replicate")


def allows_replicate_models():
    """Returns whether the current environment allows replicate models."""
    return (
        find_spec("replicate") is not None
        and "REPLICATE_API_TOKEN" in os.environ
    )


def allows_hf_models():
    """
    Returns whether the current environment allows hugging face transformer
    models.
    """
    return find_spec("transformers") is not None


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )

class VQAModel:
    """Wrapper around a VQA model."""

    def __init__(self):
        pass

    def __call__(self, sample, question):
        pass

class Idefics2_8bVQAModel(VQAModel):
    """Wrapper around Replicate Idefics2-8b model."""

    def __call__(self, sample):
        filepath = get_filepath(sample)
        question = get_question(sample)
        response = replicate.run(
            "lucataco/idefics-8b:7ab312514f213130c4a2db68b93a1719f5cc7c3246c408ba91d507b212a24303",
            input={"image": open(filepath, "rb"), "prompt": question},
        )
        return response.strip()

class ViLTVQAModel(VQAModel):
    """Wrapper around a Hugging Face ViLT VQA model."""

    def __call__(self, sample, question):
        filepath = get_filepath(sample)
        image = Image.open(filepath)
        from transformers import pipeline

        vqa_pipeline = pipeline("visual-question-answering")
        response = vqa_pipeline(image, question, top_k=1)
        return response[0]["answer"]


class BLIP2VQAModel(VQAModel):
    """Wrapper around Replicate BLIP2 VQA model."""

    def __call__(self, sample, question):
        filepath = get_filepath(sample)
        response = replicate.run(
            "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
            input={"image": open(filepath, "rb"), "question": question},
        )
        return response


class Fuyu8bVQAModel(VQAModel):
    """Wrapper around Replicate Fuyu8b model."""

    def __call__(self, sample, question):
        filepath = get_filepath(sample)
        response = replicate.run(
            "lucataco/fuyu-8b:42f23bc876570a46f5a90737086fbc4c3f79dd11753a28eaa39544dd391815e9",
            input={"image": open(filepath, "rb"), "prompt": question},
        )
        return response.lstrip()


class Llava13bVQAModel(VQAModel):
    """Wrapper around Replicate Llava13b model."""

    def __call__(self, sample, question):
        filepath = get_filepath(sample)
        response = replicate.run(
            "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
            input={"image": open(filepath, "rb"), "prompt": question},
        )

        resp_string = ""
        for r in response:
            resp_string += r
        return resp_string


MODEL_MAPPING = {
    "vilt": ViLTVQAModel,
    "blip2": BLIP2VQAModel,
    "fuyu": Fuyu8bVQAModel,
    "llava": Llava13bVQAModel,
    "idefics2-8b": Idefics2_8bVQAModel,
}


def _get_vqa_model(model_name):
    if model_name in MODEL_MAPPING:
        return MODEL_MAPPING[model_name]()

    raise ValueError(f"Model {model_name} not found.")


def run_vqa(sample, question, model_name):
    model = _get_vqa_model(model_name)
    return model(sample, question)


def _add_replicate_models(model_choices):
    if allows_replicate_models():
        model_choices.add_choice("blip2", label="BLIP2")
        model_choices.add_choice("fuyu", label="Fuyu8b")
        model_choices.add_choice("llava", label="Llava13b")
        model_choices.add_choice("idefics2-8b", label="Idefics2-8b")


def _add_hf_models(model_choices):
    if allows_hf_models():
        model_choices.add_choice("vilt", label="ViLT")


class VQA(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="answer_visual_question",
            label="VQA: Answer question about selected image",
            dynamic=True,
        )
        _config.icon = "/assets/question_icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="VQA", description="Ask a question about the selected image!"
        )

        rep_flag = allows_replicate_models()
        hf_flag = allows_hf_models()
        if not rep_flag and not hf_flag:
            inputs.message(
                "message",
                label="No models available. Please set up your environment variables.",
            )
            return types.Property(inputs)

        model_choices = types.RadioGroup()
        _add_replicate_models(model_choices)
        _add_hf_models(model_choices)

        inputs.enum(
            "model_name",
            model_choices.values(),
            label="Model",
            view=model_choices,
            required=True,
        )

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label=f"You must select a sample to use this operator"
                ),
            )
        elif num_selected > 1:
            inputs.str(
                "too_many_samples_warning",
                view=types.Warning(
                    label=f"You must select only one sample to use this operator"
                ),
            )

        else:
            inputs.str("question", label="Question", required=True)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        sample = ctx.dataset[ctx.selected[0]]
        question = ctx.params.get("question", "None provided")
        model_name = ctx.params.get("model_name", "blip2")
        answer = run_vqa(sample, question, model_name)

        return {"question": question, "answer": answer}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("question", label="Question")
        outputs.str("answer", label="Answer")
        header = "Visual Question Answering!"
        return types.Property(outputs, view=types.View(label=header))


def register(plugin):
    plugin.register(VQA)
