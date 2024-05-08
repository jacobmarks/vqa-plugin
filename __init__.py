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

HF_MODELS = {"vilt": "ViLT", "moondream2": "Moondream2"}

REPLICATE_MODELS = {
    "blip2": "BLIP2",
    "fuyu": "Fuyu8b",
    "idefics2-8b": "Idefics2-8b",
    "llava": "Llava13b",
    "moondream2": "Moondream2",
}

DEFAULT_MODEL_NAME = "llava"


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

    def __call__(self, sample, question, ctx=None):
        pass


class Idefics2_8bVQAModel(VQAModel):
    """Wrapper around Replicate Idefics2-8b model."""

    def __call__(self, sample, question, ctx=None):
        filepath = get_filepath(sample)
        response = replicate.run(
            "lucataco/idefics-8b:7ab312514f213130c4a2db68b93a1719f5cc7c3246c408ba91d507b212a24303",
            input={"image": open(filepath, "rb"), "prompt": question},
        )
        return response.strip()


class ViLTVQAModel(VQAModel):
    """Wrapper around a Hugging Face ViLT VQA model."""

    def __call__(self, sample, question, ctx=None):
        filepath = get_filepath(sample)
        image = Image.open(filepath)
        from transformers import pipeline

        vqa_pipeline = pipeline("visual-question-answering")
        response = vqa_pipeline(image, question, top_k=1)
        return response[0]["answer"]


class BLIP2VQAModel(VQAModel):
    """Wrapper around Replicate BLIP2 VQA model."""

    def __call__(self, sample, question, ctx=None):
        filepath = get_filepath(sample)
        response = replicate.run(
            "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
            input={"image": open(filepath, "rb"), "question": question},
        )
        return response


class Fuyu8bVQAModel(VQAModel):
    """Wrapper around Replicate Fuyu8b model."""

    def __call__(self, sample, question, ctx=None):
        filepath = get_filepath(sample)
        response = replicate.run(
            "lucataco/fuyu-8b:42f23bc876570a46f5a90737086fbc4c3f79dd11753a28eaa39544dd391815e9",
            input={"image": open(filepath, "rb"), "prompt": question},
        )
        return response.lstrip()


class Moondream2Model(VQAModel):
    """Wrapper around Replicate Fuyu8b model."""

    def __call__(self, sample, question, ctx=None):
        default_distro = (
            "Replicate" if allows_replicate_models() else "Transformers"
        )
        distro = (
            ctx.params.get("model_distribution", default_distro)
            if ctx
            else default_distro
        )
        if distro == "Replicate":
            return self._call_replicate(sample, question)
        else:
            return self._call_transformers(sample, question)

    def _call_replicate(self, sample, question):
        filepath = get_filepath(sample)
        input = {
            "image": open(filepath, "rb"),
            "prompt": question,
        }

        output = replicate.run(
            "lucataco/moondream2:392a53ac3f36d630d2d07ce0e78142acaccc338d6caeeb8ca552fe5baca2781e",
            input=input,
        )
        return "".join(output)

    def _call_transformers(self, sample, question):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "vikhyatk/moondream2"
        revision = "2024-04-02"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        image = Image.open(get_filepath(sample))
        enc_image = model.encode_image(image)
        return model.answer_question(enc_image, question, tokenizer)


class Llava13bVQAModel(VQAModel):
    """Wrapper around Replicate Llava13b model."""

    def __call__(self, sample, question, ctx=None):
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
    "moondream2": Moondream2Model,
}


def _get_vqa_model(model_name):
    if model_name in MODEL_MAPPING:
        return MODEL_MAPPING[model_name]()

    raise ValueError(f"Model {model_name} not found.")


def run_vqa(sample, question, model_name, ctx=None):
    model = _get_vqa_model(model_name)
    return model(sample, question, ctx=ctx)


def _add_replicate_models(model_choices):
    if not allows_replicate_models():
        return
    for model_name, label in REPLICATE_MODELS.items():
        model_choices.add_choice(model_name, label=label)


def _add_hf_models(model_choices):
    if not allows_hf_models():
        return

    for model_name, label in HF_MODELS.items():
        if model_name not in model_choices.values():
            model_choices.add_choice(model_name, label=label)


def _handle_multi_distro(ctx, inputs):

    model = ctx.params.get("model_name", None)
    if not model:
        return

    replicate_flag = allows_replicate_models()
    hf_flag = allows_hf_models()

    if not replicate_flag or model not in REPLICATE_MODELS:
        ctx.params["model_distribution"] = "Transformers"
    elif not hf_flag or model not in HF_MODELS:
        ctx.params["model_distribution"] = "Replicate"
    else:
        model_distribution_choices = types.Dropdown(label="Model Distribution")
        model_distribution_choices.add_choice(
            "Transformers", label="Transformers"
        )
        model_distribution_choices.add_choice("Replicate", label="Replicate")
        inputs.enum(
            "model_distribution",
            model_distribution_choices.values(),
            default="Transformers",
            view=model_distribution_choices,
        )


def _has_question_field(ctx):
    return (
        "question" in ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
    )


def _get_potential_question_fields(ctx):
    fields = [
        k
        for k in ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
        if k != "filepath"
    ]
    if len(ctx.selected) == 1:
        sample = ctx.dataset[ctx.selected[0]]
        fields = [f for f in fields if sample[f] != None]
    return sorted(fields)


def _handle_question_direct_input(ctx, inputs):
    inputs.str("question", label="Question", required=True)


def _handle_question_input(ctx, inputs):
    fields = _get_potential_question_fields(ctx)
    if len(fields) == 0:
        _handle_question_direct_input(ctx, inputs)
        return

    from_field_str = "From Field"
    direct_input_str = "Input Directly"
    from_field_default = (
        from_field_str if _has_question_field(ctx) else direct_input_str
    )
    from_field_choices = types.RadioGroup()
    from_field_choices.add_choice(
        from_field_str, label="Use question from field"
    )
    from_field_choices.add_choice(
        direct_input_str, label="Input question directly"
    )
    inputs.enum(
        "from",
        from_field_choices.values(),
        default=from_field_default,
        view=types.TabsView(),
    )

    if ctx.params.get("from", from_field_str) == from_field_str:
        question_field_choices = types.Dropdown(label="Question Field")
        fields = _get_potential_question_fields(ctx)
        default_field = "question" if "question" in fields else fields[0]

        for field in fields:
            question_field_choices.add_choice(field, label=field)
        inputs.enum(
            "question_field",
            question_field_choices.values(),
            default=default_field,
            view=question_field_choices,
        )

        question_field = ctx.params.get("question_field", None)
        if question_field and len(ctx.selected) == 1:
            sample = ctx.dataset[ctx.selected[0]]
            inputs.message(
                "question_message",
                label=f"Question: {sample[question_field]}",
            )

    else:
        inputs.str("question", label="Question", required=True)


def _get_question_from_field(ctx):
    question_field = ctx.params.get("question_field", None)
    sample = ctx.dataset[ctx.selected[0]]
    return sample[question_field]


def _get_question_from_context(ctx):
    from_choice = ctx.params.get("from", None)
    if from_choice == "From Field":
        return _get_question_from_field(ctx)
    elif from_choice == "Input Directly":
        return ctx.params.get("question", None)

    question = ctx.params.get("question", None)
    if not question:
        question = _get_question_from_field(ctx)
    return question


def _handle_output_options(ctx, inputs):
    inputs.bool(
        "add_answer_as_field",
        label="Store answer on sample?",
        view=types.CheckboxView(),
        default=False,
    )

    if not ctx.params.get("add_answer_as_field", None):
        return

    answer_field = ctx.params.get("answer_field", "<answer-field>")

    inputs.str(
        "answer_field",
        label="Answer Field",
        description=f"Answer will be stored at sample[{answer_field}]",
        required=True,
    )


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


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

        default = (
            DEFAULT_MODEL_NAME
            if DEFAULT_MODEL_NAME in model_choices.values()
            else model_choices.values()[0]
        )

        inputs.enum(
            "model_name",
            model_choices.values(),
            label="Model",
            view=model_choices,
            required=True,
            default=default,
        )

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label=(
                        "If no samples are selected, the operator "
                        "will be applied to all samples in the current view."
                    )
                ),
            )

        if num_selected > 1:
            inputs.str(
                "too_many_samples_warning",
                view=types.Warning(
                    label=f"You must select only one sample to use this operator"
                ),
            )
            return types.Property(inputs, view=form_view)

        _handle_multi_distro(ctx, inputs)
        _handle_question_input(ctx, inputs)
        _handle_output_options(ctx, inputs)
        _execution_mode(ctx, inputs)
        return types.Property(inputs, view=form_view)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def _execute_for_single_sample(self, sample, question, model_name, ctx):
        answer = run_vqa(sample, question, model_name, ctx=ctx)
        if ctx.params.get("add_answer_as_field", None):
            answer_field = ctx.params.get("answer_field", None)
            sample[answer_field] = answer
            sample.save()
            if answer_field not in ctx.dataset.get_field_schema():
                ctx.dataset.add_dynamic_sample_fields()
            ctx.ops.reload_dataset()
        return {"question": question, "answer": answer}

    def _execute_for_sample_collection(
        self, sample_collection, question, model_name, ctx
    ):
        results = []
        answer_field = (
            ctx.params.get("answer_field", None)
            if ctx.params.get("add_answer_as_field", None)
            else None
        )

        for sample in sample_collection.iter_samples(
            autosave=True, progress=True
        ):
            answer = run_vqa(sample, question, model_name, ctx=ctx)
            if answer_field:
                sample[answer_field] = answer
            else:
                results.append(answer)

        if answer_field not in ctx.dataset.get_field_schema():
            ctx.dataset.add_dynamic_sample_fields()
        ctx.ops.reload_dataset()

        return_dict = {"question": question}
        if answer_field:
            return_dict["answer_field"] = answer_field
        return return_dict

    def execute(self, ctx):
        question = _get_question_from_context(ctx)
        model_name = ctx.params.get("model_name", DEFAULT_MODEL_NAME)

        if len(ctx.selected) == 1:
            return self._execute_for_single_sample(
                ctx.dataset[ctx.selected[0]], question, model_name, ctx
            )
        else:
            return self._execute_for_sample_collection(
                ctx.view, question, model_name, ctx
            )

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("question", label="Question")
        outputs.str("answer", label="Answer")
        header = "Visual Question Answering!"
        return types.Property(outputs, view=types.View(label=header))

    def __call__(
        self,
        sample_collection,
        model_name=DEFAULT_MODEL_NAME,
        question=None,
        question_field=None,
        answer_field=None,
        delegate=False,
        **kwargs,
    ):
        ctx = dict(view=sample_collection.view())
        params = dict(kwargs)
        params["model_name"] = model_name
        params["question"] = question
        params["question_field"] = question_field
        params["answer_field"] = answer_field
        params["delegate"] = delegate
        if answer_field:
            params["add_answer_as_field"] = True

        return foo.execute_operator(self.uri, ctx, params=params)


def register(plugin):
    plugin.register(VQA)
