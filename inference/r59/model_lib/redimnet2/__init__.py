import torch

from redimnet2.redimnet2 import ReDimNet2Custom, ReDimNet2Wrap


URL_TEMPLATE = "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{weight_name}"


def _load_state_dict_from_url(url):
    try:
        return torch.hub.load_state_dict_from_url(
            url, progress=True, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.hub.load_state_dict_from_url(
            url, progress=True, map_location="cpu")


def load_custom(model_name="b0", train_type="lm", dataset="vox2"):
    weight_name = f"{model_name}-{dataset}-{train_type}.pt"
    full_state_dict = _load_state_dict_from_url(
        URL_TEMPLATE.format(weight_name=weight_name))

    model = ReDimNet2Wrap(**full_state_dict["model_config"])
    load_res = model.load_state_dict(full_state_dict["state_dict"])
    if load_res.missing_keys or load_res.unexpected_keys:
        raise RuntimeError(
            f"Failed to load {weight_name}: "
            f"missing_keys={load_res.missing_keys}, "
            f"unexpected_keys={load_res.unexpected_keys}")
    return model


def from_pretrained(model_name="b0", train_type="lm", dataset="vox2"):
    return load_custom(model_name=model_name, train_type=train_type, dataset=dataset)


def redimnet2(model_name="b0", train_type="lm", dataset="vox2", pretrained=True):
    if pretrained:
        return load_custom(model_name=model_name, train_type=train_type, dataset=dataset)
    raise ValueError(
        "Only pretrained=True is supported. "
        "Use ReDimNet2Wrap directly for custom models.")


class ReDimNet2:
    """Namespace for backward-compatible from_pretrained access."""

    from_pretrained = staticmethod(from_pretrained)
