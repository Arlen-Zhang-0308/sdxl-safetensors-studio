import sys
from types import SimpleNamespace

from app.prompt_encoding import build_sdxl_prompt_kwargs, clip_sequence_limit


class FakeVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def numel(self) -> int:
        return len(self.values)

    def detach(self):
        return self

    def clone(self):
        return FakeVector(self.values.copy())

    def __getitem__(self, item):
        result = self.values[item]
        return FakeVector(result) if isinstance(result, list) else result

    def __iter__(self):
        return iter(self.values)


class TokenBatch:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = [FakeVector(ids)]


class FakeTokenizer:
    model_max_length = 6

    def __call__(self, text: str, **kwargs) -> TokenBatch:
        return TokenBatch(list(range(len(text.split()))))

    def decode(self, ids, **kwargs) -> str:
        return " ".join(f"t{int(value)}" for value in ids)


class FakePipe:
    tokenizer = FakeTokenizer()
    tokenizer_2 = FakeTokenizer()
    text_encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=6))
    _execution_device = "cpu"

    def __init__(self) -> None:
        self.clip_skips: list[int | None] = []

    def encode_prompt(self, prompt: str, clip_skip=None, **kwargs):
        self.clip_skips.append(clip_skip)
        token_count = len(prompt.split())
        sequence = float(token_count)
        pooled = float(token_count)
        return sequence, None, pooled, None


class FakeStack:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def mean(self, dim=0) -> float:
        return sum(self.values) / len(self.values)


sys.modules.setdefault("torch", SimpleNamespace(stack=lambda values, dim=0: FakeStack(values)))


def test_clip_limit_uses_smallest_real_limit() -> None:
    assert clip_sequence_limit(FakePipe()) == 6


def test_short_prompt_keeps_native_prompt_path() -> None:
    pipe = FakePipe()
    kwargs = build_sdxl_prompt_kwargs(pipe, "one two", "", 2)
    assert kwargs["prompt"] == "one two"
    assert kwargs["negative_prompt"] is None
    assert kwargs["clip_skip"] == 2
    assert "prompt_embeds" not in kwargs


def test_long_prompt_is_chunked_and_averaged() -> None:
    pipe = FakePipe()
    kwargs = build_sdxl_prompt_kwargs(pipe, "a b c d e f g h", "n1 n2 n3 n4 n5", 2)
    assert kwargs["prompt"] is None
    assert kwargs["negative_prompt"] is None
    assert kwargs["prompt_embeds"] == 4.0
    assert kwargs["negative_prompt_embeds"] == 2.5
    assert pipe.clip_skips == [2, 2, 2, 2]


def test_z_image_uses_native_prompt_without_clip_skip() -> None:
    ZImagePipeline = type("ZImagePipeline", (), {})

    assert build_sdxl_prompt_kwargs(ZImagePipeline(), "portrait", "blur", 2) == {
        "prompt": "portrait",
        "negative_prompt": "blur",
    }
