"""Experimental explicit source association for composite FP8 weights.

Not imported by runtime defaults. The loader must supply physical payload
descriptors; no checkpoint key guessing, file reads or CUDA calls occur here.
"""

from dataclasses import dataclass

from .vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


@dataclass(frozen=True)
class Float8SourceContract:
    weight_spec: object
    sources: tuple[tuple[str, SafetensorFileSlice], ...]

    @property
    def physical_bytes(self):
        return self.weight_spec.physical_bytes


def describe_float8_sources_cpu(weight, payload_sources):
    """Validate all physical sources before returning a tensor-free contract.

    Caller must establish value equivalence and loader provenance; matching
    metadata alone cannot prove file bytes equal a converted weight. CPU
    checkpoint contracts only; does not open/register sources or attach hooks.
    """
    from ddo_vmm_allocator.quantized_payload import describe_float8_weight_only_cpu

    spec = describe_float8_weight_only_cpu(weight)
    if set(payload_sources) != {item.name for item in spec.payloads}:
        raise ValueError("Both FP8 data and scale source descriptors are required")
    registry = SafetensorSliceRegistry()
    intervals = {}
    sources = []
    for payload in spec.payloads:
        source = payload_sources[payload.name]
        if not isinstance(source, SafetensorFileSlice):
            raise ValueError("Immutable safetensors payload descriptor required")
        if (type(source.offset_bytes) is not int or source.offset_bytes < 8
                or type(source.size_bytes) is not int or source.size_bytes <= 0
                or source.shape is None or source.dtype is None):
            raise ValueError("Complete physical source metadata required")
        if not source.path.is_absolute():
            raise ValueError("Resolved absolute source path required")
        if not registry.bind(getattr(weight, payload.name), source):
            raise ValueError("Source shape, dtype or byte length differs from physical payload")
        # Reject overlapping payload source ranges within a shard. This also
        # rejects supplying one descriptor twice; aliases need separate support.
        path = str(source.path.resolve())
        start, end = source.offset_bytes, source.offset_bytes + source.size_bytes
        previous = intervals.setdefault(path, [])
        if any(start < old_end and old_start < end for old_start, old_end in previous):
            raise ValueError("Overlapping source payload ranges are unsupported")
        previous.append((start, end))
        sources.append((payload.name, source))
    return Float8SourceContract(spec, tuple(sources))
