"""Conservative loader-name association for immutable checkpoint byte ranges.

No tensor payload reads, copies, CUDA queries, architecture names or admission
policy. Payload conversions and unknown Transformers mapping APIs are refused.
"""

from .vmm_weight_store import SafetensorSliceRegistry


def associate_source_metadata(module, available, *, key_mapping=None):
    registry = SafetensorSliceRegistry()
    summary = {"bound": 0, "available": len(available), "resolver": "exact_names",
               "rejected_storage_groups": 0, "missing_storage_groups": 0,
               "ineligible_parameter_aliases": 0}
    if getattr(module, "is_quantized", False) or getattr(module, "hf_quantizer", None) is not None:
        summary["resolver"] = "unsupported_quantized_payload"
        return registry, summary
    try:
        from transformers import PreTrainedModel
    except ImportError:
        PreTrainedModel = ()
    if isinstance(module, PreTrainedModel):
        try:
            from transformers.conversion_mapping import get_model_conversion_mapping
            from transformers.core_model_loading import WeightRenaming, rename_source_key
            transforms = get_model_conversion_mapping(module, key_mapping=key_mapping)
        except (ImportError, AttributeError, TypeError) as error:
            summary["resolver"] = "unsupported_loader_mapping"
            summary["diagnostic"] = str(error)
            return registry, summary
        if not isinstance(transforms, (list, tuple)):
            summary["resolver"] = "unsupported_loader_mapping"
            return registry, summary
        if any(not isinstance(transform, WeightRenaming) for transform in transforms):
            summary["resolver"] = "unsupported_payload_transform"
            return registry, summary
        names = {name: True for name, _ in module.named_parameters(remove_duplicate=False)}

        def resolve(name):
            return rename_source_key(name, transforms, [], module.base_model_prefix, names)[0]

        summary["resolver"] = "transformers_name_mapping"
    else:
        if key_mapping is not None:
            # Unknown loaders may implement mappings differently; never guess.
            summary["resolver"] = "unsupported_loader_mapping"
            return registry, summary
        resolve = lambda name: name

    mapped = {}
    for name, descriptor in available.items():
        mapped.setdefault(resolve(name), []).append(descriptor)
    groups = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if parameter.device.type != "cpu" or not parameter.is_contiguous() or parameter.numel() == 0:
            summary["ineligible_parameter_aliases"] += 1
            continue
        key = (parameter.data_ptr(), parameter.numel() * parameter.element_size())
        group = groups.setdefault(key, {"parameters": [], "descriptors": []})
        group["parameters"].append(parameter)
        group["descriptors"].extend(mapped.get(name, []))
    # Transactional per storage: reject conflicting source aliases/mappings,
    # including different Parameters sharing the same pointer and byte length.
    for group in groups.values():
        candidates = set(group["descriptors"])
        if not candidates:
            summary["missing_storage_groups"] += 1
            continue
        if len(candidates) != 1:
            summary["rejected_storage_groups"] += 1
            continue
        descriptor = next(iter(candidates))
        # Validate every owner before publication; a shape/dtype conflict must
        # not leave the first alias bound to a potentially wrong source.
        probe = SafetensorSliceRegistry()
        if not all(probe.bind(parameter, descriptor) for parameter in group["parameters"]):
            summary["rejected_storage_groups"] += 1
            continue
        registry.bind(group["parameters"][0], descriptor)
    summary["bound"] = registry.descriptor_count
    return registry, summary
