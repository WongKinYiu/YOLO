def convert_weight(old_state_dict, new_state_dict, model_size: int = 38):
    # TODO: need to refactor
    shift = 1
    for idx in range(model_size):
        new_list, old_list = [], []
        for weight_name, weight_value in new_state_dict.items():
            if weight_name.split(".")[0] == str(idx):
                new_list.append((weight_name, None))
        for weight_name, weight_value in old_state_dict.items():
            if f"model.{idx+shift}." in weight_name:
                old_list.append((weight_name, weight_value))
        if len(new_list) == len(old_list):
            for (weight_name, _), (_, weight_value) in zip(new_list, old_list):
                new_state_dict[weight_name] = weight_value
        else:
            for weight_name, weight_value in old_list:
                if "dfl" in weight_name:
                    continue
                _, _, conv_name, conv_idx, *details = weight_name.split(".")
                if conv_name == "cv4" or conv_name == "cv5":
                    layer_idx = 22
                    shift = 2
                else:
                    layer_idx = 37

                if conv_name == "cv2" or conv_name == "cv4":
                    conv_task = "anchor_conv"
                if conv_name == "cv3" or conv_name == "cv5":
                    conv_task = "class_conv"

                weight_name = ".".join([str(layer_idx), "heads", conv_idx, conv_task, *details])
                new_state_dict[weight_name] = weight_value
    return new_state_dict


head_converter = {
    "head_conv": "m",
    "implicit_a": "ia",
    "implicit_m": "im",
}

SPP_converter = {
    "pre_conv.0": "cv1",
    "pre_conv.1": "cv3",
    "pre_conv.2": "cv4",
    "post_conv.0": "cv5",
    "post_conv.1": "cv6",
    "short_conv": "cv2",
    "merge_conv": "cv7",
}

REP_converter = {"conv1": "rbr_dense", "conv2": "rbr_1x1", "conv": "0", "bn": "1"}


def convert_weight_v7(old_state_dict, new_state_dict):
    map_weight = []
    for key_name in new_state_dict.keys():
        new_shape = new_state_dict[key_name].shape
        old_key_name = "model." + key_name
        new_key_name = key_name
        if old_key_name not in old_state_dict.keys():
            if "heads" in key_name:
                layer_idx, _, conv_idx, conv_name, *details = key_name.split(".")
                old_key_name = ".".join(["model", str(layer_idx), head_converter[conv_name], conv_idx, *details])
            elif (
                "pre_conv" in key_name
                or "post_conv" in key_name
                or "short_conv" in key_name
                or "merge_conv" in key_name
            ):
                for key, value in SPP_converter.items():
                    if key in key_name:
                        key_name = key_name.replace(key, value)
                old_key_name = "model." + key_name
            elif "conv1" in key_name or "conv2" in key_name:
                for key, value in REP_converter.items():
                    if key in key_name:
                        key_name = key_name.replace(key, value)
                old_key_name = "model." + key_name
        map_weight.append(old_key_name)
        assert old_key_name in old_state_dict.keys(), f"Weight Name Mismatch!! {old_key_name}"
        old_shape = old_state_dict[old_key_name].shape
        assert new_shape == old_shape, "Weight Shape Mismatch!! {old_key_name}"
        new_state_dict[new_key_name] = old_state_dict[old_key_name]
    return new_state_dict


replace_dict = {"cv": "conv", ".m.": ".bottleneck."}


def convert_weight_seg(old_state_dict, new_state_dict):
    diff = -1
    for old_weight_name in old_state_dict.keys():
        old_idx = int(old_weight_name.split(".")[1])
        if old_idx == 23:
            diff = 3
        elif old_idx == 41:
            diff = -19
        new_idx = old_idx + diff
        new_weight_name = old_weight_name.replace(f".{old_idx}.", f".{new_idx}.")
        for key, val in replace_dict.items():
            new_weight_name = new_weight_name.replace(key, val)

        if new_weight_name not in new_state_dict.keys():
            heads = "heads"
            _, _, conv_name, conv_idx, *details = old_weight_name.split(".")
            if "proto" in conv_name:
                conv_idx = "3"
                new_weight_name = ".".join(["model", str(layer_idx), heads, conv_task, *details])
                continue
            if "dfl" in old_weight_name:
                continue
            if conv_name == "cv2" or conv_name == "cv3" or conv_name == "cv6":
                layer_idx = 44
                heads = "detect.heads"
            if conv_name == "cv4" or conv_name == "cv5" or conv_name == "cv7":
                layer_idx = 25
                heads = "detect.heads"

            if conv_name == "cv2" or conv_name == "cv4":
                conv_task = "anchor_conv"
            if conv_name == "cv3" or conv_name == "cv5":
                conv_task = "class_conv"
            if conv_name == "cv6" or conv_name == "cv7":
                conv_task = "mask_conv"
                heads = "heads"

            new_weight_name = ".".join(["model", str(layer_idx), heads, conv_idx, conv_task, *details])

        if (
            new_weight_name not in new_state_dict.keys()
            or new_state_dict[new_weight_name].shape != old_state_dict[old_weight_name].shape
        ):
            print(f"new: {new_weight_name}, old: {old_weight_name}")
            print(f"{new_state_dict[new_weight_name].shape} {old_state_dict[old_weight_name].shape}")
        new_state_dict[new_weight_name] = old_state_dict[old_weight_name]
    return new_state_dict
