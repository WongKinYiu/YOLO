def convert_weight(old_state_dict, new_state_dict, model_size: int = 38):
    # TODO: need to refactor
    for idx in range(model_size):
        new_list, old_list = [], []
        for weight_name, weight_value in new_state_dict.items():
            if weight_name.split(".")[0] == str(idx):
                new_list.append((weight_name, None))
        for weight_name, weight_value in old_state_dict.items():
            if f"model.{idx+1}." in weight_name:
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
                    layer_idx = 38
                else:
                    layer_idx = 37

                if conv_name == "cv2" or conv_name == "cv4":
                    conv_task = "anchor_conv"
                if conv_name == "cv3" or conv_name == "cv5":
                    conv_task = "class_conv"

                weight_name = ".".join([str(layer_idx), "heads", conv_idx, conv_task, *details])
                new_state_dict[weight_name] = weight_value
    return new_state_dict
