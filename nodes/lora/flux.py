"""
This module provides the :class:`NunchakuFluxLoraLoader` node
for applying LoRA weights to Nunchaku FLUX models within ComfyUI.
"""

import logging
import os

from nunchaku.lora.flux import to_diffusers

from ...wrappers.flux import ComfyFluxWrapper, copy_with_ctx
from ..utils import get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFluxLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku FLUX model.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types and tooltips for the node.
        """
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku FLUX DiT Loader`."
                    },
                ),
                "lora_name": (
                    get_filename_list("loras"),
                    {"tooltip": "The file name of the LoRA (used if toggle is OFF)."},
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.05,
                        "tooltip": "Default strength. Used if 'Force Node Strength' is ON, or if text input has no strength tag.",
                    },
                ),
                "use_text_input": (
                    "BOOLEAN", 
                    {
                        "default": False, 
                        "label_on": "True", 
                        "label_off": "False",
                        "tooltip": "Enable to use the 'lora_text' input instead of the dropdown."
                    }
                ),
                "use_node_strength": (
                    "BOOLEAN", 
                    {
                        "default": False, 
                        "label_on": "True", 
                        "label_off": "False",
                        "tooltip": "If True, ignores the strength value inside the text tag and forces the slider value."
                    }
                ),
            },
            "optional": {
                "lora_text": (
                    "STRING", 
                    {"forceInput": True, "tooltip": "Input string for LoRA filename or <lora:name:strength> tag."}
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku FLUX LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model. "
        "Enable 'Use Input Slot' to drive this node via text wildcards."
    )

    def load_lora(self, model, lora_name, lora_strength, use_text_input=False, use_node_strength=False, lora_text=None):
        """
        Apply a LoRA to a Nunchaku FLUX diffusion model with Auto-Discovery for subfolders.
        """
        import re
        import os

        # Variables to hold the final decision
        target_lora_name = lora_name
        target_lora_strength = lora_strength

        # --- LOGIC: Handle Text Input & Parsing ---
        if use_text_input and lora_text and isinstance(lora_text, str) and lora_text.strip():
            # 1. Regex to find <lora:name:strength>
            match = re.search(r"<lora:([^:>]+)(?::([0-9.-]+))?>", lora_text)
            
            extracted_name = ""
            if match:
                extracted_name = match.group(1)
                
                # Check for strength in text, but ONLY update if use_node_strength is False
                if match.group(2) and not use_node_strength:
                    try:
                        target_lora_strength = float(match.group(2))
                    except ValueError:
                        pass
            else:
                # Cleanup if no tag found (e.g. "lora_name, trigger")
                if "," in lora_text:
                    extracted_name = lora_text.split(",")[0].strip()
                else:
                    extracted_name = lora_text.strip()

            # --- SMART SEARCH: Find the file in subfolders ---
            # Get list of all available LoRAs (e.g. ["folder/file.safetensors", ...])
            all_loras = get_filename_list("loras")
            
            # If the exact extracted name isn't in the list, try to find it by filename base
            if extracted_name not in all_loras:
                found_path = None
                
                # FIX: Don't use os.path.splitext blindly because it kills names like "V2.5"
                # Only strip extension if it's a known model extension.
                valid_exts = (".safetensors", ".pt", ".ckpt", ".bin")
                if extracted_name.lower().endswith(valid_exts):
                    search_base = os.path.splitext(extracted_name)[0]
                else:
                    search_base = extracted_name
                
                for candidate in all_loras:
                    # candidate is the full relative path: "02-Flux/Style/File.safetensors"
                    candidate_filename = os.path.basename(candidate)
                    # We strip the extension from the candidate file on disk to compare
                    candidate_base = os.path.splitext(candidate_filename)[0]
                    
                    # specific check: if the filename matches exactly
                    if candidate_base == search_base:
                        found_path = candidate
                        break
                
                if found_path:
                    target_lora_name = found_path
                else:
                    # If not found, pass the extracted name through 
                    target_lora_name = extracted_name
                    # Ensure extension exists for the error message/fallback
                    if not target_lora_name.lower().endswith(valid_exts):
                        target_lora_name += ".safetensors"
            else:
                target_lora_name = extracted_name
        # ------------------------------------------

        if abs(target_lora_strength) < 1e-5:
            return (model,)  # If the strength is too small, return the original model

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        # Use the determined path (which now includes subfolders)
        lora_path = get_full_path_or_raise("loras", target_lora_name)

        ret_model_wrapper, ret_model = copy_with_ctx(model_wrapper)

        ret_model_wrapper.loras = [*model_wrapper.loras, (lora_path, target_lora_strength)]
        sd = to_diffusers(lora_path)

        # To handle FLUX.1 tools LoRAs, which change the number of input channels
        if "transformer.x_embedder.lora_A.weight" in sd:
            new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
            assert new_in_channels % 4 == 0
            new_in_channels = new_in_channels // 4

            old_in_channels = ret_model.model.model_config.unet_config["in_channels"]
            if old_in_channels < new_in_channels:
                ret_model.model.model_config.unet_config["in_channels"] = new_in_channels

        return (ret_model,)

class NunchakuFluxLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku FLUX model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, providing the same effect as chaining multiple LoRA nodes.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora_stack").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the LoRA stack node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and optional LoRA inputs.
        """
        # Base inputs
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRAs will be applied to. "
                        "Make sure the model is loaded by `Nunchaku FLUX DiT Loader`."
                    },
                ),
            },
            "optional": {},
        }

        # Add fixed number of LoRA inputs (15 slots)
        for i in range(1, 16):  # Support up to 15 LoRAs
            inputs["optional"][f"lora_name_{i}"] = (
                ["None"] + get_filename_list("loras"),
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["optional"][f"lora_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i}. This value can be negative.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku FLUX LoRA Stack"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "Apply multiple LoRAs to a diffusion model in a single node. "
        "Equivalent to chaining multiple LoRA nodes but more convenient for managing many LoRAs. "
        "Supports up to 15 LoRAs simultaneously. Set unused slots to 'None' to skip them."
    )

    def load_lora_stack(self, model, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku FLUX diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        **kwargs
            Dynamic LoRA name and strength parameters.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        # Collect LoRA information to apply
        loras_to_apply = []

        for i in range(1, 16):  # Check all 15 LoRA slots
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)

            # Skip unset or None LoRAs
            if lora_name is None or lora_name == "None" or lora_name == "":
                continue

            # Skip LoRAs with zero strength
            if abs(lora_strength) < 1e-5:
                continue

            loras_to_apply.append((lora_name, lora_strength))

        # If no LoRAs need to be applied, return the original model
        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        ret_model_wrapper, ret_model = copy_with_ctx(model_wrapper)

        # Clear existing LoRA list
        ret_model_wrapper.loras = []

        # Track the maximum input channels needed
        max_in_channels = ret_model.model.model_config.unet_config["in_channels"]

        # Add all LoRAs
        for lora_name, lora_strength in loras_to_apply:
            lora_path = get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))

            # Check if input channels need to be updated
            sd = to_diffusers(lora_path)
            if "transformer.x_embedder.lora_A.weight" in sd:
                new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
                assert new_in_channels % 4 == 0
                new_in_channels = new_in_channels // 4
                max_in_channels = max(max_in_channels, new_in_channels)

        # Update the model's input channels
        if max_in_channels > ret_model.model.model_config.unet_config["in_channels"]:
            ret_model.model.model_config.unet_config["in_channels"] = max_in_channels

        return (ret_model,)
