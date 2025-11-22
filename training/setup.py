"""
* File: ./run_environment.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Run the environment regarding the architecture number providing by the user.
"""

# Import External Libraries
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class SetUpModel:
    def __init__(self, model_name: str, fine_tuned_path: str | None = None) -> None:
        """
        Setup the tokenizer and the specific model to be trained.

        Args:
            model_name (str): The model name to be trained.
            fine_tuned_path (str | None): If we use a specific fine tuned model.

        Returns:
            None.
        """
        self.model_name: str = model_name
        self.fine_tuned_path: str | None = fine_tuned_path

        # Step 1: Set the specific device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device: torch.device = device

        # Step 2: Download and load tokenizer (will use cache if already downloaded)
        tokenizer_source = self.fine_tuned_path or self.model_name
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=True
        )
        # Set specific pad tokens --padding sequences
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Download and load model + Move the model on the GPU (will use cache if already downloaded)
        self.model: PreTrainedModel | PeftModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Step 4: Load fine-tuned adapter --if provided
        if self.fine_tuned_path is not None:
            self.model = PeftModel.from_pretrained(self.model, self.fine_tuned_path)
            print("Fine tuned model setup.")

        # Step 5: Move the model to the GPU --if available
        if self.device.type != "cpu":
            self.model.to(device)  # pyright: ignore[reportArgumentType]

        # Step 6: If training --Optimization
        if self.fine_tuned_path is None:
            self._optimization_training()

    def _optimization_training(self) -> None:
        """
        Method to setup gradient checkpointing if training mode.
        Args: None.
        Returns: None.
        """
        self.model.gradient_checkpointing_enable()  # pyright: ignore[reportCallIssue]
        self.model.config.use_cache = False  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

    def setup_lora(self) -> None:
        """
        Method to setup LoRA method with specific parameters.
        Args: None.
        Returns: None.
        """
        if isinstance(self.model, PreTrainedModel):
            model = self.model

            # Step 1: Define LoRA configuration
            lora_config = LoraConfig(
                r=128,
                lora_alpha=256,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Step 2: Apply LoRA to the model
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            self.model = model  # pyright: ignore
        else:
            print("LoRa is not available because fine_tuned_path is not None.")

    def generate(self, input_text: str) -> str:
        """
        Method to generate predictions from the model.
        Specific for Qwen/Qwen2.5-1.5B Base.

        Args:
            input_text (str): The input content.

        Returns:
            None.
        """
        # Step 1: Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)  # pyright: ignore[reportArgumentType]

        # Step 2: Generate prediction
        gen_tokens = self.model.generate(  # pyright: ignore[reportCallIssue]
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Step 3: Decode to string
        prediction = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        generated_text = prediction[len(input_text) :].strip()  # Keep only the output

        return generated_text
