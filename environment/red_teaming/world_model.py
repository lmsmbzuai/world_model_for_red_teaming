"""
* File: ./environment/red_teaming/world_model.py
* Author: Loic Martins
* Date: 2025-11-20
* Description: Set up the world model --fine-tuned LLM.
"""

# Import External Libraries
import re
from typing import TYPE_CHECKING

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Import Local Modules
from experiment.config import ExperimentConfig

if TYPE_CHECKING:
    from environment.agents import RecommenderAgent


class WorldModel:
    def __init__(self, config: ExperimentConfig, agent: "RecommenderAgent") -> None:
        """
        Instantiate the world-model, based on a fine-tuned Qwen2.5-1.5B:
            - Set the specific device
            - Download and load tokenizer
            - Set specific pad tokens
            - Download and load model
            - Load fine-tuned adapter

        Args:
            config (ExperimentConfig): Specific configurations for the environment.
            agent (RecommenderAgent): The specific agent object with its attributes and methods.

        Returns:
            None.
        """
        self.agent: RecommenderAgent = agent
        self.model_name: str = config.world_model_name
        self.model_path: str = config.world_model_path
        self.seed: int = config.seed

        # Step 1: Set the specific device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")

        self.device: torch.device = device

        # Step 2: Download and load tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Set specific pad tokens --padding sequences
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Download and load model + Move the model on the GPU (will use cache if already downloaded)
        self.model: PreTrainedModel | PeftModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            device_map=None,
            low_cpu_mem_usage=True,
        ).to("cpu")  # pyright: ignore[reportArgumentType]

        # Step 4: Load fine-tuned adapter --if provided
        self.model = PeftModel.from_pretrained(self.model, self.model_path)

        # Step 5: Move the model to the GPU --if available
        if self.device.type != "cpu":
            self.model.to(device)  # pyright: ignore[reportArgumentType]

        print("====Step 3.3 - Red-Team Setup - World Model: Completed====")

    def _set_torch_seed(self) -> None:
        """
        Set seed across CPU, CUDA, and MPS if available.
        Args: None.
        Returns: None.
        """
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if torch.backends.mps.is_available():
            torch.manual_seed(self.seed)

    def format_input(
        self, message_content: str, top_m_actions: dict[str, dict[str, str | int]]
    ) -> list[str]:
        """
        Format the input for each specific action selected by the red-team agent (top_m):
            Instruction: {attack_objective}
            History: {history}
            Observation: {message_content}
            Action : {action}
            Next Observation:

        Args:
            message_content (str): The content of the current message --last observation.
            top_m_actions (dict[str, dict[str, str | int]]): The top m actions selected by the red-team agent.

        Returns:
            format_input (list[str]): List of messages, where 1 message is a format string for a specific action.
        """

        format_input: list[str] = []
        history = self.agent.conversation_history[:-1]
        for value in top_m_actions.values():
            format_input.append(f"""
                Instruction: {self.agent.config.attack_objective}
                History: {history}
                Observation: {message_content}
                Action : {value["action"]}
                Next Observation:
            """)
        return format_input

    def generate(self, input_text: str) -> str:
        """
        Generate content giving an input prompt:
            - Set the seed
            - Tokenize the input
            - Generate prediction
            - Decode to string
            - Clean the reponse

        Args:
            input_text (str): The specific input.

        Returns:
            generated_text (str): The specific response.
        """
        # Step 1: Set the seed
        self._set_torch_seed()

        # Step 2: Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)  # pyright: ignore[reportArgumentType]

        self.model = self.model.to(self.device)  # pyright: ignore[reportArgumentType]

        # Step 3: Generate prediction
        with torch.no_grad():
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

        # Step 4: Clean the reponse
        generated_text = generated_text.split('"PLANNER_AGENT",\n', 1)[-1].strip()
        generated_text = re.sub(r"\s*Instruction:\s*", "", generated_text)

        return generated_text
