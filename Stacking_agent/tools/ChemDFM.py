# ChemDFM tool for molecule description and property prediction
import json
import re
import torch
from rdkit import Chem

def canonicalize_smiles(smiles):
    """Canonicalize SMILES string using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return smiles


def get_ChemDFM(query, device=None):
    """
    Generate molecule description or predict properties using ChemDFM model

    Args:
        query: Input text containing SMILES or questions about molecules
        device: torch device (cuda/cpu)

    Returns:
        Generated response text
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer with memory optimization
        model_path = "Stacking_agent/tools/pretrained_models/ChemDFM-v1.5-8B"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if device.type == "cuda":
            # For CUDA: Use memory-efficient settings
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                offload_folder="offload_weights",
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)

        model.eval()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Prepare input with dialogue format
        round_num = 0
        input_text = f"[Round {round_num}]\nHuman: {query}\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt")
        if device.type == "cuda":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(device)

        # Generate output
        generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode output
        result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        # Extract only the assistant's response
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()

        # Clean up memory
        del model, tokenizer, inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result.strip()

    except Exception as e:
        print(f"Error in get_ChemDFM: {e}")
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"ChemDFM model not available: {str(e)}"


class ChemDFM:
    name: str = "ChemDFM"
    description: str = 'A chemistry foundation model that can describe molecules, predict properties, and answer chemistry questions. Input can be SMILES representations or chemistry-related questions. Format: "SMILES: [your_smiles]" or direct question.'

    # Class-level shared model and tokenizer (shared across all instances)
    _shared_model = None
    _shared_tokenizer = None
    _model_device = None

    def __init__(self, **tool_args):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ChemDFM using device: {self.device}")

        # Use class-level shared model (lazy loaded on first use)
        # This allows multiple instances to share the same model in memory

        # Generation parameters
        self.max_new_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.1

        # Try to load precomputed results, but handle missing files gracefully
        self.query_data = {}
        try:
            # Check for cached results
            with open("./Result/Stacking/Molecule_captioning/ChemDFM.json", 'r', encoding='utf-8') as f:
                data_test = json.load(f)
            with open("./Result/Stacking/Molecule_captioning/ChemDFM_train.json", 'r', encoding='utf-8') as f:
                data_train = json.load(f)
            data = data_test + data_train
            self.query_data = {i['SMILES']: i['answer'] for i in data}
        except FileNotFoundError:
            # If precomputed files don't exist, we'll use the model directly
            self.query_data = {}

    def _load_model(self):
        """Lazy load the model on first use with memory optimization (shared across instances)"""
        # Check if shared model is already loaded
        if ChemDFM._shared_model is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
            print("Loading ChemDFM model (shared)...")

            model_path = "Stacking_agent/tools/pretrained_models/ChemDFM-v1.5-8B"

            ChemDFM._shared_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            if self.device.type == "cuda":
                # For CUDA: Use optimized settings for memory efficiency
                # 1. Use float16 to reduce memory
                # 2. Use device_map="auto" with offloading to CPU/disk
                # 3. Enable offload_buffers to avoid OOM
                ChemDFM._shared_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    offload_folder="offload_weights",  # Offload to disk if needed
                    offload_state_dict=True,  # Offload state dict
                    low_cpu_mem_usage=True,  # Reduce CPU memory during loading
                )
            else:
                # For CPU: Use regular loading
                ChemDFM._shared_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                ChemDFM._shared_model = ChemDFM._shared_model.to(self.device)

            ChemDFM._shared_model.eval()
            ChemDFM._model_device = self.device
            print(f"ChemDFM model loaded on {self.device} (shared)")

            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"Using already loaded ChemDFM model (shared)")

    def _generate(self, query: str) -> str:
        """Generate output using the model"""
        self._load_model()

        # Prepare input with dialogue format
        round_num = 0
        input_text = f"[Round {round_num}]\nHuman: {query}\nAssistant:"

        inputs = ChemDFM._shared_tokenizer(input_text, return_tensors="pt")
        if self.device.type == "cuda":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        # Generate output
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            pad_token_id=ChemDFM._shared_tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output = ChemDFM._shared_model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode output - use batch_decode like in DFM_app.py
        result = ChemDFM._shared_tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        # Extract only the assistant's response
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()

        # Clean up memory after generation
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result.strip()

    def _run(self, query: str, **tool_args) -> str:
        # Try to extract SMILES and canonicalize if present
        smiles = None
        if "SMILES:" in query:
            smiles = query.split("SMILES:")[-1].strip()
            smiles = canonicalize_smiles(smiles)
            # Rebuild query with canonicalized SMILES
            query = f"SMILES: {smiles}"

        try:
            # Check cache first
            if smiles and smiles in self.query_data:
                return self.query_data[smiles]
            else:
                return self._generate(query)
        except Exception as e:
            print(f"Error in ChemDFM._run: {e}")
            return f"Error generating output: {str(e)}"

    def __str__(self):
        return "ChemDFM"

    def __repr__(self):
        return self.__str__()

    def wo_run(self, query):
        # Try to extract SMILES and canonicalize if present
        smiles = None
        if "SMILES:" in query:
            smiles = query.split("SMILES:")[-1].strip()
            smiles = canonicalize_smiles(smiles)
            query = f"SMILES: {smiles}"

        try:
            # Check cache first
            if smiles and smiles in self.query_data:
                return self.query_data[smiles], 0
            else:
                return self._generate(query), 0
        except Exception as e:
            print(f"Error in ChemDFM.wo_run: {e}")
            return f"Error generating output: {str(e)}", 0
