# MolXPT tool for molecule captioning
import json
import re
import torch

def get_MolXPT(query, device=None):
    """
    Generate molecule description using MolXPT model

    Args:
        query: Input text containing SMILES representation
        device: torch device (cuda/cpu)

    Returns:
        Generated description text
    """
    try:
        from transformers import AutoTokenizer, BioGptForCausalLM

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Stacking_agent/tools/pretrained_models/molxpt",
            trust_remote_code=True
        )
        model = BioGptForCausalLM.from_pretrained(
            "Stacking_agent/tools/pretrained_models/molxpt"
        )
        model = model.to(device)
        model.eval()

        # Prepare input text with MolXPT format
        input_text = f"<start-of-mol>{query}<end-of-mol> is "

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        # Generate output with adjusted parameters
        output = model.generate(
            input_ids,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode output
        result = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the input prompt from result
        if input_text in result:
            result = result.replace(input_text, "").strip()

        # Clean up output
        result = re.sub(r'<start-of-mol>.*?<end-of-mol>', '', result)
        result = re.sub(r'<start-of-mol>[^>]*$', '', result)

        # Format result
        if not result.lower().startswith(('a ', 'an ', 'the ')):
            result = "The molecule is " + result
        else:
            if result.lower().startswith('the molecule is '):
                pass
            else:
                result = "The molecule is " + result

        # Limit length
        sentences = re.split(r'(?<=[.!?])\s+', result)
        if len(sentences) > 2:
            result = ' '.join(sentences[:2])

        # Ensure proper ending
        if result and not result.endswith(('.', '!', '?')):
            last_period = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
            if last_period > 0:
                result = result[:last_period + 1]

        return result.strip()

    except Exception as e:
        print(f"Error in get_MolXPT: {e}")
        return f"MolXPT model not available: {str(e)}"


class MolXPT:
    name: str = "MolXPT"
    description: str = 'Input a molecule SMILES representation, returns a natural language description of the molecule. Note: Please input the SMILES representation in the form of "SMILES"'

    def __init__(self, **tool_args):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MolXPT using device: {self.device}")

        # Lazy loading: model will be loaded on first use
        self.model = None
        self.tokenizer = None

        # Generation parameters optimized for standard chemical descriptions
        self.max_new_tokens = 100  # Shorter for more concise chemical descriptions
        self.temperature = 0.4  # Lower temperature for more deterministic output
        self.top_p = 0.85  # Focused sampling
        self.repetition_penalty = 1.3  # Stronger penalty for repetition
        self.no_repeat_ngram_size = 4  # Prevent 4-gram repetition

        # Try to load precomputed results, but handle missing files gracefully
        self.query_data = {}
        try:
            with open("./Result/Stacking/Molecule_captioning/MolXPT.json", 'r', encoding='utf-8') as f:
                data_test = json.load(f)
            with open("./Result/Stacking/Molecule_captioning/MolXPT_train.json", 'r', encoding='utf-8') as f:
                data_train = json.load(f)
            data = data_test + data_train
            self.query_data = {i['SMILES']: i['answer'] for i in data}
        except FileNotFoundError:
            # If precomputed files don't exist, we'll use the model directly
            self.query_data = {}

    def _load_model(self):
        """Lazy load the model on first use"""
        if self.model is None:
            from transformers import AutoTokenizer, BioGptForCausalLM
            print("Loading MolXPT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Stacking_agent/tools/pretrained_models/molxpt",
                trust_remote_code=True
            )
            self.model = BioGptForCausalLM.from_pretrained(
                "Stacking_agent/tools/pretrained_models/molxpt"
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"MolXPT model loaded on {self.device}")

    def _generate(self, query: str) -> str:
        """Generate output using the model"""
        self._load_model()

        # Extract SMILES from query
        smiles = query.split("SMILES:")[-1].strip()

        # Prepare input with MolXPT format - match training data format
        input_text = f"<start-of-mol>{smiles}<end-of-mol> is "

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Generate output with adjusted parameters for more concise output
        output = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode output
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the input prompt from result
        if input_text in result:
            result = result.replace(input_text, "").strip()

        # Clean up the output - remove any SMILES tags that might appear
        result = re.sub(r'<start-of-mol>.*?<end-of-mol>', '', result)

        # Remove incomplete SMILES strings at the end
        result = re.sub(r'<start-of-mol>[^>]*$', '', result)

        # Format result to match training data style
        if not result.lower().startswith(('a ', 'an ', 'the ')):
            result = "The molecule is " + result
        else:
            # Ensure proper "The molecule is" prefix
            if result.lower().startswith('the molecule is '):
                pass  # Already correct
            else:
                result = "The molecule is " + result

        # Clean up and limit length - keep only first 2-3 complete sentences
        sentences = re.split(r'(?<=[.!?])\s+', result)
        if len(sentences) > 2:
            result = ' '.join(sentences[:2])

        # Ensure the result ends with proper punctuation
        if result and not result.endswith(('.', '!', '?')):
            # Find the last complete sentence
            last_period = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
            if last_period > 0:
                result = result[:last_period + 1]

        return result.strip()

    def _run(self, query: str, **tool_args) -> str:
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles]
            else:
                return self._generate(query)
        except Exception as e:
            print(f"Error in MolXPT._run: {e}")
            return f"Error generating output: {str(e)}"

    def __str__(self):
        return "MolXPT"

    def __repr__(self):
        return self.__str__()

    def wo_run(self, query):
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles], 0
            else:
                return self._generate(query), 0
        except Exception as e:
            print(f"Error in MolXPT.wo_run: {e}")
            return f"Error generating output: {str(e)}", 0
