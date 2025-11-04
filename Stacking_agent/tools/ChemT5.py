# Load model directly
import json
import re
import torch

def get_ChemT5_data():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    max_length = 512
    num_beams = 10

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("Stacking_agent/tools/pretrained_models/multitask-text-and-chemistry-t5-base-augm")
    model = AutoModelForSeq2SeqLM.from_pretrained("Stacking_agent/tools/pretrained_models/multitask-text-and-chemistry-t5-base-augm")
    model = model.to(device)

    import json
    with open("./Dataset/ReactionPrediction/test.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in data:
        instance = i['SMILES']
        input_text = f"Predict the product of the following reaction: {instance}"

        text = tokenizer(input_text, return_tensors="pt").to(device)
        output = model.generate(input_ids=text["input_ids"], max_length=max_length, num_beams=num_beams)
        output = tokenizer.decode(output[0].cpu())

        output = output.split(tokenizer.eos_token)[0]
        output = output.replace(tokenizer.pad_token,"")
        output = output.strip()

        i['answer'] = output

        with open("./Result/Stacking/ReactionPrediction/Text_ChemT5_test.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

# get_ChemT5_data()

def get_ChemT5(query, device=None):
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        max_length = 512
        num_beams = 10

        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained("Stacking_agent/tools/pretrained_models/multitask-text-and-chemistry-t5-base-augm")
        model = AutoModelForSeq2SeqLM.from_pretrained("Stacking_agent/tools/pretrained_models/multitask-text-and-chemistry-t5-base-augm")
        model = model.to(device)

        input_text = f"Caption the following molecule: {query}"

        text = tokenizer(input_text, return_tensors="pt").to(device)
        output = model.generate(input_ids=text["input_ids"], max_length=max_length, num_beams=num_beams)
        output = tokenizer.decode(output[0].cpu())

        output = output.split(tokenizer.eos_token)[0]
        output = output.replace(tokenizer.pad_token,"")
        output = output.strip()
        return output
    except Exception as e:
        print(f"Error in get_ChemT5: {e}")
        return f"ChemT5 model not available: {str(e)}"

# ['TextChemT5_0','SMILES2Description_2']
    
class TextChemT5:
    name: str = "TextChemT5"
    description: str = 'Input the question, returns answers. Note: Please input the SMILES representation in the form of "SMILES"'
    def __init__(
        self,
        **tool_args
    ):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TextChemT5 using device: {self.device}")

        # Lazy loading: model will be loaded on first use
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        self.num_beams = 10

        # Try to load precomputed results, but handle missing files gracefully
        self.query_data = {}
        try:
            with open("./Result/Stacking/Molecule_captioning/Text_ChemT5.json", 'r', encoding='utf-8') as f:
                data_test = json.load(f)
            with open("./Result/Stacking/Molecule_captioning/Text_ChemT5_train.json", 'r', encoding='utf-8') as f:
                data_train = json.load(f)
            data = data_test+data_train
            self.query_data = {i['SMILES']:i['answer'] for i in data}
        except FileNotFoundError:
            # If precomputed files don't exist, we'll use the model directlyzequnl/molxpt
            # print("Warning: Precomputed TextChemT5 results not found. Will use model directly.")
            self.query_data = {}

    def _load_model(self):
        """Lazy load the model on first use"""
        if self.model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print("Loading ChemT5 model...")
            self.tokenizer = AutoTokenizer.from_pretrained("Stacking_agent/tools/pretrained_models/multitask-text-and-chemistry-t5-base-augm")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("Stacking_agent/tools/pretrained_models/multitask-text-and-chemistry-t5-base-augm")
            self.model = self.model.to(self.device)
            print(f"ChemT5 model loaded on {self.device}")

    def _generate(self, query: str) -> str:
        """Generate output using the model"""
        self._load_model()

        input_text = f"Caption the following molecule: {query}"
        text = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=text["input_ids"], max_length=self.max_length, num_beams=self.num_beams)
        output = self.tokenizer.decode(output[0].cpu())

        output = output.split(self.tokenizer.eos_token)[0]
        output = output.replace(self.tokenizer.pad_token, "")
        output = output.strip()
        return output
        
    def _run(self, query: str,**tool_args) -> str:
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles]
            else:
                return self._generate(query)
        except Exception as e:
            print(f"Error in TextChemT5._run: {e}")
            return f"Error generating output: {str(e)}"
        
    def __str__(self):
        return "TextChemT5"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles],0
            else:
                return self._generate(query),0
        except Exception as e:
            print(f"Error in TextChemT5.wo_run: {e}")
            return f"Error generating output: {str(e)}",0