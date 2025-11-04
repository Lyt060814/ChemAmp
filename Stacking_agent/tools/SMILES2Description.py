import requests
from rdkit import Chem
from ..Basemodel import ChatModel


def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False

def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]

class SMILES2Description:
    name: str = "SMILES2Description"
    description: str = "Input only one molecule SMILES, returns Description. Note: the results returned by this tool may not necessarily be correct."
    def __init__(self, **tool_args):
        pass
    
    def _run(self, query: str,**tool_args) -> str:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/{}"
        # Query the PubChem database
        r = requests.get(url.format(query, "Description/JSON"))
        # Convert the response to a JSON object
        data = r.json()
        try:
            smi = data['InformationList']['Information'][1]['Description']
        except:
            return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input."
    
        return str(smi)
    
    def __str__(self):
        return "SMILES2Description"

    def __repr__(self):
        return self.__str__()

    def wo_run(self,query,debug=False):
        # Extract SMILES from query - it should be after the task description
        # Query format: "...Molecule SMILES: <SMILES>"

        # Try to find SMILES after common markers
        smiles_candidate = None
        all_tokens = 0

        # Split by common SMILES markers and take the last part
        markers = ["Molecule SMILES: ", "SMILES: ", "SMILES is ", "SMILES= ", "SMILES :"]
        for marker in markers:
            if marker in query:
                # Get the part after the marker
                parts = query.split(marker)
                if len(parts) > 1:
                    # Take everything after the marker, strip whitespace and newlines
                    after_marker = parts[-1].strip()
                    # SMILES is usually the first continuous string (no spaces)
                    # But it might span until newline or end of string
                    candidate = after_marker.split('\n')[0].split()[0] if after_marker else ""

                    # Validate if it's a valid SMILES
                    if candidate and is_smiles(candidate):
                        smiles_candidate = candidate
                        if debug:
                            print(f"[SMILES2Description] Extracted SMILES: {smiles_candidate}")
                        break

        # If no valid SMILES found, try using ChatModel as fallback
        if not smiles_candidate:
            if debug:
                print(f"[SMILES2Description] No SMILES found in query, using ChatModel fallback")
            model = ChatModel()
            prompt = "Please output only one molecule SMILES for use in generating Description based on the question:" + query
            response,all_tokens = model.chat(prompt=prompt,history=[])
            smiles_candidate = response.strip()

        answer = self._run(smiles_candidate)
        if answer == "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input.":
            return "",all_tokens
        return answer,all_tokens


