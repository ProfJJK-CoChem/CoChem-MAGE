# %%
import torch
from rdkit import Chem

class MageGraphBuilder:
    """
    Segment 2.2b: Topological Graph & BDE Initialization for CoChem-MAGE.
    Translates RDKit molecular topology into PyTorch-ready tensor structures
    and assigns Bond Dissociation Energy (BDE) thresholds for 70 eV EI simulation.
    """
    
    # Baseline heuristic BDE limits (in eV) for rapid topological pruning.
    # Standard 70 eV EI imparts massive excess energy, but these serve as the 
    # absolute physical floor to prevent unphysical shattering.
    BDE_THRESHOLDS_EV = {
        Chem.rdchem.BondType.SINGLE: 3.5,
        Chem.rdchem.BondType.AROMATIC: 5.0,
        Chem.rdchem.BondType.DOUBLE: 6.5,
        Chem.rdchem.BondType.TRIPLE: 8.5
    }

    def __init__(self, device=None):
        # Auto-detect CUDA for tensor mapping
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"⚙️ MAGE Graph Builder initialized. Compute device: {self.device}")

    def _estimate_edge_bde(self, bond):
        """Estimates the energy required to homolytically cleave a specific bond."""
        bond_type = bond.GetBondType()
        base_bde = self.BDE_THRESHOLDS_EV.get(bond_type, 3.0)
        
        # Implement minor thermodynamic penalties for heavy halogens
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() in [17, 35] or a2.GetAtomicNum() in [17, 35]:
            base_bde -= 0.5  # Halogen bonds are generally weaker/easier to cleave
            
        return base_bde

    def build_tensor_graph(self, mol):
        """
        Converts an RDKit Mol into a PyTorch-compatible graph dictionary.
        Extracts atomic numbers (nodes), connectivity (edges), and BDEs (edge weights).
        """
        if mol is None:
            raise ValueError("❌ MAGE Graph Error: Cannot build graph from NoneType.")

        num_nodes = mol.GetNumAtoms()
        num_edges = mol.GetNumBonds() * 2 # Undirected graph requires bidirectional edges

        # Node Features: [Atomic Number, Mass]
        node_features = torch.zeros((num_nodes, 2), dtype=torch.float32, device=self.device)
        for i, atom in enumerate(mol.GetAtoms()):
            node_features[i, 0] = atom.GetAtomicNum()
            node_features[i, 1] = atom.GetMass()

        # Edge Features: [Source, Target], [BDE Threshold]
        edge_index = torch.zeros((2, num_edges), dtype=torch.long, device=self.device)
        edge_bde = torch.zeros((num_edges,), dtype=torch.float32, device=self.device)

        idx = 0
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bde = self._estimate_edge_bde(bond)
            
            # Forward direction
            edge_index[0, idx] = start
            edge_index[1, idx] = end
            edge_bde[idx] = bde
            
            # Reverse direction
            edge_index[0, idx + 1] = end
            edge_index[1, idx + 1] = start
            edge_bde[idx + 1] = bde
            
            idx += 2

        graph_data = {
            "num_nodes": num_nodes,
            "x": node_features,
            "edge_index": edge_index,
            "edge_bde": edge_bde
        }

        return graph_data

# Execute Graph Build Test
if __name__ == "__main__":
    builder = MageGraphBuilder()
    
    # Test with Chlorobenzene
    smiles = "C1=CC=C(C=C1)Cl"
    test_mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(test_mol)
    
    print(f"\\n🧪 Building PyTorch Graph for {smiles}...")
    graph = builder.build_tensor_graph(test_mol)
    
    print(f"Nodes (Atoms): {graph['num_nodes']}")
    print(f"Edge Index Shape: {graph['edge_index'].shape}")
    print(f"Average Graph BDE: {graph['edge_bde'].mean().item():.2f} eV")
    
    # Memory footprint sanity check
    mem_bytes = graph['x'].element_size() * graph['x'].nelement() + \
                graph['edge_index'].element_size() * graph['edge_index'].nelement()
    print(f"✅ VRAM/RAM Footprint for single molecule: {mem_bytes} bytes")
# %%