from typing import Dict, Any, List

class NeuralNetworkVisualizer:
    """
    Generates Mermaid diagrams for neural network architectures and circuits.
    """

    @staticmethod
    def generate_model_architecture(model_name: str, layers: int, heads: int) -> str:
        """
        Generates a Mermaid diagram for the high-level architecture.
        """
        mermaid = ["graph TD", f"  Input[Input Tokens] --> Embed[Embedding Layer]"]
        
        last_node = "Embed"
        for i in range(layers):
            block = f"Layer{i}"
            mermaid.append(f"  {last_node} --> {block}[Transformer Block {i}]")
            mermaid.append(f"  subgraph {block}")
            mermaid.append(f"    {block}_Attn[Multi-Head Attention ({heads} heads)]")
            mermaid.append(f"    {block}_MLP[MLP Layer]")
            mermaid.append(f"    {block}_Attn --> {block}_MLP")
            mermaid.append("  end")
            last_node = f"{block}_MLP"
            
        mermaid.append(f"  {last_node} --> Unembed[Unembedding Layer] --> Output[Logits]")
        return "\n".join(mermaid)

    @staticmethod
    def generate_circuit_diagram(circuit_name: str) -> str:
        """
        Generates a Mermaid diagram for a specific discovered circuit (e.g., IOI).
        """
        if circuit_name.upper() == "IOI":
            return """
graph LR
    subgraph Layer 0-8
        PREV[Previous Token Heads]
        POS[Duplicate Token Heads]
    end

    subgraph Layer 9-10
        NM99[Name Mover L9H9]
        NM96[Name Mover L9H6]
        NM100[Name Mover L10H0]
    end

    subgraph Layer 10-11
        NEG[Negative Name Movers]
    end

    Input --> PREV
    Input --> POS
    PREV --> NM99
    PREV --> NM96
    POS --> NM99
    NM99 --> NEG
    NM100 --> Output
    NM96 --> Output
    NM99 --> Output
"""
        return "graph TD\n  Start --> End"
