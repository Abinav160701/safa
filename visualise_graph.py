from fashion_langgraph import assistant_graph
import matplotlib.pyplot as plt
from PIL import Image
import io

def create_graph_visualization():
    # Generate the graph
    graph_image = assistant_graph.get_graph().draw_mermaid_png()
    
    # Convert to PIL Image for display
    img = Image.open(io.BytesIO(graph_image))
    
    # Display with matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Fashion Assistant LangGraph Flow", fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save high-quality version
    plt.savefig("fashion_assistant_graph_hq.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_graph_visualization()