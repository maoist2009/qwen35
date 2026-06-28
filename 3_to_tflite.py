from ai_edge_torch import convert

def run():
    convert(
        "model.pt2",
        output_path="model.tflite"
    )

if __name__ == "__main__":
    run()
