from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.qualcomm import target

def run():
    t = target.Target(target.SocModel.SM8750)

    aot_compile(
        "model.tflite",
        output_dir="out",
        target=[t]
    )

    print("DONE -> out/model.litertlm")

if __name__ == "__main__":
    run()
