import os

os.system("python 1_load_merge.py")
os.system("python 2_export_graph.py")
os.system("python 3_to_tflite.py")
os.system("python 4_litert_compile.py")
